//! MJCF XML parser for MuJoCo model files.

const std = @import("std");
const schema = @import("schema.zig");
const Scene = @import("../world/scene.zig").Scene;
const body_mod = @import("../physics/body.zig");
const joint_mod = @import("../physics/joint.zig");
const primitives = @import("../collision/primitives.zig");
const sensors = @import("../world/sensors.zig");
const constants = @import("../physics/constants.zig");

pub const ParseError = error{
    InvalidXml,
    UnexpectedElement,
    MissingAttribute,
    InvalidValue,
    FileNotFound,
    OutOfMemory,
};

/// Parse an MJCF file and return a Scene.
pub fn parseFile(allocator: std.mem.Allocator, path: []const u8) !Scene {
    const file = std.fs.cwd().openFile(path, .{}) catch {
        return ParseError.FileNotFound;
    };
    defer file.close();

    const content = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch {
        return ParseError.OutOfMemory;
    };
    defer allocator.free(content);

    // Get directory containing the file for resolving includes
    const dir = std.fs.path.dirname(path) orelse ".";

    return parseStringWithDir(allocator, content, dir);
}

/// Parse MJCF XML string and return a Scene (uses current directory for includes).
pub fn parseString(allocator: std.mem.Allocator, xml: []const u8) !Scene {
    return parseStringWithDir(allocator, xml, ".");
}

/// Parse MJCF XML string with directory context for resolving includes.
pub fn parseStringWithDir(allocator: std.mem.Allocator, xml: []const u8, model_dir: []const u8) !Scene {
    var model = try parseXmlWithIncludes(allocator, xml, model_dir);
    defer model.deinit(allocator);

    model.model_dir = model_dir;
    return convertToScene(allocator, &model);
}

/// Parse XML into MjcfModel with include file support.
fn parseXmlWithIncludes(allocator: std.mem.Allocator, xml: []const u8, model_dir: []const u8) !schema.MjcfModel {
    // First pass: process includes and merge into single XML
    const merged = try processIncludes(allocator, xml, model_dir);
    defer if (merged.ptr != xml.ptr) allocator.free(merged);

    return parseXml(allocator, merged);
}

/// Process include directives recursively.
fn processIncludes(allocator: std.mem.Allocator, xml: []const u8, model_dir: []const u8) ![]const u8 {
    // Find include directives
    var result: std.ArrayListUnmanaged(u8) = .{};
    errdefer result.deinit(allocator);

    var pos: usize = 0;
    while (pos < xml.len) {
        // Look for <include
        if (std.mem.indexOf(u8, xml[pos..], "<include")) |include_start| {
            const abs_start = pos + include_start;

            // Copy everything before the include
            try result.appendSlice(allocator, xml[pos..abs_start]);

            // Find the end of the include tag
            const tag_end = std.mem.indexOf(u8, xml[abs_start..], "/>") orelse
                std.mem.indexOf(u8, xml[abs_start..], ">") orelse {
                pos = abs_start + 8;
                continue;
            };

            const include_tag = xml[abs_start .. abs_start + tag_end + 2];

            // Extract file attribute
            if (extractAttr(include_tag, "file")) |file_path| {
                // Resolve relative path
                const full_path = try std.fs.path.join(allocator, &.{ model_dir, file_path });
                defer allocator.free(full_path);

                // Load the included file
                const included_content = loadIncludedFile(allocator, full_path) catch |err| {
                    std.debug.print("Warning: Could not include file '{s}': {}\n", .{ full_path, err });
                    pos = abs_start + tag_end + 2;
                    continue;
                };
                defer allocator.free(included_content);

                // Recursively process includes in the included file
                const included_dir = std.fs.path.dirname(full_path) orelse model_dir;
                const processed = try processIncludes(allocator, included_content, included_dir);
                defer if (processed.ptr != included_content.ptr) allocator.free(processed);

                // Strip mujoco tags from included content and append
                const stripped = stripMujocoTags(processed);
                try result.appendSlice(allocator, stripped);
            }

            pos = abs_start + tag_end + 2;
        } else {
            // No more includes, copy the rest
            try result.appendSlice(allocator, xml[pos..]);
            break;
        }
    }

    if (result.items.len == 0) {
        return xml;
    }

    return try result.toOwnedSlice(allocator);
}

/// Load content from an included file.
fn loadIncludedFile(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();

    return try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
}

/// Extract attribute value from a tag.
fn extractAttr(tag: []const u8, name: []const u8) ?[]const u8 {
    // Look for name="value" or name='value'
    var search_pos: usize = 0;
    while (std.mem.indexOf(u8, tag[search_pos..], name)) |name_pos| {
        const abs_pos = search_pos + name_pos;
        var pos = abs_pos + name.len;

        // Skip whitespace and =
        while (pos < tag.len and (tag[pos] == ' ' or tag[pos] == '=' or tag[pos] == '\t')) {
            pos += 1;
        }

        if (pos >= tag.len) break;

        const quote = tag[pos];
        if (quote != '"' and quote != '\'') {
            search_pos = pos;
            continue;
        }
        pos += 1;

        const value_start = pos;
        while (pos < tag.len and tag[pos] != quote) {
            pos += 1;
        }

        return tag[value_start..pos];
    }
    return null;
}

/// Strip outer mujoco tags from included content.
fn stripMujocoTags(content: []const u8) []const u8 {
    // Find opening <mujoco> tag
    const start_tag = std.mem.indexOf(u8, content, "<mujoco") orelse return content;
    var start = std.mem.indexOf(u8, content[start_tag..], ">") orelse return content;
    start = start_tag + start + 1;

    // Find closing </mujoco> tag
    const end = std.mem.lastIndexOf(u8, content, "</mujoco>") orelse return content;

    return content[start..end];
}

/// Parse XML into MjcfModel.
fn parseXml(allocator: std.mem.Allocator, xml: []const u8) !schema.MjcfModel {
    var model = schema.MjcfModel{};

    var tokenizer = XmlTokenizer.init(xml);

    // Find root mujoco element
    while (tokenizer.next()) |token| {
        if (token.kind == .element_start and std.mem.eql(u8, token.name, "mujoco")) {
            model.model_name = getAttr(token.attrs, "model") orelse "";
            try parseMujocoContent(&tokenizer, allocator, &model);
            break;
        }
    }

    return model;
}

fn parseMujocoContent(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, model: *schema.MjcfModel) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, "mujoco")) {
            break;
        }

        if (token.kind == .element_start) {
            if (std.mem.eql(u8, token.name, "option")) {
                model.option = parseOption(token.attrs);
            } else if (std.mem.eql(u8, token.name, "default")) {
                try parseDefaults(tokenizer, allocator, &model.defaults);
            } else if (std.mem.eql(u8, token.name, "asset")) {
                try parseAssets(tokenizer, allocator, model);
            } else if (std.mem.eql(u8, token.name, "worldbody")) {
                try parseWorldbody(tokenizer, allocator, &model.worldbody, &model.defaults);
            } else if (std.mem.eql(u8, token.name, "actuator")) {
                try parseActuators(tokenizer, allocator, &model.actuators);
            } else if (std.mem.eql(u8, token.name, "sensor")) {
                try parseSensors(tokenizer, allocator, &model.sensors);
            } else if (std.mem.eql(u8, token.name, "tendon")) {
                try parseTendons(tokenizer, allocator, &model.tendons);
            } else if (std.mem.eql(u8, token.name, "equality")) {
                try parseEqualities(tokenizer, allocator, &model.equalities);
            }
        }
    }
}

fn parseAssets(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, model: *schema.MjcfModel) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, "asset")) {
            break;
        }

        if (token.kind == .element_start) {
            if (std.mem.eql(u8, token.name, "mesh")) {
                var mesh = schema.MjcfMesh{};
                mesh.name = getAttr(token.attrs, "name") orelse "";
                mesh.file = getAttr(token.attrs, "file") orelse "";
                if (getAttr(token.attrs, "scale")) |v| {
                    mesh.scale = parseVec3(v) orelse .{ 1, 1, 1 };
                }
                try model.meshes.append(allocator, mesh);
            } else if (std.mem.eql(u8, token.name, "texture")) {
                var tex = schema.MjcfTexture{};
                tex.name = getAttr(token.attrs, "name") orelse "";
                tex.file = getAttr(token.attrs, "file") orelse "";
                if (getAttr(token.attrs, "type")) |v| {
                    tex.texture_type = schema.TextureType.fromString(v) orelse ._2d;
                }
                if (getAttr(token.attrs, "builtin")) |v| {
                    tex.builtin = schema.BuiltinTexture.fromString(v) orelse .none;
                }
                if (getAttr(token.attrs, "rgb1")) |v| {
                    tex.rgb1 = parseVec3(v) orelse .{ 0.8, 0.8, 0.8 };
                }
                if (getAttr(token.attrs, "rgb2")) |v| {
                    tex.rgb2 = parseVec3(v) orelse .{ 0.5, 0.5, 0.5 };
                }
                if (getAttr(token.attrs, "width")) |v| {
                    tex.width = parseInt(v) orelse 512;
                }
                if (getAttr(token.attrs, "height")) |v| {
                    tex.height = parseInt(v) orelse 512;
                }
                if (getAttr(token.attrs, "mark")) |v| {
                    tex.mark = v;
                }
                if (getAttr(token.attrs, "markrgb")) |v| {
                    tex.markrgb = parseVec3(v) orelse .{ 0, 0, 0 };
                }
                if (getAttr(token.attrs, "random")) |v| {
                    tex.random = parseFloat(v) orelse 0.01;
                }
                try model.textures.append(allocator, tex);
            } else if (std.mem.eql(u8, token.name, "material")) {
                var mat = schema.MjcfMaterial{};
                mat.name = getAttr(token.attrs, "name") orelse "";
                mat.texture = getAttr(token.attrs, "texture") orelse "";
                if (getAttr(token.attrs, "texrepeat")) |v| {
                    mat.texrepeat = parseVec2(v) orelse .{ 1, 1 };
                }
                if (getAttr(token.attrs, "texuniform")) |v| {
                    mat.texuniform = std.mem.eql(u8, v, "true") or std.mem.eql(u8, v, "1");
                }
                if (getAttr(token.attrs, "emission")) |v| {
                    mat.emission = parseFloat(v) orelse 0;
                }
                if (getAttr(token.attrs, "specular")) |v| {
                    mat.specular = parseFloat(v) orelse 0.5;
                }
                if (getAttr(token.attrs, "shininess")) |v| {
                    mat.shininess = parseFloat(v) orelse 0.5;
                }
                if (getAttr(token.attrs, "reflectance")) |v| {
                    mat.reflectance = parseFloat(v) orelse 0;
                }
                if (getAttr(token.attrs, "rgba")) |v| {
                    mat.rgba = parseRgba(v) orelse .{ 1, 1, 1, 1 };
                }
                try model.materials.append(allocator, mat);
            }
        }
    }
}

fn parseOption(attrs: []const u8) schema.MjcfOption {
    var option = schema.MjcfOption{};

    if (getAttr(attrs, "timestep")) |v| {
        option.timestep = parseFloat(v) orelse 0.002;
    }

    if (getAttr(attrs, "gravity")) |v| {
        option.gravity = parseVec3(v) orelse .{ 0, 0, -9.81 };
    }

    return option;
}

fn parseWorldbody(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, worldbody: *schema.MjcfBody, defaults: *const std.StringHashMapUnmanaged(schema.MjcfDefault)) !void {
    try parseBodyContent(tokenizer, allocator, worldbody, "worldbody", defaults, "main");
}

fn parseBodyContent(
    tokenizer: *XmlTokenizer,
    allocator: std.mem.Allocator,
    body: *schema.MjcfBody,
    end_tag: []const u8,
    defaults: *const std.StringHashMapUnmanaged(schema.MjcfDefault),
    parent_class: []const u8,
) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, end_tag)) {
            break;
        }

        if (token.kind == .element_start) {
            if (std.mem.eql(u8, token.name, "body")) {
                var child = schema.MjcfBody{};
                child.name = getAttr(token.attrs, "name") orelse "";
                if (getAttr(token.attrs, "pos")) |v| {
                    child.pos = parseVec3(v) orelse .{ 0, 0, 0 };
                }
                if (getAttr(token.attrs, "quat")) |v| {
                    child.quat = parseQuat(v) orelse .{ 1, 0, 0, 0 };
                }
                if (getAttr(token.attrs, "euler")) |v| {
                    child.euler = parseVec3(v);
                }
                // Check for childclass attribute
                const child_class = getAttr(token.attrs, "childclass") orelse parent_class;
                child.childclass = child_class;
                try parseBodyContent(tokenizer, allocator, &child, "body", defaults, child_class);
                try body.children.append(allocator, child);
            } else if (std.mem.eql(u8, token.name, "joint")) {
                const joint = parseJointWithDefaults(token.attrs, defaults, parent_class);
                try body.joints.append(allocator, joint);
            } else if (std.mem.eql(u8, token.name, "geom")) {
                const geom = parseGeomWithDefaults(token.attrs, defaults, parent_class);
                try body.geoms.append(allocator, geom);
            } else if (std.mem.eql(u8, token.name, "site")) {
                const site = parseSite(token.attrs);
                try body.sites.append(allocator, site);
            }
        }
    }
}

fn parseJoint(attrs: []const u8) schema.MjcfJoint {
    var joint = schema.MjcfJoint{};
    return applyJointAttrs(&joint, attrs);
}

fn parseJointWithDefaults(attrs: []const u8, defaults: *const std.StringHashMapUnmanaged(schema.MjcfDefault), class: []const u8) schema.MjcfJoint {
    // First check for explicit class attribute
    const use_class = getAttr(attrs, "class") orelse class;

    // Start with defaults if available
    var joint = if (defaults.get(use_class)) |def| def.joint else schema.MjcfJoint{};

    // Apply attributes from the element (overriding defaults)
    return applyJointAttrs(&joint, attrs);
}

fn applyJointAttrs(joint: *schema.MjcfJoint, attrs: []const u8) schema.MjcfJoint {
    joint.name = getAttr(attrs, "name") orelse joint.name;

    if (getAttr(attrs, "type")) |v| {
        joint.joint_type = schema.JointType.fromString(v) orelse joint.joint_type;
    }

    if (getAttr(attrs, "pos")) |v| {
        joint.pos = parseVec3(v) orelse joint.pos;
    }

    if (getAttr(attrs, "axis")) |v| {
        joint.axis = parseVec3(v) orelse joint.axis;
    }

    if (getAttr(attrs, "range")) |v| {
        joint.range = parseVec2(v);
        joint.limited = joint.range != null;
    }

    if (getAttr(attrs, "damping")) |v| {
        joint.damping = parseFloat(v) orelse joint.damping;
    }

    if (getAttr(attrs, "stiffness")) |v| {
        joint.stiffness = parseFloat(v) orelse joint.stiffness;
    }

    if (getAttr(attrs, "armature")) |v| {
        joint.armature = parseFloat(v) orelse joint.armature;
    }

    if (getAttr(attrs, "ref")) |v| {
        joint.ref = parseFloat(v) orelse joint.ref;
    }

    return joint.*;
}

fn parseGeom(attrs: []const u8) schema.MjcfGeom {
    var geom = schema.MjcfGeom{};
    return applyGeomAttrs(&geom, attrs);
}

fn parseGeomWithDefaults(attrs: []const u8, defaults: *const std.StringHashMapUnmanaged(schema.MjcfDefault), class: []const u8) schema.MjcfGeom {
    // First check for explicit class attribute
    const use_class = getAttr(attrs, "class") orelse class;

    // Start with defaults if available
    var geom = if (defaults.get(use_class)) |def| def.geom else schema.MjcfGeom{};

    // Apply attributes from the element (overriding defaults)
    return applyGeomAttrs(&geom, attrs);
}

fn applyGeomAttrs(geom: *schema.MjcfGeom, attrs: []const u8) schema.MjcfGeom {
    geom.name = getAttr(attrs, "name") orelse geom.name;

    if (getAttr(attrs, "type")) |v| {
        geom.geom_type = schema.GeomType.fromString(v) orelse geom.geom_type;
    }

    if (getAttr(attrs, "pos")) |v| {
        geom.pos = parseVec3(v) orelse geom.pos;
    }

    if (getAttr(attrs, "quat")) |v| {
        geom.quat = parseQuat(v) orelse geom.quat;
    }

    if (getAttr(attrs, "size")) |v| {
        geom.size = parseSize(v);
    }

    if (getAttr(attrs, "fromto")) |v| {
        geom.fromto = parseVec6(v);
    }

    if (getAttr(attrs, "mass")) |v| {
        geom.mass = parseFloat(v);
    }

    if (getAttr(attrs, "density")) |v| {
        geom.density = parseFloat(v) orelse geom.density;
    }

    if (getAttr(attrs, "friction")) |v| {
        geom.friction = parseVec3(v) orelse geom.friction;
    }

    if (getAttr(attrs, "rgba")) |v| {
        geom.rgba = parseRgba(v) orelse geom.rgba;
    }

    if (getAttr(attrs, "contype")) |v| {
        geom.contype = parseInt(v) orelse geom.contype;
    }

    if (getAttr(attrs, "conaffinity")) |v| {
        geom.conaffinity = parseInt(v) orelse geom.conaffinity;
    }

    if (getAttr(attrs, "mesh")) |v| {
        geom.mesh = v;
        geom.geom_type = .mesh; // Auto-set type when mesh is specified
    }

    if (getAttr(attrs, "material")) |v| {
        geom.material = v;
    }

    return geom.*;
}

fn parseSite(attrs: []const u8) schema.MjcfSite {
    var site = schema.MjcfSite{};
    site.name = getAttr(attrs, "name") orelse "";
    if (getAttr(attrs, "pos")) |v| {
        site.pos = parseVec3(v) orelse .{ 0, 0, 0 };
    }
    return site;
}

fn parseActuators(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, actuators: *std.ArrayListUnmanaged(schema.MjcfActuator)) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, "actuator")) {
            break;
        }

        if (token.kind == .element_start) {
            if (std.mem.eql(u8, token.name, "motor") or
                std.mem.eql(u8, token.name, "position") or
                std.mem.eql(u8, token.name, "velocity") or
                std.mem.eql(u8, token.name, "general"))
            {
                const actuator = parseActuator(token.name, token.attrs);
                try actuators.append(allocator, actuator);
            }
        }
    }
}

fn parseActuator(tag: []const u8, attrs: []const u8) schema.MjcfActuator {
    var actuator = schema.MjcfActuator{};

    actuator.name = getAttr(attrs, "name") orelse "";
    actuator.actuator_type = schema.ActuatorType.fromString(tag) orelse .motor;
    actuator.joint = getAttr(attrs, "joint") orelse "";

    if (getAttr(attrs, "ctrlrange")) |v| {
        actuator.ctrlrange = parseVec2(v);
        actuator.ctrllimited = actuator.ctrlrange != null;
    }

    if (getAttr(attrs, "forcerange")) |v| {
        actuator.forcerange = parseVec2(v);
    }

    if (getAttr(attrs, "gear")) |v| {
        actuator.gear = parseFloat(v) orelse 1;
    }

    if (getAttr(attrs, "kp")) |v| {
        actuator.kp = parseFloat(v) orelse 0;
    }

    if (getAttr(attrs, "kv")) |v| {
        actuator.kv = parseFloat(v) orelse 0;
    }

    return actuator;
}

fn parseSensors(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, sensor_list: *std.ArrayListUnmanaged(schema.MjcfSensor)) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, "sensor")) {
            break;
        }

        if (token.kind == .element_start) {
            var sensor = schema.MjcfSensor{};
            sensor.name = getAttr(token.attrs, "name") orelse "";
            sensor.sensor_type = schema.SensorType.fromString(token.name) orelse .jointpos;
            sensor.joint = getAttr(token.attrs, "joint") orelse "";
            sensor.site = getAttr(token.attrs, "site") orelse "";

            if (getAttr(token.attrs, "noise")) |v| {
                sensor.noise = parseFloat(v) orelse 0;
            }

            try sensor_list.append(allocator, sensor);
        }
    }
}

fn parseTendons(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, tendon_list: *std.ArrayListUnmanaged(schema.MjcfTendon)) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, "tendon")) {
            break;
        }

        if (token.kind == .element_start) {
            if (std.mem.eql(u8, token.name, "fixed") or std.mem.eql(u8, token.name, "spatial")) {
                var tendon = schema.MjcfTendon{};
                tendon.name = getAttr(token.attrs, "name") orelse "";
                tendon.tendon_type = schema.TendonType.fromString(token.name) orelse .fixed;

                if (getAttr(token.attrs, "stiffness")) |v| {
                    tendon.stiffness = parseFloat(v) orelse 0;
                }
                if (getAttr(token.attrs, "damping")) |v| {
                    tendon.damping = parseFloat(v) orelse 0;
                }
                if (getAttr(token.attrs, "range")) |v| {
                    if (parseVec2(v)) |range| {
                        tendon.range_lower = range[0];
                        tendon.range_upper = range[1];
                        tendon.limited = true;
                    }
                }
                if (getAttr(token.attrs, "width")) |v| {
                    tendon.width = parseFloat(v) orelse 0.003;
                }
                if (getAttr(token.attrs, "rgba")) |v| {
                    tendon.rgba = parseRgba(v) orelse tendon.rgba;
                }
                if (getAttr(token.attrs, "margin")) |v| {
                    tendon.margin = parseFloat(v) orelse 0;
                }

                // Parse path elements (joint or site references)
                try parseTendonPath(tokenizer, allocator, &tendon, token.name);
                try tendon_list.append(allocator, tendon);
            }
        }
    }
}

fn parseTendonPath(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, tendon: *schema.MjcfTendon, end_tag: []const u8) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, end_tag)) {
            break;
        }

        if (token.kind == .element_start) {
            if (std.mem.eql(u8, token.name, "joint")) {
                var path_elem = schema.MjcfTendonPath{};
                path_elem.joint = getAttr(token.attrs, "joint") orelse "";
                if (getAttr(token.attrs, "coef")) |v| {
                    path_elem.coef = parseFloat(v) orelse 1.0;
                }
                try tendon.path.append(allocator, path_elem);
            } else if (std.mem.eql(u8, token.name, "site")) {
                var path_elem = schema.MjcfTendonPath{};
                path_elem.site = getAttr(token.attrs, "site") orelse "";
                try tendon.path.append(allocator, path_elem);
            } else if (std.mem.eql(u8, token.name, "geom")) {
                // Wrapping object (for spatial tendons)
                var path_elem = schema.MjcfTendonPath{};
                path_elem.site = getAttr(token.attrs, "geom") orelse "";
                try tendon.path.append(allocator, path_elem);
            }
        }
    }
}

fn parseEqualities(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, equality_list: *std.ArrayListUnmanaged(schema.MjcfEquality)) !void {
    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, "equality")) {
            break;
        }

        if (token.kind == .element_start) {
            var equality = schema.MjcfEquality{};
            equality.name = getAttr(token.attrs, "name") orelse "";
            equality.equality_type = schema.EqualityType.fromString(token.name) orelse .connect;

            // Body references
            equality.body1 = getAttr(token.attrs, "body1") orelse "";
            equality.body2 = getAttr(token.attrs, "body2") orelse "";

            // Joint references
            equality.joint1 = getAttr(token.attrs, "joint1") orelse "";
            equality.joint2 = getAttr(token.attrs, "joint2") orelse "";

            // Tendon reference
            equality.tendon = getAttr(token.attrs, "tendon") orelse
                getAttr(token.attrs, "tendon1") orelse "";

            // Anchor point
            if (getAttr(token.attrs, "anchor")) |v| {
                equality.anchor = parseVec3(v) orelse .{ 0, 0, 0 };
            }

            // Relative pose (for weld)
            if (getAttr(token.attrs, "relpose")) |v| {
                equality.relpose = parseRelpose(v);
            }

            // Polynomial coefficients (for joint constraints)
            if (getAttr(token.attrs, "polycoef")) |v| {
                equality.polycoef = parsePolycoef(v);
            }

            // Active state
            if (getAttr(token.attrs, "active")) |v| {
                equality.active = std.mem.eql(u8, v, "true") or std.mem.eql(u8, v, "1");
            }

            // Solver parameters
            if (getAttr(token.attrs, "solimp")) |v| {
                equality.solimp = parseSolimp(v);
            }
            if (getAttr(token.attrs, "solref")) |v| {
                equality.solref = parseSolref(v);
            }

            try equality_list.append(allocator, equality);
        }
    }
}

fn parseRelpose(s: []const u8) [7]f32 {
    var result: [7]f32 = .{ 0, 0, 0, 1, 0, 0, 0 };
    var iter = std.mem.tokenizeAny(u8, s, " \t\n");
    var i: usize = 0;
    while (iter.next()) |tok| : (i += 1) {
        if (i >= 7) break;
        result[i] = parseFloat(tok) orelse result[i];
    }
    return result;
}

fn parsePolycoef(s: []const u8) [5]f32 {
    var result: [5]f32 = .{ 0, 1, 0, 0, 0 };
    var iter = std.mem.tokenizeAny(u8, s, " \t\n");
    var i: usize = 0;
    while (iter.next()) |tok| : (i += 1) {
        if (i >= 5) break;
        result[i] = parseFloat(tok) orelse result[i];
    }
    return result;
}

fn parseSolimp(s: []const u8) [5]f32 {
    var result: [5]f32 = .{ 0.9, 0.95, 0.001, 0.5, 2 };
    var iter = std.mem.tokenizeAny(u8, s, " \t\n");
    var i: usize = 0;
    while (iter.next()) |tok| : (i += 1) {
        if (i >= 5) break;
        result[i] = parseFloat(tok) orelse result[i];
    }
    return result;
}

fn parseSolref(s: []const u8) [2]f32 {
    var result: [2]f32 = .{ 0.02, 1 };
    var iter = std.mem.tokenizeAny(u8, s, " \t\n");
    var i: usize = 0;
    while (iter.next()) |tok| : (i += 1) {
        if (i >= 2) break;
        result[i] = parseFloat(tok) orelse result[i];
    }
    return result;
}

fn parseDefaults(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, defaults: *std.StringHashMapUnmanaged(schema.MjcfDefault)) !void {
    // Parse the defaults section, supporting nested default classes
    var current_class: []const u8 = "main";
    var current_default = schema.MjcfDefault{ .class_name = current_class };

    while (tokenizer.next()) |token| {
        if (token.kind == .element_end and std.mem.eql(u8, token.name, "default")) {
            // Save the current class before exiting
            if (current_class.len > 0) {
                try defaults.put(allocator, current_class, current_default);
            }
            break;
        }

        if (token.kind == .element_start) {
            if (std.mem.eql(u8, token.name, "default")) {
                // Save current class and start new nested class
                if (current_class.len > 0) {
                    try defaults.put(allocator, current_class, current_default);
                }
                // Get class name from attribute
                current_class = getAttr(token.attrs, "class") orelse "main";
                // Inherit from current defaults or start fresh
                current_default = schema.MjcfDefault{ .class_name = current_class };
            } else if (std.mem.eql(u8, token.name, "geom")) {
                // Parse geom defaults
                current_default.geom = parseGeom(token.attrs);
            } else if (std.mem.eql(u8, token.name, "joint")) {
                // Parse joint defaults
                current_default.joint = parseJoint(token.attrs);
            } else if (std.mem.eql(u8, token.name, "site")) {
                // Parse site defaults
                current_default.site = parseSite(token.attrs);
            } else if (std.mem.eql(u8, token.name, "motor") or
                std.mem.eql(u8, token.name, "position") or
                std.mem.eql(u8, token.name, "velocity") or
                std.mem.eql(u8, token.name, "general"))
            {
                // Parse actuator defaults
                current_default.motor = parseActuator(token.name, token.attrs);
            }
        }
    }
}

/// Convert parsed MJCF model to Scene.
fn convertToScene(allocator: std.mem.Allocator, model: *const schema.MjcfModel) !Scene {
    var scene = Scene.init(allocator);

    // Set physics config from options
    scene.physics_config.timestep = model.option.timestep;
    scene.physics_config.gravity = model.option.gravity;

    // Add ground plane (worldbody geoms)
    for (model.worldbody.geoms.items) |mjcf_geom| {
        var geom = convertGeom(&mjcf_geom);
        geom.body_id = 0; // World body
        _ = try scene.addNamedGeom(mjcf_geom.name, geom);
    }

    // Process body hierarchy
    var body_index: u32 = 0;
    try processBody(allocator, &scene, &model.worldbody, -1, &body_index, .{ 0, 0, 0 }, .{ 0, 0, 0, 1 });

    // Add actuators
    for (model.actuators.items) |mjcf_actuator| {
        const joint_id = scene.getJointByName(mjcf_actuator.joint) orelse continue;

        var actuator = joint_mod.ActuatorDef{
            .name = mjcf_actuator.name,
            .joint = joint_id,
            .gear = mjcf_actuator.gear,
            .kp = mjcf_actuator.kp,
            .kv = mjcf_actuator.kv,
        };

        if (mjcf_actuator.ctrlrange) |range| {
            actuator.ctrl_min = range[0];
            actuator.ctrl_max = range[1];
        }

        if (mjcf_actuator.forcerange) |range| {
            actuator.force_min = range[0];
            actuator.force_max = range[1];
        }

        actuator.actuator_type = switch (mjcf_actuator.actuator_type) {
            .motor, .general => .motor,
            .position => .position,
            .velocity => .velocity,
        };

        _ = try scene.addActuator(actuator);
    }

    // Add sensors
    for (model.sensors.items) |mjcf_sensor| {
        const sensor = convertSensor(&mjcf_sensor, &scene);
        _ = try scene.addSensor(sensor);
    }

    // Auto-add standard sensors if none specified
    if (scene.sensor_config.count() == 0) {
        try scene.autoAddSensors();
    }

    return scene;
}

fn processBody(
    allocator: std.mem.Allocator,
    scene: *Scene,
    mjcf_body: *const schema.MjcfBody,
    parent_id: i32,
    body_index: *u32,
    parent_pos: [3]f32,
    parent_quat: [4]f32,
) !void {
    // Skip worldbody itself, but process its children
    for (mjcf_body.children.items) |*child| {
        // Compute world position
        var pos = child.pos;
        pos[0] += parent_pos[0];
        pos[1] += parent_pos[1];
        pos[2] += parent_pos[2];

        // Handle orientation - euler angles return XYZW, quat from MJCF is WXYZ
        const body_quat: [4]f32 = if (child.euler) |euler|
            eulerToQuat(euler) // Already returns XYZW
        else
            mjcfQuatToZeno(child.quat); // Convert MJCF WXYZ to XYZW

        // Combine with parent quaternion (simplified - would need proper quat multiply)
        _ = parent_quat;

        // Determine body type
        var body_type: body_mod.BodyType = .dynamic;
        for (child.joints.items) |_| {
            // Has joints, so it's dynamic
            break;
        } else {
            // No joints and has parent = fixed to parent
            if (parent_id >= 0) {
                body_type = .kinematic;
            }
        }

        // Add body
        const current_body_id = try scene.addBody(.{
            .name = child.name,
            .position = pos,
            .quaternion = body_quat,
            .parent_id = parent_id,
            .body_type = body_type,
        });
        body_index.* += 1;

        // Add joints
        for (child.joints.items) |mjcf_joint| {
            var joint_def = joint_mod.JointDef{
                .name = mjcf_joint.name,
                .parent_body = if (parent_id >= 0) @intCast(parent_id) else 0,
                .child_body = current_body_id,
                .anchor_parent = mjcf_joint.pos,
                .anchor_child = .{ 0, 0, 0 },
                .axis = mjcf_joint.axis,
                .damping = mjcf_joint.damping,
                .stiffness = mjcf_joint.stiffness,
                .armature = mjcf_joint.armature,
                .ref_position = mjcf_joint.ref,
            };

            joint_def.joint_type = switch (mjcf_joint.joint_type) {
                .free => .free,
                .ball => .ball,
                .slide => .prismatic,
                .hinge => .revolute,
            };

            if (mjcf_joint.range) |range| {
                joint_def.limit_lower = range[0];
                joint_def.limit_upper = range[1];
                joint_def.limited = true;
            }

            _ = try scene.addJoint(joint_def);
        }

        // Add geoms
        for (child.geoms.items) |mjcf_geom| {
            var geom = convertGeom(&mjcf_geom);
            geom.body_id = current_body_id;
            _ = try scene.addNamedGeom(mjcf_geom.name, geom);

            // Update body mass from geom
            if (mjcf_geom.mass) |mass| {
                scene.bodies.items[current_body_id].mass += mass;
            }
        }

        // Process children recursively
        try processBody(allocator, scene, child, @intCast(current_body_id), body_index, pos, body_quat);
    }
}

fn convertGeom(mjcf_geom: *const schema.MjcfGeom) primitives.Geom {
    var geom = primitives.Geom{
        .local_pos = mjcf_geom.pos,
        .local_quat = mjcfQuatToZeno(mjcf_geom.quat),
        .friction = mjcf_geom.friction[0],
        .group = mjcf_geom.contype,
        .mask = mjcf_geom.conaffinity,
    };

    switch (mjcf_geom.geom_type) {
        .sphere => {
            geom.geom_type = .sphere;
            geom.size = .{ mjcf_geom.size[0], 0, 0 };
        },
        .capsule => {
            geom.geom_type = .capsule;
            if (mjcf_geom.fromto) |fromto| {
                // Calculate from fromto
                const p1: [3]f32 = .{ fromto[0], fromto[1], fromto[2] };
                const p2: [3]f32 = .{ fromto[3], fromto[4], fromto[5] };
                const dx = p2[0] - p1[0];
                const dy = p2[1] - p1[1];
                const dz = p2[2] - p1[2];
                const length = @sqrt(dx * dx + dy * dy + dz * dz);

                geom.local_pos = .{
                    (p1[0] + p2[0]) * 0.5,
                    (p1[1] + p2[1]) * 0.5,
                    (p1[2] + p2[2]) * 0.5,
                };
                geom.size = .{ mjcf_geom.size[0], length * 0.5, 0 };
            } else {
                geom.size = .{ mjcf_geom.size[0], mjcf_geom.size[1], 0 };
            }
        },
        .box => {
            geom.geom_type = .box;
            geom.size = mjcf_geom.size;
        },
        .cylinder => {
            geom.geom_type = .cylinder;
            geom.size = .{ mjcf_geom.size[0], mjcf_geom.size[1], 0 };
        },
        .plane => {
            geom.geom_type = .plane;
            geom.size = .{ 0, 0, 0 };
        },
        .mesh => {
            // Not supported, use sphere approximation
            geom.geom_type = .sphere;
            geom.size = .{ 0.1, 0, 0 };
        },
    }

    return geom;
}

fn convertSensor(mjcf_sensor: *const schema.MjcfSensor, scene: *const Scene) sensors.Sensor {
    var sensor = sensors.Sensor{
        .name = mjcf_sensor.name,
        .noise = mjcf_sensor.noise,
        .cutoff = mjcf_sensor.cutoff,
    };

    switch (mjcf_sensor.sensor_type) {
        .jointpos => {
            sensor.sensor_type = .joint_pos;
            sensor.object_id = scene.getJointByName(mjcf_sensor.joint) orelse 0;
            sensor.dim = 1;
        },
        .jointvel => {
            sensor.sensor_type = .joint_vel;
            sensor.object_id = scene.getJointByName(mjcf_sensor.joint) orelse 0;
            sensor.dim = 1;
        },
        .accelerometer => {
            sensor.sensor_type = .accelerometer;
            sensor.dim = 3;
        },
        .gyro => {
            sensor.sensor_type = .gyro;
            sensor.dim = 3;
        },
        .touch => {
            sensor.sensor_type = .touch;
            sensor.dim = 1;
        },
        .force => {
            sensor.sensor_type = .force;
            sensor.dim = 3;
        },
        .torque => {
            sensor.sensor_type = .torque;
            sensor.dim = 3;
        },
        .framepos => {
            sensor.sensor_type = .framepos;
            sensor.dim = 3;
        },
        .framequat => {
            sensor.sensor_type = .framequat;
            sensor.dim = 4;
        },
        .framelinvel => {
            sensor.sensor_type = .framelinvel;
            sensor.dim = 3;
        },
        .frameangvel => {
            sensor.sensor_type = .frameangvel;
            sensor.dim = 3;
        },
        .subtreecom => {
            sensor.sensor_type = .subtreecom;
            sensor.dim = 3;
        },
    }

    return sensor;
}

// MuJoCo uses wxyz quaternion, we use xyzw
fn mjcfQuatToZeno(mjcf_quat: [4]f32) [4]f32 {
    return .{ mjcf_quat[1], mjcf_quat[2], mjcf_quat[3], mjcf_quat[0] };
}

fn eulerToQuat(euler: [3]f32) [4]f32 {
    const cr = @cos(euler[0] * 0.5);
    const sr = @sin(euler[0] * 0.5);
    const cp = @cos(euler[1] * 0.5);
    const sp = @sin(euler[1] * 0.5);
    const cy = @cos(euler[2] * 0.5);
    const sy = @sin(euler[2] * 0.5);

    return .{
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    };
}

// Simple XML tokenizer

const XmlToken = struct {
    kind: enum { element_start, element_end, text, comment },
    name: []const u8,
    attrs: []const u8,
};

const XmlTokenizer = struct {
    source: []const u8,
    pos: usize,

    pub fn init(source: []const u8) XmlTokenizer {
        return .{ .source = source, .pos = 0 };
    }

    pub fn next(self: *XmlTokenizer) ?XmlToken {
        self.skipWhitespace();

        if (self.pos >= self.source.len) return null;

        if (self.source[self.pos] == '<') {
            return self.parseTag();
        } else {
            return self.parseText();
        }
    }

    fn parseTag(self: *XmlTokenizer) ?XmlToken {
        self.pos += 1; // Skip '<'

        if (self.pos >= self.source.len) return null;

        // Check for comment
        if (self.remaining().len >= 3 and std.mem.eql(u8, self.remaining()[0..3], "!--")) {
            return self.parseComment();
        }

        // Check for closing tag
        const is_end = self.source[self.pos] == '/';
        if (is_end) self.pos += 1;

        // Parse tag name
        const name_start = self.pos;
        while (self.pos < self.source.len and !isWhitespace(self.source[self.pos]) and
            self.source[self.pos] != '>' and self.source[self.pos] != '/')
        {
            self.pos += 1;
        }
        const name = self.source[name_start..self.pos];

        // Parse attributes
        self.skipWhitespace();
        const attrs_start = self.pos;

        // Find end of tag
        var is_self_closing = false;
        while (self.pos < self.source.len and self.source[self.pos] != '>') {
            if (self.source[self.pos] == '/') {
                is_self_closing = true;
            }
            self.pos += 1;
        }

        const attrs_end = if (is_self_closing) self.pos - 1 else self.pos;
        const attrs = self.source[attrs_start..attrs_end];

        if (self.pos < self.source.len) self.pos += 1; // Skip '>'

        if (is_end) {
            return XmlToken{ .kind = .element_end, .name = name, .attrs = "" };
        } else if (is_self_closing) {
            // Return as start then immediately return end
            return XmlToken{ .kind = .element_start, .name = name, .attrs = attrs };
        } else {
            return XmlToken{ .kind = .element_start, .name = name, .attrs = attrs };
        }
    }

    fn parseComment(self: *XmlTokenizer) ?XmlToken {
        // Skip until -->
        while (self.pos + 2 < self.source.len) {
            if (self.source[self.pos] == '-' and self.source[self.pos + 1] == '-' and self.source[self.pos + 2] == '>') {
                self.pos += 3;
                break;
            }
            self.pos += 1;
        }
        return XmlToken{ .kind = .comment, .name = "", .attrs = "" };
    }

    fn parseText(self: *XmlTokenizer) ?XmlToken {
        const start = self.pos;
        while (self.pos < self.source.len and self.source[self.pos] != '<') {
            self.pos += 1;
        }
        return XmlToken{ .kind = .text, .name = self.source[start..self.pos], .attrs = "" };
    }

    fn skipWhitespace(self: *XmlTokenizer) void {
        while (self.pos < self.source.len and isWhitespace(self.source[self.pos])) {
            self.pos += 1;
        }
    }

    fn remaining(self: *const XmlTokenizer) []const u8 {
        return self.source[self.pos..];
    }
};

fn isWhitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

fn getAttr(attrs: []const u8, name: []const u8) ?[]const u8 {
    var pos: usize = 0;

    while (pos < attrs.len) {
        // Skip whitespace
        while (pos < attrs.len and isWhitespace(attrs[pos])) {
            pos += 1;
        }

        // Parse attribute name
        const name_start = pos;
        while (pos < attrs.len and attrs[pos] != '=' and !isWhitespace(attrs[pos])) {
            pos += 1;
        }
        const attr_name = attrs[name_start..pos];

        // Skip to '='
        while (pos < attrs.len and (isWhitespace(attrs[pos]) or attrs[pos] == '=')) {
            pos += 1;
        }

        // Parse value
        if (pos >= attrs.len) break;

        const quote = attrs[pos];
        if (quote != '"' and quote != '\'') continue;
        pos += 1;

        const value_start = pos;
        while (pos < attrs.len and attrs[pos] != quote) {
            pos += 1;
        }
        const value = attrs[value_start..pos];
        pos += 1;

        if (std.mem.eql(u8, attr_name, name)) {
            return value;
        }
    }

    return null;
}

fn parseFloat(s: []const u8) ?f32 {
    return std.fmt.parseFloat(f32, s) catch null;
}

fn parseInt(s: []const u8) ?u32 {
    return std.fmt.parseInt(u32, s, 10) catch null;
}

fn parseVec2(s: []const u8) ?[2]f32 {
    var iter = std.mem.splitScalar(u8, s, ' ');
    const x = parseFloat(iter.next() orelse return null) orelse return null;
    const y = parseFloat(iter.next() orelse return null) orelse return null;
    return .{ x, y };
}

fn parseVec3(s: []const u8) ?[3]f32 {
    var iter = std.mem.splitScalar(u8, s, ' ');
    const x = parseFloat(iter.next() orelse return null) orelse return null;
    const y = parseFloat(iter.next() orelse return null) orelse return null;
    const z = parseFloat(iter.next() orelse return null) orelse return null;
    return .{ x, y, z };
}

fn parseQuat(s: []const u8) ?[4]f32 {
    var iter = std.mem.splitScalar(u8, s, ' ');
    const w = parseFloat(iter.next() orelse return null) orelse return null;
    const x = parseFloat(iter.next() orelse return null) orelse return null;
    const y = parseFloat(iter.next() orelse return null) orelse return null;
    const z = parseFloat(iter.next() orelse return null) orelse return null;
    return .{ w, x, y, z };
}

fn parseVec6(s: []const u8) ?[6]f32 {
    var iter = std.mem.splitScalar(u8, s, ' ');
    var result: [6]f32 = undefined;
    for (&result) |*v| {
        v.* = parseFloat(iter.next() orelse return null) orelse return null;
    }
    return result;
}

fn parseSize(s: []const u8) [3]f32 {
    var result: [3]f32 = .{ 0, 0, 0 };
    var iter = std.mem.splitScalar(u8, s, ' ');
    var i: usize = 0;
    while (iter.next()) |part| {
        if (i >= 3) break;
        result[i] = parseFloat(part) orelse 0;
        i += 1;
    }
    return result;
}

fn parseRgba(s: []const u8) ?[4]f32 {
    var iter = std.mem.splitScalar(u8, s, ' ');
    const r = parseFloat(iter.next() orelse return null) orelse return null;
    const g = parseFloat(iter.next() orelse return null) orelse return null;
    const b = parseFloat(iter.next() orelse return null) orelse return null;
    const a = parseFloat(iter.next() orelse "1") orelse 1;
    return .{ r, g, b, a };
}
