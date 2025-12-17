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

    return parseString(allocator, content);
}

/// Parse MJCF XML string and return a Scene.
pub fn parseString(allocator: std.mem.Allocator, xml: []const u8) !Scene {
    var model = try parseXml(allocator, xml);
    defer model.deinit(allocator);

    return convertToScene(allocator, &model);
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
            } else if (std.mem.eql(u8, token.name, "worldbody")) {
                try parseWorldbody(tokenizer, allocator, &model.worldbody);
            } else if (std.mem.eql(u8, token.name, "actuator")) {
                try parseActuators(tokenizer, allocator, &model.actuators);
            } else if (std.mem.eql(u8, token.name, "sensor")) {
                try parseSensors(tokenizer, allocator, &model.sensors);
            } else if (std.mem.eql(u8, token.name, "default")) {
                try parseDefaults(tokenizer, allocator, &model.defaults);
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

fn parseWorldbody(tokenizer: *XmlTokenizer, allocator: std.mem.Allocator, worldbody: *schema.MjcfBody) !void {
    try parseBodyContent(tokenizer, allocator, worldbody, "worldbody");
}

fn parseBodyContent(
    tokenizer: *XmlTokenizer,
    allocator: std.mem.Allocator,
    body: *schema.MjcfBody,
    end_tag: []const u8,
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
                try parseBodyContent(tokenizer, allocator, &child, "body");
                try body.children.append(allocator, child);
            } else if (std.mem.eql(u8, token.name, "joint")) {
                const joint = parseJoint(token.attrs);
                try body.joints.append(allocator, joint);
            } else if (std.mem.eql(u8, token.name, "geom")) {
                const geom = parseGeom(token.attrs);
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

    joint.name = getAttr(attrs, "name") orelse "";

    if (getAttr(attrs, "type")) |v| {
        joint.joint_type = schema.JointType.fromString(v) orelse .hinge;
    }

    if (getAttr(attrs, "pos")) |v| {
        joint.pos = parseVec3(v) orelse .{ 0, 0, 0 };
    }

    if (getAttr(attrs, "axis")) |v| {
        joint.axis = parseVec3(v) orelse .{ 0, 0, 1 };
    }

    if (getAttr(attrs, "range")) |v| {
        joint.range = parseVec2(v);
        joint.limited = joint.range != null;
    }

    if (getAttr(attrs, "damping")) |v| {
        joint.damping = parseFloat(v) orelse 0;
    }

    if (getAttr(attrs, "stiffness")) |v| {
        joint.stiffness = parseFloat(v) orelse 0;
    }

    if (getAttr(attrs, "armature")) |v| {
        joint.armature = parseFloat(v) orelse 0;
    }

    if (getAttr(attrs, "ref")) |v| {
        joint.ref = parseFloat(v) orelse 0;
    }

    return joint;
}

fn parseGeom(attrs: []const u8) schema.MjcfGeom {
    var geom = schema.MjcfGeom{};

    geom.name = getAttr(attrs, "name") orelse "";

    if (getAttr(attrs, "type")) |v| {
        geom.geom_type = schema.GeomType.fromString(v) orelse .sphere;
    }

    if (getAttr(attrs, "pos")) |v| {
        geom.pos = parseVec3(v) orelse .{ 0, 0, 0 };
    }

    if (getAttr(attrs, "quat")) |v| {
        geom.quat = parseQuat(v) orelse .{ 1, 0, 0, 0 };
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
        geom.density = parseFloat(v) orelse 1000;
    }

    if (getAttr(attrs, "friction")) |v| {
        geom.friction = parseVec3(v) orelse .{ 1, 0.005, 0.0001 };
    }

    if (getAttr(attrs, "contype")) |v| {
        geom.contype = parseInt(v) orelse 1;
    }

    if (getAttr(attrs, "conaffinity")) |v| {
        geom.conaffinity = parseInt(v) orelse 1;
    }

    return geom;
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

fn parseDefaults(tokenizer: *XmlTokenizer, _: std.mem.Allocator, _: *std.StringHashMapUnmanaged(schema.MjcfDefault)) !void {
    // Skip defaults for now - consume until end tag
    var depth: u32 = 1;
    while (tokenizer.next()) |token| {
        if (token.kind == .element_start) {
            depth += 1;
        } else if (token.kind == .element_end) {
            depth -= 1;
            if (depth == 0) break;
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

        // Handle euler angles if specified
        var quat = child.quat;
        if (child.euler) |euler| {
            quat = eulerToQuat(euler);
        }

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
            .quaternion = quat,
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
        try processBody(allocator, scene, child, @intCast(current_body_id), body_index, pos, quat);
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
