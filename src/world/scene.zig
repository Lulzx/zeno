//! Scene description - static definition of the simulation world.
//! Contains bodies, joints, geoms, actuators, and sensors.

const std = @import("std");
const body_mod = @import("../physics/body.zig");
const joint_mod = @import("../physics/joint.zig");
const primitives = @import("../collision/primitives.zig");
const sensors = @import("sensors.zig");
const constants = @import("../physics/constants.zig");

/// Complete scene description.
pub const Scene = struct {
    /// Body definitions.
    bodies: std.ArrayListUnmanaged(body_mod.BodyDef) = .{},
    /// Joint definitions.
    joints: std.ArrayListUnmanaged(joint_mod.JointDef) = .{},
    /// Geometry definitions.
    geoms: std.ArrayListUnmanaged(primitives.Geom) = .{},
    /// Actuator definitions.
    actuators: std.ArrayListUnmanaged(joint_mod.ActuatorDef) = .{},
    /// Sensor configuration.
    sensor_config: sensors.SensorConfig = .{},

    /// Physics configuration.
    physics_config: constants.PhysicsConfig = .{},

    /// Name to index mappings.
    body_names: std.StringHashMapUnmanaged(u32) = .{},
    joint_names: std.StringHashMapUnmanaged(u32) = .{},
    geom_names: std.StringHashMapUnmanaged(u32) = .{},
    actuator_names: std.StringHashMapUnmanaged(u32) = .{},

    /// Allocator.
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Scene {
        return .{
            .allocator = allocator,
        };
    }

    /// Add a body to the scene.
    pub fn addBody(self: *Scene, body_def: body_mod.BodyDef) !u32 {
        const id: u32 = @intCast(self.bodies.items.len);
        try self.bodies.append(self.allocator, body_def);

        if (body_def.name.len > 0) {
            const name_copy = try self.allocator.dupe(u8, body_def.name);
            try self.body_names.put(self.allocator, name_copy, id);
        }

        return id;
    }

    /// Add a joint to the scene.
    pub fn addJoint(self: *Scene, joint_def: joint_mod.JointDef) !u32 {
        const id: u32 = @intCast(self.joints.items.len);
        try self.joints.append(self.allocator, joint_def);

        if (joint_def.name.len > 0) {
            const name_copy = try self.allocator.dupe(u8, joint_def.name);
            try self.joint_names.put(self.allocator, name_copy, id);
        }

        return id;
    }

    /// Add a geometry to the scene.
    pub fn addGeom(self: *Scene, geom: primitives.Geom) !u32 {
        const id: u32 = @intCast(self.geoms.items.len);
        try self.geoms.append(self.allocator, geom);
        return id;
    }

    /// Add a named geometry.
    pub fn addNamedGeom(self: *Scene, name: []const u8, geom: primitives.Geom) !u32 {
        const id = try self.addGeom(geom);
        if (name.len > 0) {
            const name_copy = try self.allocator.dupe(u8, name);
            try self.geom_names.put(self.allocator, name_copy, id);
        }
        return id;
    }

    /// Add an actuator.
    pub fn addActuator(self: *Scene, actuator: joint_mod.ActuatorDef) !u32 {
        const id: u32 = @intCast(self.actuators.items.len);
        try self.actuators.append(self.allocator, actuator);

        if (actuator.name.len > 0) {
            const name_copy = try self.allocator.dupe(u8, actuator.name);
            try self.actuator_names.put(self.allocator, name_copy, id);
        }

        return id;
    }

    /// Add a sensor.
    pub fn addSensor(self: *Scene, sensor: sensors.Sensor) !u32 {
        return self.sensor_config.addSensor(self.allocator, sensor);
    }

    /// Get body by name.
    pub fn getBodyByName(self: *const Scene, name: []const u8) ?u32 {
        return self.body_names.get(name);
    }

    /// Get joint by name.
    pub fn getJointByName(self: *const Scene, name: []const u8) ?u32 {
        return self.joint_names.get(name);
    }

    /// Get geom by name.
    pub fn getGeomByName(self: *const Scene, name: []const u8) ?u32 {
        return self.geom_names.get(name);
    }

    /// Get actuator by name.
    pub fn getActuatorByName(self: *const Scene, name: []const u8) ?u32 {
        return self.actuator_names.get(name);
    }

    /// Get number of bodies.
    pub fn numBodies(self: *const Scene) u32 {
        return @intCast(self.bodies.items.len);
    }

    /// Get number of joints.
    pub fn numJoints(self: *const Scene) u32 {
        return @intCast(self.joints.items.len);
    }

    /// Get number of geometries.
    pub fn numGeoms(self: *const Scene) u32 {
        return @intCast(self.geoms.items.len);
    }

    /// Get number of actuators.
    pub fn numActuators(self: *const Scene) u32 {
        return @intCast(self.actuators.items.len);
    }

    /// Get observation dimension.
    pub fn obsDim(self: *const Scene) u32 {
        return self.sensor_config.total_dim;
    }

    /// Get action dimension.
    pub fn actionDim(self: *const Scene) u32 {
        return @intCast(self.actuators.items.len);
    }

    /// Compute total DOF.
    pub fn totalDof(self: *const Scene) u32 {
        var dof: u32 = 0;
        for (self.joints.items) |joint| {
            dof += joint.dof();
        }
        return dof;
    }

    /// Validate scene consistency.
    pub fn validate(self: *const Scene) bool {
        // Check body references in joints
        for (self.joints.items) |joint| {
            if (joint.parent_body >= self.bodies.items.len) return false;
            if (joint.child_body >= self.bodies.items.len) return false;
        }

        // Check body references in geoms
        for (self.geoms.items) |geom| {
            if (geom.body_id >= self.bodies.items.len) return false;
        }

        // Check joint references in actuators
        for (self.actuators.items) |actuator| {
            if (actuator.joint >= self.joints.items.len) return false;
        }

        return true;
    }

    /// Create a simple ground plane.
    pub fn addGroundPlane(self: *Scene) !void {
        // Add static world body
        const world_body_id = try self.addBody(.{
            .name = "world",
            .body_type = .static,
            .position = .{ 0, 0, 0 },
        });

        // Add ground plane geom
        var ground = primitives.Geom.plane(.{ 0, 0, 1 }, 0);
        ground.body_id = world_body_id;
        ground.friction = 1.0;
        _ = try self.addNamedGeom("ground", ground);
    }

    /// Build kinematic tree (compute body transforms).
    pub fn buildKinematicTree(self: *Scene) !void {
        // For each body, compute transform relative to parent
        for (self.bodies.items) |*body_def| {
            if (body_def.parent_id >= 0) {
                // Child body - transform is relative to parent
                // This is handled during forward kinematics
            }
        }
    }

    /// Auto-add sensors for all joints and root body.
    pub fn autoAddSensors(self: *Scene) !void {
        // Find root body (first dynamic body with free joint or no parent)
        var root_body: u32 = 0;
        for (self.bodies.items, 0..) |body_def, i| {
            if (body_def.body_type == .dynamic and body_def.parent_id < 0) {
                root_body = @intCast(i);
                break;
            }
        }

        try self.sensor_config.addStandardSensors(
            self.allocator,
            @intCast(self.joints.items.len),
            root_body,
        );
    }

    pub fn deinit(self: *Scene) void {
        self.bodies.deinit(self.allocator);
        self.joints.deinit(self.allocator);
        self.geoms.deinit(self.allocator);
        self.actuators.deinit(self.allocator);
        self.sensor_config.deinit(self.allocator);

        var body_iter = self.body_names.iterator();
        while (body_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.body_names.deinit(self.allocator);

        var joint_iter = self.joint_names.iterator();
        while (joint_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.joint_names.deinit(self.allocator);

        var geom_iter = self.geom_names.iterator();
        while (geom_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.geom_names.deinit(self.allocator);

        var act_iter = self.actuator_names.iterator();
        while (act_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.actuator_names.deinit(self.allocator);
    }
};

/// Builder pattern for creating scenes programmatically.
pub const SceneBuilder = struct {
    scene: Scene,
    current_body: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator) SceneBuilder {
        return .{
            .scene = Scene.init(allocator),
        };
    }

    /// Set physics configuration.
    pub fn setPhysics(self: *SceneBuilder, config: constants.PhysicsConfig) *SceneBuilder {
        self.scene.physics_config = config;
        return self;
    }

    /// Add a body and set it as current.
    pub fn body(self: *SceneBuilder, name: []const u8, pos: [3]f32) *SceneBuilder {
        const id = self.scene.addBody(.{
            .name = name,
            .position = pos,
        }) catch return self;
        self.current_body = id;
        return self;
    }

    /// Add a static body.
    pub fn staticBody(self: *SceneBuilder, name: []const u8, pos: [3]f32) *SceneBuilder {
        const id = self.scene.addBody(.{
            .name = name,
            .position = pos,
            .body_type = .static,
        }) catch return self;
        self.current_body = id;
        return self;
    }

    /// Set mass of current body.
    pub fn mass(self: *SceneBuilder, m: f32) *SceneBuilder {
        if (self.current_body) |id| {
            self.scene.bodies.items[id].mass = m;
        }
        return self;
    }

    /// Add a sphere geom to current body.
    pub fn sphere(self: *SceneBuilder, radius: f32) *SceneBuilder {
        if (self.current_body) |body_id| {
            var geom = primitives.Geom.sphere(radius);
            geom.body_id = body_id;
            _ = self.scene.addGeom(geom) catch {};
        }
        return self;
    }

    /// Add a capsule geom to current body.
    pub fn capsule(self: *SceneBuilder, radius: f32, half_length: f32) *SceneBuilder {
        if (self.current_body) |body_id| {
            var geom = primitives.Geom.capsule(radius, half_length);
            geom.body_id = body_id;
            _ = self.scene.addGeom(geom) catch {};
        }
        return self;
    }

    /// Add a box geom to current body.
    pub fn box(self: *SceneBuilder, hx: f32, hy: f32, hz: f32) *SceneBuilder {
        if (self.current_body) |body_id| {
            var geom = primitives.Geom.box(hx, hy, hz);
            geom.body_id = body_id;
            _ = self.scene.addGeom(geom) catch {};
        }
        return self;
    }

    /// Add a revolute joint.
    pub fn hinge(
        self: *SceneBuilder,
        name: []const u8,
        parent: u32,
        axis: [3]f32,
    ) *SceneBuilder {
        if (self.current_body) |child| {
            _ = self.scene.addJoint(.{
                .name = name,
                .joint_type = .revolute,
                .parent_body = parent,
                .child_body = child,
                .axis = axis,
            }) catch {};
        }
        return self;
    }

    /// Add a free joint to current body.
    pub fn freeJoint(self: *SceneBuilder) *SceneBuilder {
        if (self.current_body) |body_id| {
            _ = self.scene.addJoint(.{
                .joint_type = .free,
                .parent_body = 0,
                .child_body = body_id,
            }) catch {};
        }
        return self;
    }

    /// Add a motor actuator.
    pub fn motor(self: *SceneBuilder, joint_name: []const u8, gear: f32) *SceneBuilder {
        if (self.scene.getJointByName(joint_name)) |joint_id| {
            _ = self.scene.addActuator(.{
                .joint = joint_id,
                .gear = gear,
            }) catch {};
        }
        return self;
    }

    /// Add ground plane.
    pub fn ground(self: *SceneBuilder) *SceneBuilder {
        self.scene.addGroundPlane() catch {};
        return self;
    }

    /// Auto-add standard sensors.
    pub fn autoSensors(self: *SceneBuilder) *SceneBuilder {
        self.scene.autoAddSensors() catch {};
        return self;
    }

    /// Build and return the scene.
    pub fn build(self: *SceneBuilder) Scene {
        return self.scene;
    }
};
