//! MJCF schema definitions matching MuJoCo XML format.

const std = @import("std");

/// MJCF geom types.
pub const GeomType = enum {
    plane,
    sphere,
    capsule,
    cylinder,
    box,
    mesh,

    pub fn fromString(s: []const u8) ?GeomType {
        const map = std.StaticStringMap(GeomType).initComptime(.{
            .{ "plane", .plane },
            .{ "sphere", .sphere },
            .{ "capsule", .capsule },
            .{ "cylinder", .cylinder },
            .{ "box", .box },
            .{ "mesh", .mesh },
        });
        return map.get(s);
    }
};

/// MJCF joint types.
pub const JointType = enum {
    free,
    ball,
    slide,
    hinge,

    pub fn fromString(s: []const u8) ?JointType {
        const map = std.StaticStringMap(JointType).initComptime(.{
            .{ "free", .free },
            .{ "ball", .ball },
            .{ "slide", .slide },
            .{ "hinge", .hinge },
        });
        return map.get(s);
    }
};

/// MJCF actuator types.
pub const ActuatorType = enum {
    motor,
    position,
    velocity,
    general,

    pub fn fromString(s: []const u8) ?ActuatorType {
        const map = std.StaticStringMap(ActuatorType).initComptime(.{
            .{ "motor", .motor },
            .{ "position", .position },
            .{ "velocity", .velocity },
            .{ "general", .general },
        });
        return map.get(s);
    }
};

/// MJCF sensor types.
pub const SensorType = enum {
    jointpos,
    jointvel,
    accelerometer,
    gyro,
    touch,
    force,
    torque,
    framepos,
    framequat,
    framelinvel,
    frameangvel,
    subtreecom,

    pub fn fromString(s: []const u8) ?SensorType {
        const map = std.StaticStringMap(SensorType).initComptime(.{
            .{ "jointpos", .jointpos },
            .{ "jointvel", .jointvel },
            .{ "accelerometer", .accelerometer },
            .{ "gyro", .gyro },
            .{ "touch", .touch },
            .{ "force", .force },
            .{ "torque", .torque },
            .{ "framepos", .framepos },
            .{ "framequat", .framequat },
            .{ "framelinvel", .framelinvel },
            .{ "frameangvel", .frameangvel },
            .{ "subtreecom", .subtreecom },
        });
        return map.get(s);
    }
};

/// Parsed MJCF option element.
pub const MjcfOption = struct {
    timestep: f32 = 0.002,
    gravity: [3]f32 = .{ 0, 0, -9.81 },
    integrator: []const u8 = "Euler",
    collision: []const u8 = "all",
    cone: []const u8 = "pyramidal",
    jacobian: []const u8 = "dense",
    solver: []const u8 = "Newton",
    iterations: u32 = 100,
    tolerance: f32 = 1e-8,
};

/// Parsed MJCF body element.
pub const MjcfBody = struct {
    name: []const u8 = "",
    pos: [3]f32 = .{ 0, 0, 0 },
    quat: [4]f32 = .{ 1, 0, 0, 0 }, // MuJoCo uses wxyz
    euler: ?[3]f32 = null,
    childclass: []const u8 = "",
    mocap: bool = false,

    joints: std.ArrayListUnmanaged(MjcfJoint) = .{},
    geoms: std.ArrayListUnmanaged(MjcfGeom) = .{},
    children: std.ArrayListUnmanaged(MjcfBody) = .{},
    sites: std.ArrayListUnmanaged(MjcfSite) = .{},

    pub fn deinit(self: *MjcfBody, allocator: std.mem.Allocator) void {
        for (self.children.items) |*child| {
            child.deinit(allocator);
        }
        self.joints.deinit(allocator);
        self.geoms.deinit(allocator);
        self.children.deinit(allocator);
        self.sites.deinit(allocator);
    }
};

/// Parsed MJCF joint element.
pub const MjcfJoint = struct {
    name: []const u8 = "",
    joint_type: JointType = .hinge,
    pos: [3]f32 = .{ 0, 0, 0 },
    axis: [3]f32 = .{ 0, 0, 1 },
    range: ?[2]f32 = null,
    limited: bool = false,
    damping: f32 = 0,
    stiffness: f32 = 0,
    armature: f32 = 0,
    frictionloss: f32 = 0,
    ref: f32 = 0,
};

/// Parsed MJCF geom element.
pub const MjcfGeom = struct {
    name: []const u8 = "",
    geom_type: GeomType = .sphere,
    pos: [3]f32 = .{ 0, 0, 0 },
    quat: [4]f32 = .{ 1, 0, 0, 0 },
    size: [3]f32 = .{ 0.05, 0, 0 },
    fromto: ?[6]f32 = null,
    mass: ?f32 = null,
    density: f32 = 1000,
    friction: [3]f32 = .{ 1, 0.005, 0.0001 },
    contype: u32 = 1,
    conaffinity: u32 = 1,
    condim: u32 = 3,
    rgba: [4]f32 = .{ 0.5, 0.5, 0.5, 1 },
    group: u32 = 0,
};

/// Parsed MJCF site element.
pub const MjcfSite = struct {
    name: []const u8 = "",
    pos: [3]f32 = .{ 0, 0, 0 },
    quat: [4]f32 = .{ 1, 0, 0, 0 },
    size: [3]f32 = .{ 0.005, 0, 0 },
};

/// Parsed MJCF actuator element.
pub const MjcfActuator = struct {
    name: []const u8 = "",
    actuator_type: ActuatorType = .motor,
    joint: []const u8 = "",
    ctrlrange: ?[2]f32 = null,
    forcerange: ?[2]f32 = null,
    gear: f32 = 1,
    ctrllimited: bool = false,
    kp: f32 = 0,
    kv: f32 = 0,
};

/// Parsed MJCF sensor element.
pub const MjcfSensor = struct {
    name: []const u8 = "",
    sensor_type: SensorType = .jointpos,
    site: []const u8 = "",
    joint: []const u8 = "",
    objtype: []const u8 = "",
    objname: []const u8 = "",
    noise: f32 = 0,
    cutoff: f32 = 0,
};

/// Parsed MJCF default class.
pub const MjcfDefault = struct {
    class_name: []const u8 = "main",
    joint: MjcfJoint = .{},
    geom: MjcfGeom = .{},
    site: MjcfSite = .{},
    motor: MjcfActuator = .{},
};

/// Complete parsed MJCF model.
pub const MjcfModel = struct {
    model_name: []const u8 = "",
    option: MjcfOption = .{},
    defaults: std.StringHashMapUnmanaged(MjcfDefault) = .{},
    worldbody: MjcfBody = .{},
    actuators: std.ArrayListUnmanaged(MjcfActuator) = .{},
    sensors: std.ArrayListUnmanaged(MjcfSensor) = .{},

    pub fn deinit(self: *MjcfModel, allocator: std.mem.Allocator) void {
        self.worldbody.deinit(allocator);
        self.actuators.deinit(allocator);
        self.sensors.deinit(allocator);
        self.defaults.deinit(allocator);
    }
};
