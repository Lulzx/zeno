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
    mesh: []const u8 = "", // Reference to mesh asset name
    material: []const u8 = "", // Reference to material asset name
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

/// Parsed MJCF mesh asset.
pub const MjcfMesh = struct {
    name: []const u8 = "",
    file: []const u8 = "",
    scale: [3]f32 = .{ 1, 1, 1 },
};

/// Texture type enumeration.
pub const TextureType = enum {
    _2d,
    cube,
    skybox,

    pub fn fromString(s: []const u8) ?TextureType {
        const map = std.StaticStringMap(TextureType).initComptime(.{
            .{ "2d", ._2d },
            .{ "cube", .cube },
            .{ "skybox", .skybox },
        });
        return map.get(s);
    }
};

/// Builtin texture type.
pub const BuiltinTexture = enum {
    none,
    gradient,
    checker,
    flat,

    pub fn fromString(s: []const u8) ?BuiltinTexture {
        const map = std.StaticStringMap(BuiltinTexture).initComptime(.{
            .{ "none", .none },
            .{ "gradient", .gradient },
            .{ "checker", .checker },
            .{ "flat", .flat },
        });
        return map.get(s);
    }
};

/// Parsed MJCF texture asset.
pub const MjcfTexture = struct {
    name: []const u8 = "",
    texture_type: TextureType = ._2d,
    builtin: BuiltinTexture = .none,
    file: []const u8 = "",
    /// RGB color for flat/gradient textures.
    rgb1: [3]f32 = .{ 0.8, 0.8, 0.8 },
    rgb2: [3]f32 = .{ 0.5, 0.5, 0.5 },
    /// Width and height for builtin textures.
    width: u32 = 512,
    height: u32 = 512,
    /// Mark color for procedural textures.
    mark: []const u8 = "none",
    markrgb: [3]f32 = .{ 0, 0, 0 },
    /// Random seed for procedural textures.
    random: f32 = 0.01,
    /// Grid size for skybox.
    gridsize: [2]u32 = .{ 1, 1 },
    /// Grid layout for cube/skybox textures.
    gridlayout: []const u8 = ".+.+++.+.",
};

/// Parsed MJCF material asset.
pub const MjcfMaterial = struct {
    name: []const u8 = "",
    /// Reference to texture name.
    texture: []const u8 = "",
    /// Texture repetition in U and V.
    texrepeat: [2]f32 = .{ 1, 1 },
    /// Texture uniform scaling.
    texuniform: bool = false,
    /// RGBA color for emission.
    emission: f32 = 0,
    /// RGBA color for specular.
    specular: f32 = 0.5,
    /// Shininess exponent.
    shininess: f32 = 0.5,
    /// Reflectance.
    reflectance: f32 = 0,
    /// RGBA color.
    rgba: [4]f32 = .{ 1, 1, 1, 1 },
};

/// Tendon type enumeration.
pub const TendonType = enum {
    fixed,
    spatial,

    pub fn fromString(s: []const u8) ?TendonType {
        const map = std.StaticStringMap(TendonType).initComptime(.{
            .{ "fixed", .fixed },
            .{ "spatial", .spatial },
        });
        return map.get(s);
    }
};

/// Equality constraint type.
pub const EqualityType = enum {
    connect,
    weld,
    joint,
    tendon,
    distance,

    pub fn fromString(s: []const u8) ?EqualityType {
        const map = std.StaticStringMap(EqualityType).initComptime(.{
            .{ "connect", .connect },
            .{ "weld", .weld },
            .{ "joint", .joint },
            .{ "tendon", .tendon },
            .{ "distance", .distance },
        });
        return map.get(s);
    }
};

/// Tendon path element (joint or site reference).
pub const MjcfTendonPath = struct {
    /// Joint name (for fixed tendon).
    joint: []const u8 = "",
    /// Site name (for spatial tendon).
    site: []const u8 = "",
    /// Coefficient (for fixed tendon joint).
    coef: f32 = 1.0,
};

/// Parsed MJCF tendon element.
pub const MjcfTendon = struct {
    name: []const u8 = "",
    tendon_type: TendonType = .fixed,
    /// Path elements.
    path: std.ArrayListUnmanaged(MjcfTendonPath) = .{},
    /// Stiffness coefficient.
    stiffness: f32 = 0.0,
    /// Damping coefficient.
    damping: f32 = 0.0,
    /// Lower length limit.
    range_lower: f32 = 0.0,
    /// Upper length limit.
    range_upper: f32 = 0.0,
    /// Enable length limits.
    limited: bool = false,
    /// Tendon width for visualization.
    width: f32 = 0.003,
    /// RGBA color.
    rgba: [4]f32 = .{ 0.5, 0.4, 0.3, 1.0 },
    /// Margin for limit constraints.
    margin: f32 = 0.0,

    pub fn deinit(self: *MjcfTendon, allocator: std.mem.Allocator) void {
        self.path.deinit(allocator);
    }
};

/// Parsed MJCF equality constraint element.
pub const MjcfEquality = struct {
    name: []const u8 = "",
    equality_type: EqualityType = .connect,
    /// First body name (for connect/weld).
    body1: []const u8 = "",
    /// Second body name (for connect/weld).
    body2: []const u8 = "",
    /// First joint name (for joint constraint).
    joint1: []const u8 = "",
    /// Second joint name (for joint constraint).
    joint2: []const u8 = "",
    /// Tendon name (for tendon constraint).
    tendon: []const u8 = "",
    /// Anchor point on first body.
    anchor: [3]f32 = .{ 0, 0, 0 },
    /// Relative pose quaternion (wxyz) for weld.
    relpose: [7]f32 = .{ 0, 0, 0, 1, 0, 0, 0 }, // pos + quat
    /// Polynomial coefficients for joint constraint.
    polycoef: [5]f32 = .{ 0, 1, 0, 0, 0 },
    /// Constraint is active.
    active: bool = true,
    /// Solver impedance parameters.
    solimp: [5]f32 = .{ 0.9, 0.95, 0.001, 0.5, 2 },
    /// Solver reference parameters.
    solref: [2]f32 = .{ 0.02, 1 },
};

/// Complete parsed MJCF model.
pub const MjcfModel = struct {
    model_name: []const u8 = "",
    option: MjcfOption = .{},
    defaults: std.StringHashMapUnmanaged(MjcfDefault) = .{},
    worldbody: MjcfBody = .{},
    actuators: std.ArrayListUnmanaged(MjcfActuator) = .{},
    sensors: std.ArrayListUnmanaged(MjcfSensor) = .{},
    meshes: std.ArrayListUnmanaged(MjcfMesh) = .{},
    textures: std.ArrayListUnmanaged(MjcfTexture) = .{},
    materials: std.ArrayListUnmanaged(MjcfMaterial) = .{},
    tendons: std.ArrayListUnmanaged(MjcfTendon) = .{},
    equalities: std.ArrayListUnmanaged(MjcfEquality) = .{},
    model_dir: []const u8 = "", // Directory containing the model for relative paths

    pub fn deinit(self: *MjcfModel, allocator: std.mem.Allocator) void {
        self.worldbody.deinit(allocator);
        self.actuators.deinit(allocator);
        self.sensors.deinit(allocator);
        self.defaults.deinit(allocator);
        self.meshes.deinit(allocator);
        self.textures.deinit(allocator);
        self.materials.deinit(allocator);
        for (self.tendons.items) |*t| {
            t.deinit(allocator);
        }
        self.tendons.deinit(allocator);
        self.equalities.deinit(allocator);
    }
};
