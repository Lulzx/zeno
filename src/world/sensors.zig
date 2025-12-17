//! Sensor definitions for observation computation.

const std = @import("std");

/// Sensor type enumeration.
pub const SensorType = enum(u8) {
    /// Joint position sensor.
    joint_pos = 0,
    /// Joint velocity sensor.
    joint_vel = 1,
    /// Accelerometer (linear acceleration).
    accelerometer = 2,
    /// Gyroscope (angular velocity).
    gyro = 3,
    /// Touch sensor (contact force).
    touch = 4,
    /// Force sensor.
    force = 5,
    /// Torque sensor.
    torque = 6,
    /// Position sensor (body position).
    framepos = 7,
    /// Orientation sensor (body quaternion).
    framequat = 8,
    /// Linear velocity sensor.
    framelinvel = 9,
    /// Angular velocity sensor.
    frameangvel = 10,
    /// Subtree center of mass.
    subtreecom = 11,
    /// User-defined sensor.
    user = 12,
};

/// Sensor definition.
pub const Sensor = struct {
    /// Sensor name.
    name: []const u8 = "",
    /// Sensor type.
    sensor_type: SensorType = .joint_pos,
    /// Target object ID (joint, body, or geom depending on type).
    object_id: u32 = 0,
    /// Output dimension.
    dim: u32 = 1,
    /// Noise standard deviation (0 = no noise).
    noise: f32 = 0.0,
    /// Cutoff frequency for low-pass filter (0 = no filter).
    cutoff: f32 = 0.0,
    /// Sensor-specific data.
    data: [4]f32 = .{ 0, 0, 0, 0 },

    /// Get output dimension for this sensor type.
    pub fn outputDim(sensor_type: SensorType) u32 {
        return switch (sensor_type) {
            .joint_pos, .joint_vel => 1,
            .accelerometer, .gyro => 3,
            .touch => 1,
            .force, .torque => 3,
            .framepos, .framelinvel, .frameangvel => 3,
            .framequat => 4,
            .subtreecom => 3,
            .user => 1,
        };
    }

    /// Create a joint position sensor.
    pub fn jointPos(joint_id: u32) Sensor {
        return .{
            .sensor_type = .joint_pos,
            .object_id = joint_id,
            .dim = 1,
        };
    }

    /// Create a joint velocity sensor.
    pub fn jointVel(joint_id: u32) Sensor {
        return .{
            .sensor_type = .joint_vel,
            .object_id = joint_id,
            .dim = 1,
        };
    }

    /// Create an accelerometer.
    pub fn accelerometer(body_id: u32) Sensor {
        return .{
            .sensor_type = .accelerometer,
            .object_id = body_id,
            .dim = 3,
        };
    }

    /// Create a gyroscope.
    pub fn gyro(body_id: u32) Sensor {
        return .{
            .sensor_type = .gyro,
            .object_id = body_id,
            .dim = 3,
        };
    }

    /// Create a body position sensor.
    pub fn framePos(body_id: u32) Sensor {
        return .{
            .sensor_type = .framepos,
            .object_id = body_id,
            .dim = 3,
        };
    }

    /// Create a body orientation sensor.
    pub fn frameQuat(body_id: u32) Sensor {
        return .{
            .sensor_type = .framequat,
            .object_id = body_id,
            .dim = 4,
        };
    }

    /// Create a body linear velocity sensor.
    pub fn frameLinVel(body_id: u32) Sensor {
        return .{
            .sensor_type = .framelinvel,
            .object_id = body_id,
            .dim = 3,
        };
    }

    /// Create a body angular velocity sensor.
    pub fn frameAngVel(body_id: u32) Sensor {
        return .{
            .sensor_type = .frameangvel,
            .object_id = body_id,
            .dim = 3,
        };
    }

    /// Create a touch sensor.
    pub fn touch(geom_id: u32) Sensor {
        return .{
            .sensor_type = .touch,
            .object_id = geom_id,
            .dim = 1,
        };
    }
};

/// GPU-friendly sensor data.
pub const SensorGPU = extern struct {
    /// Type and object ID.
    type_object: [4]u32 align(16),
    /// Parameters (noise, cutoff, dim, offset).
    params: [4]f32 align(16),

    pub fn fromSensor(s: *const Sensor, output_offset: u32) SensorGPU {
        return .{
            .type_object = .{ @intFromEnum(s.sensor_type), s.object_id, s.dim, 0 },
            .params = .{ s.noise, s.cutoff, @floatFromInt(s.dim), @floatFromInt(output_offset) },
        };
    }
};

/// Sensor configuration for a model.
pub const SensorConfig = struct {
    sensors: std.ArrayListUnmanaged(Sensor) = .{},
    total_dim: u32 = 0,
    allocator: ?std.mem.Allocator = null,

    /// Add a sensor and return its output offset.
    pub fn addSensor(self: *SensorConfig, allocator: std.mem.Allocator, sensor: Sensor) !u32 {
        if (self.allocator == null) self.allocator = allocator;
        const offset = self.total_dim;
        try self.sensors.append(allocator, sensor);
        self.total_dim += sensor.dim;
        return offset;
    }

    /// Add default sensors for a joint.
    pub fn addJointSensors(self: *SensorConfig, allocator: std.mem.Allocator, joint_id: u32) !void {
        _ = try self.addSensor(allocator, Sensor.jointPos(joint_id));
        _ = try self.addSensor(allocator, Sensor.jointVel(joint_id));
    }

    /// Add default sensors for a body (accelerometer + gyro).
    pub fn addBodySensors(self: *SensorConfig, allocator: std.mem.Allocator, body_id: u32) !void {
        _ = try self.addSensor(allocator, Sensor.accelerometer(body_id));
        _ = try self.addSensor(allocator, Sensor.gyro(body_id));
    }

    /// Add all standard sensors for an articulated body.
    pub fn addStandardSensors(
        self: *SensorConfig,
        allocator: std.mem.Allocator,
        num_joints: u32,
        root_body_id: u32,
    ) !void {
        // Root body sensors
        _ = try self.addSensor(allocator, Sensor.framePos(root_body_id));
        _ = try self.addSensor(allocator, Sensor.frameQuat(root_body_id));
        _ = try self.addSensor(allocator, Sensor.frameLinVel(root_body_id));
        _ = try self.addSensor(allocator, Sensor.frameAngVel(root_body_id));

        // Joint sensors
        for (0..num_joints) |j| {
            try self.addJointSensors(allocator, @intCast(j));
        }
    }

    /// Get sensor by index.
    pub fn getSensor(self: *const SensorConfig, index: usize) ?*const Sensor {
        if (index >= self.sensors.items.len) return null;
        return &self.sensors.items[index];
    }

    /// Get number of sensors.
    pub fn count(self: *const SensorConfig) u32 {
        return @intCast(self.sensors.items.len);
    }

    pub fn deinit(self: *SensorConfig, allocator: std.mem.Allocator) void {
        self.sensors.deinit(allocator);
    }
};

/// Observation space definition.
pub const ObservationSpace = struct {
    /// Total observation dimension.
    dim: u32,
    /// Lower bounds (optional).
    low: ?[]f32 = null,
    /// Upper bounds (optional).
    high: ?[]f32 = null,
    /// Feature names.
    names: ?[][]const u8 = null,

    pub fn init(dim: u32) ObservationSpace {
        return .{ .dim = dim };
    }

    pub fn withBounds(dim: u32, allocator: std.mem.Allocator, low_val: f32, high_val: f32) !ObservationSpace {
        var space = ObservationSpace{ .dim = dim };
        space.low = try allocator.alloc(f32, dim);
        space.high = try allocator.alloc(f32, dim);
        @memset(space.low.?, low_val);
        @memset(space.high.?, high_val);
        return space;
    }
};

/// Action space definition.
pub const ActionSpace = struct {
    /// Action dimension.
    dim: u32,
    /// Lower bounds.
    low: []f32,
    /// Upper bounds.
    high: []f32,
    /// Action names.
    names: ?[][]const u8 = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, dim: u32, low_val: f32, high_val: f32) !ActionSpace {
        const low = try allocator.alloc(f32, dim);
        const high = try allocator.alloc(f32, dim);
        @memset(low, low_val);
        @memset(high, high_val);

        return .{
            .dim = dim,
            .low = low,
            .high = high,
            .allocator = allocator,
        };
    }

    /// Clip actions to valid range.
    pub fn clip(self: *const ActionSpace, actions: []f32) void {
        for (actions, 0..) |*a, i| {
            a.* = std.math.clamp(a.*, self.low[i], self.high[i]);
        }
    }

    /// Normalize actions from [-1, 1] to actual range.
    pub fn denormalize(self: *const ActionSpace, normalized: []const f32, output: []f32) void {
        for (normalized, 0..) |n, i| {
            const range = self.high[i] - self.low[i];
            const mid = (self.high[i] + self.low[i]) * 0.5;
            output[i] = mid + n * range * 0.5;
        }
    }

    /// Normalize actions from actual range to [-1, 1].
    pub fn normalize(self: *const ActionSpace, actions: []const f32, output: []f32) void {
        for (actions, 0..) |a, i| {
            const range = self.high[i] - self.low[i];
            const mid = (self.high[i] + self.low[i]) * 0.5;
            output[i] = (a - mid) / (range * 0.5);
        }
    }

    pub fn deinit(self: *ActionSpace) void {
        self.allocator.free(self.low);
        self.allocator.free(self.high);
    }
};
