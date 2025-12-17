//! Tests for sensor system.

const std = @import("std");
const testing = std.testing;

const sensors = @import("zeno").world.sensors;
const Sensor = sensors.Sensor;
const SensorType = sensors.SensorType;
const SensorConfig = sensors.SensorConfig;

test "sensor type enum values" {
    try testing.expectEqual(@as(u8, 0), @intFromEnum(SensorType.joint_pos));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(SensorType.joint_vel));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(SensorType.accelerometer));
    try testing.expectEqual(@as(u8, 3), @intFromEnum(SensorType.gyro));
}

test "joint position sensor creation" {
    const sensor = Sensor.jointPos(0);

    try testing.expectEqual(SensorType.joint_pos, sensor.sensor_type);
    try testing.expectEqual(@as(u32, 0), sensor.object_id);
    try testing.expectEqual(@as(u32, 1), sensor.dim);
}

test "joint velocity sensor creation" {
    const sensor = Sensor.jointVel(1);

    try testing.expectEqual(SensorType.joint_vel, sensor.sensor_type);
    try testing.expectEqual(@as(u32, 1), sensor.object_id);
}

test "accelerometer sensor creation" {
    const sensor = Sensor.accelerometer(0);

    try testing.expectEqual(SensorType.accelerometer, sensor.sensor_type);
    try testing.expectEqual(@as(u32, 3), sensor.dim);
}

test "gyro sensor creation" {
    const sensor = Sensor.gyro(0);

    try testing.expectEqual(SensorType.gyro, sensor.sensor_type);
    try testing.expectEqual(@as(u32, 3), sensor.dim);
}

test "sensor output dimension" {
    try testing.expectEqual(@as(u32, 1), Sensor.outputDim(.joint_pos));
    try testing.expectEqual(@as(u32, 1), Sensor.outputDim(.joint_vel));
    try testing.expectEqual(@as(u32, 3), Sensor.outputDim(.accelerometer));
    try testing.expectEqual(@as(u32, 3), Sensor.outputDim(.gyro));
    try testing.expectEqual(@as(u32, 3), Sensor.outputDim(.framepos));
    try testing.expectEqual(@as(u32, 4), Sensor.outputDim(.framequat));
    try testing.expectEqual(@as(u32, 3), Sensor.outputDim(.framelinvel));
    try testing.expectEqual(@as(u32, 3), Sensor.outputDim(.frameangvel));
}

test "sensor config initialization" {
    var config = SensorConfig{};

    try testing.expect(config.count() == 0);
}

test "sensor config add sensor" {
    var config = SensorConfig{};
    defer config.deinit(testing.allocator);

    const id = try config.addSensor(testing.allocator, Sensor.jointPos(0));

    try testing.expectEqual(@as(u32, 0), id);
    try testing.expectEqual(@as(u32, 1), config.count());
}

test "sensor config multiple sensors" {
    var config = SensorConfig{};
    defer config.deinit(testing.allocator);

    _ = try config.addSensor(testing.allocator, Sensor.jointPos(0));
    _ = try config.addSensor(testing.allocator, Sensor.jointVel(0));
    _ = try config.addSensor(testing.allocator, Sensor.accelerometer(0));

    try testing.expectEqual(@as(u32, 3), config.count());
}

test "sensor config total dimension" {
    var config = SensorConfig{};
    defer config.deinit(testing.allocator);

    // Add sensors with different output dimensions
    _ = try config.addSensor(testing.allocator, Sensor.jointPos(0)); // dim=1
    _ = try config.addSensor(testing.allocator, Sensor.accelerometer(0)); // dim=3
    _ = try config.addSensor(testing.allocator, .{
        .sensor_type = .framequat,
        .object_id = 0,
        .dim = 4,
    }); // dim=4

    try testing.expectEqual(@as(u32, 8), config.total_dim); // 1 + 3 + 4
}

test "sensor with noise" {
    const sensor = Sensor{
        .sensor_type = .joint_pos,
        .object_id = 0,
        .dim = 1,
        .noise = 0.01,
    };

    try testing.expectApproxEqAbs(@as(f32, 0.01), sensor.noise, 0.001);
}

test "sensor with cutoff frequency" {
    const sensor = Sensor{
        .sensor_type = .accelerometer,
        .object_id = 0,
        .dim = 3,
        .cutoff = 100.0, // Hz
    };

    try testing.expectApproxEqAbs(@as(f32, 100.0), sensor.cutoff, 0.1);
}

test "sensor default values" {
    const sensor = Sensor{
        .sensor_type = .joint_pos,
    };

    try testing.expect(sensor.noise == 0.0);
    try testing.expect(sensor.cutoff == 0.0);
    try testing.expectEqual(@as(u32, 0), sensor.object_id);
}

test "imu sensor configuration" {
    var config = SensorConfig{};
    defer config.deinit(testing.allocator);

    _ = try config.addSensor(testing.allocator, Sensor.accelerometer(0));
    _ = try config.addSensor(testing.allocator, Sensor.gyro(0));

    try testing.expectEqual(@as(u32, 2), config.count());
    try testing.expectEqual(@as(u32, 6), config.total_dim); // 3 + 3
}

test "robot joint sensors" {
    var config = SensorConfig{};
    defer config.deinit(testing.allocator);

    // Add joint position and velocity sensors for 3 joints
    for (0..3) |i| {
        _ = try config.addSensor(testing.allocator, Sensor.jointPos(@intCast(i)));
        _ = try config.addSensor(testing.allocator, Sensor.jointVel(@intCast(i)));
    }

    try testing.expectEqual(@as(u32, 6), config.count()); // 3 pos + 3 vel
    try testing.expectEqual(@as(u32, 6), config.total_dim); // all dim=1
}

test "all sensor types" {
    const types = [_]SensorType{
        .joint_pos,
        .joint_vel,
        .accelerometer,
        .gyro,
        .touch,
        .force,
        .torque,
        .framepos,
        .framequat,
        .framelinvel,
        .frameangvel,
        .subtreecom,
        .user,
    };

    for (types) |t| {
        const dim = Sensor.outputDim(t);
        try testing.expect(dim >= 1);
        try testing.expect(dim <= 4);
    }
}
