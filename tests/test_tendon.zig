//! Tests for tendon constraints.

const std = @import("std");
const testing = std.testing;

const tendon = @import("zeno").physics.tendon;
const TendonDef = tendon.TendonDef;
const TendonType = tendon.TendonType;
const TendonPathElement = tendon.TendonPathElement;
const WrapType = tendon.WrapType;

test "tendon type enum" {
    try testing.expectEqual(@as(u8, 0), @intFromEnum(TendonType.fixed));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(TendonType.spatial));
}

test "wrap type enum" {
    try testing.expectEqual(@as(u8, 0), @intFromEnum(WrapType.none));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(WrapType.sphere));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(WrapType.cylinder));
}

test "fixed tendon creation" {
    const def = TendonDef{
        .name = "fixed_tendon",
        .tendon_type = .fixed,
        .stiffness = 1000.0,
        .damping = 10.0,
    };

    try testing.expectEqual(TendonType.fixed, def.tendon_type);
    try testing.expectApproxEqAbs(@as(f32, 1000.0), def.stiffness, 0.1);
    try testing.expectApproxEqAbs(@as(f32, 10.0), def.damping, 0.1);
}

test "spatial tendon creation" {
    const def = TendonDef{
        .name = "spatial_tendon",
        .tendon_type = .spatial,
        .stiffness = 500.0,
        .damping = 5.0,
    };

    try testing.expectEqual(TendonType.spatial, def.tendon_type);
}

test "tendon path element" {
    const elem = TendonPathElement{
        .index = 1,
        .coef = 0.5,
        .wrap_type = .none,
    };

    try testing.expectEqual(@as(u32, 1), elem.index);
    try testing.expectApproxEqAbs(@as(f32, 0.5), elem.coef, 0.001);
}

test "tendon path with wrapping" {
    const elem = TendonPathElement{
        .index = 0,
        .coef = 1.0,
        .wrap_type = .sphere,
        .wrap_obj = 2,
    };

    try testing.expectEqual(WrapType.sphere, elem.wrap_type);
    try testing.expectEqual(@as(u32, 2), elem.wrap_obj);
}

test "tendon default parameters" {
    const def = TendonDef{
        .name = "default",
    };

    // Check defaults
    try testing.expect(def.stiffness == 0.0);
    try testing.expect(def.damping == 0.0);
    try testing.expect(!def.limited);
}

test "tendon with length limits" {
    const def = TendonDef{
        .name = "limited_tendon",
        .tendon_type = .spatial,
        .stiffness = 100.0,
        .limit_lower = 0.05,
        .limit_upper = 0.15,
        .limited = true,
    };

    try testing.expect(def.limited);
    try testing.expectApproxEqAbs(@as(f32, 0.05), def.limit_lower, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.15), def.limit_upper, 0.001);
}

test "tendon within limits check" {
    const def = TendonDef{
        .tendon_type = .fixed,
        .limit_lower = 0.0,
        .limit_upper = 1.0,
        .limited = true,
    };

    try testing.expect(def.isWithinLimits(0.5));
    try testing.expect(def.isWithinLimits(0.0));
    try testing.expect(def.isWithinLimits(1.0));
    try testing.expect(!def.isWithinLimits(-0.1));
    try testing.expect(!def.isWithinLimits(1.1));
}

test "tendon limit violation" {
    const def = TendonDef{
        .tendon_type = .fixed,
        .limit_lower = 0.0,
        .limit_upper = 1.0,
        .limited = true,
    };

    try testing.expectApproxEqAbs(@as(f32, 0.0), def.limitViolation(0.5), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.1), def.limitViolation(-0.1), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.2), def.limitViolation(1.2), 0.001);
}

test "fixed tendon length computation" {
    const path = [_]TendonPathElement{
        .{ .index = 0, .coef = 1.0 },
        .{ .index = 1, .coef = -1.0 },
    };

    const def = TendonDef{
        .tendon_type = .fixed,
        .path = &path,
    };

    const joint_positions = [_]f32{ 0.5, 0.3 };
    const length = def.computeLength(&joint_positions);

    // Length = 1.0 * 0.5 + (-1.0) * 0.3 = 0.2
    try testing.expectApproxEqAbs(@as(f32, 0.2), length, 0.001);
}

test "unlimited tendon always within limits" {
    const def = TendonDef{
        .tendon_type = .fixed,
        .limited = false,
    };

    try testing.expect(def.isWithinLimits(-1000.0));
    try testing.expect(def.isWithinLimits(1000.0));
    try testing.expectApproxEqAbs(@as(f32, 0.0), def.limitViolation(1000.0), 0.001);
}

test "tendon rest length" {
    const def = TendonDef{
        .tendon_type = .spatial,
        .rest_length = 0.1,
        .stiffness = 100.0,
    };

    try testing.expectApproxEqAbs(@as(f32, 0.1), def.rest_length, 0.001);
}

test "tendon visualization parameters" {
    const def = TendonDef{
        .name = "visual_tendon",
        .width = 0.005,
        .rgba = .{ 1.0, 0.0, 0.0, 1.0 },
    };

    try testing.expectApproxEqAbs(@as(f32, 0.005), def.width, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), def.rgba[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), def.rgba[1], 0.001);
}

test "tendon margin" {
    const def = TendonDef{
        .tendon_type = .fixed,
        .margin = 0.01,
    };

    try testing.expectApproxEqAbs(@as(f32, 0.01), def.margin, 0.001);
}

test "antagonist tendon pair" {
    // Two tendons with opposite coefficients
    const path1 = [_]TendonPathElement{
        .{ .index = 0, .coef = 1.0 },
    };
    const path2 = [_]TendonPathElement{
        .{ .index = 0, .coef = -1.0 },
    };

    const t1 = TendonDef{
        .name = "agonist",
        .tendon_type = .fixed,
        .path = &path1,
    };
    const t2 = TendonDef{
        .name = "antagonist",
        .tendon_type = .fixed,
        .path = &path2,
    };

    const joint_pos = [_]f32{0.5};

    try testing.expectApproxEqAbs(@as(f32, 0.5), t1.computeLength(&joint_pos), 0.001);
    try testing.expectApproxEqAbs(@as(f32, -0.5), t2.computeLength(&joint_pos), 0.001);
}
