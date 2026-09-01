//! Tests for image decoding and render materials.

const std = @import("std");
const zeno = @import("zeno");
const material = zeno.render.material;

const png = [_]u8{
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0xf4, 0x22, 0x7f,
    0x8a, 0x00, 0x00, 0x00, 0x0f, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0,
    0x1f, 0x08, 0x1b, 0x00, 0x10, 0x79, 0x03, 0x7e, 0x7d, 0x63, 0xce, 0xd7, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

test "material module is available from the public Zig API" {
    const gpu = material.MaterialGPU.fromMaterial(&.{});
    try std.testing.expectEqual(@as(f32, 1), gpu.base_color[0]);
}

test "ImageIO decodes PNG to straight-alpha RGBA8" {
    var decoded = try material.decodeImage(std.testing.allocator, &png);
    defer decoded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), decoded.width);
    try std.testing.expectEqual(@as(u32, 1), decoded.height);
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255 }, decoded.pixels[0..4]);
    try std.testing.expectApproxEqAbs(@as(f32, 255), @as(f32, @floatFromInt(decoded.pixels[5])), 1);
    try std.testing.expectApproxEqAbs(@as(f32, 128), @as(f32, @floatFromInt(decoded.pixels[7])), 1);
}

test "invalid encoded image is rejected" {
    try std.testing.expectError(error.UnsupportedImageFormat, material.decodeImage(std.testing.allocator, "not an image"));
}

test "decoded pixels upload to an RGBA8 Metal texture" {
    var device = zeno.metal.device.Device.init(std.testing.allocator) catch |err| {
        std.log.warn("Metal not available: {}", .{err});
        return;
    };
    defer device.deinit();

    var decoded = try material.decodeImage(std.testing.allocator, &png);
    defer decoded.deinit(std.testing.allocator);

    var library = material.MaterialLibrary.init(std.testing.allocator, device.device);
    defer library.deinit();

    const index = try library.addTextureFromData("decoded", decoded.pixels, decoded.width, decoded.height, .rgba8);
    try std.testing.expectEqual(@as(u32, 0), index);
    try std.testing.expectEqual(@as(usize, 1), library.textureCount());
    try std.testing.expectEqual(@as(u32, 2), library.textures.items[index].width);
    try std.testing.expectEqual(@as(u32, 1), library.textures.items[index].height);
}

test "raw texture upload validates byte length" {
    var device = zeno.metal.device.Device.init(std.testing.allocator) catch return;
    defer device.deinit();

    var library = material.MaterialLibrary.init(std.testing.allocator, device.device);
    defer library.deinit();

    try std.testing.expectError(
        error.InvalidTextureDataLength,
        library.addTextureFromData("short", &.{ 1, 2, 3 }, 1, 1, .rgba8),
    );
}
