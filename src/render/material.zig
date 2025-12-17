//! Material and Texture system for rendering.
//! Supports PBR materials with diffuse, normal, roughness, and metallic maps.

const std = @import("std");
const objc = @import("../objc.zig");

/// Texture format.
pub const TextureFormat = enum {
    rgba8,
    rgba16f,
    r8,
    rg8,
    depth32f,
};

/// Texture filter mode.
pub const FilterMode = enum {
    nearest,
    linear,
    trilinear,
};

/// Texture wrap mode.
pub const WrapMode = enum {
    repeat,
    clamp,
    mirror,
};

/// Texture definition.
pub const TextureDef = struct {
    width: u32 = 1,
    height: u32 = 1,
    format: TextureFormat = .rgba8,
    filter: FilterMode = .linear,
    wrap: WrapMode = .repeat,
    generate_mipmaps: bool = true,
};

/// GPU Texture handle.
pub const Texture = struct {
    handle: objc.id,
    width: u32,
    height: u32,
    format: TextureFormat,

    pub fn init(device: objc.id, def: TextureDef, data: ?[]const u8) !Texture {
        // Create texture descriptor
        const desc_class = objc.objc_getClass("MTLTextureDescriptor") orelse return error.NoTextureDescriptor;
        const desc = objc.msgSend(desc_class, objc.sel("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"), .{
            formatToMTL(def.format),
            @as(u64, def.width),
            @as(u64, def.height),
            def.generate_mipmaps,
        }) orelse return error.FailedToCreateDescriptor;

        // Set usage
        objc.msgSendVoid(desc, objc.sel("setUsage:"), .{@as(u64, 0x0001 | 0x0002)}); // ShaderRead | ShaderWrite

        // Create texture
        const texture = objc.msgSend(device, objc.sel("newTextureWithDescriptor:"), .{desc}) orelse return error.FailedToCreateTexture;

        // Upload data if provided
        if (data) |pixels| {
            const bytes_per_pixel: u64 = switch (def.format) {
                .rgba8 => 4,
                .rgba16f => 8,
                .r8 => 1,
                .rg8 => 2,
                .depth32f => 4,
            };

            const region = MTLRegion{
                .origin = .{ .x = 0, .y = 0, .z = 0 },
                .size = .{ .width = def.width, .height = def.height, .depth = 1 },
            };

            objc.msgSendVoid(texture, objc.sel("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"), .{
                region,
                @as(u64, 0),
                pixels.ptr,
                @as(u64, def.width) * bytes_per_pixel,
            });
        }

        return .{
            .handle = texture,
            .width = def.width,
            .height = def.height,
            .format = def.format,
        };
    }

    pub fn deinit(self: *Texture) void {
        objc.release(self.handle);
    }

    pub fn getHandle(self: *const Texture) objc.id {
        return self.handle;
    }
};

/// Metal region structure.
const MTLRegion = extern struct {
    origin: MTLOrigin,
    size: MTLSize,
};

const MTLOrigin = extern struct {
    x: u64,
    y: u64,
    z: u64,
};

const MTLSize = extern struct {
    width: u64,
    height: u64,
    depth: u64,
};

fn formatToMTL(format: TextureFormat) u64 {
    return switch (format) {
        .rgba8 => 80, // BGRA8Unorm
        .rgba16f => 115, // RGBA16Float
        .r8 => 10, // R8Unorm
        .rg8 => 30, // RG8Unorm
        .depth32f => 252, // Depth32Float
    };
}

/// PBR Material definition.
pub const Material = struct {
    /// Material name.
    name: []const u8 = "",

    /// Base color (albedo).
    base_color: [4]f32 = .{ 1, 1, 1, 1 },
    /// Metallic factor (0-1).
    metallic: f32 = 0.0,
    /// Roughness factor (0-1).
    roughness: f32 = 0.5,
    /// Emissive color.
    emissive: [3]f32 = .{ 0, 0, 0 },
    /// Normal map strength.
    normal_strength: f32 = 1.0,
    /// Ambient occlusion strength.
    ao_strength: f32 = 1.0,

    /// Texture indices (-1 = no texture).
    base_color_texture: i32 = -1,
    metallic_roughness_texture: i32 = -1,
    normal_texture: i32 = -1,
    emissive_texture: i32 = -1,
    ao_texture: i32 = -1,

    /// UV scale.
    uv_scale: [2]f32 = .{ 1, 1 },
    /// UV offset.
    uv_offset: [2]f32 = .{ 0, 0 },

    /// Is transparent.
    transparent: bool = false,
    /// Double-sided.
    double_sided: bool = false,
};

/// GPU-optimized material data.
pub const MaterialGPU = extern struct {
    /// Base color rgba.
    base_color: [4]f32 align(16),
    /// Metallic, roughness, normal_strength, ao_strength.
    params: [4]f32 align(16),
    /// Emissive rgb + padding.
    emissive: [4]f32 align(16),
    /// UV scale xy + offset xy.
    uv_transform: [4]f32 align(16),
    /// Texture indices (base, metallic_roughness, normal, emissive).
    texture_indices: [4]i32 align(16),
    /// Flags (transparent, double_sided, padding).
    flags: [4]u32 align(16),

    pub fn fromMaterial(m: *const Material) MaterialGPU {
        return .{
            .base_color = m.base_color,
            .params = .{ m.metallic, m.roughness, m.normal_strength, m.ao_strength },
            .emissive = .{ m.emissive[0], m.emissive[1], m.emissive[2], 0 },
            .uv_transform = .{ m.uv_scale[0], m.uv_scale[1], m.uv_offset[0], m.uv_offset[1] },
            .texture_indices = .{
                m.base_color_texture,
                m.metallic_roughness_texture,
                m.normal_texture,
                m.emissive_texture,
            },
            .flags = .{
                @intFromBool(m.transparent),
                @intFromBool(m.double_sided),
                0,
                0,
            },
        };
    }
};

/// Material library for managing materials and textures.
pub const MaterialLibrary = struct {
    allocator: std.mem.Allocator,
    device: objc.id,

    /// Materials.
    materials: std.ArrayListUnmanaged(Material),
    /// Material name to index lookup.
    material_map: std.StringHashMapUnmanaged(u32),

    /// Textures.
    textures: std.ArrayListUnmanaged(Texture),
    /// Texture name to index lookup.
    texture_map: std.StringHashMapUnmanaged(u32),

    /// Default white texture.
    default_white: ?u32 = null,
    /// Default normal texture.
    default_normal: ?u32 = null,
    /// Default black texture.
    default_black: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, device: objc.id) MaterialLibrary {
        return .{
            .allocator = allocator,
            .device = device,
            .materials = .{},
            .material_map = .{},
            .textures = .{},
            .texture_map = .{},
        };
    }

    pub fn deinit(self: *MaterialLibrary) void {
        for (self.textures.items) |*t| {
            t.deinit();
        }
        self.textures.deinit(self.allocator);
        self.materials.deinit(self.allocator);
        self.material_map.deinit(self.allocator);
        self.texture_map.deinit(self.allocator);
    }

    /// Create default textures.
    pub fn createDefaults(self: *MaterialLibrary) !void {
        // White texture (1x1)
        const white_data = [_]u8{ 255, 255, 255, 255 };
        self.default_white = try self.addTextureFromData("__default_white", &white_data, 1, 1, .rgba8);

        // Normal texture (flat normal pointing up)
        const normal_data = [_]u8{ 128, 128, 255, 255 };
        self.default_normal = try self.addTextureFromData("__default_normal", &normal_data, 1, 1, .rgba8);

        // Black texture
        const black_data = [_]u8{ 0, 0, 0, 255 };
        self.default_black = try self.addTextureFromData("__default_black", &black_data, 1, 1, .rgba8);

        // Default material
        _ = try self.addMaterial(.{
            .name = "__default",
            .base_color = .{ 0.8, 0.8, 0.8, 1.0 },
            .roughness = 0.5,
            .metallic = 0.0,
        });
    }

    /// Add a material.
    pub fn addMaterial(self: *MaterialLibrary, mat: Material) !u32 {
        const idx: u32 = @intCast(self.materials.items.len);
        try self.materials.append(self.allocator, mat);

        if (mat.name.len > 0) {
            try self.material_map.put(self.allocator, mat.name, idx);
        }

        return idx;
    }

    /// Get material by name.
    pub fn getMaterial(self: *const MaterialLibrary, name: []const u8) ?*const Material {
        if (self.material_map.get(name)) |idx| {
            return &self.materials.items[idx];
        }
        return null;
    }

    /// Get material index by name.
    pub fn getMaterialIndex(self: *const MaterialLibrary, name: []const u8) ?u32 {
        return self.material_map.get(name);
    }

    /// Add texture from raw data.
    pub fn addTextureFromData(
        self: *MaterialLibrary,
        name: []const u8,
        data: []const u8,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) !u32 {
        const idx: u32 = @intCast(self.textures.items.len);

        const texture = try Texture.init(self.device, .{
            .width = width,
            .height = height,
            .format = format,
        }, data);

        try self.textures.append(self.allocator, texture);

        if (name.len > 0) {
            try self.texture_map.put(self.allocator, name, idx);
        }

        return idx;
    }

    /// Load texture from file (PNG).
    pub fn loadTexture(self: *MaterialLibrary, name: []const u8, path: []const u8) !u32 {
        // Read file
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const data = try self.allocator.alloc(u8, stat.size);
        defer self.allocator.free(data);

        _ = try file.readAll(data);

        // Simple PNG decoder would go here
        // For now, assume raw RGBA data
        // In production, use a proper PNG decoder

        // Placeholder: create 1x1 texture
        const white = [_]u8{ 255, 255, 255, 255 };
        return try self.addTextureFromData(name, &white, 1, 1, .rgba8);
    }

    /// Get texture count.
    pub fn textureCount(self: *const MaterialLibrary) usize {
        return self.textures.items.len;
    }

    /// Get material count.
    pub fn materialCount(self: *const MaterialLibrary) usize {
        return self.materials.items.len;
    }
};

/// Builtin procedural texture generators.
pub const ProceduralTextures = struct {
    /// Generate checkerboard pattern.
    pub fn checkerboard(allocator: std.mem.Allocator, size: u32, tile_size: u32, color1: [4]u8, color2: [4]u8) ![]u8 {
        const data = try allocator.alloc(u8, size * size * 4);

        for (0..size) |y| {
            for (0..size) |x| {
                const idx = (y * size + x) * 4;
                const tile_x = x / tile_size;
                const tile_y = y / tile_size;
                const color = if ((tile_x + tile_y) % 2 == 0) color1 else color2;

                data[idx + 0] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
                data[idx + 3] = color[3];
            }
        }

        return data;
    }

    /// Generate gradient texture.
    pub fn gradient(allocator: std.mem.Allocator, width: u32, height: u32, start: [4]u8, end: [4]u8, horizontal: bool) ![]u8 {
        const data = try allocator.alloc(u8, width * height * 4);

        for (0..height) |y| {
            for (0..width) |x| {
                const idx = (y * width + x) * 4;
                const t = if (horizontal)
                    @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(width - 1))
                else
                    @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(height - 1));

                data[idx + 0] = @intFromFloat(@as(f32, @floatFromInt(start[0])) * (1 - t) + @as(f32, @floatFromInt(end[0])) * t);
                data[idx + 1] = @intFromFloat(@as(f32, @floatFromInt(start[1])) * (1 - t) + @as(f32, @floatFromInt(end[1])) * t);
                data[idx + 2] = @intFromFloat(@as(f32, @floatFromInt(start[2])) * (1 - t) + @as(f32, @floatFromInt(end[2])) * t);
                data[idx + 3] = @intFromFloat(@as(f32, @floatFromInt(start[3])) * (1 - t) + @as(f32, @floatFromInt(end[3])) * t);
            }
        }

        return data;
    }

    /// Generate noise texture (simple value noise).
    pub fn noise(allocator: std.mem.Allocator, size: u32, scale: f32, seed: u32) ![]u8 {
        const data = try allocator.alloc(u8, size * size * 4);

        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        for (0..size) |y| {
            for (0..size) |x| {
                const idx = (y * size + x) * 4;

                // Simple value noise
                const fx = @as(f32, @floatFromInt(x)) * scale;
                const fy = @as(f32, @floatFromInt(y)) * scale;

                const noise_val = valueNoise(fx, fy, random);
                const val: u8 = @intFromFloat(noise_val * 255);

                data[idx + 0] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
                data[idx + 3] = 255;
            }
        }

        return data;
    }

    fn valueNoise(x: f32, y: f32, random: std.Random) f32 {
        _ = random;
        // Simple pseudo-random based on position
        const ix: i32 = @intFromFloat(@floor(x));
        const iy: i32 = @intFromFloat(@floor(y));

        const fx = x - @floor(x);
        const fy = y - @floor(y);

        // Smooth interpolation
        const u = fx * fx * (3 - 2 * fx);
        const v = fy * fy * (3 - 2 * fy);

        // Hash corners
        const n00 = hashNoise(ix, iy);
        const n10 = hashNoise(ix + 1, iy);
        const n01 = hashNoise(ix, iy + 1);
        const n11 = hashNoise(ix + 1, iy + 1);

        // Bilinear interpolation
        const nx0 = n00 * (1 - u) + n10 * u;
        const nx1 = n01 * (1 - u) + n11 * u;

        return nx0 * (1 - v) + nx1 * v;
    }

    fn hashNoise(x: i32, y: i32) f32 {
        const ux: u32 = @bitCast(x);
        const uy: u32 = @bitCast(y);
        var h = ((ux *% 1597334673) ^ (uy *% 3812015801));
        h = h ^ (h >> 16);
        h *%= 0x85ebca6b;
        h ^= h >> 13;
        return @as(f32, @floatFromInt(h & 0xFFFF)) / 65535.0;
    }
};

// Tests

test "create material" {
    const mat = Material{
        .name = "test",
        .base_color = .{ 1, 0, 0, 1 },
        .metallic = 0.5,
        .roughness = 0.3,
    };

    const gpu = MaterialGPU.fromMaterial(&mat);
    try std.testing.expectEqual(gpu.base_color[0], 1.0);
    try std.testing.expectEqual(gpu.params[0], 0.5);
    try std.testing.expectEqual(gpu.params[1], 0.3);
}

test "procedural checkerboard" {
    const allocator = std.testing.allocator;

    const data = try ProceduralTextures.checkerboard(
        allocator,
        8,
        2,
        .{ 255, 255, 255, 255 },
        .{ 0, 0, 0, 255 },
    );
    defer allocator.free(data);

    try std.testing.expectEqual(data.len, 8 * 8 * 4);
    // Top-left should be white
    try std.testing.expectEqual(data[0], 255);
}
