//! Zeno High-Performance Renderer
//! Zero-copy Metal rendering with physics buffer sharing

const std = @import("std");
const objc = @import("../objc.zig");
const Buffer = @import("../metal/buffer.zig").Buffer;
const BufferOptions = @import("../metal/buffer.zig").BufferOptions;
const primitives = @import("../collision/primitives.zig");

/// Camera uniforms for GPU
pub const CameraUniforms = extern struct {
    view_matrix: [16]f32 align(16),
    projection_matrix: [16]f32 align(16),
    view_projection: [16]f32 align(16),
    camera_pos: [3]f32 align(16),
    time: f32,
};

/// Instance data for rendering
pub const InstanceData = extern struct {
    position: [4]f32 align(16),
    quaternion: [4]f32 align(16),
    size: [4]f32 align(16),
    color: [4]f32 align(16),
    geom_type: u32,
    body_id: u32,
    env_id: u32,
    _pad: u32 = 0,
};

/// Camera for 3D viewing
pub const Camera = struct {
    position: [3]f32 = .{ 3, 3, 2 },
    target: [3]f32 = .{ 0, 0, 0.5 },
    up: [3]f32 = .{ 0, 0, 1 },
    fov: f32 = 60.0,
    near: f32 = 0.01,
    far: f32 = 1000.0,
    aspect: f32 = 16.0 / 9.0,

    // Orbit controls
    distance: f32 = 5.0,
    azimuth: f32 = 0.5,
    elevation: f32 = 0.4,

    pub fn updateOrbit(self: *Camera) void {
        self.position[0] = self.target[0] + self.distance * @cos(self.elevation) * @cos(self.azimuth);
        self.position[1] = self.target[1] + self.distance * @cos(self.elevation) * @sin(self.azimuth);
        self.position[2] = self.target[2] + self.distance * @sin(self.elevation);
    }

    pub fn getViewMatrix(self: *const Camera) [16]f32 {
        return lookAt(self.position, self.target, self.up);
    }

    pub fn getProjectionMatrix(self: *const Camera) [16]f32 {
        return perspective(self.fov, self.aspect, self.near, self.far);
    }

    pub fn getUniforms(self: *const Camera, time: f32) CameraUniforms {
        const view = self.getViewMatrix();
        const proj = self.getProjectionMatrix();
        return .{
            .view_matrix = view,
            .projection_matrix = proj,
            .view_projection = multiplyMat4(proj, view),
            .camera_pos = self.position,
            .time = time,
        };
    }
};

/// High-performance renderer
pub const Renderer = struct {
    device: objc.id,
    command_queue: objc.id,
    library: objc.id,

    // Pipelines
    geometry_pipeline: objc.id,
    ground_pipeline: objc.id,
    line_pipeline: objc.id,

    // Buffers
    camera_buffer: Buffer,
    instance_buffer: Buffer,
    vertex_buffer: Buffer,
    index_buffer: Buffer,

    // Geometry data
    sphere_vertex_count: u32,
    sphere_index_count: u32,
    capsule_vertex_count: u32,
    capsule_index_count: u32,
    box_vertex_count: u32,
    box_index_count: u32,

    // State
    camera: Camera,
    max_instances: u32,
    instance_count: u32 = 0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, device: objc.id, max_instances: u32) !Renderer {
        // Create command queue
        const queue = objc.msgSend(device, objc.sel("newCommandQueue"), .{}) orelse
            return error.FailedToCreateCommandQueue;

        // Load shaders
        const shader_source = @embedFile("shaders.metal");
        const library = try loadShaderLibrary(device, shader_source);

        // Create pipelines
        const geometry_pipeline = try createRenderPipeline(device, library, "vertexMain", "fragmentMain");
        const ground_pipeline = try createRenderPipeline(device, library, "groundVertex", "groundFragment");
        const line_pipeline = try createLinePipeline(device, library);

        // Create buffers
        const opts = BufferOptions{ .storage_mode = .shared };
        const camera_buffer = try Buffer.init(device, @sizeOf(CameraUniforms), opts);
        const instance_buffer = try Buffer.init(device, max_instances * @sizeOf(InstanceData), opts);

        // Generate geometry
        var vertex_data = std.ArrayList(f32).init(allocator);
        defer vertex_data.deinit();
        var index_data = std.ArrayList(u32).init(allocator);
        defer index_data.deinit();

        // Sphere (UV sphere)
        const sphere_start_v = vertex_data.items.len / 8;
        const sphere_start_i = index_data.items.len;
        try generateSphere(&vertex_data, &index_data, 16, 12);
        const sphere_vertex_count: u32 = @intCast((vertex_data.items.len / 8) - sphere_start_v);
        const sphere_index_count: u32 = @intCast(index_data.items.len - sphere_start_i);

        // Capsule
        const capsule_start_i = index_data.items.len;
        try generateCapsule(&vertex_data, &index_data, 12, 6);
        const capsule_index_count: u32 = @intCast(index_data.items.len - capsule_start_i);

        // Box
        const box_start_i = index_data.items.len;
        try generateBox(&vertex_data, &index_data);
        const box_index_count: u32 = @intCast(index_data.items.len - box_start_i);

        // Create vertex/index buffers
        const vertex_bytes = std.mem.sliceAsBytes(vertex_data.items);
        const index_bytes = std.mem.sliceAsBytes(index_data.items);

        const vertex_buffer = try Buffer.initWithData(device, vertex_bytes, opts);
        const index_buffer = try Buffer.initWithData(device, index_bytes, opts);

        return .{
            .device = device,
            .command_queue = queue,
            .library = library,
            .geometry_pipeline = geometry_pipeline,
            .ground_pipeline = ground_pipeline,
            .line_pipeline = line_pipeline,
            .camera_buffer = camera_buffer,
            .instance_buffer = instance_buffer,
            .vertex_buffer = vertex_buffer,
            .index_buffer = index_buffer,
            .sphere_vertex_count = sphere_vertex_count,
            .sphere_index_count = sphere_index_count,
            .capsule_vertex_count = 0, // Calculated from indices
            .capsule_index_count = capsule_index_count,
            .box_vertex_count = 0,
            .box_index_count = box_index_count,
            .camera = Camera{},
            .max_instances = max_instances,
            .allocator = allocator,
        };
    }

    /// Update instances from physics state (zero-copy when possible)
    pub fn updateFromPhysics(
        self: *Renderer,
        positions: [][4]f32,
        quaternions: [][4]f32,
        geoms: []const primitives.GeomGPU,
        num_envs: u32,
        env_to_render: u32,
    ) void {
        const instances = self.instance_buffer.getSlice(InstanceData);
        var count: u32 = 0;

        const num_bodies = @as(u32, @intCast(positions.len)) / num_envs;
        const env_offset = env_to_render * num_bodies;

        for (geoms) |geom| {
            if (count >= self.max_instances) break;

            const body_id = geom.body_id;
            const idx = env_offset + body_id;
            if (idx >= positions.len) continue;

            instances[count] = .{
                .position = .{ positions[idx][0], positions[idx][1], positions[idx][2], 1.0 },
                .quaternion = quaternions[idx],
                .size = geom.size,
                .color = geom.color,
                .geom_type = @intFromEnum(geom.geom_type),
                .body_id = body_id,
                .env_id = env_to_render,
            };
            count += 1;
        }

        self.instance_count = count;
    }

    /// Render frame to drawable
    pub fn render(self: *Renderer, drawable: objc.id, time: f32) void {
        // Update camera
        self.camera.updateOrbit();
        const uniforms = self.camera.getUniforms(time);
        const camera_slice = self.camera_buffer.getSlice(CameraUniforms);
        camera_slice[0] = uniforms;

        // Create command buffer
        const cmd_buffer = objc.msgSend(self.command_queue, objc.sel("commandBuffer"), .{}) orelse return;

        // Get render pass descriptor from drawable
        const texture = objc.msgSend(drawable, objc.sel("texture"), .{}) orelse return;

        // Create render pass descriptor
        const rpd_class = objc.objc_getClass("MTLRenderPassDescriptor") orelse return;
        const rpd = objc.msgSend(rpd_class, objc.sel("renderPassDescriptor"), .{}) orelse return;

        // Configure color attachment
        const color_attachments = objc.msgSend(rpd, objc.sel("colorAttachments"), .{}) orelse return;
        const color0 = objc.msgSend(color_attachments, objc.sel("objectAtIndexedSubscript:"), .{@as(u64, 0)}) orelse return;

        objc.msgSendVoid(color0, objc.sel("setTexture:"), .{texture});
        objc.msgSendVoid(color0, objc.sel("setLoadAction:"), .{@as(u64, 2)}); // Clear
        objc.msgSendVoid(color0, objc.sel("setStoreAction:"), .{@as(u64, 1)}); // Store
        objc.msgSendVoid(color0, objc.sel("setClearColor:"), .{MTLClearColor{ .red = 0.1, .green = 0.1, .blue = 0.15, .alpha = 1.0 }});

        // Create encoder
        const encoder = objc.msgSend(cmd_buffer, objc.sel("renderCommandEncoderWithDescriptor:"), .{rpd}) orelse return;

        // Render ground
        objc.msgSendVoid(encoder, objc.sel("setRenderPipelineState:"), .{self.ground_pipeline});
        objc.msgSendVoid(encoder, objc.sel("setVertexBuffer:offset:atIndex:"), .{ self.vertex_buffer.getHandle(), @as(u64, 0), @as(u64, 0) });
        objc.msgSendVoid(encoder, objc.sel("setVertexBuffer:offset:atIndex:"), .{ self.camera_buffer.getHandle(), @as(u64, 0), @as(u64, 1) });
        objc.msgSendVoid(encoder, objc.sel("setFragmentBuffer:offset:atIndex:"), .{ self.camera_buffer.getHandle(), @as(u64, 0), @as(u64, 1) });
        objc.msgSendVoid(encoder, objc.sel("drawPrimitives:vertexStart:vertexCount:"), .{ @as(u64, 4), @as(u64, 0), @as(u64, 6) }); // Triangle strip for quad

        // Render geometry instances
        if (self.instance_count > 0) {
            objc.msgSendVoid(encoder, objc.sel("setRenderPipelineState:"), .{self.geometry_pipeline});
            objc.msgSendVoid(encoder, objc.sel("setVertexBuffer:offset:atIndex:"), .{ self.instance_buffer.getHandle(), @as(u64, 0), @as(u64, 2) });

            // Draw spheres (index 0 in instances that are spheres)
            objc.msgSendVoid(encoder, objc.sel("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"), .{
                @as(u64, 3), // Triangles
                @as(u64, self.sphere_index_count),
                @as(u64, 0), // UInt32
                self.index_buffer.getHandle(),
                @as(u64, 0),
                @as(u64, self.instance_count),
            });
        }

        objc.msgSendVoid(encoder, objc.sel("endEncoding"), .{});

        // Present
        objc.msgSendVoid(cmd_buffer, objc.sel("presentDrawable:"), .{drawable});
        objc.msgSendVoid(cmd_buffer, objc.sel("commit"), .{});
    }

    pub fn deinit(self: *Renderer) void {
        self.camera_buffer.deinit();
        self.instance_buffer.deinit();
        self.vertex_buffer.deinit();
        self.index_buffer.deinit();
        objc.release(self.geometry_pipeline);
        objc.release(self.ground_pipeline);
        objc.release(self.line_pipeline);
        objc.release(self.library);
        objc.release(self.command_queue);
    }
};

// Metal clear color struct
const MTLClearColor = extern struct {
    red: f64,
    green: f64,
    blue: f64,
    alpha: f64,
};

fn loadShaderLibrary(device: objc.id, source: []const u8) !objc.id {
    const ns_string_class = objc.objc_getClass("NSString") orelse return error.NoNSString;
    const source_string = objc.msgSend(ns_string_class, objc.sel("stringWithUTF8String:"), .{source.ptr}) orelse return error.FailedToCreateString;

    const options_class = objc.objc_getClass("MTLCompileOptions") orelse return error.NoCompileOptions;
    const options = objc.msgSend(options_class, objc.sel("new"), .{}) orelse return error.FailedToCreateOptions;
    defer objc.release(options);

    var err: objc.id = null;
    const library = objc.msgSend(device, objc.sel("newLibraryWithSource:options:error:"), .{ source_string, options, &err });

    if (library == null) {
        if (err != null) {
            const desc = objc.msgSend(err, objc.sel("localizedDescription"), .{});
            if (desc != null) {
                const cstr = objc.msgSend(desc, objc.sel("UTF8String"), .{});
                if (cstr != null) {
                    std.debug.print("Shader compile error: {s}\n", .{@as([*:0]const u8, @ptrCast(cstr))});
                }
            }
        }
        return error.FailedToCompileShaders;
    }

    return library.?;
}

fn createRenderPipeline(device: objc.id, library: objc.id, vertex_fn: [*:0]const u8, fragment_fn: [*:0]const u8) !objc.id {
    const ns_string = objc.objc_getClass("NSString") orelse return error.NoNSString;

    const vertex_name = objc.msgSend(ns_string, objc.sel("stringWithUTF8String:"), .{vertex_fn}) orelse return error.BadString;
    const fragment_name = objc.msgSend(ns_string, objc.sel("stringWithUTF8String:"), .{fragment_fn}) orelse return error.BadString;

    const vertex_func = objc.msgSend(library, objc.sel("newFunctionWithName:"), .{vertex_name}) orelse return error.NoVertexFunction;
    defer objc.release(vertex_func);

    const fragment_func = objc.msgSend(library, objc.sel("newFunctionWithName:"), .{fragment_name}) orelse return error.NoFragmentFunction;
    defer objc.release(fragment_func);

    const desc_class = objc.objc_getClass("MTLRenderPipelineDescriptor") orelse return error.NoDescClass;
    const desc = objc.msgSend(desc_class, objc.sel("new"), .{}) orelse return error.FailedToCreateDesc;
    defer objc.release(desc);

    objc.msgSendVoid(desc, objc.sel("setVertexFunction:"), .{vertex_func});
    objc.msgSendVoid(desc, objc.sel("setFragmentFunction:"), .{fragment_func});

    // Set pixel format
    const color_attachments = objc.msgSend(desc, objc.sel("colorAttachments"), .{}) orelse return error.NoColorAttachments;
    const color0 = objc.msgSend(color_attachments, objc.sel("objectAtIndexedSubscript:"), .{@as(u64, 0)}) orelse return error.NoColor0;
    objc.msgSendVoid(color0, objc.sel("setPixelFormat:"), .{@as(u64, 80)}); // BGRA8Unorm

    var err: objc.id = null;
    const pipeline = objc.msgSend(device, objc.sel("newRenderPipelineStateWithDescriptor:error:"), .{ desc, &err });

    if (pipeline == null) return error.FailedToCreatePipeline;
    return pipeline.?;
}

fn createLinePipeline(device: objc.id, library: objc.id) !objc.id {
    return createRenderPipeline(device, library, "lineVertex", "lineFragment");
}

// Geometry generation
fn generateSphere(vertices: *std.ArrayList(f32), indices: *std.ArrayList(u32), segments: u32, rings: u32) !void {
    const base_vertex: u32 = @intCast(vertices.items.len / 8);

    // Generate vertices
    for (0..rings + 1) |j| {
        const v = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(rings));
        const phi = v * std.math.pi;

        for (0..segments + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
            const theta = u * std.math.pi * 2.0;

            const x = @sin(phi) * @cos(theta);
            const y = @sin(phi) * @sin(theta);
            const z = @cos(phi);

            // Position
            try vertices.append(x);
            try vertices.append(y);
            try vertices.append(z);
            // Normal (same as position for unit sphere)
            try vertices.append(x);
            try vertices.append(y);
            try vertices.append(z);
            // UV
            try vertices.append(u);
            try vertices.append(v);
        }
    }

    // Generate indices
    for (0..rings) |j| {
        for (0..segments) |i| {
            const i0 = base_vertex + @as(u32, @intCast(j * (segments + 1) + i));
            const i1 = i0 + 1;
            const i2 = i0 + @as(u32, @intCast(segments + 1));
            const i3 = i2 + 1;

            try indices.append(i0);
            try indices.append(i2);
            try indices.append(i1);

            try indices.append(i1);
            try indices.append(i2);
            try indices.append(i3);
        }
    }
}

fn generateCapsule(vertices: *std.ArrayList(f32), indices: *std.ArrayList(u32), segments: u32, rings: u32) !void {
    // Proper capsule: two hemispheres connected by a cylinder
    // The capsule is aligned along Z-axis with total height 2 (1 for each cap + 0 for cylinder at unit scale)
    // Scale will be applied in the shader
    const base_vertex: u32 = @intCast(vertices.items.len / 8);
    const half_height: f32 = 0.5; // Half the cylinder height (between hemispheres)

    // Top hemisphere (Z > 0)
    for (0..rings / 2 + 1) |j| {
        const v = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(rings));
        const phi = v * std.math.pi; // 0 to PI/2 for top hemisphere

        for (0..segments + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
            const theta = u * std.math.pi * 2.0;

            const nx = @sin(phi) * @cos(theta);
            const ny = @sin(phi) * @sin(theta);
            const nz = @cos(phi);

            // Position = normal * radius + offset
            const x = nx;
            const y = ny;
            const z = nz + half_height; // Offset up

            try vertices.append(x);
            try vertices.append(y);
            try vertices.append(z);
            try vertices.append(nx);
            try vertices.append(ny);
            try vertices.append(nz);
            try vertices.append(u);
            try vertices.append(v * 0.5);
        }
    }

    const top_hemisphere_verts = (rings / 2 + 1) * (segments + 1);

    // Cylinder middle section (2 rings)
    for (0..2) |j| {
        const z_offset = if (j == 0) half_height else -half_height;

        for (0..segments + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
            const theta = u * std.math.pi * 2.0;

            const nx = @cos(theta);
            const ny = @sin(theta);

            try vertices.append(nx); // x = normal * radius
            try vertices.append(ny);
            try vertices.append(z_offset);
            try vertices.append(nx);
            try vertices.append(ny);
            try vertices.append(0); // Normal points outward radially
            try vertices.append(u);
            try vertices.append(0.5);
        }
    }

    const cylinder_verts = 2 * (segments + 1);

    // Bottom hemisphere (Z < 0)
    for (0..rings / 2 + 1) |j| {
        const v = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(rings));
        const phi = std.math.pi / 2.0 + v * std.math.pi / 2.0; // PI/2 to PI for bottom

        for (0..segments + 1) |i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
            const theta = u * std.math.pi * 2.0;

            const nx = @sin(phi) * @cos(theta);
            const ny = @sin(phi) * @sin(theta);
            const nz = @cos(phi);

            const x = nx;
            const y = ny;
            const z = nz - half_height; // Offset down

            try vertices.append(x);
            try vertices.append(y);
            try vertices.append(z);
            try vertices.append(nx);
            try vertices.append(ny);
            try vertices.append(nz);
            try vertices.append(u);
            try vertices.append(0.5 + v * 0.5);
        }
    }

    // Generate indices for top hemisphere
    for (0..rings / 2) |j| {
        for (0..segments) |i| {
            const i0 = base_vertex + @as(u32, @intCast(j * (segments + 1) + i));
            const i1 = i0 + 1;
            const i2 = i0 + @as(u32, @intCast(segments + 1));
            const i3 = i2 + 1;

            try indices.append(i0);
            try indices.append(i2);
            try indices.append(i1);
            try indices.append(i1);
            try indices.append(i2);
            try indices.append(i3);
        }
    }

    // Generate indices for cylinder
    const cyl_base = base_vertex + @as(u32, @intCast(top_hemisphere_verts));
    for (0..segments) |i| {
        const i0 = cyl_base + @as(u32, @intCast(i));
        const i1 = i0 + 1;
        const i2 = i0 + @as(u32, @intCast(segments + 1));
        const i3 = i2 + 1;

        try indices.append(i0);
        try indices.append(i2);
        try indices.append(i1);
        try indices.append(i1);
        try indices.append(i2);
        try indices.append(i3);
    }

    // Generate indices for bottom hemisphere
    const bot_base = base_vertex + @as(u32, @intCast(top_hemisphere_verts + cylinder_verts));
    for (0..rings / 2) |j| {
        for (0..segments) |i| {
            const i0 = bot_base + @as(u32, @intCast(j * (segments + 1) + i));
            const i1 = i0 + 1;
            const i2 = i0 + @as(u32, @intCast(segments + 1));
            const i3 = i2 + 1;

            try indices.append(i0);
            try indices.append(i2);
            try indices.append(i1);
            try indices.append(i1);
            try indices.append(i2);
            try indices.append(i3);
        }
    }
}

fn generateBox(vertices: *std.ArrayList(f32), indices: *std.ArrayList(u32)) !void {
    const base_vertex: u32 = @intCast(vertices.items.len / 8);

    // Box vertices (24 vertices for proper normals)
    const positions = [_][3]f32{
        // Front face
        .{ -1, -1, 1 }, .{ 1, -1, 1 }, .{ 1, 1, 1 }, .{ -1, 1, 1 },
        // Back face
        .{ 1, -1, -1 }, .{ -1, -1, -1 }, .{ -1, 1, -1 }, .{ 1, 1, -1 },
        // Top face
        .{ -1, 1, 1 }, .{ 1, 1, 1 }, .{ 1, 1, -1 }, .{ -1, 1, -1 },
        // Bottom face
        .{ -1, -1, -1 }, .{ 1, -1, -1 }, .{ 1, -1, 1 }, .{ -1, -1, 1 },
        // Right face
        .{ 1, -1, 1 }, .{ 1, -1, -1 }, .{ 1, 1, -1 }, .{ 1, 1, 1 },
        // Left face
        .{ -1, -1, -1 }, .{ -1, -1, 1 }, .{ -1, 1, 1 }, .{ -1, 1, -1 },
    };

    const normals = [_][3]f32{
        .{ 0, 0, 1 },  .{ 0, 0, 1 },  .{ 0, 0, 1 },  .{ 0, 0, 1 },
        .{ 0, 0, -1 }, .{ 0, 0, -1 }, .{ 0, 0, -1 }, .{ 0, 0, -1 },
        .{ 0, 1, 0 },  .{ 0, 1, 0 },  .{ 0, 1, 0 },  .{ 0, 1, 0 },
        .{ 0, -1, 0 }, .{ 0, -1, 0 }, .{ 0, -1, 0 }, .{ 0, -1, 0 },
        .{ 1, 0, 0 },  .{ 1, 0, 0 },  .{ 1, 0, 0 },  .{ 1, 0, 0 },
        .{ -1, 0, 0 }, .{ -1, 0, 0 }, .{ -1, 0, 0 }, .{ -1, 0, 0 },
    };

    for (0..24) |i| {
        try vertices.append(positions[i][0]);
        try vertices.append(positions[i][1]);
        try vertices.append(positions[i][2]);
        try vertices.append(normals[i][0]);
        try vertices.append(normals[i][1]);
        try vertices.append(normals[i][2]);
        try vertices.append(0); // UV
        try vertices.append(0);
    }

    // Indices (6 faces, 2 triangles each)
    for (0..6) |face| {
        const base = base_vertex + @as(u32, @intCast(face * 4));
        try indices.append(base);
        try indices.append(base + 1);
        try indices.append(base + 2);
        try indices.append(base);
        try indices.append(base + 2);
        try indices.append(base + 3);
    }
}

// Matrix math
fn lookAt(eye: [3]f32, target: [3]f32, up: [3]f32) [16]f32 {
    const f = normalize3(sub3(target, eye));
    const s = normalize3(cross3(f, up));
    const u = cross3(s, f);

    return .{
        s[0],              u[0],              -f[0],             0,
        s[1],              u[1],              -f[1],             0,
        s[2],              u[2],              -f[2],             0,
        -dot3(s, eye),     -dot3(u, eye),     dot3(f, eye),      1,
    };
}

fn perspective(fov_deg: f32, aspect: f32, near: f32, far: f32) [16]f32 {
    const fov_rad = fov_deg * std.math.pi / 180.0;
    const f = 1.0 / @tan(fov_rad / 2.0);

    return .{
        f / aspect, 0, 0,                                0,
        0,          f, 0,                                0,
        0,          0, (far + near) / (near - far),      -1,
        0,          0, (2 * far * near) / (near - far),  0,
    };
}

fn multiplyMat4(a: [16]f32, b: [16]f32) [16]f32 {
    var result: [16]f32 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            var sum: f32 = 0;
            for (0..4) |k| {
                sum += a[i * 4 + k] * b[k * 4 + j];
            }
            result[i * 4 + j] = sum;
        }
    }
    return result;
}

fn sub3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn cross3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn dot3(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn normalize3(v: [3]f32) [3]f32 {
    const len = @sqrt(dot3(v, v));
    if (len == 0) return v;
    return .{ v[0] / len, v[1] / len, v[2] / len };
}
