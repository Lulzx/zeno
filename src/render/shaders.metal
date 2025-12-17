//  Zeno Renderer - High-performance Metal shaders
//  Zero-copy rendering with physics buffer sharing

#include <metal_stdlib>
using namespace metal;

// Vertex data structures
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float3 color;
    float3 viewDir;
};

// Uniforms
struct CameraUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjection;
    float3 cameraPos;
    float time;
};

struct InstanceData {
    float4 position;     // xyz = position, w = scale
    float4 quaternion;   // xyzw quaternion
    float4 size;         // geometry-specific sizes
    float4 color;        // rgba
    uint geomType;       // 0=sphere, 1=capsule, 2=box, 3=cylinder, 4=plane
    uint bodyId;
    uint envId;
    uint _pad;
};

// Quaternion rotation
float3 rotateByQuat(float3 v, float4 q) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

float4x4 quatToMatrix(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    return float4x4(
        float4(1.0 - (yy + zz), xy + wz, xz - wy, 0.0),
        float4(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0),
        float4(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}

// Main vertex shader for instanced geometry
vertex VertexOut vertexMain(
    VertexIn in [[stage_in]],
    constant CameraUniforms& camera [[buffer(1)]],
    constant InstanceData* instances [[buffer(2)]],
    uint instanceId [[instance_id]]
) {
    InstanceData inst = instances[instanceId];

    // Apply instance scale based on geometry type
    float3 scaledPos = in.position;
    if (inst.geomType == 0) { // Sphere
        scaledPos *= inst.size.x; // radius
    } else if (inst.geomType == 1) { // Capsule
        scaledPos.xy *= inst.size.x; // radius
        scaledPos.z *= inst.size.y; // half-length
    } else if (inst.geomType == 2) { // Box
        scaledPos *= inst.size.xyz; // half-extents
    } else if (inst.geomType == 3) { // Cylinder
        scaledPos.xy *= inst.size.x; // radius
        scaledPos.z *= inst.size.y; // half-height
    }

    // Apply rotation
    float3 rotatedPos = rotateByQuat(scaledPos, inst.quaternion);
    float3 rotatedNormal = rotateByQuat(in.normal, inst.quaternion);

    // Apply translation
    float3 worldPos = rotatedPos + inst.position.xyz;

    VertexOut out;
    out.position = camera.viewProjection * float4(worldPos, 1.0);
    out.worldPos = worldPos;
    out.normal = normalize(rotatedNormal);
    out.color = inst.color.rgb;
    out.viewDir = normalize(camera.cameraPos - worldPos);

    return out;
}

// Fragment shader with PBR-lite lighting
fragment float4 fragmentMain(
    VertexOut in [[stage_in]],
    constant CameraUniforms& camera [[buffer(1)]]
) {
    // Light direction (sun-like)
    float3 lightDir = normalize(float3(0.5, 0.3, 1.0));
    float3 lightColor = float3(1.0, 0.98, 0.95);

    // Ambient
    float3 ambient = 0.15 * in.color;

    // Diffuse (Lambert)
    float NdotL = max(dot(in.normal, lightDir), 0.0);
    float3 diffuse = NdotL * in.color * lightColor;

    // Specular (Blinn-Phong)
    float3 halfVec = normalize(lightDir + in.viewDir);
    float NdotH = max(dot(in.normal, halfVec), 0.0);
    float spec = pow(NdotH, 32.0);
    float3 specular = spec * lightColor * 0.3;

    // Fresnel rim
    float fresnel = pow(1.0 - max(dot(in.normal, in.viewDir), 0.0), 3.0);
    float3 rim = fresnel * 0.1 * lightColor;

    // Ground shadow approximation
    float shadow = smoothstep(-0.5, 0.5, in.worldPos.z);

    float3 finalColor = (ambient + diffuse * shadow + specular + rim);

    // Gamma correction
    finalColor = pow(finalColor, float3(1.0/2.2));

    return float4(finalColor, 1.0);
}

// Ground plane shader
vertex VertexOut groundVertex(
    VertexIn in [[stage_in]],
    constant CameraUniforms& camera [[buffer(1)]]
) {
    float3 worldPos = in.position * 50.0; // Large ground plane
    worldPos.z = 0.0;

    VertexOut out;
    out.position = camera.viewProjection * float4(worldPos, 1.0);
    out.worldPos = worldPos;
    out.normal = float3(0, 0, 1);
    out.color = float3(0.4, 0.45, 0.4);
    out.viewDir = normalize(camera.cameraPos - worldPos);

    return out;
}

fragment float4 groundFragment(
    VertexOut in [[stage_in]],
    constant CameraUniforms& camera [[buffer(1)]]
) {
    // Checkerboard pattern
    float2 uv = in.worldPos.xy;
    float checker = fmod(floor(uv.x) + floor(uv.y), 2.0);
    float3 color1 = float3(0.35, 0.4, 0.35);
    float3 color2 = float3(0.45, 0.5, 0.45);
    float3 baseColor = mix(color1, color2, checker);

    // Distance fade
    float dist = length(in.worldPos.xy - camera.cameraPos.xy);
    float fade = 1.0 - smoothstep(20.0, 50.0, dist);

    // Simple lighting
    float3 lightDir = normalize(float3(0.5, 0.3, 1.0));
    float NdotL = max(dot(in.normal, lightDir), 0.0);
    float3 finalColor = baseColor * (0.5 + 0.5 * NdotL) * fade;

    // Grid lines
    float2 grid = abs(fract(uv) - 0.5);
    float gridLine = 1.0 - smoothstep(0.0, 0.02, min(grid.x, grid.y));
    finalColor = mix(finalColor, float3(0.3), gridLine * 0.3 * fade);

    return float4(pow(finalColor, float3(1.0/2.2)), fade);
}

// Simple line shader for debug visualization
struct LineVertexOut {
    float4 position [[position]];
    float4 color;
};

vertex LineVertexOut lineVertex(
    uint vertexId [[vertex_id]],
    constant float4* lineData [[buffer(0)]],
    constant CameraUniforms& camera [[buffer(1)]]
) {
    // lineData: [start, end, color, color] per line
    uint lineId = vertexId / 2;
    uint pointId = vertexId % 2;

    float3 pos = lineData[lineId * 4 + pointId].xyz;
    float4 color = lineData[lineId * 4 + 2];

    LineVertexOut out;
    out.position = camera.viewProjection * float4(pos, 1.0);
    out.color = color;
    return out;
}

fragment float4 lineFragment(LineVertexOut in [[stage_in]]) {
    return in.color;
}
