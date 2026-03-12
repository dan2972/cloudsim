// tracer.slang

struct Camera {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
};

float4 render(
    uint2 tid: SV_DispatchThreadID,
    uniform Camera camera,
    uniform float2 resolution)
{
    uint2 pixel = tid;

    // 1. Normalize pixel coordinates to [-1, 1] range
    float2 uv = (float2(pixel) / resolution) * 2.0f - 1.0f;
    uv.y *= -1.0; // Flip Y for screen space
    float aspect = resolution.x / resolution.y;
    uv.x *= aspect;

    // 2. Setup Ray (simple pinhole camera)
    float3 rayOrigin = camera.position;
    float3 rayDir = normalize(camera.forward + uv.x * camera.right + uv.y * camera.up);

    // 3. Ray-Sphere Intersection (Sphere at 0,0,0 with radius 1.0)
    float3 sphereCenter = float3(0, 0, 0);
    float radius = 1.0;
    
    float3 oc = rayOrigin - sphereCenter;
    float b = dot(oc, rayDir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;

    float4 color = float4(0.1, 0.1, 0.1, 1.0); // Dark background

    // simple gradient background
    float3 unitDirection = normalize(rayDir);
    float t = 0.5f * (unitDirection.y + 1.0f);
    color = (1.0f - t) * float4(1.0) + t * float4(0.5, 0.7, 1.0, 1.0);

    if (discriminant > 0) {
        // We hit the sphere! Simple shading based on distance
        float t = -b - sqrt(discriminant);
        if (t > 0) {
            float3 hitPos = rayOrigin + rayDir * t;
            float3 normal = normalize(hitPos - sphereCenter);
            float light = max(dot(normal, normalize(float3(1, 1, 1))), 0.2);
            color = float4(light, 0.3, 0.2, 1.0); // Reddish sphere
        }
    }

    return color;
}