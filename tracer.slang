struct Camera {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
};

RWTexture2D<float4> output;
Texture3D<float> volume;
SamplerState volumeSampler;

uniform Camera camera;
uniform float2 resolution;

float4 get_background_color(float3 rayDir) {
    float3 unitDirection = normalize(rayDir);
    float t = 0.5f * (unitDirection.y + 1.0f);
    return (1.0f - t) * float4(75/255.0, 100/255.0, 166/255.0, 1.0) + t * float4(0.0, 47.0/255.0, 113.0/255.0, 1.0);
}

// Helper: Ray-AABB intersection to find entry/exit points
bool rayAABB(float3 ro, float3 rd, float3 boxMin, float3 boxMax, out float tMin, out float tMax) {
    float3 invDir = 1.0 / rd;
    float3 t0 = (boxMin - ro) * invDir;
    float3 t1 = (boxMax - ro) * invDir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    tMin = max(max(tmin.x, tmin.y), tmin.z);
    tMax = min(min(tmax.x, tmax.y), tmax.z);
    return tMax > max(tMin, 0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void render(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= uint(resolution.x) || tid.y >= uint(resolution.y)) return;

    float2 uv = (float2(tid.xy) / resolution) * 2.0f - 1.0f;
    float aspect = resolution.x / resolution.y;
    float3 rayDir = normalize(camera.forward + uv.x * camera.right * aspect - uv.y * camera.up);
    float3 rayOrigin = camera.position;

    uint3 texSize;
    volume.GetDimensions(texSize.x, texSize.y, texSize.z);
    float3 boxMin = float3(0.0, 0.0, 0.0);
    float3 boxMax = float3(texSize) / float3(50);
    
    float tNear, tFar;
    float4 bgColor = get_background_color(rayDir);
    float4 finalColor = bgColor;

    if (rayAABB(rayOrigin, rayDir, boxMin, boxMax, tNear, tFar)) {
        float tStart = max(tNear, 0.0);
        float tEnd = tFar;
        
        float totalDist = tEnd - tStart;
        
        const int numSamples = 128; 
        float stepSize = totalDist / float(numSamples);
        
        float t = tStart;
        float transmittance = 1.0;
        float3 scatteredLight = 0.0;
        float absorption = 10.0;

        for (int i = 0; i < numSamples; i++) {
            if (transmittance < 0.01) break;

            float3 pos = rayOrigin + rayDir * (t + stepSize * 0.5);
            float3 texCoord = (pos - boxMin) / (boxMax - boxMin);
            
            float density = volume.SampleLevel(volumeSampler, texCoord, 0).r;

            if (density > 0.001) {
                float lightSample = density * stepSize * absorption;
                transmittance *= exp(-lightSample);
                scatteredLight += transmittance * lightSample * float3(1.0); 
            }

            t += stepSize;
        }
        
        finalColor.rgb = scatteredLight + (transmittance * bgColor.rgb);
    }

    output[tid.xy] = finalColor;
}