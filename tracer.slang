import RNG;

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
uniform float3 sunDirection;

float4 getBgColor(float3 rayOrigin, float3 rayDir) {
    float3 unitDirection = normalize(rayDir);
    float t = 0.5f * (unitDirection.y + 1.0f);
    float4 sky = (1.0f - t) * float4(75/255.0, 100/255.0, 166/255.0, 1.0) + t * float4(0.0, 47.0/255.0, 113.0/255.0, 1.0);

    // draw the floor with checker pattern
    if (rayDir.y < 0) {
        float tFloor = -(rayOrigin.y + 10.0) / rayDir.y;
        if (tFloor > 0.0) {
            float3 floorPos = rayOrigin + rayDir * tFloor;
            floorPos.y -= 10.0;
            int checker = (int(floor(floorPos.x)) + int(floor(floorPos.z))) % 2;
            float3 floorColor = checker == 0 ? float3(0.8) : float3(0.2);
            sky.rgb = lerp(sky.rgb, floorColor, exp(-tFloor * 0.05));
        }
    }

    return sky;
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

float3 sampleInScatterSun(float3 toSun, float3 rayOrigin, float3 boxMin, float3 boxMax, int numSamples) {
    float tNear, tFar;
    float transmittance = 1.0;

    if (rayAABB(rayOrigin, toSun, boxMin, boxMax, tNear, tFar)) {
        float stepSize = tFar / float(numSamples);
        float t = 0.0;

        for (int i = 0; i < numSamples; i++) {
            if (transmittance < 0.01) break;
            float3 pos = rayOrigin + toSun * t;

            float3 texCoord = (pos - boxMin) / (boxMax - boxMin);
            float density = volume.SampleLevel(volumeSampler, texCoord, 0).r;
            if (density > 0.001) {
                transmittance *= exp(-density * stepSize);
            }

            t += stepSize;
        }
    }
    return transmittance * float3(1.0, 1.0, 0.8) * 20; // Sunlight color
}

float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.141592 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void render(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= uint(resolution.x) || tid.y >= uint(resolution.y)) return;

    float2 uv = (float2(tid.xy) / resolution) * 2.0f - 1.0f;
    float aspect = resolution.x / resolution.y;
    float2 view = float2(uv.x * aspect, uv.y) * tan(radians(70.0) * 0.5);
    float3 rayDir = normalize(camera.forward + view.x * camera.right - view.y * camera.up);
    float3 rayOrigin = camera.position;

    uint3 texSize;
    volume.GetDimensions(texSize.x, texSize.y, texSize.z);
    float3 boxMin = float3(0.0, 0.0, 0.0);
    float3 boxMax = float3(texSize) / float3(10);
    
    float tNear, tFar;
    float4 bgColor = getBgColor(rayOrigin, rayDir);
    float4 finalColor = bgColor;

    float3 lightDir = normalize(sunDirection);

    if (rayAABB(rayOrigin, rayDir, boxMin, boxMax, tNear, tFar)) {
        float tStart = max(tNear, 0.0);
        float tEnd = tFar;
        
        float totalDist = tEnd - tStart;
        
        const int numSamples = 128; 
        float stepSize = totalDist / float(numSamples);

        RNG rng = RNG(tid.xy, (tid.y * uint(resolution.x) + tid.x) * 10000);
        float jitter = rng.next_1d();
        
        float t = tStart + jitter * stepSize;
        float transmittance = 1.0;
        float3 L = float3(0);

        for (int i = 0; i < numSamples; i++) {
            if (transmittance < 0.01) break;

            float3 pos = rayOrigin + rayDir * t;
            float3 texCoord = (pos - boxMin) / (boxMax - boxMin);
            
            float density = volume.SampleLevel(volumeSampler, texCoord, 0).r;

            if (density > 0.001) {
                transmittance *= exp(-density * stepSize);
                float3 sunLight = sampleInScatterSun(lightDir, pos, boxMin, boxMax, 64);
                float3 sunScatter = sunLight * phaseHG(dot(rayDir, lightDir), 0.5);
                L += transmittance * density * stepSize * (sunScatter + float3(0.1, 0.15, 0.25) * density);
            }

            t += stepSize;
        }

        finalColor.rgb = L + (transmittance * bgColor.rgb);
    }

    output[tid.xy] = finalColor;
}