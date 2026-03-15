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
uniform float phaseParam;
uniform float densityScale;
uniform float ambientIntensity;

static const float PI = 3.14159265359;

float3 getSunColor() {
    float3 sunDir = normalize(sunDirection);
    float sunVisibility = smoothstep(-0.15, 0.5, sunDir.y);
    float3 daylight = float3(1.0, 0.9, 0.75);
    float3 sunset = float3(1.0, 0.35, 0.1);

    float3 color = lerp(sunset, daylight, sunVisibility);
    return color * sunVisibility;
}

float4 getBgColor(float3 rayOrigin, float3 rayDir) {
    float3 unitDir = normalize(rayDir);
    float3 sunDir = normalize(sunDirection);
    float sunVis = smoothstep(0, 0.8, sunDir.y);

    // Sky gradient colors
    float3 zenith = lerp(float3(0.01, 0.02, 0.05), float3(0.1, 0.3, 0.6), sunVis);
    float3 horizon = lerp(float3(0.2, 0.05, 0.02), float3(0.6, 0.7, 0.9), sunVis);

    float3 skyRGB = lerp(zenith, horizon, pow(1.0 - max(unitDir.y, 0.0), 3.0));

    // Sun Disk
    float sunDot = dot(unitDir, sunDir);
    float3 sunColor = getSunColor();
    float sunDisk = smoothstep(0.9995, 0.9999, sunDot);
    float sunGlow = pow(max(0.0, sunDot), 500.0);

    skyRGB += sunColor * (sunDisk + sunGlow * 0.4);

    if (rayDir.y < 0) {
        float tFloor = -(rayOrigin.y) / rayDir.y;
        if (tFloor > 0.0) {
            float3 floorPos = rayOrigin + rayDir * tFloor;
            int checker = (int(floor(floorPos.x)) + int(floor(floorPos.z))) % 2;
            float3 floorColor = (checker == 0 ? float3(0.6) : float3(0.2)) * max(0.05, sunVis);
            skyRGB = lerp(skyRGB, floorColor, exp(-tFloor * 0.02));
        }
    }
    return float4(skyRGB, 1.0);
}

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
    return transmittance;
}

float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

float Ei(float z)
{
    return 0.5772156649015328606065 + log(1e-4 + abs(z)) + z * (1.0 + z * (0.25 + z * ((1.0 / 18.0) + z * ((1.0 / 96.0) + z * (1.0 / 600.0)))));
}

float3 ambientInScatter(float3 pos, float density, float3 bbMin, float3 bbMax, float3 sunColor, float3 horizonColor) {
    float Hp = bbMax.y - bbMin.y;
    float a = -density * Hp;
    float3 isoScatterTop = sunColor * max(0.0, exp(a) - a * Ei(a));
    float Hb = pos.y - bbMin.y;
    a = -density * Hb;
    float3 isoScatterBottom = horizonColor * max(0.0, exp(a) - a * Ei(a));
    return isoScatterTop + isoScatterBottom;
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
    float3 boxOffset = float3(0, 10, 0);
    float3 boxMin = float3(0.0, 0.0, 0.0) + boxOffset;
    float3 boxMax = float3(texSize) / float3(8) + boxOffset;
    
    float tNear, tFar;
    float4 bgColor = getBgColor(rayOrigin, rayDir);
    float4 finalColor = bgColor;

    float3 lightDir = normalize(sunDirection);

    float3 sunColor = getSunColor();

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

            float density = volume.SampleLevel(volumeSampler, texCoord, 0).r * densityScale;

            if (density > 0.001) {
                transmittance *= exp(-density * stepSize);
                float3 sunLight = sampleInScatterSun(lightDir, pos, boxMin, boxMax, 64) * sunColor * 50;
                float3 sunScatter = sunLight * phaseHG(dot(rayDir, lightDir), phaseParam);
                // float3 ambientLight = ambientInScatter(pos, density, boxMin, boxMax, sunColor, float3(1)) * (1 / (4 * PI));
                float3 ambientLight = sunColor * ambientIntensity;
                L += transmittance * density * stepSize * (sunScatter + ambientLight);
            }

            t += stepSize;
        }

        finalColor.rgb = L + (transmittance * bgColor.rgb);
    }

    // hdr tonemapping
    float3 hdr = float3(1.0) - exp(-finalColor.rgb * 1);
    // float3 gamma = pow(hdr, float3(1.0/2.2));
    output[tid.xy] = float4(hdr.rgb, 1.0);
}