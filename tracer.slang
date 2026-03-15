import RNG;
import helpers;

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
uniform float scatterStrength;

static const float planetRadius = 100000.0;
static const float atmosphereRadius = 101000.0;
static const int numAtmosphereSamples = 16;
static const float atmosphereDensityFalloff = 15.0;
static const float3 wavelengths = float3(700, 530, 440); // RGB wavelengths in nm
static const float3 planetCenter = float3(0, -planetRadius, 0);

static const float4 planetColor = float4(0.5, 0.2, 0.2, 1.0);

bool raySphere(float3 ro, float3 rd, float3 sphereCenter, float radius, out float tMin, out float tMax) {
    float3 oc = ro - sphereCenter;
    float a = dot(rd, rd);
    float b = 2.0 * dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        tMin = 0.0;
        tMax = 0.0;
        return false;
    }

    float sqrtDisc = sqrt(discriminant);
    tMin = (-b - sqrtDisc) / (2.0 * a);
    tMax = (-b + sqrtDisc) / (2.0 * a);
    if (tMax < 0.0) return false;
    if (tMin < 0.0) tMin = 0.0;
    return true;
}

float atmosphereDensity(float3 pos) {
    float height = length(pos - planetCenter) - planetRadius;
    float heightNorm = height / (atmosphereRadius - planetRadius);
    return exp(-heightNorm * atmosphereDensityFalloff) * (1-heightNorm);
}

float opticalDepth(float3 start, float3 dir, float maxDist) {
    float depth = 0.0;
    const int numSamples = 16;
    float stepSize = maxDist / float(numSamples);
    float t = 0.0;

    for (int i = 0; i < numSamples; i++) {
        float3 pos = start + dir * t;
        depth += atmosphereDensity(pos) * stepSize;
        t += stepSize;
    }
    return depth;
}

float4 getAtmosphereColor(float3 rayOrigin, float3 rayDir) {
    float3 pos = rayOrigin;
    
    float tMin, tMax;
    bool hitSurface = raySphere(pos, rayDir, planetCenter, planetRadius, tMin, tMax);
    float distToSurface = tMin;

    if (!raySphere(pos, rayDir, planetCenter, atmosphereRadius, tMin, tMax))
        return float4(0); // Ray misses atmosphere
    
    float distToAtmosphere = tMin;
    float distThroughAtmosphere = tMax - distToAtmosphere;
    if (hitSurface)
        distThroughAtmosphere = distToSurface - distToAtmosphere;

    float3 inScatteredLight = 0.0;
    float viewRayOpticalDepth = 0.0;

    float3 scatteringCoeffs = pow(100 / wavelengths, 4) * scatterStrength;

    float t = distToAtmosphere;
    for (int i = 0; i < numAtmosphereSamples; i++) {

        pos = rayOrigin + rayDir * t;
        float stepSize = distThroughAtmosphere / float(numAtmosphereSamples);

        float _, sunRayLength;
        raySphere(pos, sunDirection, planetCenter, atmosphereRadius, _, sunRayLength);
        float sunRayOpticalDepth = opticalDepth(pos, sunDirection, sunRayLength);
        viewRayOpticalDepth += atmosphereDensity(pos) * stepSize;
        float3 transmittance = exp(-(sunRayOpticalDepth + viewRayOpticalDepth) * scatteringCoeffs);
        float density = atmosphereDensity(pos);

        inScatteredLight += transmittance * density * scatteringCoeffs * stepSize;
        t += stepSize;
    }

    float4 color = float4(inScatteredLight, 1.0);
    // Sun Disk
    float3 unitDir = normalize(rayDir);
    float3 sunDir = normalize(sunDirection);
    float sunDot = dot(unitDir, sunDir);
    float3 sunColor = getSunColor();
    float sunDisk = smoothstep(0.9995, 0.9999, sunDot);
    float sunGlow = pow(max(0.0, sunDot), 500.0);
    color.rgb += sunColor * (sunDisk + sunGlow * 0.4);
    return color;
}

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
    float4 bgColor = getAtmosphereColor(rayOrigin, rayDir);
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