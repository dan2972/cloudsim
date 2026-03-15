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
uniform float sunIntensity;

static const float transmittanceEpsilon = 0.001;

static const float3 volumeBoxOffset = float3(50, 50, -50);

static const float planetRadius = 6360e3;
static const float atmosphereRadius = 6420e3;
static const int numAtmosphereSamples = 16;
static const int numSunRaySamples = 8;
static const float3 planetCenter = float3(0, -planetRadius, 0);
static const float Hr = 7994;
static const float Hm = 1200;
static const float3 scatterR = float3(3.8e-6, 1.5e-5, 3.2e-5); // betaR
static const float3 scatterM = float3(21e-6f); // betaM

static const float4 planetColor = float4(0.5, 0.2, 0.2, 1.0);

float4 getAtmosphereColor(float3 rayOrigin, float3 rayDir) {
    float3 pos = rayOrigin;
    
    float tMin, tMax;
    bool hitSurface = raySphere(pos, rayDir, planetCenter, planetRadius, tMin, tMax);
    float distToSurface = tMin;

    if (!raySphere(pos, rayDir, planetCenter, atmosphereRadius, tMin, tMax))
        return float4(0); // ray misses atmosphere
    
    float distToAtmosphere = tMin;
    float distThroughAtmosphere = tMax - distToAtmosphere;
    if (hitSurface)
        distThroughAtmosphere = distToSurface - distToAtmosphere;

    float3 sumR = float3(0); // rayleigh contribution
    float3 sumM = float3(0); // mie contribution
    float opticalDepthR = 0.0;
    float opticalDepthM = 0.0;

    float cosTheta = dot(rayDir, sunDirection);

    float phaseR = phaseHG(cosTheta, 0.0);
    float phaseM = phaseHG(cosTheta, 0.8);

    float t = distToAtmosphere;
    for (int i = 0; i < numAtmosphereSamples; i++) {

        pos = rayOrigin + rayDir * t;
        float stepSize = distThroughAtmosphere / float(numAtmosphereSamples);

        float height = length(pos - planetCenter) - planetRadius;

        float hr = exp(-height / Hr) * stepSize;
        float hm = exp(-height / Hm) * stepSize;
        opticalDepthR += hr;
        opticalDepthM += hm;

        float t0Light, t1Light;
        raySphere(pos, sunDirection, planetCenter, atmosphereRadius, t0Light, t1Light); // always hits since we're inside atmosphere
        float opticalDepthSunR = 0.0;
        float opticalDepthSunM = 0.0;
        float stepSizeLight = t1Light / float(numSunRaySamples);
        // take samples along sun ray to get optical depth to sun
        bool hitGround = false;
        for (int j = 0; j < numSunRaySamples; j++) {
            float3 sunPos = pos + sunDirection * stepSizeLight * j;
            float heightSun = length(sunPos - planetCenter) - planetRadius;
            if (heightSun < 0) {hitGround = true; break;}
            opticalDepthSunM += exp(-heightSun / Hm) * stepSizeLight;
            opticalDepthSunR += exp(-heightSun / Hr) * stepSizeLight;
        }

        if (!hitGround) {
            float3 tau = scatterR * (opticalDepthR + opticalDepthSunR) + scatterM * (opticalDepthM + opticalDepthSunM);
            float3 attenuation = exp(-tau);
            sumR += attenuation * hr;
            sumM += attenuation * hm;
        }

        t += stepSize;
    }

    float3 color = (sumR * scatterR * phaseR  + sumM * scatterM * phaseM) * sunIntensity;
    return float4(color, 1.0);
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
            if (transmittance < transmittanceEpsilon) break;
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
    float3 boxMin = float3(0.0, 0.0, 0.0) + volumeBoxOffset;
    float3 boxMax = float3(texSize) / float3(8) + volumeBoxOffset;

    float tNear, tFar;
    float4 bgColor = getAtmosphereColor(rayOrigin, rayDir);
    float4 finalColor = bgColor;

    float3 lightDir = normalize(sunDirection);

    float3 sunColor = getAtmosphereColor(rayOrigin, sunDirection).rgb;

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
            if (transmittance < transmittanceEpsilon) break;

            float3 pos = rayOrigin + rayDir * t;
            float3 texCoord = (pos - boxMin) / (boxMax - boxMin);

            float density = volume.SampleLevel(volumeSampler, texCoord, 0).r * densityScale;

            if (density > 0.001) {
                transmittance *= exp(-density * stepSize);
                float3 sunLight = sampleInScatterSun(lightDir, pos, boxMin, boxMax, 64) * sunColor;
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