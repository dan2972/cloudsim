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
uniform float cloudScatterCoeff;
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

// H assumed to be thickness of the atmosphere if its density were uniform
// (acts as a scale factor for how quickly density falls off with height)
float atmosphereDensity(float height, float H) {
    return exp(-height / H);
}

// returns false if ray hits ground, true if it reaches sun
bool atmosphereTraceToSun(float3 pos, out float opticalDepthSunR, out float opticalDepthSunM) {
    float t0Light, t1Light;
    raySphere(pos, sunDirection, planetCenter, atmosphereRadius, t0Light, t1Light); // always hits since we're inside atmosphere
    opticalDepthSunR = 0.0;
    opticalDepthSunM = 0.0;
    float stepSize = t1Light / float(numSunRaySamples);

    for (int j = 0; j < numSunRaySamples; j++) {
        float3 stepPos = pos + sunDirection * stepSize * j;
        float height = length(stepPos - planetCenter) - planetRadius;
        if (height < 0) { return false;}
        opticalDepthSunM += atmosphereDensity(height, Hm) * stepSize;
        opticalDepthSunR += atmosphereDensity(height, Hr) * stepSize;
    }
    return true;
}

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

    float3 LR = float3(0); // rayleigh contribution
    float3 LM = float3(0); // mie contribution
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

        float densityR = atmosphereDensity(height, Hr) * stepSize;
        float densityM = atmosphereDensity(height, Hm) * stepSize;
        opticalDepthR += densityR;
        opticalDepthM += densityM;

        float opticalDepthSunR, opticalDepthSunM;
        bool hitGround = !atmosphereTraceToSun(pos, opticalDepthSunR, opticalDepthSunM);

        if (!hitGround) {
            float3 tau = scatterR * (opticalDepthR + opticalDepthSunR) + scatterM * (opticalDepthM + opticalDepthSunM);
            float3 transmittance = exp(-tau);
            LR += transmittance * densityR;
            LM += transmittance * densityM;
        }

        t += stepSize;
    }

    float3 color = (LR * scatterR * phaseR  + LM * scatterM * phaseM) * sunIntensity;
    return float4(color, 1.0);
}

float3 volumeTraceToSun(float3 toSun, float3 rayOrigin, float3 boxMin, float3 boxMax, int numSamples) {
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
            float sigma_s = density * cloudScatterCoeff; // scattering coefficient

            if (density > 0.001) {
                transmittance *= exp(-density * stepSize);
                float3 sunLight = volumeTraceToSun(lightDir, pos, boxMin, boxMax, 64) * sunColor;
                float3 sunScatter = sunLight * phaseHG(dot(rayDir, lightDir), 0.2);
                // float3 ambientLight = ambientInScatter(pos, density, boxMin, boxMax, sunColor, float3(1)) * (1 / (4 * PI));
                float3 ambientLight = sunColor * ambientIntensity;
                L += transmittance * sigma_s * stepSize * (sunScatter + ambientLight);
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