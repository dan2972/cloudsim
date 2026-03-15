static const float PI = 3.14159265359;

float phaseHGR(float cosTheta, float g) {
    float g2 = g * g;
    return (1.5 * (1.0 - g2) * (1.0 + cosTheta * cosTheta)) /
           ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

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