#pragma once

#include "types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>


class GTAQuat
{
private:
static constexpr float EPSILON = 0.00000202655792236328125f;

public:
GTAQuat()
: q(1.0f, 0.0f, 0.0f, 0.0f)
{
}

GTAQuat(float w, float x, float y, float z)
: q(w, x, y, z)
{
}

GTAQuat(float x, float y, float z)
: GTAQuat(Vector3(x, y, z))
{
}

GTAQuat(Vector3 degrees)
{
Vector3 radians = glm::radians(degrees);
Vector3 c = cos(radians * -0.5f);
Vector3 s = sin(radians * -0.5f);

q.w = c.x * c.y * c.z + s.x * s.y * s.z;
q.x = c.x * s.y * s.z + s.x * c.y * c.z;
q.y = c.x * s.y * c.z - s.x * c.y * s.z;
q.z = c.x * c.y * s.z - s.x * s.y * c.z;
}

Vector3 ToEuler() const
{
float temp = 2 * q.y * q.z - 2 * q.x * q.w;
float rx, ry, rz;

if (temp >= 1.0f - EPSILON)
{
rx = 90.0f;
ry = -glm::degrees(atan2(glm::clamp(q.y, -1.0f, 1.0f), glm::clamp(q.w, -1.0f, 1.0f)));
rz = -glm::degrees(atan2(glm::clamp(q.z, -1.0f, 1.0f), glm::clamp(q.w, -1.0f, 1.0f)));
}
else if (-temp >= 1.0f - EPSILON)
{
rx = -90.0f;
ry = -glm::degrees(atan2(glm::clamp(q.y, -1.0f, 1.0f), glm::clamp(q.w, -1.0f, 1.0f)));
rz = -glm::degrees(atan2(glm::clamp(q.z, -1.0f, 1.0f), glm::clamp(q.w, -1.0f, 1.0f)));
}
else
{
rx = glm::degrees(asin(glm::clamp(temp, -1.0f, 1.0f)));
ry = -glm::degrees(atan2(glm::clamp(q.x * q.z + q.y * q.w, -1.0f, 1.0f), glm::clamp(0.5f - q.x * q.x - q.y * q.y, -1.0f, 1.0f)));
rz = -glm::degrees(atan2(glm::clamp(q.x * q.y + q.z * q.w, -1.0f, 1.0f), glm::clamp(0.5f - q.x * q.x - q.z * q.z, -1.0f, 1.0f)));
}

return mod(Vector3(rx, ry, rz), 360.0f);
}

GTAQuat operator*(const GTAQuat& other) const
{
glm::quat res = q * other.q;
return GTAQuat(res.w, res.x, res.y, res.z);
}

GTAQuat& operator*=(const GTAQuat& other)
{
q *= other.q;
return *this;
}

glm::quat q;
};
