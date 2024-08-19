#pragma once
#include "Math.hpp"


struct Ray {
Vec3 origin;
Vec3 direction;
Ray() = default;
constexpr Ray(const Vec3& origin, const Vec3& direction)
: origin(origin), direction(direction) {}
};

inline std::ostream& operator<<(std::ostream& os, const Ray& ray) {
os << "Ray("
<< "origin:("    << ray.origin    << "),"
<< "direction:(" << ray.direction << ")"
<< ")";
return os;
}



class IObject;

struct Intersection {
Vec3 point;
Vec3 normal;
float t;
const IObject* object;

constexpr Intersection() : Intersection(Vec3(), Vec3(), -1.00f, nullptr) {}
constexpr Intersection(const Vec3& point, const Vec3& normal, float t, IObject* object) :
point (point), normal(normal), t(t), object(object) {}
};
inline std::ostream& operator<<(std::ostream& os, const Intersection& intersectInfo) {
os << "Intersection("
<< "point:("  << intersectInfo.point  << "),"
<< "normal:(" << intersectInfo.normal << "),"
<< "t:"       << intersectInfo.t
<< ")";
return os;
}
