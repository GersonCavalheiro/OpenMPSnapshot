#pragma once
#include "Math.hpp"
#include "Material.hpp"
#include "Ray.hpp"


class IObject {
public:
virtual ~IObject() = default;
virtual bool intersect(const Ray& ray, Intersection& result) const = 0;
virtual std::string description() const = 0;

constexpr const Vec3&     position() const { return position_; }
constexpr const Material& material() const { return material_; }

protected:
Vec3     position_{};
Material material_{};
};
inline std::ostream& operator<<(std::ostream& os, const IObject& object) {
os << object.description();
return os;
}



class Sphere final : public virtual IObject {
public:
Sphere(const Vec3& center, float radius, const Material& material);

virtual bool intersect(const Ray& ray, Intersection& result) const override;
virtual std::string description() const override;

bool contains(const Vec3& point) const;

Vec3  center() const;
float radius() const;

private:
Vec3 center_;
float radius_;
};



class Triangle final : public virtual IObject {
public:
Triangle(const Vec3& vert0, const Vec3& vert1, const Vec3& vert2, const Material& material);

virtual bool intersect(const Ray& ray, Intersection& result) const override;
virtual std::string description() const override;

bool contains(const Vec3& point) const;

Vec3 vert0()       const;
Vec3 vert1()       const;
Vec3 vert2()       const;
Vec3 center()      const;
Vec3 planeNormal() const;

private:
Vec3 vert0_;
Vec3 vert1_;
Vec3 vert2_;
Vec3 edge0_;
Vec3 edge1_;
Vec3 edge2_;
Vec3 planeNormal_;
Vec3 center_;

Vec3 computeCentroid() const;
static bool isValidTriangle(const Vec3& vert0, const Vec3& vert1, const Vec3& vert2);
};
