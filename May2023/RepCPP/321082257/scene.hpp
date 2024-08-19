#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "color.hpp"
#include "point.hpp"
#include "vec3.hpp"

#include "ray.hpp"

#include "disk.hpp"
#include "plane.hpp"
#include "sphere.hpp"

using namespace std::string_literals;

class Scene {
protected:
Sphere m_light;
std::vector<Plane> m_planes;
std::vector<Sphere> m_spheres;
std::vector<Cube> m_cubes;
RGB m_backgroundColor;

public:
Scene(Sphere light, Plane plane, Sphere sphere, Cube cube, RGB backgroundColor)
: m_light { light }
, m_backgroundColor { backgroundColor }
{
m_planes.push_back(plane);
m_spheres.push_back(sphere);
m_cubes.push_back(cube);
}

Scene(Sphere light, Plane plane, std::vector<Sphere> spheres, Cube cube, RGB backgroundColor)
: m_light { light }
, m_spheres { spheres }
, m_backgroundColor { backgroundColor }
{
m_planes.push_back(plane);
m_cubes.push_back(cube);
}

Scene(Sphere light, std::vector<Plane> planes, std::vector<Sphere> spheres,
std::vector<Cube> cubes, RGB backgroundColor)
: m_light { light }
, m_planes { planes }
, m_spheres { spheres }
, m_cubes { cubes }
, m_backgroundColor { backgroundColor }
{
}

const std::string str() const;
const RGB color(const Point3 fromPoint, int x, int y) const;

const Scene light_move(const Vec3 offset) const;
const Scene sphere_move(const size_t index, const Vec3 offset) const;
};

const Scene Scene::sphere_move(const size_t index, const Vec3 offset) const
{
if (m_spheres.empty()) {
return Scene { m_light, m_planes, m_spheres, m_cubes, m_backgroundColor };
}

std::vector<Sphere> newSpheres;
for (size_t i = 0; i < m_spheres.size(); ++i) {
if (i == index) {
const auto newPos = m_spheres[i].pos() + offset;
const auto newRadius = m_spheres[i].r();
Sphere newSphere = Sphere { newPos, newRadius };
newSpheres.push_back(newSphere);
} else {
newSpheres.push_back(m_spheres[i]);
}
}
return Scene { m_light, m_planes, newSpheres, m_cubes, m_backgroundColor };
}

const Scene Scene::light_move(const Vec3 offset) const
{
auto newPos = m_light.pos() + offset;
auto newRadius = m_light.r();
Sphere newLight = Sphere { newPos, newRadius };
return Scene { newLight, m_planes, m_spheres, m_cubes, m_backgroundColor };
}

inline const std::string Scene::str() const
{
std::stringstream ss;
ss << "background color: " << m_backgroundColor << "\n";
ss << "light: " << m_light << "\n";
for (const auto sphere : m_spheres) {
ss << sphere << "\n";
}
for (const auto plane : m_planes) {
ss << plane << "\n";
}
for (const auto cube : m_cubes) {
ss << cube << "\n";
}
return ss.str();
}

inline std::ostream& operator<<(std::ostream& os, const Scene& s)
{
os << s.str();
return os;
}

inline const RGB Scene::color(const Point3 fromPoint, int x, int y) const
{

std::unordered_map<double, RGB> depthColor;

const auto ray = Ray { fromPoint, Vec3 { static_cast<double>(x), static_cast<double>(y), 0 } };


double smallestDepth = 0;
bool firstFind = true;

for (const auto sphere : m_spheres) {

if (const auto maybeIntersectionPointAndNormal = ray.intersect(sphere)) {

const auto intersectionPointAndNormal = maybeIntersectionPointAndNormal.value();

const Point3 intersectionPoint = intersectionPointAndNormal.first;
const Vec3 normal = intersectionPointAndNormal.second;

const auto lightDirection = m_light.pos() - intersectionPoint;

const double dt = lightDirection.normalize().dot(normal.normalize());

RGB currentColor = (Color::red + Color::white * dt) * .5;

double depth = fromPoint.distance(intersectionPoint);

if (depth < smallestDepth || firstFind) {
smallestDepth = depth;
firstFind = false;
}

depthColor.insert(
std::make_pair<double, RGB>(std::move(depth), std::move(currentColor)));

}
}

for (const auto plane : m_planes) {

if (const auto maybeIntersectionPointAndNormal = ray.intersect(plane)) {

const auto intersectionPointAndNormal = maybeIntersectionPointAndNormal.value();

const Point3 intersectionPoint = intersectionPointAndNormal.first;
const Vec3 normal = intersectionPointAndNormal.second;

const auto lightDirection = m_light.pos() - intersectionPoint;

const double dt = lightDirection.normalize().dot(normal.normalize());

RGB currentColor
= ((Color::blueish + Color::white * dt) * .5) * .5 + m_backgroundColor * .5;

double depth = fromPoint.distance(intersectionPoint);

if (depth < smallestDepth || firstFind) {
smallestDepth = depth;
firstFind = false;
}

depthColor.insert(
std::make_pair<double, RGB>(std::move(depth), std::move(currentColor)));

}
}

for (const auto cube : m_cubes) {

if (const auto maybeIntersectionPointAndNormal = ray.intersect(cube)) {

const auto intersectionPointAndNormal = maybeIntersectionPointAndNormal.value();

const Point3 intersectionPoint = intersectionPointAndNormal.first;
const Vec3 normal = intersectionPointAndNormal.second;

const auto lightDirection = m_light.pos() - intersectionPoint;

const double dt = lightDirection.normalize().dot(normal.normalize());

RGB currentColor
= ((Color::blueish + Color::white * dt) * .5) * .5 + m_backgroundColor * .5;

double depth = fromPoint.distance(intersectionPoint);

if (depth < smallestDepth || firstFind) {
smallestDepth = depth;
firstFind = false;
}

depthColor.insert(
std::make_pair<double, RGB>(std::move(depth), std::move(currentColor)));

}
}

if (firstFind) { 
return m_backgroundColor;
}

RGB closestColor = depthColor.at(smallestDepth);
return closestColor.clamp255();
}
