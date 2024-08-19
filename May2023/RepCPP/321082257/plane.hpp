#pragma once


#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

#include "point.hpp"
#include "vec3.hpp"

using namespace std::string_literals;

class Plane {
protected:
const Point3 m_pos; 
const Vec3 m_normal; 

public:
Plane(Point3 pos, Vec3 normal)
: m_pos { pos }
, m_normal { normal }
{
}

Plane(double x, double y, double z, double nx, double ny, double nz)
: m_pos { x, y, z }
, m_normal { nx, ny, nz }
{
}

const std::string str() const;

double x() const;
double y() const;
double z() const;

const Point3 pos() const;
const Vec3 normal() const;
};

inline const std::string Plane::str() const
{
std::stringstream ss;
ss << "plane: ("s << m_pos << ", "s << m_normal << ")"s;
return ss.str();
}

inline std::ostream& operator<<(std::ostream& os, const Plane& s)
{
os << s.str();
return os;
}

double Plane::x() const { return m_pos.x(); }

double Plane::y() const { return m_pos.y(); }

double Plane::z() const { return m_pos.z(); }

const Point3 Plane::pos() const { return m_pos; }

const Vec3 Plane::normal() const { return m_normal; }
