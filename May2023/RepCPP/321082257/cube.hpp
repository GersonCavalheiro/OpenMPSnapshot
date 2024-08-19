#pragma once

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "points.hpp"
#include "vec3.hpp"

using namespace std::string_literals;

class Cube {
protected:
const Vec3 m_pos; 
const double m_whd[3]; 

public:
Cube(double _x, double _y, double _z, double _w, double _h, double _d)
: m_pos { _x, _y, _z }
, m_whd { _w, _h, _d }
{
}

Cube(double _x, double _y, double _z, double _whd) 
: m_pos { _x, _y, _z }
, m_whd { _whd, _whd, _whd }
{
}

Cube(const Vec3 _pos, double _w, double _h, double _d)
: m_pos { _pos }
, m_whd { _w, _h, _d }
{
}

Cube(const Vec3 _pos, double _whd) 
: m_pos { _pos }
, m_whd { _whd, _whd, _whd }
{
}

const std::string str() const;
double x() const;
double y() const;
double z() const;
double w() const;
double h() const;
double d() const;
const Vec3 pos() const;
const Vec3 normal(const Vec3 p) const;

const Vec3 p0() const;
const Vec3 p1() const;
const Vec3 p2() const;
const Vec3 p3() const;
const Vec3 p4() const;
const Vec3 p5() const;
const Vec3 p6() const;
const Vec3 p7() const;

const Points points() const;
};

inline const std::string Cube::str() const
{
std::stringstream ss;
ss << "cube: ("s << m_pos << ", "s << m_whd[0] << ", "s << m_whd[1] << ", "s << m_whd[2]
<< ")"s;
return ss.str();
}

inline std::ostream& operator<<(std::ostream& os, const Cube& s)
{
os << s.str();
return os;
}

double Cube::x() const { return m_pos.x(); }

double Cube::y() const { return m_pos.y(); }

double Cube::z() const { return m_pos.z(); }

double Cube::w() const { return m_whd[0]; }

double Cube::h() const { return m_whd[1]; }

double Cube::d() const { return m_whd[2]; }

const Vec3 Cube::pos() const { return m_pos; }


const Vec3 Cube::p0() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;


return m_pos + Vec3 { -r0, -r1, -r2 };
}

const Vec3 Cube::p1() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;


return m_pos + Vec3 { r0, -r1, -r2 };
}

const Vec3 Cube::p2() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;


return m_pos + Vec3 { r0, -r1, r2 };
}

const Vec3 Cube::p3() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;


return m_pos + Vec3 { -r0, -r1, r2 };
}

const Vec3 Cube::p4() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;


return m_pos + Vec3 { -r0, r1, -r2 };
}

const Vec3 Cube::p5() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;


return m_pos + Vec3 { r0, r1, -r2 };
}

const Vec3 Cube::p6() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;

return m_pos + Vec3 { r0, r1, r2 };
}

const Vec3 Cube::p7() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;

return m_pos + Vec3 { -r0, r1, r2 };
}

const Points Cube::points() const
{
const double r0 = m_whd[0] / 2.0;
const double r1 = m_whd[1] / 2.0;
const double r2 = m_whd[2] / 2.0;

return Points {
std::move(m_pos + Vec3 { -r0, -r1, -r2 }),
std::move(m_pos + Vec3 { r0, -r1, -r2 }),
std::move(m_pos + Vec3 { r0, -r1, r2 }),
std::move(m_pos + Vec3 { -r0, -r1, r2 }),
std::move(m_pos + Vec3 { -r0, r1, -r2 }),
std::move(m_pos + Vec3 { r0, r1, -r2 }),
std::move(m_pos + Vec3 { r0, r1, r2 }),
std::move(m_pos + Vec3 { -r0, r1, r2 }),
};
}

const Vec3 Cube::normal(const Vec3 p) const
{



Points xs = this->points();

size_t smallest_ai = index_closest(xs, p);
size_t smallest_bi = index_closest_except(xs, p, std::vector<size_t> { smallest_ai });
size_t smallest_ci
= index_closest_except(xs, p, std::vector<size_t> { smallest_ai, smallest_bi });
size_t smallest_di = index_closest_except(
xs, p, std::vector<size_t> { smallest_ai, smallest_bi, smallest_ci });

const Vec3 av = xs[smallest_ai] - m_pos;
const Vec3 bv = xs[smallest_bi] - m_pos;
const Vec3 cv = xs[smallest_ci] - m_pos;
const Vec3 dv = xs[smallest_di] - m_pos;


const Vec3 average_closest = (av + bv + cv + dv) / 4;



const Vec3 average_closest_normalized = average_closest.normalize();


return average_closest_normalized.intify();
}
