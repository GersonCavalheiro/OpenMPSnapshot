#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std::string_literals;

class Vec4 {
protected:
const double v[4];

public:
Vec4(double _x, double _y, double _z, double _t)
: v { _x, _y, _z, _t }
{
}

const Vec4 operator+(const Vec4& a) const; 
const Vec4 operator-(const Vec4& a) const; 
const Vec4 operator*(const double d) const; 
const Vec4 operator/(const double d) const; 
double dot(const Vec4& a) const; 
const Vec4 cross(const Vec4& a) const; 
double len() const; 
double len_squared() const; 
bool operator<(const Vec4& a) const; 
bool operator>(const Vec4& a) const; 
const Vec4 normalize() const; 
const std::string str() const;

double x() const;
double y() const;
double z() const;
double t() const;

double R() const;
double G() const;
double B() const;
double A() const;

const Vec4 intify() const;
};

inline const Vec4 Vec4::operator+(const Vec4& a) const
{
return Vec4 { v[0] + a.v[0], v[1] + a.v[1], v[2] + a.v[2], v[3] + a.v[3] };
}

inline const Vec4 Vec4::operator-(const Vec4& a) const
{
return Vec4 { v[0] - a.v[0], v[1] - a.v[1], v[2] - a.v[2], v[3] - a.v[3] };
}

inline const Vec4 Vec4::operator*(const double d) const
{
return Vec4 { v[0] * d, v[1] * d, v[2] * d, v[3] * d };
}

inline const Vec4 Vec4::operator/(const double d) const
{
const double r = (1.0 / d);
return Vec4 { v[0] * r, v[1] * r, v[2] * r, v[3] * r };
}

inline double Vec4::dot(const Vec4& a) const
{
return v[0] * a.v[0] + v[1] * a.v[1] + v[2] * a.v[2] + v[3] * a.v[3];
}

inline double Vec4::len() const { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

inline double Vec4::len_squared() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }

inline bool Vec4::operator<(const Vec4& a) const
{
return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
< (a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2]);
}

inline bool Vec4::operator>(const Vec4& a) const
{
return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
> (a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2] + a.v[3] * a.v[3]);
}

inline const Vec4 Vec4::normalize() const
{
const double l = this->len();
return Vec4 { v[0] / l, v[1] / l, v[2] / l, v[3] / l };
}

inline const std::string Vec4::str() const
{
std::stringstream ss;
ss << "["s << std::setprecision(3) << v[0] << ", "s << v[1] << ", "s << v[2] << ", "s << v[3]
<< "]"s;
return ss.str();
}

inline std::ostream& operator<<(std::ostream& os, const Vec4& v)
{
os << v.str();
return os;
}

inline double Vec4::x() const { return v[0]; }

inline double Vec4::y() const { return v[1]; }

inline double Vec4::z() const { return v[2]; }

inline double Vec4::t() const { return v[3]; }

inline bool operator==(const Vec4& a, const Vec4& b)
{
return a.x() == b.x() && a.y() == b.y() && a.z() == b.z() && a.t() == b.t();
}

inline double Vec4::R() const { return v[0]; }

inline double Vec4::G() const { return v[1]; }

inline double Vec4::B() const { return v[2]; }

inline double Vec4::A() const { return v[3]; }

inline const Vec4 Vec4::intify() const
{
return Vec4 { static_cast<double>(static_cast<int>(v[0])),
static_cast<double>(static_cast<int>(v[1])), static_cast<double>(static_cast<int>(v[2])),
static_cast<double>(static_cast<int>(v[3])) };
}
