#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace std::string_literals;

class Vec3 {
protected:
const double v[3];

public:
Vec3(const double _x, const double _y, const double _z)
: v { _x, _y, _z }
{
}

const Vec3 operator+(const Vec3& a) const; 
const Vec3 operator-(const Vec3& a) const; 
const Vec3 operator*(const double d) const; 
const Vec3 operator/(const double d) const; 
double dot(const Vec3& a) const; 
const Vec3 cross(const Vec3& a) const; 
double len() const; 
double len_squared() const; 
bool operator<(const Vec3& a) const; 
bool operator>(const Vec3& a) const; 
const Vec3 normalize() const; 
const std::string str() const;

double x() const;
double y() const;
double z() const;

const Vec3 clamp255() const;
const std::string ppm() const;

double R() const;
double G() const;
double B() const;

double distance(const Vec3& a) const; 
double distance_squared(const Vec3& a) const; 

const Vec3 intify() const; 

};

inline const Vec3 Vec3::operator+(const Vec3& a) const
{
return Vec3 { v[0] + a.v[0], v[1] + a.v[1], v[2] + a.v[2] };
}

inline const Vec3 Vec3::operator-(const Vec3& a) const
{
return Vec3 { v[0] - a.v[0], v[1] - a.v[1], v[2] - a.v[2] };
}

inline const Vec3 Vec3::operator*(const double d) const
{
return Vec3 { v[0] * d, v[1] * d, v[2] * d };
}

inline const Vec3 Vec3::operator/(const double d) const
{
const double r = (1.0 / d);
return Vec3 { v[0] * r, v[1] * r, v[2] * r };
}

inline double Vec3::dot(const Vec3& a) const
{
return v[0] * a.v[0] + v[1] * a.v[1] + v[2] * a.v[2];
}

inline const Vec3 Vec3::cross(const Vec3& a) const
{
return Vec3 { v[1] * a.v[2] - v[2] * a.v[1], v[2] * a.v[0] - v[0] * a.v[2],
v[0] * a.v[1] - v[1] * a.v[0] };
}

inline double Vec3::len() const { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

inline double Vec3::len_squared() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }

inline bool Vec3::operator<(const Vec3& a) const
{
return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
< (a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2]);
}

inline bool Vec3::operator>(const Vec3& a) const
{
return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
> (a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2]);
}

inline const Vec3 Vec3::normalize() const
{
const double l = this->len();
return Vec3 { v[0] / l, v[1] / l, v[2] / l };
}

inline const std::string Vec3::str() const
{
std::stringstream ss;
ss << "["s << std::setprecision(3) << v[0] << ", "s << v[1] << ", "s << v[2] << "]"s;
return ss.str();
}

inline double Vec3::x() const { return v[0]; }
inline double Vec3::y() const { return v[1]; }
inline double Vec3::z() const { return v[2]; }

inline std::ostream& operator<<(std::ostream& os, const Vec3& v)
{
os << v.str();
return os;
}

inline bool operator==(const Vec3& a, const Vec3& b)
{
return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
}

inline const Vec3 Vec3::clamp255() const
{
return Vec3 { (v[0] > 255) ? 255
: (v[0] < 0)       ? 0
: v[0],
(v[1] > 255)     ? 255
: (v[1] < 0) ? 0
: v[1],
(v[2] > 255)     ? 255
: (v[2] < 0) ? 0
: v[2] };
}

inline const std::string Vec3::ppm() const
{
return std::to_string(static_cast<int>(v[0])) + " "s + std::to_string(static_cast<int>(v[1]))
+ " "s + std::to_string(static_cast<int>(v[2]));
}

inline double Vec3::R() const { return v[0]; }

inline double Vec3::G() const { return v[1]; }

inline double Vec3::B() const { return v[2]; }

inline const Vec3 operator*(double d, const Vec3 v) { return v * d; }

inline const Vec3 operator/(double d, const Vec3 v) { return v * (1 / d); }

double Vec3::distance(const Vec3& a) const 
{
return std::sqrt((v[0] - a.v[0]) * (v[0] - a.v[0]) + (v[1] - a.v[1]) * (v[1] - a.v[1])
+ (v[2] - a.v[2]) * (v[2] - a.v[2]));
}

double Vec3::distance_squared(const Vec3& a) const 
{
return (v[0] - a.v[0]) * (v[0] - a.v[0]) + (v[1] - a.v[1]) * (v[1] - a.v[1])
+ (v[2] - a.v[2]) * (v[2] - a.v[2]);
}

inline const Vec3 Vec3::intify() const
{
return Vec3 { static_cast<double>(static_cast<int>(v[0])),
static_cast<double>(static_cast<int>(v[1])), static_cast<double>(static_cast<int>(v[2])) };
}
