#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std::string_literals;

class Vec2 {
protected:
const double v[2];

public:
Vec2(double _x, double _y)
: v { _x, _y }
{
}

const Vec2 operator+(const Vec2& a) const; 
const Vec2 operator-(const Vec2& a) const; 
const Vec2 operator*(const double d) const; 
const Vec2 operator/(const double d) const; 
double dot(const Vec2& a) const; 
double len() const; 
double len_squared() const; 
bool operator<(const Vec2& a) const; 
bool operator>(const Vec2& a) const; 
const Vec2 normalize() const; 
const std::string str() const;

double x() const;
double y() const;

double R() const;
double G() const;

const Vec2 intify() const;
};

inline const Vec2 Vec2::operator+(const Vec2& a) const
{
return Vec2 { v[0] + a.v[0], v[1] + a.v[1] };
}

inline const Vec2 Vec2::operator-(const Vec2& a) const
{
return Vec2 { v[0] - a.v[0], v[1] - a.v[1] };
}

inline const Vec2 Vec2::operator*(const double d) const { return Vec2 { v[0] * d, v[1] * d }; }

inline const Vec2 Vec2::operator/(const double d) const
{
const double r = 1.0 / d;
return Vec2 { v[0] * r, v[1] * r };
}

inline double Vec2::dot(const Vec2& a) const { return v[0] * a.v[0] + v[1] * a.v[1]; }

inline double Vec2::len() const { return std::sqrt(v[0] * v[0] + v[1] * v[1]); }

inline double Vec2::len_squared() const { return v[0] * v[0] + v[1] * v[1]; }

inline bool Vec2::operator<(const Vec2& a) const
{
return (v[0] * v[0] + v[1] * v[1]) < (a.v[0] * a.v[0] + a.v[1] * a.v[1]);
}

inline bool Vec2::operator>(const Vec2& a) const
{
return (v[0] * v[0] + v[1] * v[1]) > (a.v[0] * a.v[0] + a.v[1] * a.v[1]);
}

inline const Vec2 Vec2::normalize() const
{
const double l = this->len();
return Vec2 { v[0] / l, v[1] / l };
}

inline const std::string Vec2::str() const
{
std::stringstream ss;
ss << "["s << std::setprecision(3) << v[0] << ", "s << v[1] << "]"s;
return ss.str();
}

inline std::ostream& operator<<(std::ostream& os, const Vec2& v)
{
os << v.str();
return os;
}

inline double Vec2::x() const { return v[0]; }
inline double Vec2::y() const { return v[1]; }

inline bool operator==(const Vec2& a, const Vec2& b) { return a.x() == b.x() && a.y() == b.y(); }

inline double Vec2::R() const { return v[0]; }

inline double Vec2::G() const { return v[1]; }
inline const Vec2 Vec2::intify() const
{
return Vec2 { static_cast<double>(static_cast<int>(v[0])),
static_cast<double>(static_cast<int>(v[1])) };
}
