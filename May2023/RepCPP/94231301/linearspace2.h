
#pragma once

#include "vec2.h"

namespace embree
{

template<typename T> struct LinearSpace2
{
typedef T Vector;
typedef typename T::Scalar Scalar;


__forceinline LinearSpace2           ( ) {}
__forceinline LinearSpace2           ( const LinearSpace2& other ) { vx = other.vx; vy = other.vy; }
__forceinline LinearSpace2& operator=( const LinearSpace2& other ) { vx = other.vx; vy = other.vy; return *this; }

template<typename L1> __forceinline LinearSpace2( const LinearSpace2<L1>& s ) : vx(s.vx), vy(s.vy) {}


__forceinline LinearSpace2(const Vector& vx, const Vector& vy)
: vx(vx), vy(vy) {}


__forceinline LinearSpace2(const Scalar& m00, const Scalar& m01, 
const Scalar& m10, const Scalar& m11)
: vx(m00,m10), vy(m01,m11) {}


__forceinline const Scalar det() const { return vx.x*vy.y - vx.y*vy.x; }


__forceinline const LinearSpace2 adjoint() const { return LinearSpace2(vy.y,-vy.x,-vx.y,vx.x); }


__forceinline const LinearSpace2 inverse() const { return adjoint()/det(); }


__forceinline const LinearSpace2 transposed() const { return LinearSpace2(vx.x,vx.y,vy.x,vy.y); }


__forceinline Vector row0() const { return Vector(vx.x,vy.x); }


__forceinline Vector row1() const { return Vector(vx.y,vy.y); }


__forceinline LinearSpace2( ZeroTy ) : vx(zero), vy(zero) {}
__forceinline LinearSpace2( OneTy ) : vx(one, zero), vy(zero, one) {}


static __forceinline LinearSpace2 scale(const Vector& s) {
return LinearSpace2(s.x,   0,
0  , s.y);
}


static __forceinline LinearSpace2 rotate(const Scalar& r) {
Scalar s = sin(r), c = cos(r);
return LinearSpace2(c, -s,
s,  c);
}


LinearSpace2 orthogonal() const 
{
LinearSpace2 m = *this;

Scalar mirror(one);
if (m.det() < Scalar(zero)) {
m.vx = -m.vx;
mirror = -mirror;
}

for (int i = 0; i < 99; i++) {
const LinearSpace2 m_next = 0.5 * (m + m.transposed().inverse());
const LinearSpace2 d = m_next - m;
m = m_next;
if (max(dot(d.vx, d.vx), dot(d.vy, d.vy)) < 1e-8)
break;
}

return LinearSpace2(mirror*m.vx, m.vy);
}

public:


Vector vx,vy;
};


template<typename T> __forceinline LinearSpace2<T> operator -( const LinearSpace2<T>& a ) { return LinearSpace2<T>(-a.vx,-a.vy); }
template<typename T> __forceinline LinearSpace2<T> operator +( const LinearSpace2<T>& a ) { return LinearSpace2<T>(+a.vx,+a.vy); }
template<typename T> __forceinline LinearSpace2<T> rcp       ( const LinearSpace2<T>& a ) { return a.inverse(); }


template<typename T> __forceinline LinearSpace2<T> operator +( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return LinearSpace2<T>(a.vx+b.vx,a.vy+b.vy); }
template<typename T> __forceinline LinearSpace2<T> operator -( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return LinearSpace2<T>(a.vx-b.vx,a.vy-b.vy); }

template<typename T> __forceinline LinearSpace2<T> operator*(const typename T::Scalar & a, const LinearSpace2<T>& b) { return LinearSpace2<T>(a*b.vx, a*b.vy); }
template<typename T> __forceinline T               operator*(const LinearSpace2<T>& a, const T              & b) { return b.x*a.vx + b.y*a.vy; }
template<typename T> __forceinline LinearSpace2<T> operator*(const LinearSpace2<T>& a, const LinearSpace2<T>& b) { return LinearSpace2<T>(a*b.vx, a*b.vy); }

template<typename T> __forceinline LinearSpace2<T> operator/(const LinearSpace2<T>& a, const typename T::Scalar & b) { return LinearSpace2<T>(a.vx/b, a.vy/b); }
template<typename T> __forceinline LinearSpace2<T> operator/(const LinearSpace2<T>& a, const LinearSpace2<T>& b) { return a * rcp(b); }

template<typename T> __forceinline LinearSpace2<T>& operator *=( LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a = a * b; }
template<typename T> __forceinline LinearSpace2<T>& operator /=( LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a = a / b; }


template<typename T> __forceinline bool operator ==( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a.vx == b.vx && a.vy == b.vy; }
template<typename T> __forceinline bool operator !=( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a.vx != b.vx || a.vy != b.vy; }


template<typename T> static std::ostream& operator<<(std::ostream& cout, const LinearSpace2<T>& m) {
return cout << "{ vx = " << m.vx << ", vy = " << m.vy << "}";
}


typedef LinearSpace2<Vec2f> LinearSpace2f;
}
