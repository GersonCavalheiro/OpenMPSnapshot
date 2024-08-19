
#pragma once

#include "linearspace2.h"
#include "linearspace3.h"
#include "quaternion.h"
#include "bbox.h"

namespace embree
{
#define VectorT typename L::Vector
#define ScalarT typename L::Vector::Scalar


template<typename L>
struct AffineSpaceT
{
L l;           
VectorT p;     


__forceinline AffineSpaceT           ( )                           { }
__forceinline AffineSpaceT           ( const AffineSpaceT& other ) { l = other.l; p = other.p; }
__forceinline AffineSpaceT           ( const L           & other ) { l = other  ; p = VectorT(zero); }
__forceinline AffineSpaceT& operator=( const AffineSpaceT& other ) { l = other.l; p = other.p; return *this; }

__forceinline AffineSpaceT( const VectorT& vx, const VectorT& vy, const VectorT& vz, const VectorT& p ) : l(vx,vy,vz), p(p) {}
__forceinline AffineSpaceT( const L& l, const VectorT& p ) : l(l), p(p) {}

template<typename L1> __forceinline AffineSpaceT( const AffineSpaceT<L1>& s ) : l(s.l), p(s.p) {}


__forceinline AffineSpaceT( ZeroTy ) : l(zero), p(zero) {}
__forceinline AffineSpaceT( OneTy )  : l(one),  p(zero) {}


static __forceinline AffineSpaceT scale(const VectorT& s) { return L::scale(s); }


static __forceinline AffineSpaceT translate(const VectorT& p) { return AffineSpaceT(one,p); }


static __forceinline AffineSpaceT rotate(const ScalarT& r) { return L::rotate(r); }


static __forceinline AffineSpaceT rotate(const VectorT& u, const ScalarT& r) { return L::rotate(u,r); }


static __forceinline AffineSpaceT rotate(const VectorT& p, const VectorT& u, const ScalarT& r) { return translate(+p) * rotate(u,r) * translate(-p);  }


static __forceinline AffineSpaceT lookat(const VectorT& eye, const VectorT& point, const VectorT& up) {
VectorT Z = normalize(point-eye);
VectorT U = normalize(cross(up,Z));
VectorT V = normalize(cross(Z,U));
return AffineSpaceT(L(U,V,Z),eye);
}

};


template<typename L> __forceinline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(-a.l,-a.p); }
template<typename L> __forceinline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(+a.l,+a.p); }
template<typename L> __forceinline AffineSpaceT<L>        rcp( const AffineSpaceT<L>& a ) { L il = rcp(a.l); return AffineSpaceT<L>(il,-(il*a.p)); }


template<typename L> __forceinline const AffineSpaceT<L> operator +( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l+b.l,a.p+b.p); }
template<typename L> __forceinline const AffineSpaceT<L> operator -( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l-b.l,a.p-b.p); }

template<typename L> __forceinline const AffineSpaceT<L> operator *( const ScalarT        & a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a*b.l,a*b.p); }
template<typename L> __forceinline const AffineSpaceT<L> operator *( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l*b.l,a.l*b.p+a.p); }
template<typename L> __forceinline const AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a * rcp(b); }
template<typename L> __forceinline const AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const ScalarT        & b ) { return a * rcp(b); }

template<typename L> __forceinline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a * b; }
template<typename L> __forceinline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a * b; }
template<typename L> __forceinline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a / b; }
template<typename L> __forceinline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a / b; }

template<typename L> __forceinline VectorT xfmPoint (const AffineSpaceT<L>& m, const VectorT& p) { return madd(VectorT(p.x),m.l.vx,madd(VectorT(p.y),m.l.vy,madd(VectorT(p.z),m.l.vz,m.p))); }
template<typename L> __forceinline VectorT xfmVector(const AffineSpaceT<L>& m, const VectorT& v) { return xfmVector(m.l,v); }
template<typename L> __forceinline VectorT xfmNormal(const AffineSpaceT<L>& m, const VectorT& n) { return xfmNormal(m.l,n); }

__forceinline const BBox<Vec3fa> xfmBounds(const AffineSpaceT<LinearSpace3<Vec3fa> >& m, const BBox<Vec3fa>& b) 
{ 
BBox3fa dst = empty;
const Vec3fa p0(b.lower.x,b.lower.y,b.lower.z); dst.extend(xfmPoint(m,p0));
const Vec3fa p1(b.lower.x,b.lower.y,b.upper.z); dst.extend(xfmPoint(m,p1));
const Vec3fa p2(b.lower.x,b.upper.y,b.lower.z); dst.extend(xfmPoint(m,p2));
const Vec3fa p3(b.lower.x,b.upper.y,b.upper.z); dst.extend(xfmPoint(m,p3));
const Vec3fa p4(b.upper.x,b.lower.y,b.lower.z); dst.extend(xfmPoint(m,p4));
const Vec3fa p5(b.upper.x,b.lower.y,b.upper.z); dst.extend(xfmPoint(m,p5));
const Vec3fa p6(b.upper.x,b.upper.y,b.lower.z); dst.extend(xfmPoint(m,p6));
const Vec3fa p7(b.upper.x,b.upper.y,b.upper.z); dst.extend(xfmPoint(m,p7));
return dst;
}


template<typename L> __forceinline bool operator ==( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l == b.l && a.p == b.p; }
template<typename L> __forceinline bool operator !=( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l != b.l || a.p != b.p; }


template<typename L> __forceinline AffineSpaceT<L> select ( const typename L::Vector::Scalar::Bool& s, const AffineSpaceT<L>& t, const AffineSpaceT<L>& f ) {
return AffineSpaceT<L>(select(s,t.l,f.l),select(s,t.p,f.p));
}


template<typename T>
__forceinline AffineSpaceT<T> lerp(const AffineSpaceT<T>& l0, const AffineSpaceT<T>& l1, const float t) {
return AffineSpaceT<T>(lerp(l0.l,l1.l,t),lerp(l0.p,l1.p,t));
}


template<typename L> static std::ostream& operator<<(std::ostream& cout, const AffineSpaceT<L>& m) {
return cout << "{ l = " << m.l << ", p = " << m.p << " }";
}


typedef AffineSpaceT<LinearSpace2f> AffineSpace2f;
typedef AffineSpaceT<LinearSpace3f> AffineSpace3f;
typedef AffineSpaceT<LinearSpace3fa> AffineSpace3fa;
typedef AffineSpaceT<Quaternion3f > OrthonormalSpace3f;


template<> __forceinline AffineSpace2f AffineSpace2f::rotate(const Vec2f& p, const float& r) { return translate(+p) * AffineSpace2f(LinearSpace2f::rotate(r)) * translate(-p); }

#undef VectorT
#undef ScalarT
}
