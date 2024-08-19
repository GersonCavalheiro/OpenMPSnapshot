
#ifndef GLM_GTC_half_float
#define GLM_GTC_half_float GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTC_half_float extension included")
#endif

namespace glm{
namespace detail
{
#if(GLM_COMPONENT == GLM_COMPONENT_CXX98)
template <>
struct tvec2<half>
{
enum ctor{null};
typedef half value_type;
typedef std::size_t size_type;

GLM_FUNC_DECL size_type length() const;
static GLM_FUNC_DECL size_type value_size();

typedef tvec2<half> type;
typedef tvec2<bool> bool_type;


half x, y;


GLM_FUNC_DECL half & operator[](size_type i);
GLM_FUNC_DECL half const & operator[](size_type i) const;


GLM_FUNC_DECL tvec2();
GLM_FUNC_DECL tvec2(tvec2<half> const & v);


GLM_FUNC_DECL explicit tvec2(ctor);
GLM_FUNC_DECL explicit tvec2(
half const & s);
GLM_FUNC_DECL explicit tvec2(
half const & s1, 
half const & s2);


GLM_FUNC_DECL tvec2(tref2<half> const & r);


template <typename U> 
GLM_FUNC_DECL explicit tvec2(U const & x);
template <typename U, typename V> 
GLM_FUNC_DECL explicit tvec2(U const & x, V const & y);			


template <typename U> 
GLM_FUNC_DECL explicit tvec2(tvec2<U> const & v);
template <typename U> 
GLM_FUNC_DECL explicit tvec2(tvec3<U> const & v);
template <typename U> 
GLM_FUNC_DECL explicit tvec2(tvec4<U> const & v);


GLM_FUNC_DECL tvec2<half>& operator= (tvec2<half> const & v);

GLM_FUNC_DECL tvec2<half>& operator+=(half const & s);
GLM_FUNC_DECL tvec2<half>& operator+=(tvec2<half> const & v);
GLM_FUNC_DECL tvec2<half>& operator-=(half const & s);
GLM_FUNC_DECL tvec2<half>& operator-=(tvec2<half> const & v);
GLM_FUNC_DECL tvec2<half>& operator*=(half const & s);
GLM_FUNC_DECL tvec2<half>& operator*=(tvec2<half> const & v);
GLM_FUNC_DECL tvec2<half>& operator/=(half const & s);
GLM_FUNC_DECL tvec2<half>& operator/=(tvec2<half> const & v);
GLM_FUNC_DECL tvec2<half>& operator++();
GLM_FUNC_DECL tvec2<half>& operator--();


GLM_FUNC_DECL half swizzle(comp X) const;
GLM_FUNC_DECL tvec2<half> swizzle(comp X, comp Y) const;
GLM_FUNC_DECL tvec3<half> swizzle(comp X, comp Y, comp Z) const;
GLM_FUNC_DECL tvec4<half> swizzle(comp X, comp Y, comp Z, comp W) const;
GLM_FUNC_DECL tref2<half> swizzle(comp X, comp Y);
};

template <>
struct tvec3<half>
{
enum ctor{null};
typedef half value_type;
typedef std::size_t size_type;
GLM_FUNC_DECL size_type length() const;
static GLM_FUNC_DECL size_type value_size();

typedef tvec3<half> type;
typedef tvec3<bool> bool_type;


half x, y, z;


GLM_FUNC_DECL half & operator[](size_type i);
GLM_FUNC_DECL half const & operator[](size_type i) const;


GLM_FUNC_DECL tvec3();
GLM_FUNC_DECL tvec3(tvec3<half> const & v);


GLM_FUNC_DECL explicit tvec3(ctor);
GLM_FUNC_DECL explicit tvec3(
half const & s);
GLM_FUNC_DECL explicit tvec3(
half const & s1, 
half const & s2, 
half const & s3);


GLM_FUNC_DECL tvec3(tref3<half> const & r);


template <typename U> 
GLM_FUNC_DECL explicit tvec3(U const & x);
template <typename U, typename V, typename W> 
GLM_FUNC_DECL explicit tvec3(U const & x, V const & y, W const & z);			


template <typename A, typename B> 
GLM_FUNC_DECL explicit tvec3(tvec2<A> const & v, B const & s);
template <typename A, typename B> 
GLM_FUNC_DECL explicit tvec3(A const & s, tvec2<B> const & v);
template <typename U> 
GLM_FUNC_DECL explicit tvec3(tvec3<U> const & v);
template <typename U> 
GLM_FUNC_DECL explicit tvec3(tvec4<U> const & v);


GLM_FUNC_DECL tvec3<half>& operator= (tvec3<half> const & v);

GLM_FUNC_DECL tvec3<half>& operator+=(half const & s);
GLM_FUNC_DECL tvec3<half>& operator+=(tvec3<half> const & v);
GLM_FUNC_DECL tvec3<half>& operator-=(half const & s);
GLM_FUNC_DECL tvec3<half>& operator-=(tvec3<half> const & v);
GLM_FUNC_DECL tvec3<half>& operator*=(half const & s);
GLM_FUNC_DECL tvec3<half>& operator*=(tvec3<half> const & v);
GLM_FUNC_DECL tvec3<half>& operator/=(half const & s);
GLM_FUNC_DECL tvec3<half>& operator/=(tvec3<half> const & v);
GLM_FUNC_DECL tvec3<half>& operator++();
GLM_FUNC_DECL tvec3<half>& operator--();


GLM_FUNC_DECL half swizzle(comp X) const;
GLM_FUNC_DECL tvec2<half> swizzle(comp X, comp Y) const;
GLM_FUNC_DECL tvec3<half> swizzle(comp X, comp Y, comp Z) const;
GLM_FUNC_DECL tvec4<half> swizzle(comp X, comp Y, comp Z, comp W) const;
GLM_FUNC_DECL tref3<half> swizzle(comp X, comp Y, comp Z);
};

template <>
struct tvec4<half>
{
enum ctor{null};
typedef half value_type;
typedef std::size_t size_type;
GLM_FUNC_DECL size_type length() const;
static GLM_FUNC_DECL size_type value_size();

typedef tvec4<half> type;
typedef tvec4<bool> bool_type;


half x, y, z, w;


GLM_FUNC_DECL half & operator[](size_type i);
GLM_FUNC_DECL half const & operator[](size_type i) const;


GLM_FUNC_DECL tvec4();
GLM_FUNC_DECL tvec4(tvec4<half> const & v);


GLM_FUNC_DECL explicit tvec4(ctor);
GLM_FUNC_DECL explicit tvec4(
half const & s);
GLM_FUNC_DECL explicit tvec4(
half const & s0, 
half const & s1, 
half const & s2, 
half const & s3);


GLM_FUNC_DECL tvec4(tref4<half> const & r);


template <typename U> 
GLM_FUNC_DECL explicit tvec4(U const & x);
template <typename A, typename B, typename C, typename D> 
GLM_FUNC_DECL explicit tvec4(A const & x, B const & y, C const & z, D const & w);			


template <typename A, typename B, typename C> 
GLM_FUNC_DECL explicit tvec4(tvec2<A> const & v, B const & s1, C const & s2);
template <typename A, typename B, typename C> 
GLM_FUNC_DECL explicit tvec4(A const & s1, tvec2<B> const & v, C const & s2);
template <typename A, typename B, typename C> 
GLM_FUNC_DECL explicit tvec4(A const & s1, B const & s2, tvec2<C> const & v);
template <typename A, typename B> 
GLM_FUNC_DECL explicit tvec4(tvec3<A> const & v, B const & s);
template <typename A, typename B> 
GLM_FUNC_DECL explicit tvec4(A const & s, tvec3<B> const & v);
template <typename A, typename B> 
GLM_FUNC_DECL explicit tvec4(tvec2<A> const & v1, tvec2<B> const & v2);
template <typename U> 
GLM_FUNC_DECL explicit tvec4(tvec4<U> const & v);


GLM_FUNC_DECL tvec4<half>& operator= (tvec4<half> const & v);

GLM_FUNC_DECL tvec4<half>& operator+=(half const & s);
GLM_FUNC_DECL tvec4<half>& operator+=(tvec4<half> const & v);
GLM_FUNC_DECL tvec4<half>& operator-=(half const & s);
GLM_FUNC_DECL tvec4<half>& operator-=(tvec4<half> const & v);
GLM_FUNC_DECL tvec4<half>& operator*=(half const & s);
GLM_FUNC_DECL tvec4<half>& operator*=(tvec4<half> const & v);
GLM_FUNC_DECL tvec4<half>& operator/=(half const & s);
GLM_FUNC_DECL tvec4<half>& operator/=(tvec4<half> const & v);
GLM_FUNC_DECL tvec4<half>& operator++();
GLM_FUNC_DECL tvec4<half>& operator--();


GLM_FUNC_DECL half swizzle(comp X) const;
GLM_FUNC_DECL tvec2<half> swizzle(comp X, comp Y) const;
GLM_FUNC_DECL tvec3<half> swizzle(comp X, comp Y, comp Z) const;
GLM_FUNC_DECL tvec4<half> swizzle(comp X, comp Y, comp Z, comp W) const;
GLM_FUNC_DECL tref4<half> swizzle(comp X, comp Y, comp Z, comp W);
};
#endif
}


typedef detail::half					half;

typedef detail::tvec2<detail::half>	hvec2;

typedef detail::tvec3<detail::half>	hvec3;

typedef detail::tvec4<detail::half>	hvec4;

typedef detail::tmat2x2<detail::half>	hmat2;

typedef detail::tmat3x3<detail::half>	hmat3;

typedef detail::tmat4x4<detail::half>	hmat4;

typedef detail::tmat2x2<detail::half>	hmat2x2;

typedef detail::tmat2x3<detail::half>	hmat2x3;

typedef detail::tmat2x4<detail::half>	hmat2x4;

typedef detail::tmat3x2<detail::half>	hmat3x2;

typedef detail::tmat3x3<detail::half>	hmat3x3;

typedef detail::tmat3x4<detail::half>	hmat3x4;

typedef detail::tmat4x2<detail::half>	hmat4x2;    

typedef detail::tmat4x3<detail::half>	hmat4x3;

typedef detail::tmat4x4<detail::half>	hmat4x4;

GLM_FUNC_DECL half abs(half const & x);

GLM_FUNC_DECL hvec2 abs(hvec2 const & x);

GLM_FUNC_DECL hvec3 abs(hvec3 const & x);

GLM_FUNC_DECL hvec4 abs(hvec4 const & x);

GLM_FUNC_DECL half mix(half const & x, half const & y, bool const & a);

}

#include "half_float.inl"

#endif
