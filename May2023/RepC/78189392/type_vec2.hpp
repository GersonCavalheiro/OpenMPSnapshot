#pragma once
#include "type_vec.hpp"
#ifdef GLM_SWIZZLE
#	if GLM_HAS_ANONYMOUS_UNION
#		include "_swizzle.hpp"
#	else
#		include "_swizzle_func.hpp"
#	endif
#endif 
#include <cstddef>
namespace glm
{
template <typename T, precision P = defaultp>
struct tvec2
{
typedef tvec2<T, P> type;
typedef tvec2<bool, P> bool_type;
typedef T value_type;
#		ifdef GLM_META_PROG_HELPERS
static GLM_RELAXED_CONSTEXPR length_t components = 2;
static GLM_RELAXED_CONSTEXPR precision prec = P;
#		endif
#		if GLM_HAS_ANONYMOUS_UNION
union
{
struct{ T x, y; };
struct{ T r, g; };
struct{ T s, t; };
#				ifdef GLM_SWIZZLE
_GLM_SWIZZLE2_2_MEMBERS(T, P, tvec2, x, y)
_GLM_SWIZZLE2_2_MEMBERS(T, P, tvec2, r, g)
_GLM_SWIZZLE2_2_MEMBERS(T, P, tvec2, s, t)
_GLM_SWIZZLE2_3_MEMBERS(T, P, tvec3, x, y)
_GLM_SWIZZLE2_3_MEMBERS(T, P, tvec3, r, g)
_GLM_SWIZZLE2_3_MEMBERS(T, P, tvec3, s, t)
_GLM_SWIZZLE2_4_MEMBERS(T, P, tvec4, x, y)
_GLM_SWIZZLE2_4_MEMBERS(T, P, tvec4, r, g)
_GLM_SWIZZLE2_4_MEMBERS(T, P, tvec4, s, t)
#				endif
};
#		else
union {T x, r, s;};
union {T y, g, t;};
#			ifdef GLM_SWIZZLE
GLM_SWIZZLE_GEN_VEC_FROM_VEC2(T, P, tvec2, tvec2, tvec3, tvec4)
#			endif
#		endif
#		ifdef GLM_FORCE_SIZE_FUNC
typedef size_t size_type;
GLM_FUNC_DECL GLM_CONSTEXPR size_type size() const;
GLM_FUNC_DECL T & operator[](size_type i);
GLM_FUNC_DECL T const & operator[](size_type i) const;
#		else
typedef length_t length_type;
GLM_FUNC_DECL GLM_CONSTEXPR length_type length() const;
GLM_FUNC_DECL T & operator[](length_type i);
GLM_FUNC_DECL T const & operator[](length_type i) const;
#		endif
GLM_FUNC_DECL tvec2() GLM_DEFAULT_CTOR;
GLM_FUNC_DECL tvec2(tvec2<T, P> const & v) GLM_DEFAULT;
template <precision Q>
GLM_FUNC_DECL tvec2(tvec2<T, Q> const & v);
GLM_FUNC_DECL explicit tvec2(ctor);
GLM_FUNC_DECL explicit tvec2(T const & scalar);
GLM_FUNC_DECL tvec2(T const & s1, T const & s2);
template <typename A, typename B>
GLM_FUNC_DECL tvec2(A const & x, B const & y);
template <typename A, typename B>
GLM_FUNC_DECL tvec2(tvec1<A, P> const & v1, tvec1<B, P> const & v2);
template <typename U, precision Q>
GLM_FUNC_DECL explicit tvec2(tvec3<U, Q> const & v);
template <typename U, precision Q>
GLM_FUNC_DECL explicit tvec2(tvec4<U, Q> const & v);
template <typename U, precision Q>
GLM_FUNC_DECL GLM_EXPLICIT tvec2(tvec2<U, Q> const & v);
#		if GLM_HAS_ANONYMOUS_UNION && defined(GLM_SWIZZLE)
template <int E0, int E1>
GLM_FUNC_DECL tvec2(detail::_swizzle<2, T, P, tvec2<T, P>, E0, E1,-1,-2> const & that)
{
*this = that();
}
#		endif
GLM_FUNC_DECL tvec2<T, P>& operator=(tvec2<T, P> const & v) GLM_DEFAULT;
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator=(tvec2<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator+=(U scalar);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator+=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator+=(tvec2<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator-=(U scalar);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator-=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator-=(tvec2<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator*=(U scalar);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator*=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator*=(tvec2<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator/=(U scalar);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator/=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec2<T, P>& operator/=(tvec2<U, P> const & v);
GLM_FUNC_DECL tvec2<T, P> & operator++();
GLM_FUNC_DECL tvec2<T, P> & operator--();
GLM_FUNC_DECL tvec2<T, P> operator++(int);
GLM_FUNC_DECL tvec2<T, P> operator--(int);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator%=(U scalar);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator%=(tvec1<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator%=(tvec2<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator&=(U scalar);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator&=(tvec1<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator&=(tvec2<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator|=(U scalar);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator|=(tvec1<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator|=(tvec2<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator^=(U scalar);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator^=(tvec1<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator^=(tvec2<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator<<=(U scalar);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator<<=(tvec1<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator<<=(tvec2<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator>>=(U scalar);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator>>=(tvec1<U, P> const & v);
template <typename U> 
GLM_FUNC_DECL tvec2<T, P> & operator>>=(tvec2<U, P> const & v);
};
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator+(tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator-(tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator+(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator+(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator+(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator+(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator+(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator-(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator-(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator-(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator-(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator-(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator*(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator*(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator*(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator*(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator*(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator/(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator/(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator/(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator/(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator/(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator%(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator%(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator%(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator%(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator%(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator&(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator&(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator&(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator&(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator&(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator|(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator|(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator|(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator|(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator|(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator^(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator^(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator^(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator^(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator^(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator<<(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator<<(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator<<(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator<<(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator<<(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator>>(tvec2<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator>>(tvec2<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator>>(T const & scalar, tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator>>(tvec1<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator>>(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec2<T, P> operator~(tvec2<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL bool operator==(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL bool operator!=(tvec2<T, P> const & v1, tvec2<T, P> const & v2);
}
#ifndef GLM_EXTERNAL_TEMPLATE
#include "type_vec2.inl"
#endif
