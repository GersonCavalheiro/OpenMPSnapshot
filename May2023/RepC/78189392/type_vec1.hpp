#pragma once
#include "../fwd.hpp"
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
struct tvec1
{
typedef tvec1<T, P> type;
typedef tvec1<bool, P> bool_type;
typedef T value_type;
#		ifdef GLM_META_PROG_HELPERS
static GLM_RELAXED_CONSTEXPR length_t components = 1;
static GLM_RELAXED_CONSTEXPR precision prec = P;
#		endif
#		if GLM_HAS_ANONYMOUS_UNION
union
{
T x;
T r;
T s;
};
#		else
union {T x, r, s;};
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
GLM_FUNC_DECL tvec1() GLM_DEFAULT_CTOR;
GLM_FUNC_DECL tvec1(tvec1<T, P> const & v) GLM_DEFAULT;
template <precision Q>
GLM_FUNC_DECL tvec1(tvec1<T, Q> const & v);
GLM_FUNC_DECL explicit tvec1(ctor);
GLM_FUNC_DECL explicit tvec1(T const & scalar);
template <typename U, precision Q>
GLM_FUNC_DECL explicit tvec1(tvec2<U, Q> const & v);
template <typename U, precision Q>
GLM_FUNC_DECL explicit tvec1(tvec3<U, Q> const & v);
template <typename U, precision Q>
GLM_FUNC_DECL explicit tvec1(tvec4<U, Q> const & v);
template <typename U, precision Q>
GLM_FUNC_DECL GLM_EXPLICIT tvec1(tvec1<U, Q> const & v);
#		if(GLM_HAS_ANONYMOUS_UNION && defined(GLM_SWIZZLE))
template <int E0>
GLM_FUNC_DECL tvec1(detail::_swizzle<1, T, P, tvec1<T, P>, E0, -1,-2,-3> const & that)
{
*this = that();
}
#		endif
GLM_FUNC_DECL tvec1<T, P> & operator=(tvec1<T, P> const & v) GLM_DEFAULT;
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator+=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator+=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator-=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator-=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator*=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator*=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator/=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator/=(tvec1<U, P> const & v);
GLM_FUNC_DECL tvec1<T, P> & operator++();
GLM_FUNC_DECL tvec1<T, P> & operator--();
GLM_FUNC_DECL tvec1<T, P> operator++(int);
GLM_FUNC_DECL tvec1<T, P> operator--(int);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator%=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator%=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator&=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator&=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator|=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator|=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator^=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator^=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator<<=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator<<=(tvec1<U, P> const & v);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator>>=(U const & scalar);
template <typename U>
GLM_FUNC_DECL tvec1<T, P> & operator>>=(tvec1<U, P> const & v);
};
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator+(tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator-(tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator+(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator+(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator+(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator-(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator-(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator-	(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator*(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator*(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator*(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator/(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator/(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator/(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator%(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator%(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator%(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator&(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator&(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator&(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator|(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator|(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator|(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator^(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator^(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator^(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator<<(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator<<(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator<<(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator>>(tvec1<T, P> const & v, T const & scalar);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator>>(T const & scalar, tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator>>(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL tvec1<T, P> operator~(tvec1<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL bool operator==(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
template <typename T, precision P>
GLM_FUNC_DECL bool operator!=(tvec1<T, P> const & v1, tvec1<T, P> const & v2);
}
#ifndef GLM_EXTERNAL_TEMPLATE
#include "type_vec1.inl"
#endif
