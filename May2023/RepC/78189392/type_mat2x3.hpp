#pragma once
#include "../fwd.hpp"
#include "type_vec2.hpp"
#include "type_vec3.hpp"
#include "type_mat.hpp"
#include <limits>
#include <cstddef>
namespace glm
{
template <typename T, precision P = defaultp>
struct tmat2x3
{
typedef tvec3<T, P> col_type;
typedef tvec2<T, P> row_type;
typedef tmat2x3<T, P> type;
typedef tmat3x2<T, P> transpose_type;
typedef T value_type;
#		ifdef GLM_META_PROG_HELPERS
static GLM_RELAXED_CONSTEXPR length_t components = 2;
static GLM_RELAXED_CONSTEXPR length_t cols = 2;
static GLM_RELAXED_CONSTEXPR length_t rows = 3;
static GLM_RELAXED_CONSTEXPR precision prec = P;
#		endif
private:
col_type value[2];
public:
GLM_FUNC_DECL tmat2x3() GLM_DEFAULT_CTOR;
GLM_FUNC_DECL tmat2x3(tmat2x3<T, P> const & m) GLM_DEFAULT;
template <precision Q>
GLM_FUNC_DECL tmat2x3(tmat2x3<T, Q> const & m);
GLM_FUNC_DECL explicit tmat2x3(ctor);
GLM_FUNC_DECL explicit tmat2x3(T const & s);
GLM_FUNC_DECL tmat2x3(
T const & x0, T const & y0, T const & z0,
T const & x1, T const & y1, T const & z1);
GLM_FUNC_DECL tmat2x3(
col_type const & v0,
col_type const & v1);
template <typename X1, typename Y1, typename Z1, typename X2, typename Y2, typename Z2>
GLM_FUNC_DECL tmat2x3(
X1 const & x1, Y1 const & y1, Z1 const & z1,
X2 const & x2, Y2 const & y2, Z2 const & z2);
template <typename U, typename V>
GLM_FUNC_DECL tmat2x3(
tvec3<U, P> const & v1,
tvec3<V, P> const & v2);
template <typename U, precision Q>
GLM_FUNC_DECL GLM_EXPLICIT tmat2x3(tmat2x3<U, Q> const & m);
GLM_FUNC_DECL explicit tmat2x3(tmat2x2<T, P> const & x);
GLM_FUNC_DECL explicit tmat2x3(tmat3x3<T, P> const & x);
GLM_FUNC_DECL explicit tmat2x3(tmat4x4<T, P> const & x);
GLM_FUNC_DECL explicit tmat2x3(tmat2x4<T, P> const & x);
GLM_FUNC_DECL explicit tmat2x3(tmat3x2<T, P> const & x);
GLM_FUNC_DECL explicit tmat2x3(tmat3x4<T, P> const & x);
GLM_FUNC_DECL explicit tmat2x3(tmat4x2<T, P> const & x);
GLM_FUNC_DECL explicit tmat2x3(tmat4x3<T, P> const & x);
#		ifdef GLM_FORCE_SIZE_FUNC
typedef size_t size_type;
GLM_FUNC_DECL GLM_CONSTEXPR size_t size() const;
GLM_FUNC_DECL col_type & operator[](size_type i);
GLM_FUNC_DECL col_type const & operator[](size_type i) const;
#		else
typedef length_t length_type;
GLM_FUNC_DECL GLM_CONSTEXPR length_type length() const;
GLM_FUNC_DECL col_type & operator[](length_type i);
GLM_FUNC_DECL col_type const & operator[](length_type i) const;
#		endif
GLM_FUNC_DECL tmat2x3<T, P> & operator=(tmat2x3<T, P> const & m) GLM_DEFAULT;
template <typename U>
GLM_FUNC_DECL tmat2x3<T, P> & operator=(tmat2x3<U, P> const & m);
template <typename U>
GLM_FUNC_DECL tmat2x3<T, P> & operator+=(U s);
template <typename U>
GLM_FUNC_DECL tmat2x3<T, P> & operator+=(tmat2x3<U, P> const & m);
template <typename U>
GLM_FUNC_DECL tmat2x3<T, P> & operator-=(U s);
template <typename U>
GLM_FUNC_DECL tmat2x3<T, P> & operator-=(tmat2x3<U, P> const & m);
template <typename U>
GLM_FUNC_DECL tmat2x3<T, P> & operator*=(U s);
template <typename U>
GLM_FUNC_DECL tmat2x3<T, P> & operator/=(U s);
GLM_FUNC_DECL tmat2x3<T, P> & operator++ ();
GLM_FUNC_DECL tmat2x3<T, P> & operator-- ();
GLM_FUNC_DECL tmat2x3<T, P> operator++(int);
GLM_FUNC_DECL tmat2x3<T, P> operator--(int);
};
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator+(tmat2x3<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator-(tmat2x3<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator+(tmat2x3<T, P> const & m, T const & s);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator+(tmat2x3<T, P> const & m1, tmat2x3<T, P> const & m2);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator-(tmat2x3<T, P> const & m, T const & s);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator-(tmat2x3<T, P> const & m1, tmat2x3<T, P> const & m2);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator*(tmat2x3<T, P> const & m, T const & s);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator*(T const & s, tmat2x3<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL typename tmat2x3<T, P>::col_type operator*(tmat2x3<T, P> const & m, typename tmat2x3<T, P>::row_type const & v);
template <typename T, precision P>
GLM_FUNC_DECL typename tmat2x3<T, P>::row_type operator*(typename tmat2x3<T, P>::col_type const & v, tmat2x3<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator*(tmat2x3<T, P> const & m1, tmat2x2<T, P> const & m2);
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> operator*(tmat2x3<T, P> const & m1, tmat3x2<T, P> const & m2);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x3<T, P> operator*(tmat2x3<T, P> const & m1, tmat4x2<T, P> const & m2);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator/(tmat2x3<T, P> const & m, T const & s);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x3<T, P> operator/(T const & s, tmat2x3<T, P> const & m);
template <typename T, precision P>
GLM_FUNC_DECL bool operator==(tmat2x3<T, P> const & m1, tmat2x3<T, P> const & m2);
template <typename T, precision P>
GLM_FUNC_DECL bool operator!=(tmat2x3<T, P> const & m1, tmat2x3<T, P> const & m2);
}
#ifndef GLM_EXTERNAL_TEMPLATE
#include "type_mat2x3.inl"
#endif
