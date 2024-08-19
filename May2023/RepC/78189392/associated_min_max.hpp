#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_associated_min_max extension included")
#endif
namespace glm
{
template<typename T, typename U, precision P>
GLM_FUNC_DECL U associatedMin(T x, U a, T y, U b);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL tvec2<U, P> associatedMin(
vecType<T, P> const & x, vecType<U, P> const & a,
vecType<T, P> const & y, vecType<U, P> const & b);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMin(
T x, const vecType<U, P>& a,
T y, const vecType<U, P>& b);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMin(
vecType<T, P> const & x, U a,
vecType<T, P> const & y, U b);
template<typename T, typename U>
GLM_FUNC_DECL U associatedMin(
T x, U a,
T y, U b,
T z, U c);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMin(
vecType<T, P> const & x, vecType<U, P> const & a,
vecType<T, P> const & y, vecType<U, P> const & b,
vecType<T, P> const & z, vecType<U, P> const & c);
template<typename T, typename U>
GLM_FUNC_DECL U associatedMin(
T x, U a,
T y, U b,
T z, U c,
T w, U d);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMin(
vecType<T, P> const & x, vecType<U, P> const & a,
vecType<T, P> const & y, vecType<U, P> const & b,
vecType<T, P> const & z, vecType<U, P> const & c,
vecType<T, P> const & w, vecType<U, P> const & d);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMin(
T x, vecType<U, P> const & a,
T y, vecType<U, P> const & b,
T z, vecType<U, P> const & c,
T w, vecType<U, P> const & d);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMin(
vecType<T, P> const & x, U a,
vecType<T, P> const & y, U b,
vecType<T, P> const & z, U c,
vecType<T, P> const & w, U d);
template<typename T, typename U>
GLM_FUNC_DECL U associatedMax(T x, U a, T y, U b);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL tvec2<U, P> associatedMax(
vecType<T, P> const & x, vecType<U, P> const & a,
vecType<T, P> const & y, vecType<U, P> const & b);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> associatedMax(
T x, vecType<U, P> const & a,
T y, vecType<U, P> const & b);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMax(
vecType<T, P> const & x, U a,
vecType<T, P> const & y, U b);
template<typename T, typename U>
GLM_FUNC_DECL U associatedMax(
T x, U a,
T y, U b,
T z, U c);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMax(
vecType<T, P> const & x, vecType<U, P> const & a,
vecType<T, P> const & y, vecType<U, P> const & b,
vecType<T, P> const & z, vecType<U, P> const & c);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> associatedMax(
T x, vecType<U, P> const & a,
T y, vecType<U, P> const & b,
T z, vecType<U, P> const & c);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMax(
vecType<T, P> const & x, U a,
vecType<T, P> const & y, U b,
vecType<T, P> const & z, U c);
template<typename T, typename U>
GLM_FUNC_DECL U associatedMax(
T x, U a,
T y, U b,
T z, U c,
T w, U d);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMax(
vecType<T, P> const & x, vecType<U, P> const & a,
vecType<T, P> const & y, vecType<U, P> const & b,
vecType<T, P> const & z, vecType<U, P> const & c,
vecType<T, P> const & w, vecType<U, P> const & d);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMax(
T x, vecType<U, P> const & a,
T y, vecType<U, P> const & b,
T z, vecType<U, P> const & c,
T w, vecType<U, P> const & d);
template<typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<U, P> associatedMax(
vecType<T, P> const & x, U a,
vecType<T, P> const & y, U b,
vecType<T, P> const & z, U c,
vecType<T, P> const & w, U d);
} 
#include "associated_min_max.inl"
