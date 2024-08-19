#pragma once
#include "setup.hpp"
#include "precision.hpp"
#include "type_int.hpp"
#include "_fixes.hpp"
namespace glm
{
template <typename genType>
GLM_FUNC_DECL genType abs(genType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> abs(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> sign(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> floor(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> trunc(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> round(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> roundEven(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> ceil(vecType<T, P> const & x);
template <typename genType>
GLM_FUNC_DECL genType fract(genType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> fract(vecType<T, P> const & x);
template <typename genType>
GLM_FUNC_DECL genType mod(genType x, genType y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> mod(vecType<T, P> const & x, T y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> mod(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename genType>
GLM_FUNC_DECL genType modf(genType x, genType & i);
template <typename genType>
GLM_FUNC_DECL genType min(genType x, genType y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> min(vecType<T, P> const & x, T y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> min(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename genType>
GLM_FUNC_DECL genType max(genType x, genType y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> max(vecType<T, P> const & x, T y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> max(vecType<T, P> const & x, vecType<T, P> const & y);
template <typename genType>
GLM_FUNC_DECL genType clamp(genType x, genType minVal, genType maxVal);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> clamp(vecType<T, P> const & x, T minVal, T maxVal);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> clamp(vecType<T, P> const & x, vecType<T, P> const & minVal, vecType<T, P> const & maxVal);
template <typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> mix(vecType<T, P> const & x, vecType<T, P> const & y, vecType<U, P> const & a);
template <typename T, typename U, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> mix(vecType<T, P> const & x, vecType<T, P> const & y, U a);
template <typename genTypeT, typename genTypeU>
GLM_FUNC_DECL genTypeT mix(genTypeT x, genTypeT y, genTypeU a);
template <typename genType>
GLM_FUNC_DECL genType step(genType edge, genType x);
template <template <typename, precision> class vecType, typename T, precision P>
GLM_FUNC_DECL vecType<T, P> step(T edge, vecType<T, P> const & x);
template <template <typename, precision> class vecType, typename T, precision P>
GLM_FUNC_DECL vecType<T, P> step(vecType<T, P> const & edge, vecType<T, P> const & x);
template <typename genType>
GLM_FUNC_DECL genType smoothstep(genType edge0, genType edge1, genType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> smoothstep(T edge0, T edge1, vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> smoothstep(vecType<T, P> const & edge0, vecType<T, P> const & edge1, vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> isnan(vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> isinf(vecType<T, P> const & x);
GLM_FUNC_DECL int floatBitsToInt(float const & v);
template <template <typename, precision> class vecType, precision P>
GLM_FUNC_DECL vecType<int, P> floatBitsToInt(vecType<float, P> const & v);
GLM_FUNC_DECL uint floatBitsToUint(float const & v);
template <template <typename, precision> class vecType, precision P>
GLM_FUNC_DECL vecType<uint, P> floatBitsToUint(vecType<float, P> const & v);
GLM_FUNC_DECL float intBitsToFloat(int const & v);
template <template <typename, precision> class vecType, precision P>
GLM_FUNC_DECL vecType<float, P> intBitsToFloat(vecType<int, P> const & v);
GLM_FUNC_DECL float uintBitsToFloat(uint const & v);
template <template <typename, precision> class vecType, precision P>
GLM_FUNC_DECL vecType<float, P> uintBitsToFloat(vecType<uint, P> const & v);
template <typename genType>
GLM_FUNC_DECL genType fma(genType const & a, genType const & b, genType const & c);
template <typename genType, typename genIType>
GLM_FUNC_DECL genType frexp(genType const & x, genIType & exp);
template <typename genType, typename genIType>
GLM_FUNC_DECL genType ldexp(genType const & x, genIType const & exp);
}
#include "func_common.inl"
