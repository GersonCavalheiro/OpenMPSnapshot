#pragma once
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"
#include "../detail/_vectorize.hpp"
#include "../vector_relational.hpp"
#include "../common.hpp"
#include <limits>
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_integer extension included")
#endif
namespace glm
{
template <typename genIUType>
GLM_FUNC_DECL bool isPowerOfTwo(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> isPowerOfTwo(vecType<T, P> const & value);
template <typename genIUType>
GLM_FUNC_DECL genIUType ceilPowerOfTwo(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> ceilPowerOfTwo(vecType<T, P> const & value);
template <typename genIUType>
GLM_FUNC_DECL genIUType floorPowerOfTwo(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> floorPowerOfTwo(vecType<T, P> const & value);
template <typename genIUType>
GLM_FUNC_DECL genIUType roundPowerOfTwo(genIUType Value);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> roundPowerOfTwo(vecType<T, P> const & value);
template <typename genIUType>
GLM_FUNC_DECL bool isMultiple(genIUType Value, genIUType Multiple);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> isMultiple(vecType<T, P> const & Value, T Multiple);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> isMultiple(vecType<T, P> const & Value, vecType<T, P> const & Multiple);
template <typename genType>
GLM_FUNC_DECL genType ceilMultiple(genType Source, genType Multiple);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> ceilMultiple(vecType<T, P> const & Source, vecType<T, P> const & Multiple);
template <typename genType>
GLM_FUNC_DECL genType floorMultiple(
genType Source,
genType Multiple);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> floorMultiple(
vecType<T, P> const & Source,
vecType<T, P> const & Multiple);
template <typename genType>
GLM_FUNC_DECL genType roundMultiple(
genType Source,
genType Multiple);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> roundMultiple(
vecType<T, P> const & Source,
vecType<T, P> const & Multiple);
} 
#include "round.inl"
