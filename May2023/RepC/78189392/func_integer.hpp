#pragma once
#include "setup.hpp"
#include "precision.hpp"
#include "func_common.hpp"
#include "func_vector_relational.hpp"
namespace glm
{
template <precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<uint, P> uaddCarry(
vecType<uint, P> const & x,
vecType<uint, P> const & y,
vecType<uint, P> & carry);
template <precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<uint, P> usubBorrow(
vecType<uint, P> const & x,
vecType<uint, P> const & y,
vecType<uint, P> & borrow);
template <precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL void umulExtended(
vecType<uint, P> const & x,
vecType<uint, P> const & y,
vecType<uint, P> & msb,
vecType<uint, P> & lsb);
template <precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL void imulExtended(
vecType<int, P> const & x,
vecType<int, P> const & y,
vecType<int, P> & msb,
vecType<int, P> & lsb);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> bitfieldExtract(
vecType<T, P> const & Value,
int Offset,
int Bits);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> bitfieldInsert(
vecType<T, P> const & Base,
vecType<T, P> const & Insert,
int Offset,
int Bits);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> bitfieldReverse(vecType<T, P> const & v);
template <typename genType>
GLM_FUNC_DECL int bitCount(genType v);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<int, P> bitCount(vecType<T, P> const & v);
template <typename genIUType>
GLM_FUNC_DECL int findLSB(genIUType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<int, P> findLSB(vecType<T, P> const & v);
template <typename genIUType>
GLM_FUNC_DECL int findMSB(genIUType x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<int, P> findMSB(vecType<T, P> const & v);
}
#include "func_integer.inl"
