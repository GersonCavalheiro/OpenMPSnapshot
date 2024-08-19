#pragma once
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"
#include "../detail/type_int.hpp"
#include "../detail/_vectorize.hpp"
#include <limits>
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_bitfield extension included")
#endif
namespace glm
{
template <typename genIUType>
GLM_FUNC_DECL genIUType mask(genIUType Bits);
template <typename T, precision P, template <typename, precision> class vecIUType>
GLM_FUNC_DECL vecIUType<T, P> mask(vecIUType<T, P> const & v);
template <typename genIUType>
GLM_FUNC_DECL genIUType bitfieldRotateRight(genIUType In, int Shift);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> bitfieldRotateRight(vecType<T, P> const & In, int Shift);
template <typename genIUType>
GLM_FUNC_DECL genIUType bitfieldRotateLeft(genIUType In, int Shift);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> bitfieldRotateLeft(vecType<T, P> const & In, int Shift);
template <typename genIUType>
GLM_FUNC_DECL genIUType bitfieldFillOne(genIUType Value, int FirstBit, int BitCount);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> bitfieldFillOne(vecType<T, P> const & Value, int FirstBit, int BitCount);
template <typename genIUType>
GLM_FUNC_DECL genIUType bitfieldFillZero(genIUType Value, int FirstBit, int BitCount);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> bitfieldFillZero(vecType<T, P> const & Value, int FirstBit, int BitCount);
GLM_FUNC_DECL int16 bitfieldInterleave(int8 x, int8 y);
GLM_FUNC_DECL uint16 bitfieldInterleave(uint8 x, uint8 y);
GLM_FUNC_DECL int32 bitfieldInterleave(int16 x, int16 y);
GLM_FUNC_DECL uint32 bitfieldInterleave(uint16 x, uint16 y);
GLM_FUNC_DECL int64 bitfieldInterleave(int32 x, int32 y);
GLM_FUNC_DECL uint64 bitfieldInterleave(uint32 x, uint32 y);
GLM_FUNC_DECL int32 bitfieldInterleave(int8 x, int8 y, int8 z);
GLM_FUNC_DECL uint32 bitfieldInterleave(uint8 x, uint8 y, uint8 z);
GLM_FUNC_DECL int64 bitfieldInterleave(int16 x, int16 y, int16 z);
GLM_FUNC_DECL uint64 bitfieldInterleave(uint16 x, uint16 y, uint16 z);
GLM_FUNC_DECL int64 bitfieldInterleave(int32 x, int32 y, int32 z);
GLM_FUNC_DECL uint64 bitfieldInterleave(uint32 x, uint32 y, uint32 z);
GLM_FUNC_DECL int32 bitfieldInterleave(int8 x, int8 y, int8 z, int8 w);
GLM_FUNC_DECL uint32 bitfieldInterleave(uint8 x, uint8 y, uint8 z, uint8 w);
GLM_FUNC_DECL int64 bitfieldInterleave(int16 x, int16 y, int16 z, int16 w);
GLM_FUNC_DECL uint64 bitfieldInterleave(uint16 x, uint16 y, uint16 z, uint16 w);
} 
#include "bitfield.inl"
