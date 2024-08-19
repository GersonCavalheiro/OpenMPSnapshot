
#ifndef GLM_GTX_bit
#define GLM_GTX_bit

#include "../detail/type_int.hpp"
#include "../detail/setup.hpp"
#include <cstddef>

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_bit extension included")
#endif

namespace glm
{

template <typename genIType>
GLM_FUNC_DECL genIType mask(genIType const & count);

template <typename genType> 
GLM_FUNC_DECL genType highestBitValue(genType const & value);

template <typename genType> 
GLM_FUNC_DECL bool isPowerOfTwo(genType const & value);

template <typename genType> 
GLM_FUNC_DECL genType powerOfTwoAbove(genType const & value);

template <typename genType> 
GLM_FUNC_DECL genType powerOfTwoBelow(genType const & value);

template <typename genType> 
GLM_FUNC_DECL genType powerOfTwoNearest(genType const & value);

template <typename genType> 
GLM_DEPRECATED GLM_FUNC_DECL genType bitRevert(genType const & value);

template <typename genType>
GLM_FUNC_DECL genType bitRotateRight(genType const & In, std::size_t Shift);

template <typename genType>
GLM_FUNC_DECL genType bitRotateLeft(genType const & In, std::size_t Shift);

template <typename genIUType>
GLM_FUNC_DECL genIUType fillBitfieldWithOne(
genIUType const & Value,
int const & FromBit, 
int const & ToBit);

template <typename genIUType>
GLM_FUNC_DECL genIUType fillBitfieldWithZero(
genIUType const & Value,
int const & FromBit, 
int const & ToBit);

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

#include "bit.inl"

#endif
