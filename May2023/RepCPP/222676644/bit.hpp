
#ifndef GLM_GTX_bit
#define GLM_GTX_bit GLM_VERSION

#include "../glm.hpp"
#include "../gtc/half_float.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_bit extension included")
#endif

namespace glm
{

template <typename genIType>
genIType mask(genIType const & count);

template <typename genIUType, typename sizeType>
GLM_DEPRECATED genIUType extractField(
genIUType const & v, 
sizeType const & first, 
sizeType const & count);

template <typename genType> 
GLM_DEPRECATED int lowestBit(genType const & value);

template <typename genType> 
GLM_DEPRECATED int highestBit(genType const & value);

template <typename genType> 
genType highestBitValue(genType const & value);

template <typename genType> 
bool isPowerOfTwo(genType const & value);

template <typename genType> 
genType powerOfTwoAbove(genType const & value);

template <typename genType> 
genType powerOfTwoBelow(genType const & value);

template <typename genType> 
genType powerOfTwoNearest(genType const & value);

template <typename genType> 
GLM_DEPRECATED genType bitRevert(genType const & value);

template <typename genType>
genType bitRotateRight(genType const & In, std::size_t Shift);

template <typename genType>
genType bitRotateLeft(genType const & In, std::size_t Shift);

template <typename genIUType>
genIUType fillBitfieldWithOne(
genIUType const & Value,
int const & FromBit, 
int const & ToBit);

template <typename genIUType>
genIUType fillBitfieldWithZero(
genIUType const & Value,
int const & FromBit, 
int const & ToBit);

} 

#include "bit.inl"

#endif
