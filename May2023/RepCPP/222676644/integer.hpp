
#ifndef GLM_GTX_integer
#define GLM_GTX_integer GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_integer extension included")
#endif

namespace glm
{

int pow(int x, int y);

int sqrt(int x);

template <typename genIUType>
genIUType log2(genIUType const & x);

unsigned int floor_log2(unsigned int x);

int mod(int x, int y);

template <typename genType> 
genType factorial(genType const & x);

typedef signed int					sint;

uint pow(uint x, uint y);

uint sqrt(uint x);

uint mod(uint x, uint y);

uint nlz(uint x);

}

#include "integer.inl"

#endif
