#pragma once
#include "../glm.hpp"
#include "../gtc/integer.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_integer extension included")
#endif
namespace glm
{
GLM_FUNC_DECL int pow(int x, int y);
GLM_FUNC_DECL int sqrt(int x);
GLM_FUNC_DECL unsigned int floor_log2(unsigned int x);
GLM_FUNC_DECL int mod(int x, int y);
template <typename genType> 
GLM_FUNC_DECL genType factorial(genType const & x);
typedef signed int					sint;
GLM_FUNC_DECL uint pow(uint x, uint y);
GLM_FUNC_DECL uint sqrt(uint x);
GLM_FUNC_DECL uint mod(uint x, uint y);
GLM_FUNC_DECL uint nlz(uint x);
}
#include "integer.inl"
