
#ifndef GLM_GTX_associated_min_max
#define GLM_GTX_associated_min_max GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_associated_min_max extension included")
#endif

namespace glm
{

template<typename genTypeT, typename genTypeU>
genTypeU associatedMin(
const genTypeT& x, const genTypeU& a, 
const genTypeT& y, const genTypeU& b);

template<typename genTypeT, typename genTypeU>
genTypeU associatedMin(
const genTypeT& x, const genTypeU& a, 
const genTypeT& y, const genTypeU& b, 
const genTypeT& z, const genTypeU& c);

template<typename genTypeT, typename genTypeU>
genTypeU associatedMin(
const genTypeT& x, const genTypeU& a, 
const genTypeT& y, const genTypeU& b, 
const genTypeT& z, const genTypeU& c, 
const genTypeT& w, const genTypeU& d);

template<typename genTypeT, typename genTypeU>
genTypeU associatedMax(
const genTypeT& x, const genTypeU& a, 
const genTypeT& y, const genTypeU& b);

template<typename genTypeT, typename genTypeU>
genTypeU associatedMax(
const genTypeT& x, const genTypeU& a, 
const genTypeT& y, const genTypeU& b, 
const genTypeT& z, const genTypeU& c);

template<typename genTypeT, typename genTypeU>
genTypeU associatedMax(
const genTypeT& x, const genTypeU& a, 
const genTypeT& y, const genTypeU& b, 
const genTypeT& z, const genTypeU& c, 
const genTypeT& w, const genTypeU& d);

} 

#include "associated_min_max.inl"

#endif
