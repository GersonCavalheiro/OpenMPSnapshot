
#ifndef GLM_GTX_vector_query
#define GLM_GTX_vector_query

#include "../glm.hpp"
#include <cfloat>
#include <limits>

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_vector_query extension included")
#endif

namespace glm
{

template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL bool areCollinear(vecType<T, P> const & v0, vecType<T, P> const & v1, T const & epsilon);

template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL bool areOrthogonal(vecType<T, P> const & v0, vecType<T, P> const & v1, T const & epsilon);

template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL bool isNormalized(vecType<T, P> const & v, T const & epsilon);

template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL bool isNull(vecType<T, P> const & v, T const & epsilon);

template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<bool, P> isCompNull(vecType<T, P> const & v, T const & epsilon);

template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL bool areOrthonormal(vecType<T, P> const & v0, vecType<T, P> const & v1, T const & epsilon);

}

#include "vector_query.inl"

#endif
