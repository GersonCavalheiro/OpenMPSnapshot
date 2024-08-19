
#ifndef GLM_GTX_matrix_query
#define GLM_GTX_matrix_query

#include "../glm.hpp"
#include "../gtx/vector_query.hpp"
#include <limits>

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_matrix_query extension included")
#endif

namespace glm
{

template<typename T, precision P>
GLM_FUNC_DECL bool isNull(detail::tmat2x2<T, P> const & m, T const & epsilon);

template<typename T, precision P>
GLM_FUNC_DECL bool isNull(detail::tmat3x3<T, P> const & m, T const & epsilon);

template<typename T, precision P>
GLM_FUNC_DECL bool isNull(detail::tmat4x4<T, P> const & m, T const & epsilon);

template<typename T, precision P, template <typename, precision> class matType>
GLM_FUNC_DECL bool isIdentity(matType<T, P> const & m, T const & epsilon);

template<typename T, precision P>
GLM_FUNC_DECL bool isNormalized(detail::tmat2x2<T, P> const & m, T const & epsilon);

template<typename T, precision P>
GLM_FUNC_DECL bool isNormalized(detail::tmat3x3<T, P> const & m, T const & epsilon);

template<typename T, precision P>
GLM_FUNC_DECL bool isNormalized(detail::tmat4x4<T, P> const & m, T const & epsilon);

template<typename T, precision P, template <typename, precision> class matType>
GLM_FUNC_DECL bool isOrthogonal(matType<T, P> const & m, T const & epsilon);

}

#include "matrix_query.inl"

#endif
