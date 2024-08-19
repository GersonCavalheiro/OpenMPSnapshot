
#ifndef GLM_GTX_euler_angles
#define GLM_GTX_euler_angles GLM_VERSION

#include "../glm.hpp"
#include "../gtc/half_float.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_euler_angles extension included")
#endif

namespace glm
{

template <typename valType> 
detail::tmat4x4<valType> eulerAngleX(
valType const & angleX);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleY(
valType const & angleY);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleZ(
valType const & angleZ);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleXY(
valType const & angleX, 
valType const & angleY);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleYX(
valType const & angleY, 
valType const & angleX);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleXZ(
valType const & angleX, 
valType const & angleZ);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleZX(
valType const & angleZ, 
valType const & angleX);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleYZ(
valType const & angleY, 
valType const & angleZ);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleZY(
valType const & angleZ, 
valType const & angleY);

template <typename valType> 
detail::tmat4x4<valType> eulerAngleYXZ(
valType const & yaw, 
valType const & pitch, 
valType const & roll);

template <typename valType> 
detail::tmat4x4<valType> yawPitchRoll(
valType const & yaw, 
valType const & pitch, 
valType const & roll);

template <typename T> 
detail::tmat2x2<T> orientate2(T const & angle);

template <typename T> 
detail::tmat3x3<T> orientate3(T const & angle);

template <typename T> 
detail::tmat3x3<T> orientate3(detail::tvec3<T> const & angles);

template <typename T> 
detail::tmat4x4<T> orientate4(detail::tvec3<T> const & angles);

}

#include "euler_angles.inl"

#endif
