#pragma once
#include "../glm.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_euler_angles extension included")
#endif
namespace glm
{
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleX(
T const & angleX);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleY(
T const & angleY);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleZ(
T const & angleZ);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleXY(
T const & angleX,
T const & angleY);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleYX(
T const & angleY,
T const & angleX);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleXZ(
T const & angleX,
T const & angleZ);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleZX(
T const & angle,
T const & angleX);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleYZ(
T const & angleY,
T const & angleZ);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleZY(
T const & angleZ,
T const & angleY);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleXYZ(
T const & t1,
T const & t2,
T const & t3);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> eulerAngleYXZ(
T const & yaw,
T const & pitch,
T const & roll);
template <typename T>
GLM_FUNC_DECL tmat4x4<T, defaultp> yawPitchRoll(
T const & yaw,
T const & pitch,
T const & roll);
template <typename T>
GLM_FUNC_DECL tmat2x2<T, defaultp> orientate2(T const & angle);
template <typename T>
GLM_FUNC_DECL tmat3x3<T, defaultp> orientate3(T const & angle);
template <typename T, precision P>
GLM_FUNC_DECL tmat3x3<T, P> orientate3(tvec3<T, P> const & angles);
template <typename T, precision P>
GLM_FUNC_DECL tmat4x4<T, P> orientate4(tvec3<T, P> const & angles);
template <typename T>
GLM_FUNC_DECL void extractEulerAngleXYZ(tmat4x4<T, defaultp> & M,
T & t1,
T & t2,
T & t3);
}
#include "euler_angles.inl"
