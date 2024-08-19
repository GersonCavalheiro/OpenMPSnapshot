
namespace glm
{
template <typename T, precision P>
GLM_FUNC_QUALIFIER detail::tmat4x4<T, P> rotateNormalizedAxis
(
detail::tmat4x4<T, P> const & m,
T const & angle,
detail::tvec3<T, P> const & v
)
{
#ifdef GLM_FORCE_RADIANS
T a = angle;
#else
#		pragma message("GLM: rotateNormalizedAxis function taking degrees as parameters is deprecated. #define GLM_FORCE_RADIANS before including GLM headers to remove this message.")
T a = radians(angle);
#endif
T c = cos(a);
T s = sin(a);

detail::tvec3<T, P> axis = v;

detail::tvec3<T, P> temp = (T(1) - c) * axis;

detail::tmat4x4<T, P> Rotate(detail::tmat4x4<T, P>::_null);
Rotate[0][0] = c + temp[0] * axis[0];
Rotate[0][1] = 0 + temp[0] * axis[1] + s * axis[2];
Rotate[0][2] = 0 + temp[0] * axis[2] - s * axis[1];

Rotate[1][0] = 0 + temp[1] * axis[0] - s * axis[2];
Rotate[1][1] = c + temp[1] * axis[1];
Rotate[1][2] = 0 + temp[1] * axis[2] + s * axis[0];

Rotate[2][0] = 0 + temp[2] * axis[0] + s * axis[1];
Rotate[2][1] = 0 + temp[2] * axis[1] - s * axis[0];
Rotate[2][2] = c + temp[2] * axis[2];

detail::tmat4x4<T, P> Result(detail::tmat4x4<T, P>::_null);
Result[0] = m[0] * Rotate[0][0] + m[1] * Rotate[0][1] + m[2] * Rotate[0][2];
Result[1] = m[0] * Rotate[1][0] + m[1] * Rotate[1][1] + m[2] * Rotate[1][2];
Result[2] = m[0] * Rotate[2][0] + m[1] * Rotate[2][1] + m[2] * Rotate[2][2];
Result[3] = m[3];
return Result;
}

template <typename T, precision P>
GLM_FUNC_QUALIFIER detail::tquat<T, P> rotateNormalizedAxis
(
detail::tquat<T, P> const & q, 
T const & angle,
detail::tvec3<T, P> const & v
)
{
detail::tvec3<T, P> Tmp = v;

#ifdef GLM_FORCE_RADIANS
T const AngleRad(angle);
#else
#		pragma message("GLM: rotateNormalizedAxis function taking degrees as parameters is deprecated. #define GLM_FORCE_RADIANS before including GLM headers to remove this message.")
T const AngleRad = radians(angle);
#endif
T const Sin = sin(AngleRad * T(0.5));

return q * detail::tquat<T, P>(cos(AngleRad * static_cast<T>(0.5)), Tmp.x * Sin, Tmp.y * Sin, Tmp.z * Sin);
}
}
