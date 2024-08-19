
#pragma once

#include "../detail/type_mat3x3.hpp"
#include "../detail/type_mat4x4.hpp"
#include "../detail/type_vec3.hpp"
#include "../detail/type_vec4.hpp"
#include "../ext/vector_relational.hpp"
#include "../ext/quaternion_relational.hpp"
#include "../gtc/constants.hpp"
#include "../gtc/matrix_transform.hpp"

namespace glm
{
template<typename T, qualifier Q>
struct qua
{

typedef qua<T, Q> type;
typedef T value_type;


#		if GLM_SILENT_WARNINGS == GLM_ENABLE
#			if GLM_COMPILER & GLM_COMPILER_GCC
#				pragma GCC diagnostic push
#				pragma GCC diagnostic ignored "-Wpedantic"
#			elif GLM_COMPILER & GLM_COMPILER_CLANG
#				pragma clang diagnostic push
#				pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#				pragma clang diagnostic ignored "-Wnested-anon-types"
#			elif GLM_COMPILER & GLM_COMPILER_VC
#				pragma warning(push)
#				pragma warning(disable: 4201)  
#			endif
#		endif

#		if GLM_LANG & GLM_LANG_CXXMS_FLAG
union
{
#				ifdef GLM_FORCE_QUAT_DATA_WXYZ
struct { T w, x, y, z; };
#				else
struct { T x, y, z, w; };
#				endif

typename detail::storage<4, T, detail::is_aligned<Q>::value>::type data;
};
#		else
#			ifdef GLM_FORCE_QUAT_DATA_WXYZ
T w, x, y, z;
#			else
T x, y, z, w;
#			endif
#		endif

#		if GLM_SILENT_WARNINGS == GLM_ENABLE
#			if GLM_COMPILER & GLM_COMPILER_CLANG
#				pragma clang diagnostic pop
#			elif GLM_COMPILER & GLM_COMPILER_GCC
#				pragma GCC diagnostic pop
#			elif GLM_COMPILER & GLM_COMPILER_VC
#				pragma warning(pop)
#			endif
#		endif


typedef length_t length_type;

GLM_FUNC_DECL static GLM_CONSTEXPR length_type length(){return 4;}

GLM_FUNC_DECL GLM_CONSTEXPR T & operator[](length_type i);
GLM_FUNC_DECL GLM_CONSTEXPR T const& operator[](length_type i) const;


GLM_FUNC_DECL GLM_CONSTEXPR qua() GLM_DEFAULT;
GLM_FUNC_DECL GLM_CONSTEXPR qua(qua<T, Q> const& q) GLM_DEFAULT;
template<qualifier P>
GLM_FUNC_DECL GLM_CONSTEXPR qua(qua<T, P> const& q);


GLM_FUNC_DECL GLM_CONSTEXPR qua(T s, vec<3, T, Q> const& v);
GLM_FUNC_DECL GLM_CONSTEXPR qua(T w, T x, T y, T z);


template<typename U, qualifier P>
GLM_FUNC_DECL GLM_CONSTEXPR GLM_EXPLICIT qua(qua<U, P> const& q);

#		if GLM_HAS_EXPLICIT_CONVERSION_OPERATORS
GLM_FUNC_DECL explicit operator mat<3, 3, T, Q>() const;
GLM_FUNC_DECL explicit operator mat<4, 4, T, Q>() const;
#		endif

GLM_FUNC_DECL qua(vec<3, T, Q> const& u, vec<3, T, Q> const& v);

GLM_FUNC_DECL GLM_CONSTEXPR GLM_EXPLICIT qua(vec<3, T, Q> const& eulerAngles);
GLM_FUNC_DECL GLM_EXPLICIT qua(mat<3, 3, T, Q> const& q);
GLM_FUNC_DECL GLM_EXPLICIT qua(mat<4, 4, T, Q> const& q);


GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q>& operator=(qua<T, Q> const& q) GLM_DEFAULT;

template<typename U>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q>& operator=(qua<U, Q> const& q);
template<typename U>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q>& operator+=(qua<U, Q> const& q);
template<typename U>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q>& operator-=(qua<U, Q> const& q);
template<typename U>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q>& operator*=(qua<U, Q> const& q);
template<typename U>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q>& operator*=(U s);
template<typename U>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q>& operator/=(U s);
};


template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator+(qua<T, Q> const& q);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator-(qua<T, Q> const& q);


template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator+(qua<T, Q> const& q, qua<T, Q> const& p);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator-(qua<T, Q> const& q, qua<T, Q> const& p);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator*(qua<T, Q> const& q, qua<T, Q> const& p);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR vec<3, T, Q> operator*(qua<T, Q> const& q, vec<3, T, Q> const& v);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR vec<3, T, Q> operator*(vec<3, T, Q> const& v, qua<T, Q> const& q);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR vec<4, T, Q> operator*(qua<T, Q> const& q, vec<4, T, Q> const& v);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR vec<4, T, Q> operator*(vec<4, T, Q> const& v, qua<T, Q> const& q);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator*(qua<T, Q> const& q, T const& s);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator*(T const& s, qua<T, Q> const& q);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> operator/(qua<T, Q> const& q, T const& s);


template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR bool operator==(qua<T, Q> const& q1, qua<T, Q> const& q2);

template<typename T, qualifier Q>
GLM_FUNC_DECL GLM_CONSTEXPR bool operator!=(qua<T, Q> const& q1, qua<T, Q> const& q2);
} 

#ifndef GLM_EXTERNAL_TEMPLATE
#include "type_quat.inl"
#endif
