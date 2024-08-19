#pragma once
#include "type_vec3.hpp"
namespace glm
{
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T length(
vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T distance(
vecType<T, P> const & p0,
vecType<T, P> const & p1);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL T dot(
vecType<T, P> const & x,
vecType<T, P> const & y);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> cross(
tvec3<T, P> const & x,
tvec3<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> normalize(
vecType<T, P> const & x);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> faceforward(
vecType<T, P> const & N,
vecType<T, P> const & I,
vecType<T, P> const & Nref);
template <typename genType>
GLM_FUNC_DECL genType reflect(
genType const & I,
genType const & N);
template <typename T, precision P, template <typename, precision> class vecType>
GLM_FUNC_DECL vecType<T, P> refract(
vecType<T, P> const & I,
vecType<T, P> const & N,
T eta);
}
#include "func_geometric.inl"
