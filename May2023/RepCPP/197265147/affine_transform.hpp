
#pragma once
#include <array>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "interpolation.hpp"


namespace affine_transform
{

namespace detail
{

template <int Dim, typename T, typename Func, typename BoundaryFunc,
typename... Xs>
constexpr void transform_loop(
Eigen::Matrix<double, Dim, 1> point,
const pybind11::detail::unchecked_reference<T, Dim>& input_image,
pybind11::detail::unchecked_mutable_reference<T, Dim>& output_image,
const std::array<Eigen::Matrix<double, Dim, 1>, Dim>& dx,
interpolation::Data<Func, Dim>& chunk, T background_value, int begin,
int end, Xs... xs)
{
for (int i = begin; i < end; ++i)
{

if constexpr (Dim == sizeof...(xs) + 1)
{
auto x_lower = std::array<int, Dim>{};
auto x_relative = std::array<double, Dim>{};
for (size_t l = 0; l < Dim; ++l)
{
x_lower[l] = point(l) - (point(l) < 0);
x_relative[l] = point(l) - x_lower[l];
}

interpolation::extract<Func, BoundaryFunc, Dim>(
chunk, input_image, x_lower, background_value);

auto interpolate = [&chunk](auto... args) {
return interpolation::apply_func(chunk, args...);
};

output_image(xs..., i) = std::apply(interpolate, x_relative);
}

else
{
transform_loop<Dim, T, Func, BoundaryFunc>(
point, input_image, output_image, dx, chunk, background_value,
0, output_image.shape(sizeof...(xs) + 1), xs..., i);
}
point += dx[sizeof...(xs)];
}
}
}; 


template <int Dim, typename T, template <typename> typename Func,
typename BoundaryFunc>
void transform(const Eigen::Matrix<double, Dim, 1>& origin,
const std::array<Eigen::Matrix<double, Dim, 1>, Dim>& dx,
const pybind11::array_t<T>& input_image,
pybind11::array_t<T>& output_image, T background_value)
{
auto input = input_image.template unchecked<Dim>();
auto output = output_image.template mutable_unchecked<Dim>();

#pragma omp parallel
{
typedef Func<T> _Func;

interpolation::Data<_Func, Dim> chunk;

int x_len = output.shape(0) / omp_get_num_threads();
int x_start = x_len * omp_get_thread_num();
int x_end = x_start + x_len;

Eigen::Matrix<double, Dim, 1> local_origin = origin + x_start * dx[0];
if (omp_get_thread_num() == omp_get_num_threads() - 1)
{
x_end = output.shape(0);
}
detail::transform_loop<Dim, T, _Func, BoundaryFunc>(
local_origin, input, output, dx, chunk, background_value, x_start,
x_end);
}
}

}; 