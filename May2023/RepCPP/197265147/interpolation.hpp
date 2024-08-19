
#pragma once
#include <array>
#include <cmath>
#include <utility>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace interpolation
{

template <typename Func, int Dim>
struct Data
{

std::array<Data<Func, Dim - 1>, Func::NUMBER_OF_VALUES> data;


template <typename... Indices>
typename Func::VALUE_TYPE operator()(size_t index_0,
Indices... indices) const
{
return data[index_0](indices...);
}


template <typename... Indices>
typename Func::VALUE_TYPE& operator()(size_t index_0, Indices... indices)
{
return data[index_0](indices...);
}
};


template <typename Func>
struct Data<Func, 1>
{

std::array<typename Func::VALUE_TYPE, Func::NUMBER_OF_VALUES> data;


typename Func::VALUE_TYPE operator()(size_t index) const
{
return data[index];
}


typename Func::VALUE_TYPE& operator()(size_t index) { return data[index]; }
};


template <typename Func>
struct Data<Func, 0>;


template <typename T>
struct linear
{

typedef T VALUE_TYPE;

static constexpr int NUMBER_OF_VALUES = 2;


T operator()(const Data<linear<T>, 1>& p, double x)
{
return static_cast<T>(p(0) * (1 - x) + p(1) * x);
}
};


template <typename T>
struct cubic
{

typedef T VALUE_TYPE;

static constexpr int NUMBER_OF_VALUES = 4;


T operator()(const Data<cubic<T>, 1>& p, double x)
{
return static_cast<T>(
p(1) + 0.5 * x *
(p(2) - p(0) +
x * (2.0 * p(0) - 5.0 * p(1) + 4.0 * p(2) - p(3) +
x * (3.0 * (p(1) - p(2)) + p(3) - p(0)))));
}
};



namespace detail
{

template <typename T, int Dim, std::size_t... I>
T get_image_value_impl(
const pybind11::detail::unchecked_reference<T, Dim>& image,
const std::array<int, Dim>& array_indices, std::index_sequence<I...>)
{
return image(array_indices[I]...);
}


template <typename T, int Dim>
T get_image_value(const pybind11::detail::unchecked_reference<T, Dim>& image,
const std::array<int, Dim>& array_indices)
{
return get_image_value_impl<T, Dim>(image, array_indices,
std::make_index_sequence<Dim>{});
}


template <typename Func, typename... Ts, std::size_t... I>
static typename Func::VALUE_TYPE
apply_func_impl(std::index_sequence<I...> indices,
const Data<Func, sizeof...(Ts) + 1>& chunk, double x, Ts... xs)
{
if constexpr (sizeof...(Ts) > 0)
{
return apply_func_impl<Func>(
indices,
{{apply_func_impl(indices, std::get<I>(chunk.data), xs...)...}}, x);
}
else
{
return Func()(chunk, x);
}
}


template <typename Func, typename BoundaryFunc, int Dim, typename... Ts>
static void extract_impl(
Data<Func, Dim>& chunk,
const pybind11::detail::unchecked_reference<typename Func::VALUE_TYPE, Dim>&
image,
const std::array<int, Dim>& lower_corner,
typename Func::VALUE_TYPE background_value, Ts... loop_indices)
{
for (int i = 0; i < Func::NUMBER_OF_VALUES; ++i)
{
if constexpr (Dim == sizeof...(Ts) + 1)
{
std::array<int, Dim> voxel_position{{loop_indices..., i}};

for (int l = 0; l < Dim; ++l)
{
voxel_position[l] += lower_corner[l];
}
chunk(loop_indices..., i) =
BoundaryFunc::template apply<typename Func::VALUE_TYPE, Dim>(image, voxel_position, background_value);
}
else
{
extract_impl<Func, BoundaryFunc, Dim>(chunk, image, lower_corner,
background_value,
loop_indices..., i);
}
}
}
}; 


struct ConstantBoundary
{

template <typename T, int Dim>
static T apply(const pybind11::detail::unchecked_reference<T, Dim>& image,
const std::array<int, Dim>& voxel_position,
T background_value)
{
for (int l = 0; l < Dim; ++l)
{
if (voxel_position[l] < 0 || voxel_position[l] >= image.shape(l))
{
return background_value;
}
}
return detail::get_image_value<T, Dim>(image, voxel_position);
}
};


template <typename Func, typename... Ts>
static typename Func::VALUE_TYPE
apply_func(const Data<Func, sizeof...(Ts) + 1>& chunk, double x, Ts... xs)
{
return detail::apply_func_impl(
std::make_index_sequence<Func::NUMBER_OF_VALUES>{}, chunk, x, xs...);
}


template <typename Func, typename BoundaryFunc, int Dim>
static void
extract(Data<Func, Dim>& chunk,
const pybind11::detail::unchecked_reference<typename Func::VALUE_TYPE,
Dim>& image,
std::array<int, Dim>& point_floored,
typename Func::VALUE_TYPE background_value)
{
for (size_t l = 0; l < Dim; ++l)
{
point_floored[l] -= (Func::NUMBER_OF_VALUES - 2) / 2;
}

detail::extract_impl<Func, BoundaryFunc, Dim>(chunk, image, point_floored,
background_value);
}

}; 
