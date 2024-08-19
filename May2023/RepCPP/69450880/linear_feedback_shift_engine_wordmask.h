

#pragma once

namespace hydra_thrust
{

namespace random
{

namespace detail
{

template<typename T, int w, int i = w-1>
struct linear_feedback_shift_engine_wordmask
{
static const T value =
(T(1u) << i) |
linear_feedback_shift_engine_wordmask<T, w, i-1>::value;
}; 

template<typename T, int w>
struct linear_feedback_shift_engine_wordmask<T, w, 0>
{
static const T value = 0;
}; 

} 

} 

} 

