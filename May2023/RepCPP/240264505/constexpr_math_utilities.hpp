

#ifndef LBT_CEM_UTILITIES
#define LBT_CEM_UTILITIES
#pragma once

#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>


namespace lbt {
namespace test {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
auto generateRandomNumber(T const lower_bound, T const upper_bound) noexcept {
std::uniform_real_distribution<T> uniform_distribution {lower_bound, upper_bound};
std::random_device random_device {};
std::default_random_engine random_engine {random_device()};
T const random_number {uniform_distribution(random_engine)};
return random_number;
}


template <typename T, typename std::enable_if_t<std::is_integral_v<T>>* = nullptr>
auto generateRandomNumber(T const lower_bound, T const upper_bound) {
if (lower_bound > upper_bound) {
throw std::invalid_argument("Range invalid: lower_bound > upper_bound (" + std::to_string(lower_bound) + " > " + std::to_string(upper_bound) + ")!");
}
std::uniform_int_distribution<T> uniform_distribution {lower_bound, upper_bound};
std::random_device random_device {};
std::default_random_engine random_engine {random_device()};
T const random_number {uniform_distribution(random_engine)};
return random_number;
}

}
}

#endif 
