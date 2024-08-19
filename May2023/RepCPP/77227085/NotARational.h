

#pragma once

#include <type_traits> 

namespace rawspeed {

template <class T> struct NotARational {
public:
using value_type = T;

T num;
T den;

NotARational() = default;
NotARational(T num_, T den_) : num(num_), den(den_) {}

template <typename T2,
std::enable_if_t<std::is_floating_point_v<T2>, bool> = true>
explicit operator T2() const {
return T2(num) / T2(den);
}
};

} 
