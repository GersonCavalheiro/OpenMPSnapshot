

#pragma once

#include "rawspeedconfig.h" 
#include "adt/Invariant.h"  
#include <type_traits>      
#include <vector>           

namespace rawspeed {

template <class T> class CroppedArray1DRef;

template <class T> class Array1DRef {
T* data = nullptr;
int numElts = 0;

friend Array1DRef<const T>; 

public:
using value_type = T;
using cvless_value_type = std::remove_cv_t<value_type>;

Array1DRef() = default;

Array1DRef(T* data, int numElts);

template <
typename T2,
std::enable_if_t<std::conjunction_v<std::is_const<T2>,
std::negation<std::is_const<T>>>,
bool> = true>
Array1DRef(Array1DRef<T2> RHS) = delete;

template <typename T2,
std::enable_if_t<
std::conjunction_v<
std::negation<std::conjunction<
std::is_const<T2>, std::negation<std::is_const<T>>>>,
std::negation<std::is_same<std::remove_const_t<T>,
std::remove_const_t<T2>>>>,
bool> = true>
Array1DRef(Array1DRef<T2> RHS) = delete;

template <
typename T2,
std::enable_if_t<
std::conjunction_v<
std::conjunction<std::negation<std::is_const<T2>>,
std::is_const<T>>,
std::is_same<std::remove_const_t<T>, std::remove_const_t<T2>>>,
bool> = true>
Array1DRef(Array1DRef<T2> RHS) 
: data(RHS.data), numElts(RHS.numElts) {}

[[nodiscard]] CroppedArray1DRef<T> getCrop(int offset, int numElts) const;

[[nodiscard]] int RAWSPEED_READONLY size() const;

[[nodiscard]] T& operator()(int eltIdx) const;

[[nodiscard]] T* begin() const;
[[nodiscard]] T* end() const;
};

template <typename T> Array1DRef(T* data_, int numElts_) -> Array1DRef<T>;

template <class T>
Array1DRef<T>::Array1DRef(T* data_, const int numElts_)
: data(data_), numElts(numElts_) {
invariant(data);
invariant(numElts >= 0);
}

template <class T>
[[nodiscard]] CroppedArray1DRef<T> Array1DRef<T>::getCrop(int offset,
int size) const {
invariant(offset >= 0);
invariant(size >= 0);
invariant(offset + size <= numElts);
return {*this, offset, size};
}

template <class T> inline T& Array1DRef<T>::operator()(const int eltIdx) const {
invariant(data);
invariant(eltIdx >= 0);
invariant(eltIdx < numElts);
return data[eltIdx];
}

template <class T> inline int Array1DRef<T>::size() const { return numElts; }

template <class T> inline T* Array1DRef<T>::begin() const {
return &operator()(0);
}
template <class T> inline T* Array1DRef<T>::end() const {
return &operator()(size() - 1) + 1;
}

} 
