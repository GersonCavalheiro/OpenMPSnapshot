

#pragma once

#include "adt/Array1DRef.h" 
#include "adt/Invariant.h"  
#include <cstddef>          
#include <type_traits>      
#include <vector>           

namespace rawspeed {

template <class T> class Array2DRef {
T* _data = nullptr;
int _pitch = 0;

friend Array2DRef<const T>; 

friend Array2DRef<std::byte>;
friend Array2DRef<const std::byte>;

public:
using value_type = T;
using cvless_value_type = std::remove_cv_t<value_type>;

int width = 0;
int height = 0;

Array2DRef() = default;

Array2DRef(T* data, int width, int height, int pitch = 0);

template <
typename T2,
std::enable_if_t<std::conjunction_v<std::is_const<T2>,
std::negation<std::is_const<T>>>,
bool> = true>
Array2DRef(Array2DRef<T2> RHS) = delete;

template <
typename T2,
std::enable_if_t<
std::conjunction_v<
std::negation<std::conjunction<std::is_const<T2>,
std::negation<std::is_const<T>>>>,
std::negation<std::is_same<std::remove_const_t<T>,
std::remove_const_t<T2>>>,
std::negation<std::is_same<std::remove_const_t<T>, std::byte>>>,
bool> = true>
Array2DRef(Array2DRef<T2> RHS) = delete;

template <
typename T2,
std::enable_if_t<
std::conjunction_v<
std::conjunction<std::negation<std::is_const<T2>>,
std::is_const<T>>,
std::is_same<std::remove_const_t<T>, std::remove_const_t<T2>>>,
bool> = true>
Array2DRef(Array2DRef<T2> RHS) 
: _data(RHS._data), _pitch(RHS._pitch), width(RHS.width),
height(RHS.height) {}

template <typename T2,
std::enable_if_t<
std::conjunction_v<
std::negation<std::conjunction<
std::is_const<T2>, std::negation<std::is_const<T>>>>,
std::negation<std::is_same<std::remove_const_t<T>,
std::remove_const_t<T2>>>,
std::is_same<std::remove_const_t<T>, std::byte>>,
bool> = true>
Array2DRef(Array2DRef<T2> RHS) 
: _data(reinterpret_cast<T*>(RHS._data)), _pitch(sizeof(T2) * RHS._pitch),
width(sizeof(T2) * RHS.width), height(RHS.height) {}

template <typename AllocatorType =
typename std::vector<cvless_value_type>::allocator_type>
static Array2DRef<T>
create(std::vector<cvless_value_type, AllocatorType>& storage, int width,
int height) {
using VectorTy = std::remove_reference_t<decltype(storage)>;
storage = VectorTy(width * height);
return {storage.data(), width, height};
}

Array1DRef<T> operator[](int row) const;

T& operator()(int row, int col) const;
};

template <class T>
Array2DRef<T>::Array2DRef(T* data, const int width_, const int height_,
const int pitch_ )
: _data(data), width(width_), height(height_) {
invariant(width >= 0);
invariant(height >= 0);
_pitch = (pitch_ == 0 ? width_ : pitch_);
}

template <class T>
inline Array1DRef<T> Array2DRef<T>::operator[](const int row) const {
invariant(_data);
invariant(row >= 0);
invariant(row < height);
return Array1DRef<T>(&_data[row * _pitch], width);
}

template <class T>
inline T& Array2DRef<T>::operator()(const int row, const int col) const {
invariant(col >= 0);
invariant(col < width);
return (operator[](row))(col);
}

} 
