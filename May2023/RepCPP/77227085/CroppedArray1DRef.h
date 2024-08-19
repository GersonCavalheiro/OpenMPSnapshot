

#pragma once

#include "rawspeedconfig.h" 
#include "adt/Array1DRef.h" 
#include "adt/Invariant.h"  
#include <type_traits>      
#include <vector>           

namespace rawspeed {

template <class T> class CroppedArray1DRef {
const Array1DRef<T> base;
int offset = 0;
int numElts = 0;

friend CroppedArray1DRef<const T>; 

friend CroppedArray1DRef<T> Array1DRef<T>::getCrop(int offset,
int numElts) const;
CroppedArray1DRef(Array1DRef<T> base, int offset, int numElts);

public:
using value_type = T;
using cvless_value_type = std::remove_cv_t<value_type>;

CroppedArray1DRef() = default;

template <
typename T2,
std::enable_if_t<std::conjunction_v<std::is_const<T2>,
std::negation<std::is_const<T>>>,
bool> = true>
CroppedArray1DRef(CroppedArray1DRef<T2> RHS) = delete;

template <typename T2,
std::enable_if_t<
std::conjunction_v<
std::negation<std::conjunction<
std::is_const<T2>, std::negation<std::is_const<T>>>>,
std::negation<std::is_same<std::remove_const_t<T>,
std::remove_const_t<T2>>>>,
bool> = true>
CroppedArray1DRef(CroppedArray1DRef<T2> RHS) = delete;

template <
typename T2,
std::enable_if_t<
std::conjunction_v<
std::conjunction<std::negation<std::is_const<T2>>,
std::is_const<T>>,
std::is_same<std::remove_const_t<T>, std::remove_const_t<T2>>>,
bool> = true>
CroppedArray1DRef( 
CroppedArray1DRef<T2> RHS)
: base(RHS.base), numElts(RHS.numElts) {}

[[nodiscard]] const T* begin() const;

[[nodiscard]] int RAWSPEED_READONLY size() const;

[[nodiscard]] T& operator()(int eltIdx) const;
};

template <typename T>
CroppedArray1DRef(Array1DRef<T> base, int offset, int numElts)
-> CroppedArray1DRef<T>;

template <class T>
CroppedArray1DRef<T>::CroppedArray1DRef(Array1DRef<T> base_, const int offset_,
const int numElts_)
: base(base_), offset(offset_), numElts(numElts_) {
invariant(offset >= 0);
invariant(numElts >= 0);
invariant(offset + numElts <= base.size());
}

template <class T> inline const T* CroppedArray1DRef<T>::begin() const {
return &operator()(0);
}

template <class T> inline int CroppedArray1DRef<T>::size() const {
return numElts;
}

template <class T>
inline T& CroppedArray1DRef<T>::operator()(const int eltIdx) const {
invariant(eltIdx >= 0);
invariant(eltIdx < numElts);
return base(offset + eltIdx);
}

} 
