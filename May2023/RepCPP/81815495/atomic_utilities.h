
#pragma once

#include <atomic>

#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#endif

#if !defined(__cpp_lib_atomic_ref) && defined(KRATOS_SMP_CXX11)
#include <boost/atomic/atomic_ref.hpp>
#endif

#include "includes/define.h"
#include "containers/array_1d.h"

namespace Kratos {

#if defined(KRATOS_SMP_CXX11)
#if defined(__cpp_lib_atomic_ref) 
template <class T>
using AtomicRef = std::atomic_ref<T>;
#else
template <class T>
using AtomicRef = boost::atomic_ref<T>;
#endif 
#endif 




template<class TDataType>
inline void AtomicAdd(TDataType& target, const TDataType& value)
{
#ifdef KRATOS_SMP_OPENMP
#pragma omp atomic
target += value;
#elif defined(KRATOS_SMP_CXX11)
AtomicRef<TDataType>{target} += value;
#else
target += value;
#endif
}


template <class TDataType, std::size_t ArraySize>
inline void AtomicAdd(array_1d<TDataType,ArraySize>& target, const array_1d<TDataType,ArraySize>& value)
{
for (std::size_t i=0; i<ArraySize; ++i) {
AtomicAdd(target[i], value[i]);
}
}


template<class TVectorType1, class TVectorType2>
inline void AtomicAddVector(TVectorType1& target, const TVectorType2& value)
{
KRATOS_DEBUG_ERROR_IF(target.size() != value.size()) << "vector size mismatch in vector AtomicAddVector- Sizes are: " << target.size() << " for target and " << value.size() << " for value " << std::endl;

for(std::size_t i=0; i<target.size(); ++i) {
AtomicAdd(target[i], value[i]);
}
}


template<class TMatrixType1, class TMatrixType2>
inline void AtomicAddMatrix(TMatrixType1& target, const TMatrixType2& value)
{
KRATOS_DEBUG_ERROR_IF(target.size1() != value.size1() || target.size2() != value.size2()) << "matrix size mismatch in matrix AtomicAddMatrix- Sizes are: " << target.size1() << "x" << target.size2() << " for target and " << value.size1() << "x" << value.size2() << " for value " << std::endl;

for(std::size_t i=0; i<target.size1(); ++i) {
for(std::size_t j=0; j<target.size2(); ++j) {
AtomicAdd(target(i,j), value(i,j));
}
}
}


template<class TDataType>
inline void AtomicSub(TDataType& target, const TDataType& value)
{
#ifdef KRATOS_SMP_OPENMP
#pragma omp atomic
target -= value;
#elif defined(KRATOS_SMP_CXX11)
AtomicRef<TDataType>{target} -= value;
#else
target -= value;
#endif
}


template <class TDataType, std::size_t ArraySize>
inline void AtomicSub(array_1d<TDataType,ArraySize>& target, const array_1d<TDataType,ArraySize>& value)
{
for(std::size_t i=0; i<ArraySize; ++i) {
AtomicSub(target[i], value[i]);
}
}


template<class TVectorType1, class TVectorType2>
inline void AtomicSubVector(TVectorType1& target, const TVectorType2& value) {
KRATOS_DEBUG_ERROR_IF(target.size() != value.size()) << "vector size mismatch in vector AtomicSubVector- Sizes are: " << target.size() << " for target and " << value.size() << " for value " << std::endl;

for(std::size_t i=0; i<target.size(); ++i) {
AtomicSub(target[i], value[i]);
}
}


template<class TMatrixType1, class TMatrixType2>
inline void AtomicSubMatrix(TMatrixType1& target, const TMatrixType2& value)
{
KRATOS_DEBUG_ERROR_IF(target.size1() != value.size1() || target.size2() != value.size2()) << "matrix size mismatch in matrix AtomicSubMatrix- Sizes are: " << target.size1() << "x" << target.size2() << " for target and " << value.size1() << "x" << value.size2() << " for value " << std::endl;

for(std::size_t i=0; i<target.size1(); ++i) {
for(std::size_t j=0; j<target.size2(); ++j) {
AtomicSub(target(i,j), value(i,j));
}
}
}


template<class TDataType>
inline void AtomicMult(TDataType& target, const TDataType& value)
{
#ifdef KRATOS_SMP_OPENMP
#pragma omp atomic
target *= value;
#elif defined(KRATOS_SMP_CXX11)
AtomicRef<TDataType> at_ref{target};
at_ref = at_ref*value;
#else
target *= value;
#endif
}


template <class TDataType, std::size_t ArraySize>
inline void AtomicMult(array_1d<TDataType,ArraySize>& target, const array_1d<TDataType,ArraySize>& value)
{
for(std::size_t i=0; i<ArraySize; ++i) {
AtomicMult(target[i], value[i]);
}
}


template<class TVectorType1, class TVectorType2>
inline void AtomicMultVector(TVectorType1& target, const TVectorType2& value)
{
KRATOS_DEBUG_ERROR_IF(target.size() != value.size()) << "vector size mismatch in vector AtomicMultVector- Sizes are: " << target.size() << " for target and " << value.size() << " for value " << std::endl;

for(std::size_t i=0; i<target.size(); ++i) {
AtomicMult(target[i], value[i]);
}
}


template<class TMatrixType1, class TMatrixType2>
inline void AtomicMultMatrix(TMatrixType1& target, const TMatrixType2& value)
{
KRATOS_DEBUG_ERROR_IF(target.size1() != value.size1() || target.size2() != value.size2()) << "matrix size mismatch in matrix AtomicMultMatrix- Sizes are: " << target.size1() << "x" << target.size2() << " for target and " << value.size1() << "x" << value.size2() << " for value " << std::endl;

for(std::size_t i=0; i<target.size1(); ++i) {
for(std::size_t j=0; j<target.size2(); ++j) {
AtomicMult(target(i,j), value(i,j));
}
}
}


template<class TDataType>
inline void AtomicDiv(TDataType& target, const TDataType& value)
{
AtomicMult(target, 1.0/value);
}


template <class TDataType, std::size_t ArraySize>
inline void AtomicDiv(array_1d<TDataType,ArraySize>& target, const array_1d<TDataType,ArraySize>& value)
{
for(std::size_t i=0; i<ArraySize; ++i) {
AtomicDiv(target[i], value[i]);
}
}


template<class TVectorType1, class TVectorType2>
inline void AtomicDivVector(TVectorType1& target, const TVectorType2& value)
{
KRATOS_DEBUG_ERROR_IF(target.size() != value.size()) << "vector size mismatch in vector AtomicDivVector- Sizes are: " << target.size() << " for target and " << value.size() << " for value " << std::endl;

for(std::size_t i=0; i<target.size(); ++i) {
AtomicDiv(target[i], value[i]);
}
}


template<class TMatrixType1, class TMatrixType2>
inline void AtomicDivMatrix(TMatrixType1& target, const TMatrixType2& value)
{
KRATOS_DEBUG_ERROR_IF(target.size1() != value.size1() || target.size2() != value.size2()) << "matrix size mismatch in matrix AtomicDivMatrix- Sizes are: " << target.size1() << "x" << target.size2() << " for target and " << value.size1() << "x" << value.size2() << " for value " << std::endl;

for(std::size_t i=0; i<target.size1(); ++i) {
for(std::size_t j=0; j<target.size2(); ++j) {
AtomicDiv(target(i,j), value(i,j));
}
}
}

}  
