#pragma once

#include <CL/sycl.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace facebook { namespace cuda {

template <typename T,
int Dim,
typename IndexT,
template <typename U> class PtrTraits>
class DeviceTensor;

namespace detail {
template <typename TensorType,
int SubDim,
template <typename U> class PtrTraits>
class DeviceSubTensor;
}

template <typename T>
struct RestrictPtrTraits {
typedef T* __restrict PtrType;
};

template <typename T>
struct DefaultPtrTraits {
typedef T* PtrType;
};


template <typename T,
int Dim,
typename IndexT = int,
template <typename U> class PtrTraits = DefaultPtrTraits>
class DeviceTensor {
public:
enum { NumDim = Dim };
typedef T DataType;
typedef IndexT IndexType;
typedef typename PtrTraits<T>::PtrType DataPtrType;
typedef DeviceTensor<T, Dim, IndexT, PtrTraits> TensorType;

DeviceTensor();

DeviceTensor(DataPtrType data,
const IndexT sizes[Dim]);

DeviceTensor(DataPtrType data,
const IndexT sizes[Dim],
const IndexT strides[Dim]);

template <int OtherDim>
bool
isSameSize(
const DeviceTensor<T, OtherDim, IndexT, PtrTraits>& rhs) const;

template <int OtherDim>
bool
isSameSizeAndStride(
const DeviceTensor<T, OtherDim, IndexT, PtrTraits>& rhs) const;

std::string toString() const;

template <typename U>
DeviceTensor<U, Dim, IndexT, PtrTraits> cast();

template <typename U>

const DeviceTensor<U, Dim, IndexT, PtrTraits> cast() const;

inline DataPtrType data() {
return data_;
}

inline const DataPtrType data() const {
return data_;
}

template <typename U>
inline typename PtrTraits<U>::PtrType dataAs() {
return reinterpret_cast<typename PtrTraits<U>::PtrType>(data_);
}

template <typename U>
inline const typename PtrTraits<const U>::PtrType dataAs() const {
return reinterpret_cast<typename PtrTraits<const U>::PtrType>(data_);
}

inline detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>
operator[](IndexT);

inline const detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>
operator[](IndexT) const;

inline int getSize(int i) const {
return size_[i];
}

inline int getStride(int i) const {
return stride_[i];
}

long numElements() const;

inline const IndexT *sizes() const {
return size_;
}

inline const IndexT *strides() const {
return stride_;
}

void permuteDims(const std::vector<int>& perm);

bool isContiguous() const;

bool isConsistentlySized(int i) const;

bool isConsistentlySized() const;

bool isContiguousDim(int i) const;

DeviceTensor<T, Dim, IndexT, PtrTraits>
transpose(int dim1, int dim2) const;

template <int NewDim>
DeviceTensor<T, NewDim, IndexT, PtrTraits> upcastOuter();

template <int NewDim>
DeviceTensor<T, NewDim, IndexT, PtrTraits> upcastInner();

template <int NewDim>

DeviceTensor<T, NewDim, IndexT, PtrTraits> downcastOuter();

template <int NewDim>

DeviceTensor<T, NewDim, IndexT, PtrTraits> downcastInner();

template <int SubDim>
DeviceTensor<T, SubDim, IndexT, PtrTraits>
view(DataPtrType at);

template <int SubDim>
DeviceTensor<T, SubDim, IndexT, PtrTraits>
view();

void zero(sycl::queue *stream = 0);

private:
DataPtrType data_;

IndexT stride_[Dim];

IndexT size_[Dim];
};

namespace detail {

template <typename TensorType, template <typename U> class PtrTraits>
class DeviceSubTensor<TensorType, 0, PtrTraits> {
public:
DeviceSubTensor<TensorType, 0, PtrTraits>
operator=(typename TensorType::DataType val) {
*data_ = val;
return *this;
}

operator typename TensorType::DataType&() {
return *data_;
}

operator const typename TensorType::DataType&() const {
return *data_;
}

typename TensorType::DataType* operator&() {
return data_;
}

const typename TensorType::DataType* operator&() const {
return data_;
}

inline typename TensorType::DataPtrType data() {
return data_;
}

inline const typename TensorType::DataPtrType data() const {
return data_;
}

template <typename T>
T& as() {
return *dataAs<T>();
}

template <typename T>
const T& as() const {
return *dataAs<T>();
}

template <typename T>
inline typename PtrTraits<T>::PtrType dataAs() {
return reinterpret_cast<typename PtrTraits<T>::PtrType>(data_);
}

template <typename T>
inline typename PtrTraits<const T>::PtrType dataAs() const {
return reinterpret_cast<typename PtrTraits<const T>::PtrType>(data_);
}

inline typename TensorType::DataType ldg() const {
return *data_;
}

template <typename T> inline T ldgAs() const {
return as<T>();
}

private:
friend class DeviceSubTensor<TensorType, 1, PtrTraits>;

friend class DeviceTensor<typename TensorType::DataType,
1,
typename TensorType::IndexType,
PtrTraits>;

inline DeviceSubTensor(TensorType &t,
typename TensorType::DataPtrType data)
: tensor_(t), data_(data) {
}

TensorType& tensor_;

typename TensorType::DataPtrType const data_;
};

template <typename TensorType,
int SubDim,
template <typename U> class PtrTraits>
class DeviceSubTensor {
public:
inline DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>
operator[](typename TensorType::IndexType index) {
return DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>(
tensor_,
data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
}

inline const DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>
operator[](typename TensorType::IndexType index) const {
return DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>(
tensor_,
data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
}

typename TensorType::DataType* operator&() {
return data_;
}

const typename TensorType::DataType* operator&() const {
return data_;
}

inline typename TensorType::DataPtrType data() {
return data_;
}

inline const typename TensorType::DataPtrType data() const {
return data_;
}

template <typename T>
T& as() {
return *dataAs<T>();
}

template <typename T>
const T& as() const {
return *dataAs<T>();
}

template <typename T>
inline typename PtrTraits<T>::PtrType dataAs() {
return reinterpret_cast<typename PtrTraits<T>::PtrType>(data_);
}

template <typename T>
inline typename PtrTraits<const T>::PtrType dataAs() const {
return reinterpret_cast<typename PtrTraits<const T>::PtrType>(data_);
}

inline typename TensorType::DataType ldg() const {
return *data_;
}

template <typename T> inline T ldgAs() const {
return as<T>();
}

DeviceTensor<typename TensorType::DataType,
SubDim,
typename TensorType::IndexType,
PtrTraits> view() {
return tensor_.template view<SubDim>(data_);
}

private:
friend class DeviceSubTensor<TensorType, SubDim + 1, PtrTraits>;

friend class
DeviceTensor<typename TensorType::DataType,
TensorType::NumDim,
typename TensorType::IndexType,
PtrTraits>;

inline DeviceSubTensor(TensorType &t,
typename TensorType::DataPtrType data)
: tensor_(t), data_(data) {
}

TensorType& tensor_;

typename TensorType::DataPtrType const data_;
};

} 

template <typename T, int Dim, typename IndexT,
template <typename U> class PtrTraits>
inline detail::DeviceSubTensor<DeviceTensor<T, Dim, IndexT, PtrTraits>,
Dim - 1, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index) {
return detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>(
detail::DeviceSubTensor<TensorType, Dim, PtrTraits>(
*this, data_)[index]);
}

template <typename T, int Dim, typename IndexT,
template <typename U> class PtrTraits>
inline const
detail::DeviceSubTensor<DeviceTensor<T, Dim, IndexT, PtrTraits>, Dim - 1,
PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index) const {
return detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>(
detail::DeviceSubTensor<TensorType, Dim, PtrTraits>(
const_cast<TensorType&>(*this), data_)[index]);
}

template <typename T, int Dim,
typename IndexT, template <typename U> class PtrTraits>
std::ostream& operator<<(
std::ostream& os, const DeviceTensor<T, Dim, IndexT, PtrTraits>& t) {
os << t.toString();
return os;
}

} } 

#include "cuda/DeviceTensor-inl.dp.hpp"
