

#ifndef TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
#define TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
#ifdef INTEL_MKL

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(INTEL_MKL_ML_ONLY) || defined(INTEL_MKL_DNN_ONLY)
#ifndef INTEL_MKL
#error "INTEL_MKL_{ML,DNN}_ONLY require INTEL_MKL"
#endif
#endif

#if defined(INTEL_MKL_ML_ONLY) && defined(INTEL_MKL_DNN_ONLY)
#error "at most one of INTEL_MKL_ML_ONLY and INTEL_MKL_DNN_ONLY may be defined"
#endif

#ifdef INTEL_MKL_ML_ONLY
#error \
"Compiling for INTEL MKL ML only is no longer supported.Please use MKL DNN (the default option for --config=mkl)"
#endif

#ifdef INTEL_MKL_ML_ONLY
#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#include "mkl_service.h"
#include "mkl_trans.h"
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#ifndef INTEL_MKL_ML_ONLY
#include "mkldnn.hpp"
#include "tensorflow/core/lib/core/stringpiece.h"

using mkldnn::engine;
using mkldnn::memory;
using mkldnn::padding_kind;
using mkldnn::primitive;
using mkldnn::reorder;
#endif

#ifdef _WIN32
typedef unsigned int uint;
#endif

namespace tensorflow {



typedef enum { W = 0, H = 1, C = 2, N = 3 } MklDims;

typedef enum {
Dim_N = 0,
Dim_C = 1,
Dim_H = 2,
Dim_W = 3,
Dim_O = 0,
Dim_I = 1
} MklDnnDims;

typedef enum {
Dim3d_N = 0,
Dim3d_C = 1,
Dim3d_D = 2,
Dim3d_H = 3,
Dim3d_W = 4,
Dim3d_O = 0,
Dim3d_I = 1
} MklDnnDims3D;

enum class MklQuantization {
QUANTIZED_VERSION,
FP_VERSION,
};

static const int kSmallBatchSize = 32;

#ifdef INTEL_MKL_ML_ONLY
class MklShape {
public:
MklShape() {}
TF_DISALLOW_COPY_AND_ASSIGN(MklShape);  

~MklShape() {
if (sizes_) delete[] sizes_;
if (strides_) delete[] strides_;
if (mklLayout_) CHECK_EQ(dnnLayoutDelete_F32(mklLayout_), E_SUCCESS);
if (tfLayout_) CHECK_EQ(dnnLayoutDelete_F32(tfLayout_), E_SUCCESS);
if (tf_to_mkl_dim_map_) delete[] tf_to_mkl_dim_map_;
}

const bool IsMklTensor() const { return isMklTensor_; }

void SetMklTensor(const bool isMklTensor) { isMklTensor_ = isMklTensor; }

void SetDimensions(const size_t dimension) { dimension_ = dimension; }

void SetMklLayout(dnnLayout_t mklLayout) { mklLayout_ = mklLayout; }

void SetMklLayout(const void* primitive, size_t resourceType) {
CHECK_EQ(
dnnLayoutCreateFromPrimitive_F32(&mklLayout_, (dnnPrimitive_t)primitive,
(dnnResourceType_t)resourceType),
E_SUCCESS);
}

void SetTfLayout(const size_t dimension, const size_t* sizes,
const size_t* strides) {
dimension_ = dimension;
if (dimension > 0) {  
sizes_ = new size_t[dimension];
strides_ = new size_t[dimension];

for (int ii = 0; ii < dimension; ii++) {
sizes_[ii] = sizes[ii];
strides_[ii] = strides[ii];
}
CHECK_EQ(dnnLayoutCreate_F32(&tfLayout_, dimension, sizes, strides),
E_SUCCESS);
}
}

void SetTfDimOrder(const size_t dimension) {
CHECK(dimension == dimension_);
if (tf_to_mkl_dim_map_ == nullptr) {
tf_to_mkl_dim_map_ = new size_t[dimension];
}
for (size_t ii = 0; ii < dimension; ii++) {
tf_to_mkl_dim_map_[ii] = dimension - (ii + 1);
}
}

void SetTfDimOrder(const size_t dimension, const size_t* tf_to_mkl_dim_map) {
CHECK(dimension == dimension_);
if (tf_to_mkl_dim_map_ == nullptr) {
tf_to_mkl_dim_map_ = new size_t[dimension];
}
for (size_t ii = 0; ii < dimension; ii++) {
tf_to_mkl_dim_map_[ii] = tf_to_mkl_dim_map[ii];
}
}

void SetTfDimOrder(const size_t dimension, TensorFormat data_format) {
CHECK_EQ(dimension, 4);
CHECK(dimension == dimension_);
if (tf_to_mkl_dim_map_ == nullptr) {
tf_to_mkl_dim_map_ = new size_t[dimension];
}
tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'W')] = MklDims::W;
tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'H')] = MklDims::H;
tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'C')] = MklDims::C;
tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'N')] = MklDims::N;
}

const dnnLayout_t GetMklLayout() const { return mklLayout_; }
const dnnLayout_t GetTfLayout() const { return tfLayout_; }
const dnnLayout_t GetCurLayout() const {
return isMklTensor_ ? mklLayout_ : tfLayout_;
}
size_t GetDimension() const { return dimension_; }
const size_t* GetSizes() const { return sizes_; }
int64 dim_size(int index) const { return sizes_[index]; }
int64 tf_dim_size(int index) const {
return sizes_[tf_to_mkl_dim_map_[index]];
}
const size_t* GetStrides() const { return strides_; }
const size_t* GetTfToMklDimMap() const { return tf_to_mkl_dim_map_; }
size_t tf_dim_idx(int index) const { return tf_to_mkl_dim_map_[index]; }

bool IsMklChannelDim(int d) const { return tf_dim_idx(d) == MklDims::C; }
bool IsMklBatchDim(int d) const { return tf_dim_idx(d) == MklDims::N; }
bool IsMklWidthDim(int d) const { return tf_dim_idx(d) == MklDims::W; }
bool IsMklHeightDim(int d) const { return tf_dim_idx(d) == MklDims::H; }

bool IsTensorInNCHWFormat() const {
TensorFormat data_format = FORMAT_NCHW;
return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
}

bool IsTensorInNHWCFormat() const {
TensorFormat data_format = FORMAT_NHWC;
return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
}

void GetConvertedFlatData(dnnLayout_t targetLayout, void* input,
void* output) const {
dnnLayout_t curLayout;
if (isMklTensor_)
curLayout = mklLayout_;
else
curLayout = tfLayout_;
dnnPrimitive_t convert;
CHECK_EQ(dnnConversionCreate_F32(&convert, curLayout, targetLayout),
E_SUCCESS);
CHECK_EQ(dnnConversionExecute_F32(convert, input, output), E_SUCCESS);
CHECK_EQ(dnnDelete_F32(convert), E_SUCCESS);
}


#define SIZE_OF_MKL_DNN_BUF \
(dnnLayoutSerializationBufferSize_F32())  


#define SIZE_OF_MKL_SERIAL_DATA(dims) \
(2 * sizeof(size_t) + 3 * dims * sizeof(size_t) + 2 * SIZE_OF_MKL_DNN_BUF)


#define IS_MKL_TENSOR_OFFSET 0
#define DIMS_OFFSET \
(IS_MKL_TENSOR_OFFSET + sizeof(size_t))  
#define SIZES_OFFSET(dims) (DIMS_OFFSET + sizeof(size_t))
#define STRIDES_OFFSET(dims) \
(SIZES_OFFSET(dims) + dims * sizeof(size_t))  
#define MKL_LAYOUT_OFFSET(dims) \
(STRIDES_OFFSET(dims) + dims * sizeof(size_t))  
#define TF_LAYOUT_OFFSET(dims) \
(MKL_LAYOUT_OFFSET(dims) + SIZE_OF_MKL_DNN_BUF)  
#define TF_TO_MKL_DIM_MAP_OFFSET(dims) \
(TF_LAYOUT_OFFSET(dims) + SIZE_OF_MKL_DNN_BUF)


void DeSerializeMklShape(const unsigned char* buf, size_t buf_size) {
CHECK(buf_size >= sizeof(size_t)) << "Bufsize too small in DeSerialize";
isMklTensor_ =
*reinterpret_cast<const size_t*>(buf + IS_MKL_TENSOR_OFFSET) != 0;

if (isMklTensor_) {  
dimension_ = *(reinterpret_cast<const size_t*>(buf + DIMS_OFFSET));
CHECK(buf_size >= SIZE_OF_MKL_SERIAL_DATA(dimension_))
<< "Bufsize too small in DeSerialize";
sizes_ = new size_t[dimension_];
strides_ = new size_t[dimension_];
tf_to_mkl_dim_map_ = new size_t[dimension_];
for (int i = 0; i < dimension_; i++) {
sizes_[i] =
reinterpret_cast<const size_t*>(buf + SIZES_OFFSET(dimension_))[i];
strides_[i] = reinterpret_cast<const size_t*>(
buf + STRIDES_OFFSET(dimension_))[i];
tf_to_mkl_dim_map_[i] = reinterpret_cast<const size_t*>(
buf + TF_TO_MKL_DIM_MAP_OFFSET(dimension_))[i];
}
CHECK_EQ(dnnLayoutDeserialize_F32(&mklLayout_,
buf + MKL_LAYOUT_OFFSET(dimension_)),
E_SUCCESS);
CHECK_EQ(dnnLayoutDeserialize_F32(&tfLayout_,
buf + TF_LAYOUT_OFFSET(dimension_)),
E_SUCCESS);
}
}

void SerializeMklShape(unsigned char* buf, size_t buf_size) const {
CHECK(buf_size >= SIZE_OF_MKL_SERIAL_DATA(dimension_))
<< "Bufsize too small to Serialize";
*reinterpret_cast<size_t*>(buf + IS_MKL_TENSOR_OFFSET) =
isMklTensor_ ? 1 : 0;
if (isMklTensor_) {
*(reinterpret_cast<size_t*>(buf + DIMS_OFFSET)) = dimension_;
for (int i = 0; i < dimension_; i++) {
reinterpret_cast<size_t*>(buf + SIZES_OFFSET(dimension_))[i] =
sizes_[i];
reinterpret_cast<size_t*>(buf + STRIDES_OFFSET(dimension_))[i] =
strides_[i];
reinterpret_cast<size_t*>(buf +
TF_TO_MKL_DIM_MAP_OFFSET(dimension_))[i] =
tf_to_mkl_dim_map_[i];
}
CHECK_EQ(dnnLayoutSerialize_F32(mklLayout_,
buf + MKL_LAYOUT_OFFSET(dimension_)),
E_SUCCESS);
CHECK_EQ(
dnnLayoutSerialize_F32(tfLayout_, buf + TF_LAYOUT_OFFSET(dimension_)),
E_SUCCESS);
}
}

private:
bool isMklTensor_ =
false;  
dnnLayout_t mklLayout_ = nullptr;  
dnnLayout_t tfLayout_ = nullptr;   
size_t dimension_ = 0;
size_t* sizes_ = nullptr;    
size_t* strides_ = nullptr;  
size_t* tf_to_mkl_dim_map_ =
nullptr;  
};

#else

TensorFormat MklDnn3DDataFormatToTFDataFormat(memory::format format);
TensorFormat MklDnnDataFormatToTFDataFormat(memory::format format);
memory::dims CalculateTFStrides(const memory::dims& dims_tf_order);
memory::desc CreateBlockedMemDescHelper(const memory::dims& dim,
const memory::dims& strides,
memory::data_type dtype);

class MklDnnShape {
private:
typedef struct {
bool is_mkl_tensor_ = false;
size_t dimension_ = 0;
mkldnn_dims_t sizes_;  
memory::format tf_data_format_ = memory::format::format_undef;
memory::data_type T_ = memory::data_type::data_undef;
mkldnn_memory_desc_t mkl_md_;
mkldnn_dims_t map_;
} MklShapeData;
MklShapeData data_;

typedef std::remove_extent<mkldnn_dims_t>::type mkldnn_dim_t;
#define INVALID_DIM_SIZE -1

public:
MklDnnShape() {
for (size_t i = 0; i < sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
++i) {
data_.sizes_[i] = -1;
}
for (size_t i = 0; i < sizeof(data_.map_) / sizeof(data_.map_[0]); ++i) {
data_.map_[i] = -1;
}
}

~MklDnnShape() {}
TF_DISALLOW_COPY_AND_ASSIGN(MklDnnShape);  

inline bool CompareMklDnnLayouts(const memory::desc& md1,
const memory::desc& md2) const {
mkldnn_memory_desc_t mdd1 = md1.data;
mkldnn_memory_desc_t mdd2 = md2.data;
const char* d1 = reinterpret_cast<const char*>(&mdd1);
const char* d2 = reinterpret_cast<const char*>(&mdd2);

size_t md_size = sizeof(mdd1);
for (size_t i = 0; i < md_size; i++) {
if (*d1++ != *d2++) {
return false;
}
}
return true;
}

inline bool operator==(const MklDnnShape& input_shape) const {
if (this->IsMklTensor() != input_shape.IsMklTensor()) {
return false;
}

if (this->IsMklTensor()) {
return this->GetTfShape() == input_shape.GetTfShape() &&
CompareMklDnnLayouts(this->GetMklLayout(),
input_shape.GetMklLayout());
}

return true;
}

inline bool operator==(const TensorShape& input_shape) const {
if (!this->IsMklTensor()) {
return false;
}

return this->GetTfShape() == input_shape;
}

inline const bool IsMklTensor() const { return data_.is_mkl_tensor_; }
inline void SetMklTensor(bool is_mkl_tensor) {
data_.is_mkl_tensor_ = is_mkl_tensor;
}

inline void SetDimensions(const size_t dimension) {
data_.dimension_ = dimension;
}
inline size_t GetDimension(char dimension) const {
int index = GetMklDnnTensorDimIndex(dimension);
CHECK(index >= 0 && index < this->GetDimension())
<< "Invalid index from the dimension: " << index << ", " << dimension;
return this->DimSize(index);
}

inline size_t GetDimension3D(char dimension) const {
int index = GetMklDnnTensor3DDimIndex(dimension);
CHECK(index >= 0 && index < this->GetDimension())
<< "Invalid index from the dimension: " << index << ", " << dimension;
return this->DimSize(index);
}

inline int32 GetMklDnnTensorDimIndex(char dimension) const {
switch (dimension) {
case 'N':
return MklDnnDims::Dim_N;
case 'C':
return MklDnnDims::Dim_C;
case 'H':
return MklDnnDims::Dim_H;
case 'W':
return MklDnnDims::Dim_W;
default:
LOG(FATAL) << "Invalid dimension: " << dimension;
return -1;  
}
}

inline int32 GetMklDnnTensor3DDimIndex(char dimension) const {
switch (dimension) {
case 'N':
return MklDnnDims3D::Dim3d_N;
case 'C':
return MklDnnDims3D::Dim3d_C;
case 'D':
return MklDnnDims3D::Dim3d_D;
case 'H':
return MklDnnDims3D::Dim3d_H;
case 'W':
return MklDnnDims3D::Dim3d_W;
default:
LOG(FATAL) << "Invalid dimension: " << dimension;
return -1;  
}
}

inline size_t GetDimension() const { return data_.dimension_; }
inline const int* GetSizes() const {
return reinterpret_cast<const int*>(&data_.sizes_[0]);
}

inline memory::dims GetSizesAsMklDnnDims() const {
memory::dims retVal;
if (data_.is_mkl_tensor_) {
size_t dimensions = sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
for (size_t i = 0; i < dimensions; i++) {
if (data_.sizes_[i] != INVALID_DIM_SIZE)
retVal.push_back(data_.sizes_[i]);
}
} else {
CHECK_EQ(data_.is_mkl_tensor_, true);
}
return retVal;
}

inline int64 DimSize(int index) const {
CHECK_LT(index, sizeof(data_.sizes_) / sizeof(data_.sizes_[0]));
return data_.sizes_[index];
}

inline TensorShape GetTfShape() const {
CHECK_EQ(data_.is_mkl_tensor_, true);

std::vector<int32> shape(data_.dimension_, -1);
if (data_.tf_data_format_ != memory::format::blocked) {
for (size_t idx = 0; idx < data_.dimension_; ++idx) {
shape[idx] = data_.sizes_[TfDimIdx(idx)];
}
} else {
for (size_t idx = 0; idx < data_.dimension_; ++idx) {
shape[idx] = data_.sizes_[idx];
}
}

TensorShape ts;
bool ret = TensorShapeUtils::MakeShape(shape, &ts).ok();
CHECK_EQ(ret, true);
return ts;
}

inline void SetElemType(memory::data_type dt) { data_.T_ = dt; }
inline const memory::data_type GetElemType() { return data_.T_; }

inline void SetMklLayout(memory::primitive_desc* pd) {
CHECK_NOTNULL(pd);
data_.mkl_md_ = pd->desc().data;
}

inline void SetMklLayout(memory::desc* md) {
CHECK_NOTNULL(md);
data_.mkl_md_ = md->data;
}

inline const memory::desc GetMklLayout() const {
return memory::desc(data_.mkl_md_);
}

inline memory::format GetTfDataFormat() const {
return data_.tf_data_format_;
}
inline void SetTfLayout(size_t dims, const memory::dims& sizes,
memory::format format) {
CHECK_EQ(dims, sizes.size());
data_.dimension_ = dims;
for (size_t ii = 0; ii < dims; ii++) {
data_.sizes_[ii] = sizes[ii];
}
data_.tf_data_format_ = format;
if (format != memory::format::blocked) {
SetTfDimOrder(dims, format);
}
}

inline const memory::desc GetTfLayout() const {
memory::dims dims;
for (size_t ii = 0; ii < data_.dimension_; ii++) {
dims.push_back(data_.sizes_[ii]);
}

if (data_.tf_data_format_ == memory::format::blocked) {
auto strides = CalculateTFStrides(dims);
return CreateBlockedMemDescHelper(dims, strides, data_.T_);
} else {
return memory::desc(dims, data_.T_, data_.tf_data_format_);
}
}

inline const memory::desc GetCurLayout() const {
return IsMklTensor() ? GetMklLayout() : GetTfLayout();
}

inline void SetTfDimOrder(const size_t dimension, const mkldnn_dims_t map) {
CHECK(dimension == data_.dimension_);
for (size_t ii = 0; ii < dimension; ii++) {
data_.map_[ii] = map[ii];
}
}

inline void SetTfDimOrder(const size_t dimension, TensorFormat data_format) {
if (dimension == 5) {
CHECK(dimension == data_.dimension_);
data_.map_[GetTensorDimIndex<3>(data_format, '0')] =
MklDnnDims3D::Dim3d_D;
data_.map_[GetTensorDimIndex<3>(data_format, '1')] =
MklDnnDims3D::Dim3d_H;
data_.map_[GetTensorDimIndex<3>(data_format, '2')] =
MklDnnDims3D::Dim3d_W;
data_.map_[GetTensorDimIndex<3>(data_format, 'C')] =
MklDnnDims3D::Dim3d_C;
data_.map_[GetTensorDimIndex<3>(data_format, 'N')] =
MklDnnDims3D::Dim3d_N;
} else {
CHECK_EQ(dimension, 4);
CHECK(dimension == data_.dimension_);
data_.map_[GetTensorDimIndex<2>(data_format, 'W')] = MklDnnDims::Dim_W;
data_.map_[GetTensorDimIndex<2>(data_format, 'H')] = MklDnnDims::Dim_H;
data_.map_[GetTensorDimIndex<2>(data_format, 'C')] = MklDnnDims::Dim_C;
data_.map_[GetTensorDimIndex<2>(data_format, 'N')] = MklDnnDims::Dim_N;
}
}

inline void SetTfDimOrder(const size_t dimension, memory::format format) {
TensorFormat data_format = MklDnnDataFormatToTFDataFormat(format);
SetTfDimOrder(dimension, data_format);
}

inline const mkldnn_dim_t* GetTfToMklDimMap() const { return &data_.map_[0]; }
inline size_t TfDimIdx(int index) const { return data_.map_[index]; }
inline int64 TfDimSize(int index) const {
return data_.sizes_[TfDimIdx(index)];
}

inline bool IsMklChannelDim(int d) const {
return TfDimIdx(d) == MklDnnDims::Dim_C;
}
inline bool IsMklBatchDim(int d) const {
return TfDimIdx(d) == MklDnnDims::Dim_N;
}
inline bool IsMklWidthDim(int d) const {
return TfDimIdx(d) == MklDnnDims::Dim_W;
}
inline bool IsMklHeightDim(int d) const {
return TfDimIdx(d) == MklDnnDims::Dim_H;
}

inline bool IsTensorInNCHWFormat() const {
TensorFormat data_format = FORMAT_NCHW;
return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
}

inline bool IsTensorInNHWCFormat() const {
TensorFormat data_format = FORMAT_NHWC;
return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
}


inline size_t GetSerializeBufferSize() const { return sizeof(MklShapeData); }

void SerializeMklDnnShape(unsigned char* buf, size_t buf_size) const {
CHECK(buf_size >= GetSerializeBufferSize())
<< "Buffer size is too small to SerializeMklDnnShape";
*reinterpret_cast<MklShapeData*>(buf) = data_;
}

void DeSerializeMklDnnShape(const unsigned char* buf, size_t buf_size) {
CHECK(buf_size >= sizeof(data_.is_mkl_tensor_))
<< "Buffer size is too small in DeSerializeMklDnnShape";

const bool is_mkl_tensor = *reinterpret_cast<const bool*>(buf);
if (is_mkl_tensor) {  
CHECK(buf_size >= GetSerializeBufferSize())
<< "Buffer size is too small in DeSerializeMklDnnShape";
data_ = *reinterpret_cast<const MklShapeData*>(buf);
}
}
};

#endif


#ifndef INTEL_MKL_ML_ONLY
typedef std::vector<MklDnnShape> MklDnnShapeList;
#else
typedef std::vector<MklShape> MklShapeList;
#endif

#ifdef INTEL_MKL_ML_ONLY
inline bool AreAllMklTensors(const MklShapeList& shapes) {
for (auto& s : shapes) {
if (!s.IsMklTensor()) {
return false;
}
}
return true;
}

template <typename T>
inline Tensor ConvertMklToTF(OpKernelContext* context, const Tensor& mkl_tensor,
const MklShape& mkl_shape) {
Tensor output_tensor;
TensorShape output_shape;

for (size_t j = 0; j < mkl_shape.GetDimension(); j++) {
output_shape.AddDim(mkl_shape.GetSizes()[mkl_shape.tf_dim_idx(j)]);
}

context->allocate_temp(DataTypeToEnum<T>::v(), output_shape, &output_tensor);

dnnLayout_t output_layout = static_cast<dnnLayout_t>(mkl_shape.GetTfLayout());
void* input_buffer = const_cast<T*>(mkl_tensor.flat<T>().data());
void* output_buffer = const_cast<T*>(output_tensor.flat<T>().data());

if (mkl_tensor.NumElements() != 0) {
mkl_shape.GetConvertedFlatData(output_layout, input_buffer, output_buffer);
}

return output_tensor;
}
#else
using mkldnn::stream;
template <typename T>
class MklDnnData;

template <typename T>
inline Tensor ConvertMklToTF(OpKernelContext* context, const Tensor& mkl_tensor,
const MklDnnShape& mkl_shape) {
Tensor output_tensor;
try {
if (!mkl_shape.IsMklTensor())
return mkl_tensor;  

TensorShape output_shape = mkl_shape.GetTfShape();
;

context->allocate_temp(DataTypeToEnum<T>::v(), output_shape,
&output_tensor);

auto cpu_engine = engine(engine::cpu, 0);
MklDnnData<T> input(&cpu_engine);

auto input_mkl_md = mkl_shape.GetMklLayout();
auto output_tf_md = mkl_shape.GetTfLayout();
auto output_tf_pd = memory::primitive_desc(output_tf_md, cpu_engine);
input.SetUsrMem(input_mkl_md, &mkl_tensor);

if (input.IsReorderNeeded(output_tf_pd)) {
std::vector<primitive> net;
CHECK_EQ(input.CheckReorderToOpMem(output_tf_pd, &output_tensor, &net),
true);
stream(stream::kind::eager).submit(net).wait();
} else {
CHECK(output_tensor.CopyFrom(mkl_tensor, output_shape));
}
} catch (mkldnn::error& e) {
string error_msg = "Status: " + std::to_string(e.status) +
", message: " + string(e.message) + ", in file " +
string(__FILE__) + ":" + std::to_string(__LINE__);
LOG(FATAL) << "Operation received an exception: " << error_msg;
}
return output_tensor;
}
#endif

#ifdef INTEL_MKL_ML_ONLY
inline void GetMklShape(OpKernelContext* ctext, int n, MklShape* mklshape) {
mklshape->DeSerializeMklShape(
ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
.flat<uint8>()
.data(),
ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
.flat<uint8>()
.size() *
sizeof(uint8));
}
#else
inline void GetMklShape(OpKernelContext* ctext, int n, MklDnnShape* mklshape) {
mklshape->DeSerializeMklDnnShape(
ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
.flat<uint8>()
.data(),
ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
.flat<uint8>()
.size() *
sizeof(uint8));
}
#endif

inline const Tensor& MklGetInput(OpKernelContext* ctext, int n) {
return ctext->input(GetTensorDataIndex(n, ctext->num_inputs()));
}

inline void GetMklInputList(OpKernelContext* ctext, StringPiece name,
OpInputList* input_tensors) {
CHECK_NOTNULL(input_tensors);
ctext->input_list(name, input_tensors);
}

#ifdef INTEL_MKL_ML_ONLY

inline void GetMklShapeList(OpKernelContext* ctext, StringPiece name,
MklShapeList* mkl_shapes) {
OpInputList input_mkl_tensors;
GetMklInputList(ctext, strings::StrCat("mkl_", name), &input_mkl_tensors);

for (int i = 0; i < input_mkl_tensors.size(); i++) {
(*mkl_shapes)[i].DeSerializeMklShape(
input_mkl_tensors[i].flat<uint8>().data(),
input_mkl_tensors[i].flat<uint8>().size() * sizeof(uint8));
}
}

#else

inline void GetMklShapeList(OpKernelContext* ctext, StringPiece name,
MklDnnShapeList* mkl_shapes) {
OpInputList input_mkl_tensors;
GetMklInputList(ctext, strings::StrCat("mkl_", name), &input_mkl_tensors);

for (int i = 0; i < input_mkl_tensors.size(); i++) {
(*mkl_shapes)[i].DeSerializeMklDnnShape(
input_mkl_tensors[i].flat<uint8>().data(),
input_mkl_tensors[i].flat<uint8>().size() * sizeof(uint8));
}
}

#endif

#ifndef INTEL_MKL_ML_ONLY
inline TensorShape GetTfShape(OpKernelContext* context, size_t input_idx) {
CHECK_NOTNULL(context);
CHECK_LT(input_idx, context->num_inputs());

MklDnnShape input_mkl_shape;
GetMklShape(context, input_idx, &input_mkl_shape);
if (input_mkl_shape.IsMklTensor()) {
return input_mkl_shape.GetTfShape();
} else {
const Tensor& t = MklGetInput(context, input_idx);
return t.shape();
}
}
#endif

#ifdef INTEL_MKL_ML_ONLY
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
const MklShape& mkl_shape) {
Tensor* second_tensor = nullptr;
TensorShape second_shape;
second_shape.AddDim(SIZE_OF_MKL_SERIAL_DATA(mkl_shape.GetDimension()));
OP_REQUIRES_OK(ctext, ctext->allocate_output(
GetTensorMetaDataIndex(n, ctext->num_outputs()),
second_shape, &second_tensor));
mkl_shape.SerializeMklShape(
second_tensor->flat<uint8>().data(),
second_tensor->flat<uint8>().size() * sizeof(uint8));
}

#else
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
const MklDnnShape& mkl_shape) {
Tensor* second_tensor = nullptr;
TensorShape second_shape;
second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
OP_REQUIRES_OK(ctext, ctext->allocate_output(
GetTensorMetaDataIndex(n, ctext->num_outputs()),
second_shape, &second_tensor));
mkl_shape.SerializeMklDnnShape(
second_tensor->flat<uint8>().data(),
second_tensor->flat<uint8>().size() * sizeof(uint8));
}
#endif

#ifdef INTEL_MKL_ML_ONLY
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
Tensor** output,
const TensorShape& tf_shape,
const MklShape& mkl_shape) {
Tensor* second_tensor = nullptr;
TensorShape second_shape;
second_shape.AddDim(SIZE_OF_MKL_SERIAL_DATA(mkl_shape.GetDimension()));
OP_REQUIRES_OK(
ctext, ctext->allocate_output(GetTensorDataIndex(n, ctext->num_outputs()),
tf_shape, output));
OP_REQUIRES_OK(ctext, ctext->allocate_output(
GetTensorMetaDataIndex(n, ctext->num_outputs()),
second_shape, &second_tensor));
mkl_shape.SerializeMklShape(
second_tensor->flat<uint8>().data(),
second_tensor->flat<uint8>().size() * sizeof(uint8));
}

#else
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
Tensor** output,
const TensorShape& tf_shape,
const MklDnnShape& mkl_shape) {
Tensor* second_tensor = nullptr;
TensorShape second_shape;
second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
OP_REQUIRES_OK(
ctext, ctext->allocate_output(GetTensorDataIndex(n, ctext->num_outputs()),
tf_shape, output));
OP_REQUIRES_OK(ctext, ctext->allocate_output(
GetTensorMetaDataIndex(n, ctext->num_outputs()),
second_shape, &second_tensor));
mkl_shape.SerializeMklDnnShape(
second_tensor->flat<uint8>().data(),
second_tensor->flat<uint8>().size() * sizeof(uint8));
}
#endif

#ifndef INTEL_MKL_ML_ONLY
template <typename T>
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
const memory::primitive_desc& pd, void** buf_out) {
TensorShape tf_shape;

tf_shape.AddDim(pd.get_size() / sizeof(T) + 1);
OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
tf_shape, tensor_out));
*buf_out = static_cast<void*>(tensor_out->flat<T>().data());
}
#else
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
dnnLayout_t lt_buff, void** buf_out) {
TensorShape tf_shape;

tf_shape.AddDim(
dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(lt_buff)) /
sizeof(float) +
1);
OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::v(),
tf_shape, tensor_out));
*buf_out = static_cast<void*>(tensor_out->flat<float>().data());
}

#endif
template <typename T>
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
TensorShape tf_shape) {
OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
tf_shape, tensor_out));
}

inline void GetStridesFromSizes(TensorFormat data_format, size_t* strides,
const size_t* sizes) {
if (data_format == FORMAT_NHWC) {
strides[0] = sizes[2];
strides[1] = sizes[0] * sizes[2];
strides[2] = 1;
strides[3] = sizes[0] * sizes[1] * sizes[2];
} else {
strides[0] = 1;
strides[1] = sizes[0];
strides[2] = sizes[0] * sizes[1];
strides[3] = sizes[0] * sizes[1] * sizes[2];
}
}

#ifdef INTEL_MKL_ML_ONLY
inline void MklSizesToTFSizes(OpKernelContext* context,
TensorFormat data_format_,
const MklShape& mkl_shape,
TensorShape* tf_shape) {
size_t tf_dim = mkl_shape.GetDimension();
const size_t* tf_sizes = mkl_shape.GetSizes();

OP_REQUIRES(context, tf_dim == 4,
errors::InvalidArgument("MKLSizesToTFSizes: size must be 4-dim"));
std::vector<int32> sizes;

sizes.push_back(tf_sizes[3]);

if (data_format_ == FORMAT_NHWC) {
sizes.push_back(tf_sizes[1]);
sizes.push_back(tf_sizes[0]);
sizes.push_back(tf_sizes[2]);
} else {
sizes.push_back(tf_sizes[2]);
sizes.push_back(tf_sizes[1]);
sizes.push_back(tf_sizes[0]);
}

OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(sizes, tf_shape));
}
#endif

inline int32 GetMklTensorDimIndex(char dimension) {
switch (dimension) {
case 'N':
return MklDims::N;
case 'C':
return MklDims::C;
case 'H':
return MklDims::H;
case 'W':
return MklDims::W;
default:
LOG(FATAL) << "Invalid dimension: " << dimension;
return -1;  
}
}

#ifdef INTEL_MKL_ML_ONLY
inline int64 GetMklTensorDim(const MklShape& mkl_shape, char dimension) {
int index = GetMklTensorDimIndex(dimension);
CHECK(index >= 0 && index < mkl_shape.GetDimension())
<< "Invalid index from the dimension: " << index << ", " << dimension;
return mkl_shape.dim_size(index);
}
#endif

inline void CopyMklTensorInToOut(OpKernelContext* context, int idx_in,
int idx_out) {
int num_inputs = context->num_inputs();
int num_outputs = context->num_outputs();
int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
int idx_meta_in = GetTensorMetaDataIndex(idx_in, num_inputs);
int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
int idx_meta_out = GetTensorMetaDataIndex(idx_out, num_outputs);

const Tensor& data = context->input(idx_data_in);
const Tensor& meta = context->input(idx_meta_in);
Tensor output(data.dtype());
Tensor meta_output(meta.dtype());

CHECK(output.CopyFrom(data, data.shape()));
CHECK(meta_output.CopyFrom(meta, meta.shape()));
context->set_output(idx_data_out, output);
context->set_output(idx_meta_out, meta_output);
}

#ifdef INTEL_MKL_ML_ONLY
inline void CopyTfTensorInToOutWithShape(OpKernelContext* context, int idx_in,
int idx_out,
const TensorShape& shape) {
int num_inputs = context->num_inputs();
int num_outputs = context->num_outputs();
int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

const Tensor& data = context->input(idx_data_in);
MklShape mkl_shape_output;
mkl_shape_output.SetMklTensor(false);
AllocateOutputSetMklShape(context, idx_out, mkl_shape_output);
Tensor output(data.dtype());
CHECK(output.CopyFrom(data, shape));
context->set_output(idx_data_out, output);
}
#else
inline void CopyTfTensorInToOutWithShape(OpKernelContext* context, int idx_in,
int idx_out,
const TensorShape& shape) {
int num_inputs = context->num_inputs();
int num_outputs = context->num_outputs();
int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

const Tensor& data = context->input(idx_data_in);
MklDnnShape mkl_shape_output;
mkl_shape_output.SetMklTensor(false);
AllocateOutputSetMklShape(context, idx_out, mkl_shape_output);
Tensor output(data.dtype());
CHECK(output.CopyFrom(data, shape));
context->set_output(idx_data_out, output);
}
#endif

#ifdef INTEL_MKL_ML_ONLY

inline void ForwardTfTensorInToOut(OpKernelContext* context, int idx_in,
int idx_out) {
int num_inputs = context->num_inputs();
int num_outputs = context->num_outputs();
int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

MklShape mkl_shape_output;
mkl_shape_output.SetMklTensor(false);
AllocateOutputSetMklShape(context, idx_out, mkl_shape_output);
if (IsRefType(context->input_dtype(idx_data_in))) {
context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
} else {
context->set_output(idx_data_out, context->input(idx_data_in));
}
}

#else

inline void ForwardTfTensorInToOut(OpKernelContext* context, int idx_in,
int idx_out) {
int num_inputs = context->num_inputs();
int num_outputs = context->num_outputs();
int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

MklDnnShape dnn_shape_output;
dnn_shape_output.SetMklTensor(false);
AllocateOutputSetMklShape(context, idx_out, dnn_shape_output);
if (IsRefType(context->input_dtype(idx_data_in))) {
context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
} else {
context->set_output(idx_data_out, context->input(idx_data_in));
}
}

#endif

inline void ForwardMklTensorInToOut(OpKernelContext* context, int idx_in,
int idx_out) {
int num_inputs = context->num_inputs();
int num_outputs = context->num_outputs();
int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
int idx_meta_in = GetTensorMetaDataIndex(idx_in, num_inputs);
int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
int idx_meta_out = GetTensorMetaDataIndex(idx_out, num_outputs);

if (IsRefType(context->input_dtype(idx_data_in))) {
context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
context->forward_ref_input_to_ref_output(idx_meta_in, idx_meta_out);
} else {
context->set_output(idx_data_out, context->input(idx_data_in));
context->set_output(idx_meta_out, context->input(idx_meta_in));
}
}

#ifndef INTEL_MKL_ML_ONLY
inline void SetDummyMklDnnShapeOutput(OpKernelContext* context,
uint32 idx_data_out) {
MklDnnShape mkl_shape_output;
mkl_shape_output.SetMklTensor(false);
AllocateOutputSetMklShape(context, idx_data_out, mkl_shape_output);
}

inline void ForwardMklTensorInToOutWithMklShape(OpKernelContext* context,
int idx_in, int idx_out,
const MklDnnShape& mkl_shape) {
int num_inputs = context->num_inputs();
int num_outputs = context->num_outputs();
int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

AllocateOutputSetMklShape(context, idx_out, mkl_shape);

if (IsRefType(context->input_dtype(idx_data_in))) {
context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
} else {
context->set_output(idx_data_out, context->input(idx_data_in));
}
}
#endif

inline void ForwardMklMetaDataInToOut(OpKernelContext* context,
uint32 idx_data_in,
uint32_t idx_data_out) {
uint32 idx_meta_in =
GetTensorMetaDataIndex(idx_data_in, context->num_inputs());
uint32 idx_meta_out =
GetTensorMetaDataIndex(idx_data_out, context->num_outputs());

if (IsRefType(context->input_dtype(idx_data_in))) {
context->forward_ref_input_to_ref_output(idx_meta_in, idx_meta_out);
} else {
context->set_output(idx_meta_out, context->input(idx_meta_in));
}
}

#ifdef INTEL_MKL_ML_ONLY
inline void SetDummyMklShapeOutput(OpKernelContext* context,
uint32 idx_data_out) {
MklShape mkl_shape_output;
mkl_shape_output.SetMklTensor(false);
AllocateOutputSetMklShape(context, idx_data_out, mkl_shape_output);
}

inline bool MklCompareShapes(const MklShape* input_shape_0,
const MklShape* input_shape_1) {
if (input_shape_0->GetDimension() != input_shape_1->GetDimension()) {
return false;
}

size_t ndims = input_shape_0->GetDimension();
for (size_t i = 0; i < ndims; i++) {
if (input_shape_0->dim_size(i) != input_shape_1->dim_size(i)) {
return false;
}
}

return true;
}

inline bool MklCompareShapes(const MklShape* input_shape_0,
const TensorShape* input_shape_1) {
if (input_shape_0->GetDimension() != input_shape_1->dims()) {
return false;
}

size_t ndims = input_shape_0->GetDimension();
for (size_t i = 0; i < ndims; i++) {
if (input_shape_0->tf_dim_size(i) != input_shape_1->dim_size(i)) {
return false;
}
}

return true;
}

inline bool MklCompareShapes(const TensorShape* input_shape_0,
const MklShape* input_shape_1) {
return MklCompareShapes(input_shape_1, input_shape_0);
}

inline bool MklCompareShapes(const TensorShape* input_shape_0,
const TensorShape* input_shape_1) {
if (input_shape_0->dims() != input_shape_1->dims()) {
return false;
}

size_t ndims = input_shape_0->dims();
for (size_t i = 0; i < ndims; i++) {
if (input_shape_0->dim_size(i) != input_shape_1->dim_size(i)) {
return false;
}
}

return true;
}

inline void MklNHWCToNCHW(const Tensor& input, Tensor** output) {
const float* buf_in = input.flat<float>().data();
float* buf_out = (*output)->flat<float>().data();

int64 N = input.dim_size(0);
int64 H = input.dim_size(1);
int64 W = input.dim_size(2);
int64 C = input.dim_size(3);
int64 stride_n = H * W * C;
#pragma omp parallel for num_threads(16)
for (int64 n = 0; n < N; ++n) {
mkl_somatcopy('R', 'T', H * W, C, 1, buf_in + n * stride_n, C,
buf_out + n * stride_n, H * W);
}
}

inline void MklNCHWToNHWC(const Tensor& input, Tensor** output) {
const float* buf_in = input.flat<float>().data();
float* buf_out = (*output)->flat<float>().data();

int64 N = (*output)->dim_size(0);
int64 H = (*output)->dim_size(1);
int64 W = (*output)->dim_size(2);
int64 C = (*output)->dim_size(3);
int64 stride_n = H * W * C;
#pragma omp parallel for num_threads(16)
for (int64 n = 0; n < N; ++n) {
mkl_somatcopy('R', 'T', C, H * W, 1, buf_in + n * stride_n, H * W,
buf_out + n * stride_n, C);
}
}

#endif

#ifndef INTEL_MKL_ML_ONLY

template <typename T>
static memory::data_type MklDnnType();

template <>
memory::data_type MklDnnType<float>() {
return memory::data_type::f32;
}
template <>
memory::data_type MklDnnType<quint8>() {
return memory::data_type::u8;
}
template <>
memory::data_type MklDnnType<qint8>() {
return memory::data_type::s8;
}
template <>
memory::data_type MklDnnType<qint32>() {
return memory::data_type::s32;
}

inline memory::format TFDataFormatToMklDnn3DDataFormat(TensorFormat format) {
if (format == FORMAT_NHWC)
return memory::format::ndhwc;
else if (format == FORMAT_NCHW)
return memory::format::ncdhw;
TF_CHECK_OK(Status(error::Code::INVALID_ARGUMENT, "Unsupported data format"));
return memory::format::format_undef;
}

inline memory::format TFDataFormatToMklDnnDataFormat(TensorFormat format) {
if (format == FORMAT_NHWC)
return memory::format::nhwc;
else if (format == FORMAT_NCHW)
return memory::format::nchw;
TF_CHECK_OK(Status(error::Code::INVALID_ARGUMENT, "Unsupported data format"));
return memory::format::format_undef;
}

inline TensorFormat MklDnnDataFormatToTFDataFormat(memory::format format) {
if (format == memory::format::nhwc || format == memory::format::ndhwc)
return FORMAT_NHWC;
else if (format == memory::format::nchw || format == memory::format::ncdhw)
return FORMAT_NCHW;
TF_CHECK_OK(Status(error::Code::INVALID_ARGUMENT, "Unsupported data format"));

return FORMAT_NHWC;
}

inline memory::dims TFShapeToMklDnnDims(const TensorShape& shape) {
memory::dims dims(shape.dims());
for (int d = 0; d < shape.dims(); ++d) {
dims[d] = shape.dim_size(d);
}
return dims;
}

inline memory::dims TFShapeToMklDnnDimsInNCHW(const TensorShape& shape,
TensorFormat format) {
CHECK_NE(TFDataFormatToMklDnnDataFormat(format),
memory::format::format_undef);

int n = shape.dim_size(GetTensorDimIndex(format, 'N'));
int c = shape.dim_size(GetTensorDimIndex(format, 'C'));
int h = shape.dim_size(GetTensorDimIndex(format, 'H'));
int w = shape.dim_size(GetTensorDimIndex(format, 'W'));

return memory::dims({n, c, h, w});
}

inline memory::dims TFShapeToMklDnnDimsInNCDHW(const TensorShape& shape,
TensorFormat format) {
CHECK_NE(TFDataFormatToMklDnn3DDataFormat(format),
memory::format::format_undef);

int n = shape.dim_size(GetTensorDimIndex<3>(format, 'N'));
int c = shape.dim_size(GetTensorDimIndex<3>(format, 'C'));
int d = shape.dim_size(GetTensorDimIndex<3>(format, '0'));
int h = shape.dim_size(GetTensorDimIndex<3>(format, '1'));
int w = shape.dim_size(GetTensorDimIndex<3>(format, '2'));

return memory::dims({n, c, d, h, w});
}

inline memory::dims MklDnnDimsInNCHW(const memory::dims& in_dims,
TensorFormat format) {
CHECK_NE(TFDataFormatToMklDnnDataFormat(format),
memory::format::format_undef);

int n = in_dims[GetTensorDimIndex(format, 'N')];
int c = in_dims[GetTensorDimIndex(format, 'C')];
int h = in_dims[GetTensorDimIndex(format, 'H')];
int w = in_dims[GetTensorDimIndex(format, 'W')];

return memory::dims({n, c, h, w});
}

inline TensorShape MklDnnDimsToTFShape(const memory::dims& dims) {
std::vector<int32> shape(dims.size(), -1);
for (int d = 0; d < dims.size(); d++) {
shape[d] = dims[d];
}

TensorShape ret;
CHECK_EQ(TensorShapeUtils::MakeShape(shape, &ret).ok(), true);
return ret;
}

inline memory::dims CalculateTFStrides(const memory::dims& dims_tf_order) {
CHECK_GT(dims_tf_order.size(), 0);
memory::dims strides(dims_tf_order.size());
int last_dim_idx = dims_tf_order.size() - 1;
strides[last_dim_idx] = 1;
for (int d = last_dim_idx - 1; d >= 0; d--) {
strides[d] = strides[d + 1] * dims_tf_order[d + 1];
}
return strides;
}

inline padding_kind TFPaddingToMklDnnPadding(Padding pad) {
return padding_kind::zero;
}

inline memory::desc CreateBlockedMemDescHelper(const memory::dims& dim,
const memory::dims& strides,
memory::data_type dtype) {
CHECK_EQ(dim.size(), strides.size());

mkldnn_memory_desc_t md;
md.primitive_kind = mkldnn_memory;
md.ndims = dim.size();
md.format = mkldnn_blocked;
md.data_type = memory::convert_to_c(dtype);

for (size_t i = 0; i < dim.size(); i++) {
md.layout_desc.blocking.block_dims[i] = 1;
md.layout_desc.blocking.strides[1][i] = 1;
md.layout_desc.blocking.strides[0][i] = strides[i];
md.layout_desc.blocking.padding_dims[i] = dim[i];
md.layout_desc.blocking.offset_padding_to_data[i] = 0;
md.dims[i] = dim[i];
}
md.layout_desc.blocking.offset_padding = 0;

return memory::desc(md);
}

template <typename T>
inline primitive FindOrCreateReorder(const memory* from, const memory* to);

template <typename T>
class MklDnnData {
private:
memory* user_memory_;

memory* reorder_memory_;

memory::desc* op_md_;
bool bIs3D;
void* allocated_buffer_;
const engine* cpu_engine_;

public:
explicit MklDnnData(const engine* e)
: user_memory_(nullptr),
reorder_memory_(nullptr),
op_md_(nullptr),
allocated_buffer_(nullptr),
cpu_engine_(e) {}

~MklDnnData() {
if (allocated_buffer_ != nullptr) {
cpu_allocator()->DeallocateRaw(allocated_buffer_);
}
cpu_engine_ = nullptr;  
delete (user_memory_);
delete (reorder_memory_);
delete (op_md_);
}

inline void* GetTensorBuffer(const Tensor* tensor) const {
CHECK_NOTNULL(tensor);
return const_cast<void*>(
static_cast<const void*>(tensor->flat<T>().data()));
}

void SetIs3DData(bool bIs3D_) { bIs3D = bIs3D_; }

bool GetIs3D() { return bIs3D; }

inline void SetUsrMem(const memory::dims& dim, memory::format fm,
void* data_buffer = nullptr) {
auto md = memory::desc(dim, MklDnnType<T>(), fm);
SetUsrMem(md, data_buffer);
}

inline void SetUsrMem(const memory::dims& dim, memory::format fm,
const Tensor* tensor) {
CHECK_NOTNULL(tensor);
SetUsrMem(dim, fm, GetTensorBuffer(tensor));
}

static inline memory::desc CreateBlockedMemDesc(const memory::dims& dim,
const memory::dims& strides) {
return CreateBlockedMemDescHelper(dim, strides, MklDnnType<T>());
}

inline void SetUsrMem(const memory::dims& dim, const memory::dims& strides,
void* data_buffer = nullptr) {
CHECK_EQ(dim.size(), strides.size());
auto blocked_md = MklDnnData<T>::CreateBlockedMemDesc(dim, strides);
SetUsrMem(blocked_md, data_buffer);
}

inline void SetUsrMem(const memory::dims& dim, const memory::dims& strides,
const Tensor* tensor) {
CHECK_NOTNULL(tensor);
SetUsrMem(dim, strides, GetTensorBuffer(tensor));
}

inline void SetUsrMem(const memory::desc& md, void* data_buffer = nullptr) {
auto pd = memory::primitive_desc(md, *cpu_engine_);
SetUsrMem(pd, data_buffer);
}

inline void SetUsrMem(const memory::desc& md, const Tensor* tensor) {
CHECK_NOTNULL(tensor);
SetUsrMem(md, GetTensorBuffer(tensor));
}

inline void SetUsrMem(const memory::primitive_desc& pd,
void* data_buffer = nullptr) {
CHECK_NOTNULL(cpu_engine_);
if (data_buffer) {
user_memory_ = new memory(pd, data_buffer);
} else {
user_memory_ = new memory(pd);
}
}

inline void SetUsrMem(const memory::primitive_desc& pd,
const Tensor* tensor) {
CHECK_NOTNULL(tensor);
SetUsrMem(pd, GetTensorBuffer(tensor));
}

inline const memory* GetUsrMem() const { return user_memory_; }

inline const memory::primitive_desc GetUsrMemPrimDesc() const {
CHECK_NOTNULL(user_memory_);
return user_memory_->get_primitive_desc();
}

inline memory::desc GetUsrMemDesc() {
const memory::primitive_desc pd = GetUsrMemPrimDesc();
return const_cast<memory::primitive_desc*>(&pd)->desc();
}

inline void* GetUsrMemDataHandle() const {
CHECK_NOTNULL(user_memory_);
return user_memory_->get_data_handle();
}

inline void SetUsrMemDataHandle(void* data_buffer) {
CHECK_NOTNULL(user_memory_);
CHECK_NOTNULL(data_buffer);
user_memory_->set_data_handle(data_buffer);
}

inline void SetUsrMemDataHandle(const Tensor* tensor) {
CHECK_NOTNULL(user_memory_);
CHECK_NOTNULL(tensor);
user_memory_->set_data_handle(GetTensorBuffer(tensor));
}

inline void AllocateBuffer(size_t size) {
const int64 kMemoryAlginment = 64;  
allocated_buffer_ = cpu_allocator()->AllocateRaw(kMemoryAlginment, size);
}

inline void* GetAllocatedBuffer() { return allocated_buffer_; }

inline const memory& GetOpMem() const {
return reorder_memory_ ? *reorder_memory_ : *user_memory_;
}

inline void SetOpMemDesc(const memory::dims& dim, memory::format fm) {
op_md_ = new memory::desc(dim, MklDnnType<T>(), fm);
}

inline const memory::desc& GetOpMemDesc() const { return *op_md_; }

inline bool IsReorderNeeded(const memory::primitive_desc& op_pd) const {
CHECK_NOTNULL(user_memory_);
return op_pd != user_memory_->get_primitive_desc();
}

inline bool IsReorderNeeded(const memory::format& target_format) const {
CHECK_NOTNULL(user_memory_);
return target_format !=
user_memory_->get_primitive_desc().desc().data.format;
}

inline primitive CreateReorder(const memory* from, const memory* to) const {
CHECK_NOTNULL(from);
CHECK_NOTNULL(to);
return reorder(*from, *to);
}

inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
std::vector<primitive>* net) {
CHECK_NOTNULL(net);
CHECK_NOTNULL(user_memory_);
if (IsReorderNeeded(op_pd)) {
reorder_memory_ = new memory(op_pd);
net->push_back(CreateReorder(user_memory_, reorder_memory_));
return true;
}
return false;
}

inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd) {
CHECK_NOTNULL(user_memory_);
if (IsReorderNeeded(op_pd)) {
reorder_memory_ = new memory(op_pd);
std::vector<primitive> net;
net.push_back(FindOrCreateReorder<T>(user_memory_, reorder_memory_));
stream(stream::kind::eager).submit(net).wait();
return true;
}
return false;
}

inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
void* reorder_data_handle,
std::vector<primitive>* net) {
CHECK_NOTNULL(net);
CHECK_NOTNULL(reorder_data_handle);
CHECK_NOTNULL(user_memory_);
if (IsReorderNeeded(op_pd)) {
reorder_memory_ = new memory(op_pd, reorder_data_handle);
net->push_back(CreateReorder(user_memory_, reorder_memory_));
return true;
}
return false;
}

inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
void* reorder_data_handle) {
CHECK_NOTNULL(reorder_data_handle);
CHECK_NOTNULL(user_memory_);
if (IsReorderNeeded(op_pd)) {
std::vector<primitive> net;
reorder_memory_ = new memory(op_pd, reorder_data_handle);
net.push_back(FindOrCreateReorder<T>(user_memory_, reorder_memory_));
stream(stream::kind::eager).submit(net).wait();
return true;
}
return false;
}

inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
Tensor* reorder_tensor,
std::vector<primitive>* net) {
CHECK_NOTNULL(net);
CHECK_NOTNULL(reorder_tensor);
return CheckReorderToOpMem(op_pd, GetTensorBuffer(reorder_tensor), net);
}

inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
Tensor* reorder_tensor) {
CHECK_NOTNULL(reorder_tensor);
return CheckReorderToOpMem(op_pd, GetTensorBuffer(reorder_tensor));
}

inline bool PrepareReorderToUserMemIfReq(
const memory::primitive_desc& op_pd) {
CHECK_NOTNULL(user_memory_);
if (IsReorderNeeded(op_pd)) {
reorder_memory_ = new memory(op_pd);
return true;
}
return false;
}

inline void InsertReorderToUserMem(std::vector<primitive>* net) {
CHECK_NOTNULL(net);
CHECK_NOTNULL(user_memory_);
CHECK_NOTNULL(reorder_memory_);
net->push_back(CreateReorder(reorder_memory_, user_memory_));
}

inline void InsertReorderToUserMem() {
CHECK_NOTNULL(user_memory_);
CHECK_NOTNULL(reorder_memory_);
std::vector<primitive> net;
net.push_back(FindOrCreateReorder<T>(reorder_memory_, user_memory_));
stream(stream::kind::eager).submit(net).wait();
}
};

class MklPrimitive {
public:
virtual ~MklPrimitive() {}

unsigned char* DummyData = nullptr;
};

const mkldnn::memory::dims NONE_DIMS = {};

template <typename T>
class MklPrimitiveFactory {
public:
MklPrimitiveFactory() {}

~MklPrimitiveFactory() {}

MklPrimitive* GetOp(const string& key) {
auto& map = MklPrimitiveFactory<T>::GetHashMap();
auto stream_iter = map.find(key);
if (stream_iter == map.end()) {
return nullptr;
} else {
CHECK(stream_iter->second != nullptr) << "nullptr present in map";
return stream_iter->second;
}
}

void SetOp(const string& key, MklPrimitive* op) {
auto& map = MklPrimitiveFactory<T>::GetHashMap();
auto stream_iter = map.find(key);

CHECK(stream_iter == map.end());

map[key] = op;
}

static inline bool IsLegacyPlatform() {
return (!port::TestCPUFeature(port::CPUFeature::AVX512F) &&
!port::TestCPUFeature(port::CPUFeature::AVX2));
}

static inline bool IsPrimitiveMemOptEnabled() {
bool is_primitive_mem_opt_enabled = true;
TF_CHECK_OK(ReadBoolFromEnvVar("TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE", true,
&is_primitive_mem_opt_enabled));
return is_primitive_mem_opt_enabled;
}

private:
static inline std::unordered_map<string, MklPrimitive*>& GetHashMap() {
static thread_local std::unordered_map<string, MklPrimitive*> map_;
return map_;
}
};

class FactoryKeyCreator {
public:
FactoryKeyCreator() { key_.reserve(kMaxKeyLength); }

~FactoryKeyCreator() {}

void AddAsKey(const string& str) { Append(str); }

void AddAsKey(const mkldnn::memory::dims& dims) {
for (unsigned int i = 0; i < dims.size(); i++) {
AddAsKey<int>(dims[i]);
}
}

template <typename T>
void AddAsKey(const T data) {
auto buffer = reinterpret_cast<const char*>(&data);
Append(StringPiece(buffer, sizeof(T)));
}

string GetKey() { return key_; }

private:
string key_;
const char delimiter = 'x';
const int kMaxKeyLength = 256;
void Append(StringPiece s) {
key_.append(string(s));
key_.append(1, delimiter);
}
};

static inline memory::format get_desired_format(int channel,
bool is_2d = true) {
memory::format fmt_desired = memory::format::any;

if (port::TestCPUFeature(port::CPUFeature::AVX512F)) {
fmt_desired = is_2d ? memory::format::nChw16c : memory::format::nCdhw16c;
} else if (port::TestCPUFeature(port::CPUFeature::AVX2) &&
(channel % 8) == 0) {
fmt_desired = is_2d ? memory::format::nChw8c
: memory::format::ncdhw;  
} else {
fmt_desired = is_2d ? memory::format::nchw : memory::format::ncdhw;
}
return fmt_desired;
}

class MklReorderPrimitive : public MklPrimitive {
public:
explicit MklReorderPrimitive(const memory* from, const memory* to) {
Setup(from, to);
}
~MklReorderPrimitive() {}

std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }

void SetMemory(const memory* from, const memory* to) {
context_.src_mem->set_data_handle(from->get_data_handle());
context_.dst_mem->set_data_handle(to->get_data_handle());
}

private:
struct ReorderContext {
std::shared_ptr<mkldnn::memory> src_mem;
std::shared_ptr<mkldnn::memory> dst_mem;
std::shared_ptr<primitive> reorder_prim;
ReorderContext()
: src_mem(nullptr), dst_mem(nullptr), reorder_prim(nullptr) {}
} context_;

engine cpu_engine_ = engine(engine::cpu, 0);

void Setup(const memory* from, const memory* to) {
context_.src_mem.reset(new memory(
{from->get_primitive_desc().desc(), cpu_engine_}, DummyData));
context_.dst_mem.reset(
new memory({to->get_primitive_desc().desc(), cpu_engine_}, DummyData));
context_.reorder_prim = std::make_shared<mkldnn::reorder>(
reorder(*context_.src_mem, *context_.dst_mem));
}
};

template <typename T>
class MklReorderPrimitiveFactory : public MklPrimitiveFactory<T> {
public:
static MklReorderPrimitive* Get(const memory* from, const memory* to) {
auto reorderPrim = static_cast<MklReorderPrimitive*>(
MklReorderPrimitiveFactory<T>::GetInstance().GetReorder(from, to));
if (reorderPrim == nullptr) {
reorderPrim = new MklReorderPrimitive(from, to);
MklReorderPrimitiveFactory<T>::GetInstance().SetReorder(from, to,
reorderPrim);
}
reorderPrim->SetMemory(from, to);
return reorderPrim;
}

static MklReorderPrimitiveFactory& GetInstance() {
static MklReorderPrimitiveFactory instance_;
return instance_;
}

private:
MklReorderPrimitiveFactory() {}
~MklReorderPrimitiveFactory() {}

static string CreateKey(const memory* from, const memory* to) {
string prefix = "reorder";
FactoryKeyCreator key_creator;
auto const& from_desc = from->get_primitive_desc().desc().data;
auto const& to_desc = to->get_primitive_desc().desc().data;
const int KIdxFirstStride = 0;
memory::dims from_dims(from_desc.dims, &from_desc.dims[from_desc.ndims]);
memory::dims to_dims(to_desc.dims, &to_desc.dims[to_desc.ndims]);
memory::dims from_strides(
from_desc.layout_desc.blocking.strides[KIdxFirstStride],
&from_desc.layout_desc.blocking
.strides[KIdxFirstStride][from_desc.ndims]);
memory::dims to_strides(
to_desc.layout_desc.blocking.strides[KIdxFirstStride],
&to_desc.layout_desc.blocking.strides[KIdxFirstStride][to_desc.ndims]);
key_creator.AddAsKey(prefix);
key_creator.AddAsKey(static_cast<int>(from_desc.format));
key_creator.AddAsKey(static_cast<int>(from_desc.data_type));
key_creator.AddAsKey(from_dims);
key_creator.AddAsKey(from_strides);
key_creator.AddAsKey(static_cast<int>(to_desc.format));
key_creator.AddAsKey(static_cast<int>(to_desc.data_type));
key_creator.AddAsKey(to_dims);
key_creator.AddAsKey(to_strides);
return key_creator.GetKey();
}

MklPrimitive* GetReorder(const memory* from, const memory* to) {
string key = CreateKey(from, to);
return this->GetOp(key);
}

void SetReorder(const memory* from, const memory* to, MklPrimitive* op) {
string key = CreateKey(from, to);
this->SetOp(key, op);
}
};

template <typename T>
inline primitive FindOrCreateReorder(const memory* from, const memory* to) {
CHECK_NOTNULL(from);
CHECK_NOTNULL(to);
MklReorderPrimitive* reorder_prim =
MklReorderPrimitiveFactory<T>::Get(from, to);
return *reorder_prim->GetPrimitive();
}

inline bool IsConv1x1StrideNot1(memory::dims filter_dims,
memory::dims strides) {
if (filter_dims.size() != 4 || strides.size() != 2) return false;

return ((filter_dims[2] == 1) && (filter_dims[3] == 1) &&
((strides[0] != 1) || (strides[1] != 1)));
}

#endif  

}  
#endif  
#endif  
