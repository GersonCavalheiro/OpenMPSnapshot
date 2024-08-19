

#ifndef GKO_DPCPP_COMPONENTS_WARP_BLAS_DP_HPP_
#define GKO_DPCPP_COMPONENTS_WARP_BLAS_DP_HPP_


#include <cassert>
#include <type_traits>


#include <CL/sycl.hpp>


#include <ginkgo/config.hpp>


#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {



enum postprocess_transformation { and_return, and_transpose };



template <
int max_problem_size, typename Group, typename ValueType,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ void apply_gauss_jordan_transform(
const Group& __restrict__ group, int32 key_row, int32 key_col,
ValueType* __restrict__ row, bool& __restrict__ status)
{
auto key_col_elem = group.shfl(row[key_col], key_row);
if (key_col_elem == zero<ValueType>()) {
status = false;
return;
}
if (group.thread_rank() == key_row) {
key_col_elem = one<ValueType>() / key_col_elem;
} else {
key_col_elem = -row[key_col] / key_col_elem;
}
#pragma unroll
for (int32 i = 0; i < max_problem_size; ++i) {
const auto key_row_elem = group.shfl(row[i], key_row);
if (group.thread_rank() == key_row) {
row[i] = zero<ValueType>();
}
row[i] += key_col_elem * key_row_elem;
}
row[key_col] = key_col_elem;
}



template <
int max_problem_size, typename Group, typename ValueType,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ void apply_gauss_jordan_transform_with_rhs(
const Group& __restrict__ group, int32 key_row, int32 key_col,
ValueType* __restrict__ row, ValueType* __restrict__ rhs,
bool& __restrict__ status)
{
auto key_col_elem = group.shfl(row[key_col], key_row);
auto key_rhs_elem = group.shfl(rhs[0], key_row);
if (key_col_elem == zero<ValueType>()) {
status = false;
return;
}
if (group.thread_rank() == key_row) {
key_col_elem = one<ValueType>() / key_col_elem;
rhs[0] = key_rhs_elem * key_col_elem;
} else {
key_col_elem = -row[key_col] / key_col_elem;
rhs[0] += key_rhs_elem * key_col_elem;
}
#pragma unroll
for (int32 i = 0; i < max_problem_size; ++i) {
const auto key_row_elem = group.shfl(row[i], key_row);
if (group.thread_rank() == key_row) {
row[i] = zero<ValueType>();
}
row[i] += key_col_elem * key_row_elem;
}
row[key_col] = key_col_elem;
}



template <
int max_problem_size, typename Group, typename ValueType,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ bool invert_block(const Group& __restrict__ group,
uint32 problem_size,
ValueType* __restrict__ row,
uint32& __restrict__ perm,
uint32& __restrict__ trans_perm)
{
GKO_ASSERT(problem_size <= max_problem_size);
auto pivoted = group.thread_rank() >= problem_size;
auto status = true;
#ifdef GINKGO_JACOBI_FULL_OPTIMIZATIONS
#pragma unroll
#else
#pragma unroll 1
#endif
for (int32 i = 0; i < max_problem_size; ++i) {
if (i < problem_size) {
const auto piv = choose_pivot(group, row[i], pivoted);
if (group.thread_rank() == piv) {
perm = i;
pivoted = true;
}
if (group.thread_rank() == i) {
trans_perm = piv;
}
apply_gauss_jordan_transform<max_problem_size>(group, piv, i, row,
status);
}
}
return status;
}



template <postprocess_transformation mod, typename T1, typename T2, typename T3>
__dpct_inline__ auto get_row_major_index(T1 row, T2 col, T3 stride) ->
typename std::enable_if<
mod != and_transpose,
typename std::decay<decltype(row * stride + col)>::type>::type
{
return row * stride + col;
}


template <postprocess_transformation mod, typename T1, typename T2, typename T3>
__dpct_inline__ auto get_row_major_index(T1 row, T2 col, T3 stride) ->
typename std::enable_if<
mod == and_transpose,
typename std::decay<decltype(col * stride + row)>::type>::type
{
return col * stride + row;
}



template <
int max_problem_size, postprocess_transformation mod = and_return,
typename Group, typename SourceValueType, typename ResultValueType,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ void copy_matrix(const Group& __restrict__ group,
uint32 problem_size,
const SourceValueType* __restrict__ source_row,
uint32 increment, uint32 row_perm,
uint32 col_perm,
ResultValueType* __restrict__ destination,
size_type stride)
{
GKO_ASSERT(problem_size <= max_problem_size);
#pragma unroll
for (int32 i = 0; i < max_problem_size; ++i) {
if (i < problem_size) {
const auto idx = group.shfl(col_perm, i);
if (group.thread_rank() < problem_size) {
const auto val = source_row[i * increment];
destination[get_row_major_index<mod>(idx, row_perm, stride)] =
static_cast<ResultValueType>(val);
}
}
}
}



template <
int max_problem_size, typename Group, typename MatrixValueType,
typename VectorValueType,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ void multiply_transposed_vec(
const Group& __restrict__ group, uint32 problem_size,
const VectorValueType& __restrict__ vec,
const MatrixValueType* __restrict__ mtx_row, uint32 mtx_increment,
VectorValueType* __restrict__ res, uint32 res_increment)
{
GKO_ASSERT(problem_size <= max_problem_size);
auto mtx_elem = zero<VectorValueType>();
#pragma unroll
for (int32 i = 0; i < max_problem_size; ++i) {
if (i < problem_size) {
if (group.thread_rank() < problem_size) {
mtx_elem =
static_cast<VectorValueType>(mtx_row[i * mtx_increment]);
}
const auto out = ::gko::kernels::dpcpp::reduce(
group, mtx_elem * vec,
[](VectorValueType x, VectorValueType y) { return x + y; });
if (group.thread_rank() == 0) {
res[i * res_increment] = out;
}
}
}
}



template <
int max_problem_size, typename Group, typename MatrixValueType,
typename VectorValueType, typename Closure,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ void multiply_vec(const Group& __restrict__ group,
uint32 problem_size,
const VectorValueType& __restrict__ vec,
const MatrixValueType* __restrict__ mtx_row,
uint32 mtx_increment,
VectorValueType* __restrict__ res,
uint32 res_increment, Closure closure_op)
{
GKO_ASSERT(problem_size <= max_problem_size);
auto mtx_elem = zero<VectorValueType>();
auto out = zero<VectorValueType>();
#pragma unroll
for (int32 i = 0; i < max_problem_size; ++i) {
if (i < problem_size) {
if (group.thread_rank() < problem_size) {
mtx_elem =
static_cast<VectorValueType>(mtx_row[i * mtx_increment]);
}
out += mtx_elem * group.shfl(vec, i);
}
}
if (group.thread_rank() < problem_size) {
closure_op(res[group.thread_rank() * res_increment], out);
}
}



template <
int max_problem_size, typename Group, typename ValueType,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ remove_complex<ValueType> compute_infinity_norm(
const Group& group, uint32 num_rows, uint32 num_cols, const ValueType* row)
{
using result_type = remove_complex<ValueType>;
auto sum = zero<result_type>();
if (group.thread_rank() < num_rows) {
#ifdef GINKGO_JACOBI_FULL_OPTIMIZATIONS
#pragma unroll
#else
#pragma unroll 1
#endif
for (uint32 i = 0; i < max_problem_size; ++i) {
if (i < num_cols) {
sum += abs(row[i]);
}
}
}
return ::gko::kernels::dpcpp::reduce(
group, sum, [](result_type x, result_type y) { return max(x, y); });
}


}  
}  
}  


#endif  
