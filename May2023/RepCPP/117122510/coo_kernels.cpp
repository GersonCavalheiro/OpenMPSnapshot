

#include "core/matrix/coo_kernels.hpp"


#include <array>


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {

namespace omp {

namespace coo {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
const matrix::Coo<ValueType, IndexType>* a,
const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
dense::fill(exec, c, zero<ValueType>());
spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
const matrix::Dense<ValueType>* alpha,
const matrix::Coo<ValueType, IndexType>* a,
const matrix::Dense<ValueType>* b,
const matrix::Dense<ValueType>* beta,
matrix::Dense<ValueType>* c)
{
dense::scale(exec, beta, c);
advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <int block_size, typename ValueType, typename IndexType>
void spmv2_blocked(std::shared_ptr<const OmpExecutor> exec,
const matrix::Coo<ValueType, IndexType>* a,
const matrix::Dense<ValueType>* b,
matrix::Dense<ValueType>* c, ValueType scale)
{
GKO_ASSERT(b->get_size()[1] > block_size);
const auto coo_val = a->get_const_values();
const auto coo_col = a->get_const_col_idxs();
const auto coo_row = a->get_const_row_idxs();
const auto num_rhs = b->get_size()[1];
const auto rounded_rhs = num_rhs / block_size * block_size;
const auto sentinel_row = a->get_size()[0] + 1;
const auto nnz = a->get_num_stored_elements();

#pragma omp parallel
{
const auto num_threads = omp_get_num_threads();
const auto work_per_thread =
static_cast<size_type>(ceildiv(nnz, num_threads));
const auto thread_id = static_cast<size_type>(omp_get_thread_num());
const auto begin = work_per_thread * thread_id;
const auto end = std::min(begin + work_per_thread, nnz);
if (begin < end) {
const auto first = begin > 0 ? coo_row[begin - 1] : sentinel_row;
const auto last = end < nnz ? coo_row[end] : sentinel_row;
auto nz = begin;
std::array<ValueType, block_size> partial_sum;
if (first != sentinel_row) {
for (size_type rhs_base = 0; rhs_base < rounded_rhs;
rhs_base += block_size) {
partial_sum.fill(zero<ValueType>());
for (auto local_nz = nz;
local_nz < end && coo_row[local_nz] == first;
local_nz++) {
const auto col = coo_col[local_nz];
#pragma unroll
for (size_type i = 0; i < block_size; i++) {
const auto rhs = i + rhs_base;
partial_sum[i] +=
scale * coo_val[local_nz] * b->at(col, rhs);
}
}
#pragma unroll
for (size_type i = 0; i < block_size; i++) {
const auto rhs = i + rhs_base;
atomic_add(c->at(first, rhs), partial_sum[i]);
}
}
partial_sum.fill(zero<ValueType>());
for (; nz < end && coo_row[nz] == first; nz++) {
const auto row = first;
const auto col = coo_col[nz];
for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
partial_sum[rhs - rounded_rhs] +=
scale * coo_val[nz] * b->at(col, rhs);
}
}
for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
atomic_add(c->at(first, rhs),
partial_sum[rhs - rounded_rhs]);
}
}
for (; nz < end && coo_row[nz] != last; nz++) {
const auto row = coo_row[nz];
const auto col = coo_col[nz];
for (size_type rhs_base = 0; rhs_base < rounded_rhs;
rhs_base += block_size) {
#pragma unroll
for (size_type i = 0; i < block_size; i++) {
const auto rhs = i + rhs_base;
c->at(row, rhs) +=
scale * coo_val[nz] * b->at(col, rhs);
}
}
for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
c->at(row, rhs) += scale * coo_val[nz] * b->at(col, rhs);
}
}
if (last != sentinel_row) {
for (size_type rhs_base = 0; rhs_base < rounded_rhs;
rhs_base += block_size) {
partial_sum.fill(zero<ValueType>());
for (auto local_nz = nz; local_nz < end; local_nz++) {
const auto col = coo_col[local_nz];
#pragma unroll
for (size_type i = 0; i < block_size; i++) {
const auto rhs = i + rhs_base;
partial_sum[i] +=
scale * coo_val[local_nz] * b->at(col, rhs);
}
}
#pragma unroll
for (size_type i = 0; i < block_size; i++) {
const auto rhs = i + rhs_base;
const auto row = last;
atomic_add(c->at(row, rhs), partial_sum[i]);
}
}
partial_sum.fill(zero<ValueType>());
for (; nz < end; nz++) {
const auto col = coo_col[nz];
for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
partial_sum[rhs - rounded_rhs] +=
scale * coo_val[nz] * b->at(col, rhs);
}
}
for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
const auto row = last;
atomic_add(c->at(row, rhs), partial_sum[rhs - rounded_rhs]);
}
}
}
}
}


template <int num_rhs, typename ValueType, typename IndexType>
void spmv2_small_rhs(std::shared_ptr<const OmpExecutor> exec,
const matrix::Coo<ValueType, IndexType>* a,
const matrix::Dense<ValueType>* b,
matrix::Dense<ValueType>* c, ValueType scale)
{
GKO_ASSERT(b->get_size()[1] == num_rhs);
const auto coo_val = a->get_const_values();
const auto coo_col = a->get_const_col_idxs();
const auto coo_row = a->get_const_row_idxs();
const auto sentinel_row = a->get_size()[0] + 1;
const auto nnz = a->get_num_stored_elements();

#pragma omp parallel
{
const auto num_threads = omp_get_num_threads();
const auto work_per_thread =
static_cast<size_type>(ceildiv(nnz, num_threads));
const auto thread_id = static_cast<size_type>(omp_get_thread_num());
const auto begin = work_per_thread * thread_id;
const auto end = std::min(begin + work_per_thread, nnz);
if (begin < end) {
const auto first = begin > 0 ? coo_row[begin - 1] : sentinel_row;
const auto last = end < nnz ? coo_row[end] : sentinel_row;
auto nz = begin;
std::array<ValueType, num_rhs> partial_sum;
if (first != sentinel_row) {
partial_sum.fill(zero<ValueType>());
for (; nz < end && coo_row[nz] == first; nz++) {
const auto col = coo_col[nz];
#pragma unroll
for (size_type rhs = 0; rhs < num_rhs; rhs++) {
partial_sum[rhs] +=
scale * coo_val[nz] * b->at(col, rhs);
}
}
#pragma unroll
for (size_type rhs = 0; rhs < num_rhs; rhs++) {
atomic_add(c->at(first, rhs), partial_sum[rhs]);
}
}
for (; nz < end && coo_row[nz] != last; nz++) {
const auto row = coo_row[nz];
const auto col = coo_col[nz];
#pragma unroll
for (size_type rhs = 0; rhs < num_rhs; rhs++) {
c->at(row, rhs) += scale * coo_val[nz] * b->at(col, rhs);
}
}
if (last != sentinel_row) {
partial_sum.fill(zero<ValueType>());
for (; nz < end; nz++) {
const auto col = coo_col[nz];
#pragma unroll
for (size_type rhs = 0; rhs < num_rhs; rhs++) {
partial_sum[rhs] +=
scale * coo_val[nz] * b->at(col, rhs);
}
}
#pragma unroll
for (size_type rhs = 0; rhs < num_rhs; rhs++) {
const auto row = last;
atomic_add(c->at(row, rhs), partial_sum[rhs]);
}
}
}
}
}


template <typename ValueType, typename IndexType>
void generic_spmv2(std::shared_ptr<const OmpExecutor> exec,
const matrix::Coo<ValueType, IndexType>* a,
const matrix::Dense<ValueType>* b,
matrix::Dense<ValueType>* c, ValueType scale)
{
const auto num_rhs = b->get_size()[1];
if (num_rhs <= 0) {
return;
}
if (num_rhs == 1) {
spmv2_small_rhs<1>(exec, a, b, c, scale);
return;
}
if (num_rhs == 2) {
spmv2_small_rhs<2>(exec, a, b, c, scale);
return;
}
if (num_rhs == 3) {
spmv2_small_rhs<3>(exec, a, b, c, scale);
return;
}
if (num_rhs == 4) {
spmv2_small_rhs<4>(exec, a, b, c, scale);
return;
}
spmv2_blocked<4>(exec, a, b, c, scale);
}


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const OmpExecutor> exec,
const matrix::Coo<ValueType, IndexType>* a,
const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
generic_spmv2(exec, a, b, c, one<ValueType>());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const OmpExecutor> exec,
const matrix::Dense<ValueType>* alpha,
const matrix::Coo<ValueType, IndexType>* a,
const matrix::Dense<ValueType>* b,
matrix::Dense<ValueType>* c)
{
generic_spmv2(exec, a, b, c, alpha->at(0, 0));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


}  
}  
}  
}  
