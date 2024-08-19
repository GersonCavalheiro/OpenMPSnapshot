

#include "host_matrix_csr.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../matrix_formats_ind.hpp"
#include "host_conversion.hpp"
#include "host_io.hpp"
#include "host_matrix_bcsr.hpp"
#include "host_matrix_coo.hpp"
#include "host_matrix_dense.hpp"
#include "host_matrix_dia.hpp"
#include "host_matrix_ell.hpp"
#include "host_matrix_hyb.hpp"
#include "host_matrix_mcsr.hpp"
#include "host_vector.hpp"
#include "rocalution/utils/types.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <map>
#include <math.h>
#include <numeric>
#include <string.h>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_set_nested(num) ;
#endif

namespace rocalution
{

template <typename ValueType>
HostMatrixCSR<ValueType>::HostMatrixCSR()
{
LOG_INFO("no default constructor");
FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HostMatrixCSR<ValueType>::HostMatrixCSR(const Rocalution_Backend_Descriptor& local_backend)
{
log_debug(this, "HostMatrixCSR::HostMatrixCSR()", "constructor with local_backend");

this->mat_.row_offset = NULL;
this->mat_.col        = NULL;
this->mat_.val        = NULL;
this->set_backend(local_backend);

this->L_diag_unit_ = false;
this->U_diag_unit_ = false;
}

template <typename ValueType>
HostMatrixCSR<ValueType>::~HostMatrixCSR()
{
log_debug(this, "HostMatrixCSR::~HostMatrixCSR()", "destructor");

this->Clear();
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::Clear(void)
{
free_host(&this->mat_.row_offset);
free_host(&this->mat_.col);
free_host(&this->mat_.val);

this->nrow_ = 0;
this->ncol_ = 0;
this->nnz_  = 0;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Zeros(void)
{
set_to_zero_host(this->nnz_, mat_.val);

return true;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::Info(void) const
{
LOG_INFO(
"HostMatrixCSR<ValueType>, OpenMP threads: " << this->local_backend_.OpenMP_threads);
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Check(void) const
{
bool sorted = true;

if(this->nnz_ > 0)
{
assert(this->nrow_ > 0);
assert(this->ncol_ > 0);

assert(this->mat_.row_offset != NULL);
assert(this->mat_.val != NULL);
assert(this->mat_.col != NULL);

if((std::abs(this->nnz_) == std::numeric_limits<int64_t>::infinity()) || 
(this->nnz_ != this->nnz_))
{ 
LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix nnz");
return false;
}

if((std::abs(this->nrow_) == std::numeric_limits<int>::infinity()) || 
(this->nrow_ != this->nrow_))
{ 
LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix nrow");
return false;
}

if((std::abs(this->ncol_) == std::numeric_limits<int>::infinity()) || 
(this->ncol_ != this->ncol_))
{ 
LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix ncol");
return false;
}

for(int ai = 0; ai < this->nrow_ + 1; ++ai)
{
PtrType row = this->mat_.row_offset[ai];
if((row < 0) || (row > this->nnz_))
{
LOG_VERBOSE_INFO(
2,
"*** error: Matrix CSR:Check - problems with matrix row offset pointers");
return false;
}
}

for(int ai = 0; ai < this->nrow_; ++ai)
{
int s = this->mat_.col[this->mat_.row_offset[ai]];

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
++aj)
{
int col      = this->mat_.col[aj];
int prev_col = (aj > this->mat_.row_offset[ai]) ? this->mat_.col[aj - 1] : -1;

if((col < 0) || (col > this->ncol_))
{
LOG_VERBOSE_INFO(
2, "*** error: Matrix CSR:Check - problems with matrix col values");
return false;
}

if(col == prev_col)
{
LOG_VERBOSE_INFO(2,
"*** error: Matrix CSR:Check - problems with matrix col "
"values - the matrix has duplicated column entries");
return false;
}

ValueType val = this->mat_.val[aj];
if((val == std::numeric_limits<ValueType>::infinity()) || (val != val))
{
LOG_VERBOSE_INFO(
2, "*** error: Matrix CSR:Check - problems with matrix values");
return false;
}

if((aj > this->mat_.row_offset[ai]) && (s >= col))
{
sorted = false;
}

s = this->mat_.col[aj];
}
}
}
else
{
assert(this->nnz_ == 0);
assert(this->nrow_ >= 0);
assert(this->ncol_ >= 0);

if(this->nrow_ == 0 && this->ncol_ == 0)
{
assert(this->mat_.val == NULL);
assert(this->mat_.col == NULL);
}
}

if(sorted == false)
{
LOG_VERBOSE_INFO(2,
"*** warning: Matrix CSR:Check - the matrix has not sorted columns");
}

return true;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::AllocateCSR(int64_t nnz, int nrow, int ncol)
{
assert(nnz >= 0);
assert(ncol >= 0);
assert(nrow >= 0);

this->Clear();

allocate_host(nrow + 1, &this->mat_.row_offset);
allocate_host(nnz, &this->mat_.col);
allocate_host(nnz, &this->mat_.val);

set_to_zero_host(nrow + 1, mat_.row_offset);
set_to_zero_host(nnz, mat_.col);
set_to_zero_host(nnz, mat_.val);

this->nrow_ = nrow;
this->ncol_ = ncol;
this->nnz_  = nnz;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::SetDataPtrCSR(
PtrType** row_offset, int** col, ValueType** val, int64_t nnz, int nrow, int ncol)
{
assert(nnz >= 0);
assert(nrow >= 0);
assert(ncol >= 0);
assert(*row_offset != NULL);

if(nnz > 0)
{
assert(*col != NULL);
assert(*val != NULL);
}

this->Clear();

this->nrow_ = nrow;
this->ncol_ = ncol;
this->nnz_  = nnz;

this->mat_.row_offset = *row_offset;
this->mat_.col        = *col;
this->mat_.val        = *val;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::LeaveDataPtrCSR(PtrType** row_offset, int** col, ValueType** val)
{
assert(this->nrow_ >= 0);
assert(this->ncol_ >= 0);
assert(this->nnz_ >= 0);

*row_offset = this->mat_.row_offset;
*col        = this->mat_.col;
*val        = this->mat_.val;

this->mat_.row_offset = NULL;
this->mat_.col        = NULL;
this->mat_.val        = NULL;

this->nrow_ = 0;
this->ncol_ = 0;
this->nnz_  = 0;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::CopyFromCSR(const PtrType*   row_offsets,
const int*       col,
const ValueType* val)
{
assert(row_offsets != NULL);

copy_h2h(this->nrow_ + 1, row_offsets, this->mat_.row_offset);

if(this->nnz_ > 0)
{
assert(this->nrow_ > 0);
assert(this->ncol_ > 0);
assert(col != NULL);
assert(val != NULL);

copy_h2h(this->nnz_, col, this->mat_.col);
copy_h2h(this->nnz_, val, this->mat_.val);
}
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::CopyToCSR(PtrType* row_offsets, int* col, ValueType* val) const
{
assert(row_offsets != NULL);

copy_h2h(this->nrow_ + 1, this->mat_.row_offset, row_offsets);

if(this->nnz_ > 0)
{
assert(this->nrow_ > 0);
assert(this->ncol_ > 0);
assert(col != NULL);
assert(val != NULL);

copy_h2h(this->nnz_, this->mat_.col, col);
copy_h2h(this->nnz_, this->mat_.val, val);
}
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
{
assert(this->GetMatFormat() == mat.GetMatFormat());
assert(this->GetMatBlockDimension() == mat.GetMatBlockDimension());

if(const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
{
if(this->nnz_ == 0)
{
this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
}

assert(this->nnz_ == cast_mat->nnz_);
assert(this->nrow_ == cast_mat->nrow_);
assert(this->ncol_ == cast_mat->ncol_);

if(cast_mat->mat_.row_offset != NULL)
{
copy_h2h(this->nrow_ + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
}

copy_h2h(this->nnz_, cast_mat->mat_.col, this->mat_.col);
copy_h2h(this->nnz_, cast_mat->mat_.val, this->mat_.val);
}
else
{
mat.CopyTo(this);
}
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
{
mat->CopyFrom(*this);
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ReadFileCSR(const std::string& filename)
{
int64_t nrow;
int64_t ncol;
int64_t nnz;

PtrType*   ptr = NULL;
int*       col = NULL;
ValueType* val = NULL;

if(read_matrix_csr(nrow, ncol, nnz, &ptr, &col, &val, filename.c_str()) != true)
{
return false;
}

assert(nrow <= std::numeric_limits<int>::max());
assert(ncol <= std::numeric_limits<int>::max());

this->Clear();
this->SetDataPtrCSR(&ptr, &col, &val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

return true;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::CopyFromHostCSR(const PtrType*   row_offset,
const int*       col,
const ValueType* val,
int64_t          nnz,
int              nrow,
int              ncol)
{
assert(nnz >= 0);
assert(ncol >= 0);
assert(nrow >= 0);
assert(row_offset != NULL);

this->Clear();

this->nrow_ = nrow;
this->ncol_ = ncol;
this->nnz_  = nnz;

allocate_host(nrow + 1, &this->mat_.row_offset);

copy_h2h(this->nrow_ + 1, row_offset, this->mat_.row_offset);

if(nnz > 0)
{
assert(col != NULL);
assert(val != NULL);
}

allocate_host(nnz, &this->mat_.col);
allocate_host(nnz, &this->mat_.val);

copy_h2h(this->nnz_, col, this->mat_.col);
copy_h2h(this->nnz_, val, this->mat_.val);
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::WriteFileCSR(const std::string& filename) const
{
if(write_matrix_csr(this->nrow_,
this->ncol_,
this->nnz_,
this->mat_.row_offset,
this->mat_.col,
this->mat_.val,
filename.c_str())
!= true)
{
return false;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
{
this->Clear();

if(mat.GetNnz() == 0)
{
this->AllocateCSR(mat.GetNnz(), mat.GetM(), mat.GetN());

return true;
}

if(const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
{
this->CopyFrom(*cast_mat);
return true;
}

if(const HostMatrixBCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
{
this->Clear();

int     nrow = cast_mat->mat_.nrowb * cast_mat->mat_.blockdim;
int     ncol = cast_mat->mat_.ncolb * cast_mat->mat_.blockdim;
int64_t nnz  = cast_mat->mat_.nnzb * cast_mat->mat_.blockdim * cast_mat->mat_.blockdim;

if(bcsr_to_csr(this->local_backend_.OpenMP_threads,
nnz,
nrow,
ncol,
cast_mat->mat_,
&this->mat_)
== true)
{
this->nrow_ = nrow;
this->ncol_ = ncol;
this->nnz_  = nnz;

return true;
}
}

if(const HostMatrixCOO<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCOO<ValueType>*>(&mat))
{
this->Clear();

if(coo_to_csr(this->local_backend_.OpenMP_threads,
cast_mat->nnz_,
cast_mat->nrow_,
cast_mat->ncol_,
cast_mat->mat_,
&this->mat_)
== true)
{
this->nrow_ = cast_mat->nrow_;
this->ncol_ = cast_mat->ncol_;
this->nnz_  = cast_mat->nnz_;

return true;
}
}

if(const HostMatrixDENSE<ValueType>* cast_mat
= dynamic_cast<const HostMatrixDENSE<ValueType>*>(&mat))
{
this->Clear();
int64_t nnz = 0;

if(dense_to_csr(this->local_backend_.OpenMP_threads,
cast_mat->nrow_,
cast_mat->ncol_,
cast_mat->mat_,
&this->mat_,
&nnz)
== true)
{
this->nrow_ = cast_mat->nrow_;
this->ncol_ = cast_mat->ncol_;
this->nnz_  = nnz;

return true;
}
}

if(const HostMatrixDIA<ValueType>* cast_mat
= dynamic_cast<const HostMatrixDIA<ValueType>*>(&mat))
{
this->Clear();
int64_t nnz;

if(dia_to_csr(this->local_backend_.OpenMP_threads,
cast_mat->nnz_,
cast_mat->nrow_,
cast_mat->ncol_,
cast_mat->mat_,
&this->mat_,
&nnz)
== true)
{
this->nrow_ = cast_mat->nrow_;
this->ncol_ = cast_mat->ncol_;
this->nnz_  = nnz;

return true;
}
}

if(const HostMatrixELL<ValueType>* cast_mat
= dynamic_cast<const HostMatrixELL<ValueType>*>(&mat))
{
this->Clear();
int64_t nnz;

if(ell_to_csr(this->local_backend_.OpenMP_threads,
cast_mat->nnz_,
cast_mat->nrow_,
cast_mat->ncol_,
cast_mat->mat_,
&this->mat_,
&nnz)
== true)
{
this->nrow_ = cast_mat->nrow_;
this->ncol_ = cast_mat->ncol_;
this->nnz_  = nnz;

return true;
}
}

if(const HostMatrixMCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixMCSR<ValueType>*>(&mat))
{
this->Clear();

if(mcsr_to_csr(this->local_backend_.OpenMP_threads,
cast_mat->nnz_,
cast_mat->nrow_,
cast_mat->ncol_,
cast_mat->mat_,
&this->mat_)
== true)
{
this->nrow_ = cast_mat->nrow_;
this->ncol_ = cast_mat->ncol_;
this->nnz_  = cast_mat->nnz_;

return true;
}
}

if(const HostMatrixHYB<ValueType>* cast_mat
= dynamic_cast<const HostMatrixHYB<ValueType>*>(&mat))
{
this->Clear();
int64_t nnz;

if(hyb_to_csr(this->local_backend_.OpenMP_threads,
cast_mat->nnz_,
cast_mat->nrow_,
cast_mat->ncol_,
cast_mat->ell_nnz_,
cast_mat->coo_nnz_,
cast_mat->mat_,
&this->mat_,
&nnz)
== true)
{
this->nrow_ = cast_mat->nrow_;
this->ncol_ = cast_mat->ncol_;
this->nnz_  = nnz;

return true;
}
}

return false;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
BaseVector<ValueType>*       out) const
{
assert(in.GetSize() >= 0);
assert(out->GetSize() >= 0);
assert(in.GetSize() == this->ncol_);
assert(out->GetSize() == this->nrow_);

const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

assert(cast_in != NULL);
assert(cast_out != NULL);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
ValueType sum     = static_cast<ValueType>(0);
PtrType   row_beg = this->mat_.row_offset[ai];
PtrType   row_end = this->mat_.row_offset[ai + 1];

for(PtrType aj = row_beg; aj < row_end; ++aj)
{
sum += this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
}

cast_out->vec_[ai] = sum;
}
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
ValueType                    scalar,
BaseVector<ValueType>*       out) const
{
if(this->nnz_ > 0)
{
assert(in.GetSize() >= 0);
assert(out->GetSize() >= 0);
assert(in.GetSize() == this->ncol_);
assert(out->GetSize() == this->nrow_);

const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

assert(cast_in != NULL);
assert(cast_out != NULL);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
++aj)
{
cast_out->vec_[ai]
+= scalar * this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
}
}
}
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractDiagonal(BaseVector<ValueType>* vec_diag) const
{
assert(vec_diag != NULL);
assert(vec_diag->GetSize() == this->nrow_);

HostVector<ValueType>* cast_vec_diag = dynamic_cast<HostVector<ValueType>*>(vec_diag);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai == this->mat_.col[aj])
{
cast_vec_diag->vec_[ai] = this->mat_.val[aj];
break;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const
{
assert(vec_inv_diag != NULL);
assert(vec_inv_diag->GetSize() == this->nrow_);

HostVector<ValueType>* cast_vec_inv_diag
= dynamic_cast<HostVector<ValueType>*>(vec_inv_diag);

int detect_zero_diag = 0;

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai == this->mat_.col[aj])
{
if(this->mat_.val[aj] != static_cast<ValueType>(0))
{
cast_vec_inv_diag->vec_[ai]
= static_cast<ValueType>(1) / this->mat_.val[aj];
}
else
{
cast_vec_inv_diag->vec_[ai] = static_cast<ValueType>(1);
detect_zero_diag            = 1;
}

break;
}
}
}

if(detect_zero_diag == 1)
{
LOG_VERBOSE_INFO(
2,
"*** warning: in HostMatrixCSR::ExtractInverseDiagonal() a zero has been detected "
"on the diagonal. It has been replaced with one to avoid inf");
}
return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractSubMatrix(int                    row_offset,
int                    col_offset,
int                    row_size,
int                    col_size,
BaseMatrix<ValueType>* mat) const
{
assert(mat != NULL);

assert(row_offset >= 0);
assert(col_offset >= 0);

assert(this->nrow_ >= 0);
assert(this->ncol_ >= 0);

HostMatrixCSR<ValueType>* cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*>(mat);
assert(cast_mat != NULL);

int64_t mat_nnz = 0;



for(int ai = row_offset; ai < row_offset + row_size; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if((this->mat_.col[aj] >= col_offset)
&& (this->mat_.col[aj] < col_offset + col_size))
{
++mat_nnz;
}
}
}

cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

if(mat_nnz > 0)
{
PtrType mat_row_offset       = 0;
cast_mat->mat_.row_offset[0] = mat_row_offset;

for(int ai = row_offset; ai < row_offset + row_size; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
++aj)
{
if((this->mat_.col[aj] >= col_offset)
&& (this->mat_.col[aj] < col_offset + col_size))
{
cast_mat->mat_.col[mat_row_offset] = this->mat_.col[aj] - col_offset;
cast_mat->mat_.val[mat_row_offset] = this->mat_.val[aj];
++mat_row_offset;
}
}

cast_mat->mat_.row_offset[ai - row_offset + 1] = mat_row_offset;
}

cast_mat->mat_.row_offset[row_size] = mat_row_offset;
assert(mat_row_offset == mat_nnz);
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractU(BaseMatrix<ValueType>* U) const
{
assert(U != NULL);

assert(this->nrow_ > 0);
assert(this->ncol_ > 0);

HostMatrixCSR<ValueType>* cast_U = dynamic_cast<HostMatrixCSR<ValueType>*>(U);

assert(cast_U != NULL);

int64_t nnz_U = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_U)
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] > ai)
{
++nnz_U;
}
}
}

PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

allocate_host(this->nrow_ + 1, &row_offset);
allocate_host(nnz_U, &col);
allocate_host(nnz_U, &val);

PtrType nnz   = 0;
row_offset[0] = 0;
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] > ai)
{
col[nnz] = this->mat_.col[aj];
val[nnz] = this->mat_.val[aj];
++nnz;
}
}

row_offset[ai + 1] = nnz;
}

cast_U->Clear();
cast_U->SetDataPtrCSR(&row_offset, &col, &val, nnz_U, this->nrow_, this->ncol_);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType>* U) const
{
assert(U != NULL);

assert(this->nrow_ > 0);
assert(this->ncol_ > 0);

HostMatrixCSR<ValueType>* cast_U = dynamic_cast<HostMatrixCSR<ValueType>*>(U);

assert(cast_U != NULL);

int64_t nnz_U = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_U)
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] >= ai)
{
++nnz_U;
}
}
}

PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

allocate_host(this->nrow_ + 1, &row_offset);
allocate_host(nnz_U, &col);
allocate_host(nnz_U, &val);

PtrType nnz   = 0;
row_offset[0] = 0;
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] >= ai)
{
col[nnz] = this->mat_.col[aj];
val[nnz] = this->mat_.val[aj];
++nnz;
}
}

row_offset[ai + 1] = nnz;
}

cast_U->Clear();
cast_U->SetDataPtrCSR(&row_offset, &col, &val, nnz_U, this->nrow_, this->ncol_);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractL(BaseMatrix<ValueType>* L) const
{
assert(L != NULL);

assert(this->nrow_ > 0);
assert(this->ncol_ > 0);

HostMatrixCSR<ValueType>* cast_L = dynamic_cast<HostMatrixCSR<ValueType>*>(L);

assert(cast_L != NULL);

int64_t nnz_L = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_L)
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] < ai)
{
++nnz_L;
}
}
}

PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

allocate_host(this->nrow_ + 1, &row_offset);
allocate_host(nnz_L, &col);
allocate_host(nnz_L, &val);

PtrType nnz   = 0;
row_offset[0] = 0;
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] < ai)
{
col[nnz] = this->mat_.col[aj];
val[nnz] = this->mat_.val[aj];
++nnz;
}
}

row_offset[ai + 1] = nnz;
}

cast_L->Clear();
cast_L->SetDataPtrCSR(&row_offset, &col, &val, nnz_L, this->nrow_, this->ncol_);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType>* L) const
{
assert(L != NULL);

assert(this->nrow_ > 0);
assert(this->ncol_ > 0);

HostMatrixCSR<ValueType>* cast_L = dynamic_cast<HostMatrixCSR<ValueType>*>(L);

assert(cast_L != NULL);

int64_t nnz_L = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_L)
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] <= ai)
{
++nnz_L;
}
}
}

PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

allocate_host(this->nrow_ + 1, &row_offset);
allocate_host(nnz_L, &col);
allocate_host(nnz_L, &val);

PtrType nnz   = 0;
row_offset[0] = 0;
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] <= ai)
{
col[nnz] = this->mat_.col[aj];
val[nnz] = this->mat_.val[aj];
++nnz;
}
}

row_offset[ai + 1] = nnz;
}

cast_L->Clear();
cast_L->SetDataPtrCSR(&row_offset, &col, &val, nnz_L, this->nrow_, this->ncol_);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::LUSolve(const BaseVector<ValueType>& in,
BaseVector<ValueType>*       out) const
{
assert(in.GetSize() >= 0);
assert(out->GetSize() >= 0);
assert(in.GetSize() == this->ncol_);
assert(out->GetSize() == this->nrow_);

const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

assert(cast_in != NULL);
assert(cast_out != NULL);

for(int ai = 0; ai < this->nrow_; ++ai)
{
cast_out->vec_[ai] = cast_in->vec_[ai];

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] < ai)
{
cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
}
else
{
break;
}
}
}

int64_t diag_aj = this->nnz_ - 1;

for(int ai = this->nrow_ - 1; ai >= 0; --ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] > ai)
{
cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
}

if(this->mat_.col[aj] == ai)
{
diag_aj = aj;
}
}

cast_out->vec_[ai] /= this->mat_.val[diag_aj];
}

return true;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::LLAnalyse(void)
{
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::LLAnalyseClear(void)
{
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::LUAnalyse(void)
{
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::LUAnalyseClear(void)
{
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
BaseVector<ValueType>*       out) const
{
assert(in.GetSize() >= 0);
assert(out->GetSize() >= 0);
assert(in.GetSize() == this->ncol_);
assert(out->GetSize() == this->nrow_);

const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

assert(cast_in != NULL);
assert(cast_out != NULL);

for(int ai = 0; ai < this->nrow_; ++ai)
{
ValueType value    = cast_in->vec_[ai];
PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;

for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
{
value -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
}

cast_out->vec_[ai] = value / this->mat_.val[diag_idx];
}

for(int ai = this->nrow_ - 1; ai >= 0; --ai)
{
PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;
ValueType value    = cast_out->vec_[ai] / this->mat_.val[diag_idx];

for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
{
cast_out->vec_[this->mat_.col[aj]] -= value * this->mat_.val[aj];
}

cast_out->vec_[ai] = value;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
const BaseVector<ValueType>& inv_diag,
BaseVector<ValueType>*       out) const
{
assert(in.GetSize() >= 0);
assert(out->GetSize() >= 0);
assert(in.GetSize() == this->ncol_);
assert(out->GetSize() == this->nrow_);
assert(inv_diag.GetSize() == this->nrow_ || inv_diag.GetSize() == this->ncol_);

const HostVector<ValueType>* cast_in = dynamic_cast<const HostVector<ValueType>*>(&in);
const HostVector<ValueType>* cast_diag
= dynamic_cast<const HostVector<ValueType>*>(&inv_diag);
HostVector<ValueType>* cast_out = dynamic_cast<HostVector<ValueType>*>(out);

assert(cast_in != NULL);
assert(cast_out != NULL);

for(int ai = 0; ai < this->nrow_; ++ai)
{
ValueType value    = cast_in->vec_[ai];
PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;

for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
{
value -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
}

cast_out->vec_[ai] = value * cast_diag->vec_[ai];
}

for(int ai = this->nrow_ - 1; ai >= 0; --ai)
{
PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;
ValueType value    = cast_out->vec_[ai] * cast_diag->vec_[ai];

for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
{
cast_out->vec_[this->mat_.col[aj]] -= value * this->mat_.val[aj];
}

cast_out->vec_[ai] = value;
}

return true;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::LAnalyse(bool diag_unit)
{
this->L_diag_unit_ = diag_unit;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::LAnalyseClear(void)
{
this->L_diag_unit_ = true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::LSolve(const BaseVector<ValueType>& in,
BaseVector<ValueType>*       out) const
{
assert(in.GetSize() >= 0);
assert(out->GetSize() >= 0);
assert(in.GetSize() == this->ncol_);
assert(out->GetSize() == this->nrow_);

const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

assert(cast_in != NULL);
assert(cast_out != NULL);

PtrType diag_aj = 0;

for(int ai = 0; ai < this->nrow_; ++ai)
{
cast_out->vec_[ai] = cast_in->vec_[ai];

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] < ai)
{
cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
}
else
{
if(this->L_diag_unit_ == false)
{
assert(this->mat_.col[aj] == ai);
diag_aj = aj;
}
break;
}
}

if(this->L_diag_unit_ == false)
{
cast_out->vec_[ai] /= this->mat_.val[diag_aj];
}
}

return true;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::UAnalyse(bool diag_unit)
{
this->U_diag_unit_ = diag_unit;
}

template <typename ValueType>
void HostMatrixCSR<ValueType>::UAnalyseClear(void)
{
this->U_diag_unit_ = false;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::USolve(const BaseVector<ValueType>& in,
BaseVector<ValueType>*       out) const
{
assert(in.GetSize() >= 0);
assert(out->GetSize() >= 0);
assert(in.GetSize() == this->ncol_);
assert(out->GetSize() == this->nrow_);

const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

assert(cast_in != NULL);
assert(cast_out != NULL);

int64_t diag_aj = this->nnz_ - 1;

for(int ai = this->nrow_ - 1; ai >= 0; --ai)
{
cast_out->vec_[ai] = cast_in->vec_[ai];

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(this->mat_.col[aj] > ai)
{
cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
}

if(this->L_diag_unit_ == false)
{
if(this->mat_.col[aj] == ai)
{
diag_aj = aj;
}
}
}

if(this->L_diag_unit_ == false)
{
cast_out->vec_[ai] /= this->mat_.val[diag_aj];
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ILU0Factorize(void)
{
assert(this->nrow_ == this->ncol_);
assert(this->nnz_ > 0);

PtrType* diag_offset = NULL;
PtrType* nnz_entries = NULL;

allocate_host(this->nrow_, &diag_offset);
allocate_host(this->nrow_, &nnz_entries);

set_to_zero_host(this->nrow_, nnz_entries);

for(int ai = 0; ai < this->nrow_; ++ai)
{
PtrType row_start = this->mat_.row_offset[ai];
PtrType row_end   = this->mat_.row_offset[ai + 1];
PtrType j;

for(j = row_start; j < row_end; ++j)
{
nnz_entries[this->mat_.col[j]] = j;
}

for(j = row_start; j < row_end; ++j)
{
if(this->mat_.col[j] < ai)
{
int     col_j  = this->mat_.col[j];
PtrType diag_j = diag_offset[col_j];

if(this->mat_.val[diag_j] != static_cast<ValueType>(0))
{
this->mat_.val[j] = this->mat_.val[j] / this->mat_.val[diag_j];

for(PtrType k = diag_j + 1; k < this->mat_.row_offset[col_j + 1]; ++k)
{
if(nnz_entries[this->mat_.col[k]] != 0)
{
this->mat_.val[nnz_entries[this->mat_.col[k]]]
-= this->mat_.val[j] * this->mat_.val[k];
}
}
}
}
else
{
break;
}
}

diag_offset[ai] = j;

for(j = row_start; j < row_end; ++j)
{
nnz_entries[this->mat_.col[j]] = 0;
}
}

free_host(&diag_offset);
free_host(&nnz_entries);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ILUTFactorize(double t, int maxrow)
{
assert(this->nrow_ == this->ncol_);
assert(this->nnz_ > 0);

int nrow = this->nrow_;
int ncol = this->ncol_;

PtrType*   row_offset  = NULL;
PtrType*   diag_offset = NULL;
int*       nnz_entries = NULL;
bool*      nnz_pos     = (bool*)malloc(nrow * sizeof(bool));
ValueType* w           = NULL;

allocate_host(nrow + 1, &row_offset);
allocate_host(nrow, &diag_offset);
allocate_host(nrow, &nnz_entries);
allocate_host(nrow, &w);

for(int i = 0; i < nrow; ++i)
{
w[i]           = 0.0;
nnz_entries[i] = -1;
nnz_pos[i]     = false;
diag_offset[i] = 0;
}

float      nnzA       = static_cast<float>(this->nnz_);
size_t     alloc_size = static_cast<size_t>(nnzA * 1.5f);
int*       col        = (int*)malloc(alloc_size * sizeof(int));
ValueType* val        = (ValueType*)malloc(alloc_size * sizeof(ValueType));

row_offset[0] = 0;
int64_t nnz   = 0;

for(int ai = 0; ai < this->nrow_; ++ai)
{
row_offset[ai + 1] = row_offset[ai];

PtrType row_begin = this->mat_.row_offset[ai];
PtrType row_end   = this->mat_.row_offset[ai + 1];
double  row_norm  = 0.0;

int m = 0;
for(PtrType aj = row_begin; aj < row_end; ++aj)
{
int idx        = this->mat_.col[aj];
w[idx]         = this->mat_.val[aj];
nnz_entries[m] = idx;
nnz_pos[idx]   = true;

row_norm += std::abs(this->mat_.val[aj]);
++m;
}

double threshold = t * row_norm / static_cast<double>(row_end - row_begin);

for(int k = 0; k < nrow; ++k)
{
if(nnz_entries[k] == -1)
{
break;
}

int aj = nnz_entries[k];

int sidx = k;
for(int j = k + 1; j < nrow; ++j)
{
if(nnz_entries[j] == -1)
{
break;
}

if(nnz_entries[j] < aj)
{
sidx = j;
aj   = nnz_entries[j];
}
}

aj = nnz_entries[k];

if(k != sidx)
{
nnz_entries[k]    = nnz_entries[sidx];
nnz_entries[sidx] = aj;
aj                = nnz_entries[k];
}

if(aj < ai)
{
if(val[diag_offset[aj]] == static_cast<ValueType>(0))
{
LOG_INFO("(ILUT) zero row");
continue;
}

w[aj] /= val[diag_offset[aj]];

for(PtrType l = diag_offset[aj] + 1; l < row_offset[aj + 1]; ++l)
{
int       idx    = col[l];
ValueType fillin = w[aj] * val[l];

if(nnz_pos[idx] == false)
{
if(std::abs(fillin) >= threshold)
{
nnz_entries[m] = idx;
nnz_pos[idx]   = true;
w[idx] -= fillin;
++m;
}
}
else
{
w[idx] -= fillin;
}
}
}
}

for(int k = 0, num_lower = 0, num_upper = 0; k < nrow; ++k)
{
int aj = nnz_entries[k];

if(aj == -1)
{
break;
}

if(nnz_pos[aj] == false)
{
break;
}

if(aj < ai && num_lower < maxrow)
{
val[nnz] = w[aj];
col[nnz] = aj;

++row_offset[ai + 1];
++num_lower;
++nnz;

}
else if(aj > ai && num_upper < maxrow)
{
val[nnz] = w[aj];
col[nnz] = aj;

++row_offset[ai + 1];
++num_upper;
++nnz;

}
else if(aj == ai)
{
val[nnz] = w[aj];
col[nnz] = aj;

diag_offset[ai] = row_offset[ai + 1];

++row_offset[ai + 1];
++nnz;
}

w[aj]          = static_cast<ValueType>(0);
nnz_entries[k] = -1;
nnz_pos[aj]    = false;
}

if(alloc_size < static_cast<size_t>(nnz + 2 * maxrow + 1))
{
alloc_size += static_cast<size_t>(nnzA * 1.5f);
int*       col_tmp = (int*)realloc(col, alloc_size * sizeof(int));
ValueType* val_tmp = (ValueType*)realloc(val, alloc_size * sizeof(ValueType));

if(col_tmp == NULL || val_tmp == NULL)
{
free(col);
free(val);

LOG_INFO("ILUTFactorize failed on realloc");
FATAL_ERROR(__FILE__, __LINE__);
}
else
{
col = col_tmp;
val = val_tmp;
}
}
}

free_host(&w);
free_host(&diag_offset);
free_host(&nnz_entries);
free(nnz_pos);

int*       p_col = NULL;
ValueType* p_val = NULL;

allocate_host(nnz, &p_col);
allocate_host(nnz, &p_val);

copy_h2h(nnz, col, p_col);
copy_h2h(nnz, val, p_val);

free(col);
free(val);

this->Clear();
this->SetDataPtrCSR(&row_offset, &p_col, &p_val, nnz, nrow, ncol);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ICFactorize(BaseVector<ValueType>* inv_diag)
{
assert(this->nrow_ == this->ncol_);
assert(this->nnz_ > 0);

assert(inv_diag != NULL);
HostVector<ValueType>* cast_diag = dynamic_cast<HostVector<ValueType>*>(inv_diag);
assert(cast_diag != NULL);

cast_diag->Allocate(this->nrow_);

PtrType* diag_offset = NULL;
PtrType* nnz_entries = NULL;

allocate_host(this->nrow_, &diag_offset);
allocate_host(this->nrow_, &nnz_entries);

set_to_zero_host(this->nrow_, nnz_entries);

for(int i = 0; i < this->nrow_; ++i)
{
PtrType row_begin = this->mat_.row_offset[i];
PtrType row_end   = this->mat_.row_offset[i + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
nnz_entries[this->mat_.col[j]] = j;
}

ValueType sum = static_cast<ValueType>(0);

bool has_diag = false;

PtrType j;
for(j = row_begin; j < row_end; ++j)
{
int       col_j = this->mat_.col[j];
ValueType val_j = this->mat_.val[j];

if(col_j == i)
{
has_diag = true;
break;
}

if(col_j > i)
{
break;
}

PtrType row_begin_j = this->mat_.row_offset[col_j];
PtrType row_diag_j  = diag_offset[col_j];

ValueType local_sum = static_cast<ValueType>(0);
ValueType inv_diag  = this->mat_.val[row_diag_j];

if(inv_diag == static_cast<ValueType>(0))
{
LOG_INFO("IC breakdown: division by zero");
FATAL_ERROR(__FILE__, __LINE__);
}

inv_diag = static_cast<ValueType>(1) / inv_diag;

for(PtrType k = row_begin_j; k < row_diag_j; ++k)
{
int col_k = this->mat_.col[k];

if(nnz_entries[col_k] != 0)
{
PtrType idx = nnz_entries[col_k];
local_sum += this->mat_.val[k] * this->mat_.val[idx];
}
}

val_j = (val_j - local_sum) * inv_diag;
sum += val_j * val_j;

this->mat_.val[j] = val_j;
}

if(!has_diag)
{
LOG_INFO("IC breakdown: structural zero diagonal");
FATAL_ERROR(__FILE__, __LINE__);
}

ValueType diag_entry = std::sqrt(std::abs(this->mat_.val[j] - sum));
this->mat_.val[j]    = diag_entry;

if(diag_entry == static_cast<ValueType>(0))
{
LOG_INFO("IC breakdown: division by zero");
FATAL_ERROR(__FILE__, __LINE__);
}

cast_diag->vec_[i] = static_cast<ValueType>(1) / diag_entry;

diag_offset[i] = j;

for(j = row_begin; j < row_end; ++j)
{
nnz_entries[this->mat_.col[j]] = 0;
}
}

free_host(&diag_offset);
free_host(&nnz_entries);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::MultiColoring(int&             num_colors,
int**            size_colors,
BaseVector<int>* permutation) const
{
assert(*size_colors == NULL);
assert(permutation != NULL);
HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
assert(cast_perm != NULL);

int* color = NULL;
allocate_host(this->nrow_, &color);

memset(color, 0, sizeof(int) * this->nrow_);
num_colors = 0;
std::vector<bool> row_col;

for(int ai = 0; ai < this->nrow_; ++ai)
{
color[ai] = 1;
row_col.clear();
row_col.reserve(num_colors + 2);
row_col.assign(num_colors + 2, false);

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai != this->mat_.col[aj])
{
row_col[color[this->mat_.col[aj]]] = true;
}
}

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(row_col[color[ai]] == true)
{
++color[ai];
}
}

if(color[ai] > num_colors)
{
num_colors = color[ai];
}
}

allocate_host(num_colors, size_colors);
set_to_zero_host(num_colors, *size_colors);

int* offsets_color = NULL;
allocate_host(num_colors, &offsets_color);
memset(offsets_color, 0, sizeof(int) * num_colors);

for(int i = 0; i < this->nrow_; ++i)
{
++(*size_colors)[color[i] - 1];
}

int total = 0;
for(int i = 1; i < num_colors; ++i)
{
total += (*size_colors)[i - 1];
offsets_color[i] = total;
}

cast_perm->Allocate(this->nrow_);

for(int i = 0; i < permutation->GetSize(); ++i)
{
cast_perm->vec_[i] = offsets_color[color[i] - 1];
++offsets_color[color[i] - 1];
}

free_host(&color);
free_host(&offsets_color);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::MaximalIndependentSet(int&             size,
BaseVector<int>* permutation) const
{
assert(permutation != NULL);
assert(this->nrow_ == this->ncol_);

HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
assert(cast_perm != NULL);

int* mis = NULL;
allocate_host(this->nrow_, &mis);
memset(mis, 0, sizeof(int) * this->nrow_);

size = 0;

for(int ai = 0; ai < this->nrow_; ++ai)
{
if(mis[ai] == 0)
{
mis[ai] = 1;
++size;

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
++aj)
{
if(ai != this->mat_.col[aj])
{
mis[this->mat_.col[aj]] = -1;
}
}
}
}

cast_perm->Allocate(this->nrow_);

int pos = 0;
for(int ai = 0; ai < this->nrow_; ++ai)
{
if(mis[ai] == 1)
{
cast_perm->vec_[ai] = pos;
++pos;
}
else
{
cast_perm->vec_[ai] = size + ai - pos;
}
}


free_host(&mis);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ZeroBlockPermutation(int&             size,
BaseVector<int>* permutation) const
{
assert(permutation != NULL);
assert(permutation->GetSize() == this->nrow_);
assert(permutation->GetSize() == this->ncol_);

HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
assert(cast_perm != NULL);

size = 0;

for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai == this->mat_.col[aj])
{
++size;
}
}
}

int k_z  = size;
int k_nz = 0;

for(int ai = 0; ai < this->nrow_; ++ai)
{
bool hit = false;

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai == this->mat_.col[aj])
{
cast_perm->vec_[ai] = k_nz;
++k_nz;
hit = true;
}
}

if(hit == false)
{
cast_perm->vec_[ai] = k_z;
++k_z;
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& src)
{
const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&src);

assert(cast_mat != NULL);
assert(this->ncol_ == cast_mat->nrow_);

std::vector<PtrType> row_offset;
std::vector<int>*    new_col = new std::vector<int>[this->nrow_];

row_offset.resize(this->nrow_ + 1);

row_offset[0] = 0;

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int ii = this->mat_.col[j];

for(PtrType k = cast_mat->mat_.row_offset[ii];
k < cast_mat->mat_.row_offset[ii + 1];
++k)
{
new_col[i].push_back(cast_mat->mat_.col[k]);
}
}

std::sort(new_col[i].begin(), new_col[i].end());
new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()), new_col[i].end());

row_offset[i + 1] = static_cast<PtrType>(new_col[i].size());
}

for(int i = 0; i < this->nrow_; ++i)
{
row_offset[i + 1] += row_offset[i];
}

this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
PtrType jj = 0;
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
this->mat_.col[j] = new_col[i][jj];
++jj;
}
}


delete[] new_col;

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
const BaseMatrix<ValueType>& B)
{
assert((this != &A) && (this != &B));


const HostMatrixCSR<ValueType>* cast_mat_A
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
const HostMatrixCSR<ValueType>* cast_mat_B
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

assert(cast_mat_A != NULL);
assert(cast_mat_B != NULL);
assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

int n = cast_mat_A->nrow_;
int m = cast_mat_B->ncol_;

PtrType* row_offset = NULL;
allocate_host(n + 1, &row_offset);
int*       col = NULL;
ValueType* val = NULL;

set_to_zero_host(n + 1, row_offset);

#ifdef _OPENMP
#pragma omp parallel
#endif
{
std::vector<PtrType> marker(m, -1);

#ifdef _OPENMP
int nt  = omp_get_num_threads();
int tid = omp_get_thread_num();

int chunk_size  = (n + nt - 1) / nt;
int chunk_start = tid * chunk_size;
int chunk_end   = std::min(n, chunk_start + chunk_size);
#else
int chunk_start = 0;
int chunk_end   = n;
#endif

for(int ia = chunk_start; ia < chunk_end; ++ia)
{
for(PtrType ja = cast_mat_A->mat_.row_offset[ia],
ea = cast_mat_A->mat_.row_offset[ia + 1];
ja < ea;
++ja)
{
int ca = cast_mat_A->mat_.col[ja];
for(PtrType jb = cast_mat_B->mat_.row_offset[ca],
eb = cast_mat_B->mat_.row_offset[ca + 1];
jb < eb;
++jb)
{
int cb = cast_mat_B->mat_.col[jb];

if(marker[cb] != ia)
{
marker[cb] = ia;
++row_offset[ia + 1];
}
}
}
}

std::fill(marker.begin(), marker.end(), -1);

#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef _OPENMP
#pragma omp single
#endif
{
for(int i = 1; i < n + 1; ++i)
{
row_offset[i] += row_offset[i - 1];
}

allocate_host(row_offset[n], &col);
allocate_host(row_offset[n], &val);
}

for(int ia = chunk_start; ia < chunk_end; ++ia)
{
PtrType row_begin = row_offset[ia];
PtrType row_end   = row_begin;

for(PtrType ja = cast_mat_A->mat_.row_offset[ia],
ea = cast_mat_A->mat_.row_offset[ia + 1];
ja < ea;
++ja)
{
int       ca = cast_mat_A->mat_.col[ja];
ValueType va = cast_mat_A->mat_.val[ja];

for(PtrType jb = cast_mat_B->mat_.row_offset[ca],
eb = cast_mat_B->mat_.row_offset[ca + 1];
jb < eb;
++jb)
{
int       cb = cast_mat_B->mat_.col[jb];
ValueType vb = cast_mat_B->mat_.val[jb];

if(marker[cb] < row_begin)
{
marker[cb]   = row_end;
col[row_end] = cb;
val[row_end] = va * vb;
++row_end;
}
else
{
val[marker[cb]] += va * vb;
}
}
}
}
}

this->SetDataPtrCSR(
&row_offset, &col, &val, row_offset[n], cast_mat_A->nrow_, cast_mat_B->ncol_);

this->Sort();

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& A,
const BaseMatrix<ValueType>& B)
{
const HostMatrixCSR<ValueType>* cast_mat_A
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
const HostMatrixCSR<ValueType>* cast_mat_B
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

assert(cast_mat_A != NULL);
assert(cast_mat_B != NULL);
assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

std::vector<PtrType> row_offset;
std::vector<int>*    new_col = new std::vector<int>[cast_mat_A->nrow_];

row_offset.resize(cast_mat_A->nrow_ + 1);

row_offset[0] = 0;

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < cast_mat_A->nrow_; ++i)
{
for(PtrType j = cast_mat_A->mat_.row_offset[i]; j < cast_mat_A->mat_.row_offset[i + 1];
++j)
{
int ii = cast_mat_A->mat_.col[j];

for(PtrType k = cast_mat_B->mat_.row_offset[ii];
k < cast_mat_B->mat_.row_offset[ii + 1];
++k)
{
new_col[i].push_back(cast_mat_B->mat_.col[k]);
}
}

std::sort(new_col[i].begin(), new_col[i].end());
new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()), new_col[i].end());

row_offset[i + 1] = static_cast<PtrType>(new_col[i].size());
}

for(int i = 0; i < cast_mat_A->nrow_; ++i)
{
row_offset[i + 1] += row_offset[i];
}

this->AllocateCSR(row_offset[cast_mat_A->nrow_], cast_mat_A->nrow_, cast_mat_B->ncol_);

copy_h2h(cast_mat_A->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < cast_mat_A->nrow_; ++i)
{
PtrType jj = 0;
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
this->mat_.col[j] = new_col[i][jj];
++jj;
}
}


delete[] new_col;

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::NumericMatMatMult(const BaseMatrix<ValueType>& A,
const BaseMatrix<ValueType>& B)
{
const HostMatrixCSR<ValueType>* cast_mat_A
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
const HostMatrixCSR<ValueType>* cast_mat_B
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

assert(cast_mat_A != NULL);
assert(cast_mat_B != NULL);
assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);
assert(this->nrow_ == cast_mat_A->nrow_);
assert(this->ncol_ == cast_mat_B->ncol_);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < cast_mat_A->nrow_; ++i)
{
for(PtrType j = cast_mat_A->mat_.row_offset[i]; j < cast_mat_A->mat_.row_offset[i + 1];
++j)
{
int ii = cast_mat_A->mat_.col[j];

for(PtrType k = cast_mat_B->mat_.row_offset[ii];
k < cast_mat_B->mat_.row_offset[ii + 1];
++k)
{
for(PtrType p = this->mat_.row_offset[i]; p < this->mat_.row_offset[i + 1]; ++p)
{
if(cast_mat_B->mat_.col[k] == this->mat_.col[p])
{
this->mat_.val[p] += cast_mat_B->mat_.val[k] * cast_mat_A->mat_.val[j];
break;
}
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::SymbolicPower(int p)
{
assert(p > 1);


if(p == 2)
{
this->SymbolicMatMatMult(*this);
}

if(p == 3)
{
HostMatrixCSR<ValueType> tmp(this->local_backend_);
tmp.CopyFrom(*this);

this->SymbolicPower(2);
this->SymbolicMatMatMult(tmp);
}

if(p == 4)
{
this->SymbolicPower(2);
this->SymbolicPower(2);
}

if(p == 5)
{
HostMatrixCSR<ValueType> tmp(this->local_backend_);
tmp.CopyFrom(*this);

this->SymbolicPower(4);
this->SymbolicMatMatMult(tmp);
}

if(p == 6)
{
this->SymbolicPower(2);
this->SymbolicPower(3);
}

if(p == 7)
{
HostMatrixCSR<ValueType> tmp(this->local_backend_);
tmp.CopyFrom(*this);

this->SymbolicPower(6);
this->SymbolicMatMatMult(tmp);
}

if(p == 8)
{
HostMatrixCSR<ValueType> tmp(this->local_backend_);
tmp.CopyFrom(*this);

this->SymbolicPower(6);
tmp.SymbolicPower(2);

this->SymbolicMatMatMult(tmp);
}

if(p > 8)
{
HostMatrixCSR<ValueType> tmp(this->local_backend_);
tmp.CopyFrom(*this);

for(int i = 0; i < p; ++i)
{
this->SymbolicMatMatMult(tmp);
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat)
{
const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

assert(cast_mat != NULL);
assert(cast_mat->nrow_ == this->nrow_);
assert(cast_mat->ncol_ == this->ncol_);
assert(this->nnz_ > 0);
assert(cast_mat->nnz_ > 0);

PtrType*   row_offset = NULL;
PtrType*   ind_diag   = NULL;
int*       levels     = NULL;
ValueType* val        = NULL;

allocate_host(cast_mat->nrow_ + 1, &row_offset);
allocate_host(cast_mat->nrow_, &ind_diag);
allocate_host(cast_mat->nnz_, &levels);
allocate_host(cast_mat->nnz_, &val);

int     inf_level = 99999;
int64_t nnz       = 0;

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < cast_mat->nrow_; ++ai)
{
for(PtrType aj = cast_mat->mat_.row_offset[ai]; aj < cast_mat->mat_.row_offset[ai + 1];
++aj)
{
if(ai == cast_mat->mat_.col[aj])
{
ind_diag[ai] = aj;
break;
}
}
}

set_to_zero_host(cast_mat->nrow_ + 1, row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int64_t i = 0; i < cast_mat->nnz_; ++i)
{
levels[i] = inf_level;
}

set_to_zero_host(cast_mat->nnz_, val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < cast_mat->nrow_; ++ai)
{
for(PtrType aj = cast_mat->mat_.row_offset[ai]; aj < cast_mat->mat_.row_offset[ai + 1];
++aj)
{
for(PtrType ajj = this->mat_.row_offset[ai]; ajj < this->mat_.row_offset[ai + 1];
++ajj)
{
if(cast_mat->mat_.col[aj] == this->mat_.col[ajj])
{
val[aj]    = this->mat_.val[ajj];
levels[aj] = 0;
break;
}
}
}
}

for(int ai = 1; ai < cast_mat->nrow_; ++ai)
{
for(PtrType ak = cast_mat->mat_.row_offset[ai]; ai > cast_mat->mat_.col[ak]; ++ak)
{
if(levels[ak] <= p)
{
val[ak] /= val[ind_diag[cast_mat->mat_.col[ak]]];

for(PtrType aj = ak + 1; aj < cast_mat->mat_.row_offset[ai + 1]; ++aj)
{
ValueType val_kj   = static_cast<ValueType>(0);
int       level_kj = inf_level;

for(PtrType kj = cast_mat->mat_.row_offset[cast_mat->mat_.col[ak]];
kj < cast_mat->mat_.row_offset[cast_mat->mat_.col[ak] + 1];
++kj)
{
if(cast_mat->mat_.col[aj] == cast_mat->mat_.col[kj])
{
level_kj = levels[kj];
val_kj   = val[kj];
break;
}
}

int lev = level_kj + levels[ak] + 1;

if(levels[aj] > lev)
{
levels[aj] = lev;
}

val[aj] -= val[ak] * val_kj;
}
}
}

for(PtrType ak = cast_mat->mat_.row_offset[ai]; ak < cast_mat->mat_.row_offset[ai + 1];
++ak)
{
if(levels[ak] > p)
{
levels[ak] = inf_level;
val[ak]    = static_cast<ValueType>(0);
}
else
{
++row_offset[ai + 1];
}
}
}

row_offset[0] = this->mat_.row_offset[0];
row_offset[1] = this->mat_.row_offset[1];

for(int i = 0; i < cast_mat->nrow_; ++i)
{
row_offset[i + 1] += row_offset[i];
}

nnz = row_offset[cast_mat->nrow_];

this->AllocateCSR(nnz, cast_mat->nrow_, cast_mat->ncol_);

PtrType jj = 0;
for(int i = 0; i < cast_mat->nrow_; ++i)
{
for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1]; ++j)
{
if(levels[j] <= p)
{
this->mat_.col[jj] = cast_mat->mat_.col[j];
this->mat_.val[jj] = val[j];
++jj;
}
}
}

assert(jj == nnz);

copy_h2h(this->nrow_ + 1, row_offset, this->mat_.row_offset);

free_host(&row_offset);
free_host(&ind_diag);
free_host(&levels);
free_host(&val);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::MatrixAdd(const BaseMatrix<ValueType>& mat,
ValueType                    alpha,
ValueType                    beta,
bool                         structure)
{
const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

assert(cast_mat != NULL);
assert(cast_mat->nrow_ == this->nrow_);
assert(cast_mat->ncol_ == this->ncol_);
assert(this->nnz_ >= 0);
assert(cast_mat->nnz_ >= 0);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

if(structure == false)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < cast_mat->nrow_; ++ai)
{
PtrType first_col = cast_mat->mat_.row_offset[ai];

for(PtrType ajj = this->mat_.row_offset[ai]; ajj < this->mat_.row_offset[ai + 1];
++ajj)
{
for(PtrType aj = first_col; aj < cast_mat->mat_.row_offset[ai + 1]; ++aj)
{
if(cast_mat->mat_.col[aj] == this->mat_.col[ajj])
{
this->mat_.val[ajj]
= alpha * this->mat_.val[ajj] + beta * cast_mat->mat_.val[aj];
++first_col;
break;
}
}
}
}
}
else
{
std::vector<PtrType> row_offset;
std::vector<int>*    new_col = new std::vector<int>[this->nrow_];

HostMatrixCSR<ValueType> tmp(this->local_backend_);

tmp.CopyFrom(*this);

row_offset.resize(this->nrow_ + 1);

row_offset[0] = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
new_col[i].push_back(this->mat_.col[j]);
}

for(PtrType k = cast_mat->mat_.row_offset[i]; k < cast_mat->mat_.row_offset[i + 1];
++k)
{
new_col[i].push_back(cast_mat->mat_.col[k]);
}

std::sort(new_col[i].begin(), new_col[i].end());
new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()),
new_col[i].end());

row_offset[i + 1] = static_cast<PtrType>(new_col[i].size());
}

for(int i = 0; i < this->nrow_; ++i)
{
row_offset[i + 1] += row_offset[i];
}

this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
PtrType jj = 0;
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
this->mat_.col[j] = new_col[i][jj];
++jj;
}
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
PtrType Aj = tmp.mat_.row_offset[i];
PtrType Bj = cast_mat->mat_.row_offset[i];

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
for(PtrType jj = Aj; jj < tmp.mat_.row_offset[i + 1]; ++jj)
{
if(this->mat_.col[j] == tmp.mat_.col[jj])
{
this->mat_.val[j] += alpha * tmp.mat_.val[jj];
++Aj;
break;
}
}

for(PtrType jj = Bj; jj < cast_mat->mat_.row_offset[i + 1]; ++jj)
{
if(this->mat_.col[j] == cast_mat->mat_.col[jj])
{
this->mat_.val[j] += beta * cast_mat->mat_.val[jj];
++Bj;
break;
}
}
}
}
delete[] new_col;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const
{
_set_omp_backend_threads(this->local_backend_, this->nrow_);

lambda_min = static_cast<ValueType>(0);
lambda_max = static_cast<ValueType>(0);


for(int ai = 0; ai < this->nrow_; ++ai)
{
ValueType sum  = static_cast<ValueType>(0);
ValueType diag = static_cast<ValueType>(0);

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai != this->mat_.col[aj])
{
sum += std::abs(this->mat_.val[aj]);
}
else
{
diag = this->mat_.val[aj];
}
}

if(sum + diag > lambda_max)
{
lambda_max = sum + diag;
}

if(diag - sum < lambda_min)
{
lambda_min = diag - sum;
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Scale(ValueType alpha)
{
_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int64_t ai = 0; ai < this->nnz_; ++ai)
{
this->mat_.val[ai] *= alpha;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ScaleDiagonal(ValueType alpha)
{
_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai == this->mat_.col[aj])
{
this->mat_.val[aj] *= alpha;
break;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ScaleOffDiagonal(ValueType alpha)
{
_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai != this->mat_.col[aj])
{
this->mat_.val[aj] *= alpha;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AddScalar(ValueType alpha)
{
_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int64_t ai = 0; ai < this->nnz_; ++ai)
{
this->mat_.val[ai] += alpha;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AddScalarDiagonal(ValueType alpha)
{
_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai == this->mat_.col[aj])
{
this->mat_.val[aj] += alpha;
break;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AddScalarOffDiagonal(ValueType alpha)
{
_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
if(ai != this->mat_.col[aj])
{
this->mat_.val[aj] += alpha;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType>& diag)
{
assert(diag.GetSize() == this->ncol_);

const HostVector<ValueType>* cast_diag = dynamic_cast<const HostVector<ValueType>*>(&diag);
assert(cast_diag != NULL);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
this->mat_.val[aj] *= cast_diag->vec_[this->mat_.col[aj]];
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType>& diag)
{
assert(diag.GetSize() == this->ncol_);

const HostVector<ValueType>* cast_diag = dynamic_cast<const HostVector<ValueType>*>(&diag);
assert(cast_diag != NULL);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
{
this->mat_.val[aj] *= cast_diag->vec_[ai];
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Compress(double drop_off)
{
if(this->nnz_ > 0)
{
std::vector<PtrType> row_offset;

HostMatrixCSR<ValueType> tmp(this->local_backend_);

tmp.CopyFrom(*this);

row_offset.resize(this->nrow_ + 1);

row_offset[0] = 0;

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
row_offset[i + 1] = 0;

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if((std::abs(this->mat_.val[j]) > drop_off) || (this->mat_.col[j] == i))
{
row_offset[i + 1] += 1;
}
}
}

for(int i = 0; i < this->nrow_; ++i)
{
row_offset[i + 1] += row_offset[i];
}

this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
PtrType jj = this->mat_.row_offset[i];

for(PtrType j = tmp.mat_.row_offset[i]; j < tmp.mat_.row_offset[i + 1]; ++j)
{
if((std::abs(tmp.mat_.val[j]) > drop_off) || (tmp.mat_.col[j] == i))
{
this->mat_.col[jj] = tmp.mat_.col[j];
this->mat_.val[jj] = tmp.mat_.val[j];
++jj;
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Transpose(void)
{
if(this->nnz_ > 0)
{
HostMatrixCSR<ValueType> tmp(this->local_backend_);

tmp.CopyFrom(*this);
tmp.Transpose(this);
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Transpose(BaseMatrix<ValueType>* T) const
{
assert(T != NULL);

HostMatrixCSR<ValueType>* cast_T = dynamic_cast<HostMatrixCSR<ValueType>*>(T);

assert(cast_T != NULL);

if(this->nnz_ > 0)
{
cast_T->Clear();
cast_T->AllocateCSR(this->nnz_, this->ncol_, this->nrow_);

for(int64_t i = 0; i < cast_T->nnz_; ++i)
{
cast_T->mat_.row_offset[this->mat_.col[i] + 1] += 1;
}

for(int i = 0; i < cast_T->nrow_; ++i)
{
cast_T->mat_.row_offset[i + 1] += cast_T->mat_.row_offset[i];
}

for(int ai = 0; ai < cast_T->ncol_; ++ai)
{
for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
++aj)
{
int     ind_col = this->mat_.col[aj];
PtrType ind     = cast_T->mat_.row_offset[ind_col];

cast_T->mat_.col[ind] = ai;
cast_T->mat_.val[ind] = this->mat_.val[aj];

cast_T->mat_.row_offset[ind_col] += 1;
}
}

PtrType shift = 0;
for(int i = 0; i < cast_T->nrow_; ++i)
{
PtrType tmp                = cast_T->mat_.row_offset[i];
cast_T->mat_.row_offset[i] = shift;
shift                      = tmp;
}

cast_T->mat_.row_offset[cast_T->nrow_] = shift;

assert(this->nnz_ == shift);
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Sort(void)
{
if(this->nnz_ > 0)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
for(PtrType jj = this->mat_.row_offset[i];
jj < this->mat_.row_offset[i + 1] - 1;
++jj)
{
if(this->mat_.col[jj] > this->mat_.col[jj + 1])
{
int       ind = this->mat_.col[jj];
ValueType val = this->mat_.val[jj];

this->mat_.col[jj] = this->mat_.col[jj + 1];
this->mat_.val[jj] = this->mat_.val[jj + 1];

this->mat_.col[jj + 1] = ind;
this->mat_.val[jj + 1] = val;
}
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::Permute(const BaseVector<int>& permutation)
{
assert((permutation.GetSize() == this->nrow_) && (permutation.GetSize() == this->ncol_));

if(this->nnz_ > 0)
{
const HostVector<int>* cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);
assert(cast_perm != NULL);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

int* row_nnz = NULL;
allocate_host(this->nrow_, &row_nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
row_nnz[i]
= static_cast<int>(this->mat_.row_offset[i + 1] - this->mat_.row_offset[i]);
}

int* perm_row_nnz = NULL;
allocate_host(this->nrow_, &perm_row_nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
perm_row_nnz[cast_perm->vec_[i]] = row_nnz[i];
}

PtrType* perm_nnz = NULL;
allocate_host(this->nrow_ + 1, &perm_nnz);
PtrType sum = 0;

for(int i = 0; i < this->nrow_; ++i)
{
perm_nnz[i] = sum;
sum += perm_row_nnz[i];
}

perm_nnz[this->nrow_] = sum;

int*       col = NULL;
ValueType* val = NULL;
allocate_host(this->nnz_, &col);
allocate_host(this->nnz_, &val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
PtrType permIndex = perm_nnz[cast_perm->vec_[i]];
PtrType prevIndex = this->mat_.row_offset[i];

for(int j = 0; j < row_nnz[i]; ++j)
{
col[permIndex + j] = this->mat_.col[prevIndex + j];
val[permIndex + j] = this->mat_.val[prevIndex + j];
}
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < this->nrow_; ++i)
{
PtrType row_index = perm_nnz[i];

for(int j = 0; j < perm_row_nnz[i]; ++j)
{
int k     = j - 1;
int aComp = col[row_index + j];
int comp  = cast_perm->vec_[aComp];
for(; k >= 0; --k)
{
if(this->mat_.col[row_index + k] > comp)
{
this->mat_.val[row_index + k + 1] = this->mat_.val[row_index + k];
this->mat_.col[row_index + k + 1] = this->mat_.col[row_index + k];
}
else
{
break;
}
}

this->mat_.val[row_index + k + 1] = val[row_index + j];
this->mat_.col[row_index + k + 1] = comp;
}
}

free_host(&this->mat_.row_offset);
this->mat_.row_offset = perm_nnz;
free_host(&col);
free_host(&val);
free_host(&row_nnz);
free_host(&perm_row_nnz);
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::CMK(BaseVector<int>* permutation) const
{
assert(this->nnz_ > 0);
assert(permutation != NULL);

HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
assert(cast_perm != NULL);

cast_perm->Clear();
cast_perm->Allocate(this->nrow_);

int next   = 0;
int head   = 0;
int tmp    = 0;
int test   = 1;
int maxdeg = 0;

int* nd         = NULL;
int* marker     = NULL;
int* levset     = NULL;
int* nextlevset = NULL;

allocate_host(this->nrow_, &nd);
allocate_host(this->nrow_, &marker);
allocate_host(this->nrow_, &levset);
allocate_host(this->nrow_, &nextlevset);

int qlength = 1;

for(int k = 0; k < this->nrow_; ++k)
{
marker[k] = 0;
nd[k] = static_cast<int>(this->mat_.row_offset[k + 1] - this->mat_.row_offset[k] - 1);

if(nd[k] > maxdeg)
{
maxdeg = nd[k];
}
}

head               = this->mat_.col[0];
levset[0]          = head;
cast_perm->vec_[0] = 0;
++next;
marker[head] = 1;

while(next < this->nrow_)
{
int position = 0;

for(int h = 0; h < qlength; ++h)
{
head = levset[h];

for(PtrType k = this->mat_.row_offset[head]; k < this->mat_.row_offset[head + 1];
++k)
{
tmp = this->mat_.col[k];

if((marker[tmp] == 0) && (tmp != head))
{
nextlevset[position] = tmp;
marker[tmp]          = 1;
cast_perm->vec_[tmp] = next;
++next;
++position;
}
}
}

qlength = position;

while(test == 1)
{
test = 0;

for(int j = position - 1; j > 0; --j)
{
if(nd[nextlevset[j]] < nd[nextlevset[j - 1]])
{
tmp               = nextlevset[j];
nextlevset[j]     = nextlevset[j - 1];
nextlevset[j - 1] = tmp;
test              = 1;
}
}
}

for(int i = 0; i < position; ++i)
{
levset[i] = nextlevset[i];
}

if(qlength == 0)
{
for(int i = 0; i < this->nrow_; ++i)
{
if(marker[i] == 0)
{
levset[0]          = i;
qlength            = 1;
cast_perm->vec_[i] = next;
marker[i]          = 1;
++next;
}
}
}
}

free_host(&nd);
free_host(&marker);
free_host(&levset);
free_host(&nextlevset);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RCMK(BaseVector<int>* permutation) const
{
HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
assert(cast_perm != NULL);

cast_perm->Clear();
cast_perm->Allocate(this->nrow_);

HostVector<int> tmp_perm(this->local_backend_);

this->CMK(&tmp_perm);

for(int i = 0; i < this->nrow_; ++i)
{
cast_perm->vec_[i] = this->nrow_ - tmp_perm.vec_[i] - 1;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ConnectivityOrder(BaseVector<int>* permutation) const
{
HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
assert(cast_perm != NULL);

cast_perm->Clear();
cast_perm->Allocate(this->nrow_);

std::multimap<int, int> map;

for(int i = 0; i < this->nrow_; ++i)
{
map.insert(
std::pair<int, int>(this->mat_.row_offset[i + 1] - this->mat_.row_offset[i], i));
}

std::multimap<int, int>::iterator it = map.begin();

for(int i = 0; i < this->nrow_; ++i, ++it)
{
cast_perm->vec_[i] = it->second;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::CreateFromMap(const BaseVector<int>& map, int n, int m)
{
assert(map.GetSize() == n);

const HostVector<int>* cast_map = dynamic_cast<const HostVector<int>*>(&map);

assert(cast_map != NULL);

int*     row_nnz    = NULL;
PtrType* row_buffer = NULL;
allocate_host(m, &row_nnz);
allocate_host(m + 1, &row_buffer);

set_to_zero_host(m, row_nnz);

int nnz = 0;

for(int i = 0; i < n; ++i)
{
assert(cast_map->vec_[i] < m);

if(cast_map->vec_[i] < 0)
{
continue;
}

++row_nnz[cast_map->vec_[i]];
++nnz;
}

this->Clear();
this->AllocateCSR(nnz, m, n);

this->mat_.row_offset[0] = 0;
row_buffer[0]            = 0;

for(int i = 0; i < m; ++i)
{
this->mat_.row_offset[i + 1] = this->mat_.row_offset[i] + row_nnz[i];
row_buffer[i + 1]            = this->mat_.row_offset[i + 1];
}

for(int i = 0; i < nnz; ++i)
{
if(cast_map->vec_[i] < 0)
{
continue;
}

this->mat_.col[row_buffer[cast_map->vec_[i]]] = i;
this->mat_.val[i]                             = static_cast<ValueType>(1);
row_buffer[cast_map->vec_[i]]++;
}

assert(this->mat_.row_offset[m] == nnz);

free_host(&row_nnz);
free_host(&row_buffer);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::CreateFromMap(const BaseVector<int>& map,
int                    n,
int                    m,
BaseMatrix<ValueType>* pro)
{
assert(map.GetSize() == n);
assert(pro != NULL);

const HostVector<int>*    cast_map = dynamic_cast<const HostVector<int>*>(&map);
HostMatrixCSR<ValueType>* cast_pro = dynamic_cast<HostMatrixCSR<ValueType>*>(pro);

assert(cast_pro != NULL);
assert(cast_map != NULL);

this->CreateFromMap(map, n, m);

cast_pro->Clear();
cast_pro->AllocateCSR(this->nnz_, n, m);

int k = 0;

for(int i = 0; i < n; ++i)
{
cast_pro->mat_.row_offset[i + 1] = cast_pro->mat_.row_offset[i];

if(cast_map->vec_[i] < 0)
{
continue;
}

assert(cast_map->vec_[i] < m);

++cast_pro->mat_.row_offset[i + 1];
cast_pro->mat_.col[k] = cast_map->vec_[i];
cast_pro->mat_.val[k] = static_cast<ValueType>(1);
++k;
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AMGConnect(ValueType eps, BaseVector<int>* connections) const
{
assert(connections != NULL);

HostVector<int>* cast_conn = dynamic_cast<HostVector<int>*>(connections);
assert(cast_conn != NULL);

cast_conn->Clear();
cast_conn->Allocate(this->nnz_);

ValueType eps2 = eps * eps;

HostVector<ValueType> vec_diag(this->local_backend_);
vec_diag.Allocate(this->nrow_);
this->ExtractDiagonal(&vec_diag);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
ValueType eps_dia_i = eps2 * vec_diag.vec_[i];

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int       c = this->mat_.col[j];
ValueType v = this->mat_.val[j];

cast_conn->vec_[j] = (c != i) && (v * v > eps_dia_i * vec_diag.vec_[c]);
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AMGAggregate(const BaseVector<int>& connections,
BaseVector<int>*       aggregates) const
{
assert(aggregates != NULL);

HostVector<int>*       cast_agg  = dynamic_cast<HostVector<int>*>(aggregates);
const HostVector<int>* cast_conn = dynamic_cast<const HostVector<int>*>(&connections);

assert(cast_agg != NULL);
assert(cast_conn != NULL);

aggregates->Clear();
aggregates->Allocate(this->nrow_);

const int undefined = -1;
const int removed   = -2;

int max_neib = 0;
for(int i = 0; i < this->nrow_; ++i)
{
PtrType j = this->mat_.row_offset[i];
PtrType e = this->mat_.row_offset[i + 1];

max_neib = std::max(static_cast<int>(e - j), max_neib);

int state = removed;
for(; j < e; ++j)
{
if(cast_conn->vec_[j])
{
state = undefined;
break;
}
}

cast_agg->vec_[i] = state;
}

std::vector<int> neib;
neib.reserve(max_neib);

int last_g = -1;

for(int i = 0; i < this->nrow_; ++i)
{
if(cast_agg->vec_[i] != undefined)
{
continue;
}

cast_agg->vec_[i] = ++last_g;

neib.clear();

for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e; ++j)
{
int c = this->mat_.col[j];
if(cast_conn->vec_[j] && cast_agg->vec_[c] != removed)
{
cast_agg->vec_[c] = last_g;
neib.push_back(c);
}
}

for(typename std::vector<int>::const_iterator nb = neib.begin(); nb != neib.end(); ++nb)
{
for(PtrType j = this->mat_.row_offset[*nb], e = this->mat_.row_offset[*nb + 1];
j < e;
++j)
{
if(cast_conn->vec_[j] && cast_agg->vec_[this->mat_.col[j]] == undefined)
{
cast_agg->vec_[this->mat_.col[j]] = last_g;
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AMGPMISAggregate(const BaseVector<int>& connections,
BaseVector<int>*       aggregates) const
{
assert(aggregates != NULL);

HostVector<int>*       cast_agg  = dynamic_cast<HostVector<int>*>(aggregates);
const HostVector<int>* cast_conn = dynamic_cast<const HostVector<int>*>(&connections);

assert(cast_agg != NULL);
assert(cast_conn != NULL);

aggregates->Clear();
aggregates->Allocate(this->nrow_);

std::vector<mis_tuple> tuples(this->nrow_);
std::vector<mis_tuple> max_tuples(this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
int state = -2;

PtrType row_start = this->mat_.row_offset[i];
PtrType row_end   = this->mat_.row_offset[i + 1];
for(PtrType j = row_start; j < row_end; j++)
{
if(cast_conn->vec_[j] == 1)
{
state = 0;
break;
}
}

unsigned int hash = i;
hash              = ((hash >> 16) ^ hash) * 0x45d9f3b;
hash              = ((hash >> 16) ^ hash) * 0x45d9f3b;
hash              = (hash >> 16) ^ hash;

tuples[i].s = state;
tuples[i].v = hash;
tuples[i].i = i;
}

bool done = false;
int  iter = 0;
while(!done)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
max_tuples[i] = tuples[i];
}

for(int k = 0; k < 2; k++)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
mis_tuple t_max = max_tuples[i];

PtrType row_start = this->mat_.row_offset[t_max.i];
PtrType row_end   = this->mat_.row_offset[t_max.i + 1];
for(PtrType j = row_start; j < row_end; j++)
{
if(cast_conn->vec_[j] == 1)
{
int       c  = this->mat_.col[j];
mis_tuple tj = tuples[c];

if(tj.s > t_max.s)
{
t_max = tj;
}
else if(tj.s == t_max.s && (tj.v > t_max.v))
{
t_max = tj;
}
}
}

max_tuples[i] = t_max;
}
}

done = true;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
if(tuples[i].s == 0)
{
mis_tuple t_max = max_tuples[i];

if(t_max.i == i)
{
tuples[i].s       = 1;
cast_agg->vec_[i] = 1;
}
else if(t_max.s == 1)
{
tuples[i].s       = -1;
cast_agg->vec_[i] = 0;
}
else
{
done = false;
}
}
}

iter++;

if(iter > 10)
{
LOG_VERBOSE_INFO(
2,
"*** warning: HostMatrixCSR::AMGPMISAggregate() Current number of iterations: "
<< iter);
}
}

int sum = 0;
for(int i = 0; i < this->nrow_; ++i)
{
int temp          = cast_agg->vec_[i];
cast_agg->vec_[i] = sum;
sum += temp;
}

for(int k = 0; k < 2; k++)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
max_tuples[i] = tuples[i];
}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
mis_tuple t = max_tuples[i];

assert(t.s != 0);

if(t.s == -1)
{
PtrType row_start = this->mat_.row_offset[i];
PtrType row_end   = this->mat_.row_offset[i + 1];

for(PtrType j = row_start; j < row_end; j++)
{
if(cast_conn->vec_[j] == 1)
{
int c = this->mat_.col[j];

if(max_tuples[c].s == 1)
{
cast_agg->vec_[i] = cast_agg->vec_[c];
tuples[i].s       = 1;
break;
}
}
}
}
else if(t.s == -2)
{
cast_agg->vec_[i] = -2;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AMGSmoothedAggregation(ValueType              relax,
const BaseVector<int>& aggregates,
const BaseVector<int>& connections,
BaseMatrix<ValueType>* prolong,
int lumping_strat) const
{
assert(prolong != NULL);

const HostVector<int>*    cast_agg     = dynamic_cast<const HostVector<int>*>(&aggregates);
const HostVector<int>*    cast_conn    = dynamic_cast<const HostVector<int>*>(&connections);
HostMatrixCSR<ValueType>* cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

assert(cast_agg != NULL);
assert(cast_conn != NULL);
assert(cast_prolong != NULL);

cast_prolong->Clear();
cast_prolong->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);

int ncol = 0;

for(int i = 0; i < cast_agg->GetSize(); ++i)
{
if(cast_agg->vec_[i] > ncol)
{
ncol = cast_agg->vec_[i];
}
}

++ncol;

#ifdef _OPENMP
#pragma omp parallel
#endif
{
std::vector<PtrType> marker(ncol, -1);

#ifdef _OPENMP
int nt  = omp_get_num_threads();
int tid = omp_get_thread_num();

int chunk_size  = (this->nrow_ + nt - 1) / nt;
int chunk_start = tid * chunk_size;
int chunk_end   = std::min(this->nrow_, chunk_start + chunk_size);
#else
int chunk_start = 0;
int chunk_end   = this->nrow_;
#endif

for(int i = chunk_start; i < chunk_end; ++i)
{
for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
++j)
{
int c = this->mat_.col[j];

if(c != i && !cast_conn->vec_[j])
{
continue;
}

int g = cast_agg->vec_[c];

if(g >= 0 && marker[g] != i)
{
marker[g] = i;
++cast_prolong->mat_.row_offset[i + 1];
}
}
}

std::fill(marker.begin(), marker.end(), -1);

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
{
PtrType* row_offset = NULL;
allocate_host(cast_prolong->nrow_ + 1, &row_offset);

int*       col = NULL;
ValueType* val = NULL;

int64_t nnz  = 0;
int     nrow = cast_prolong->nrow_;

row_offset[0] = 0;
for(int i = 1; i < nrow + 1; ++i)
{
row_offset[i] = cast_prolong->mat_.row_offset[i] + row_offset[i - 1];
}

nnz = row_offset[nrow];

allocate_host(nnz, &col);
allocate_host(nnz, &val);

cast_prolong->Clear();
cast_prolong->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
}

for(int i = chunk_start; i < chunk_end; ++i)
{
ValueType dia = static_cast<ValueType>(0);
for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
++j)
{
if(this->mat_.col[j] == i)
{
dia += this->mat_.val[j];
}
else if(!cast_conn->vec_[j])
{
if(lumping_strat == 0)
{
dia += this->mat_.val[j];
}
else
{
dia -= this->mat_.val[j];
}
}
}

dia = static_cast<ValueType>(1) / dia;

PtrType row_begin = cast_prolong->mat_.row_offset[i];
PtrType row_end   = row_begin;

for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
++j)
{
int c = this->mat_.col[j];

if(c != i && !cast_conn->vec_[j])
{
continue;
}

int g = cast_agg->vec_[c];
if(g < 0)
{
continue;
}

ValueType v = (c == i) ? static_cast<ValueType>(1) - relax
: -relax * dia * this->mat_.val[j];

if(marker[g] < row_begin)
{
marker[g]                       = row_end;
cast_prolong->mat_.col[row_end] = g;
cast_prolong->mat_.val[row_end] = v;
++row_end;
}
else
{
cast_prolong->mat_.val[marker[g]] += v;
}
}
}
}

cast_prolong->Sort();

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::AMGAggregation(const BaseVector<int>& aggregates,
BaseMatrix<ValueType>* prolong) const
{
assert(prolong != NULL);

const HostVector<int>*    cast_agg     = dynamic_cast<const HostVector<int>*>(&aggregates);
HostMatrixCSR<ValueType>* cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

assert(cast_agg != NULL);
assert(cast_prolong != NULL);

int ncol = 0;

for(int i = 0; i < cast_agg->GetSize(); ++i)
{
if(cast_agg->vec_[i] > ncol)
{
ncol = cast_agg->vec_[i];
}
}

++ncol;

PtrType* row_offset = NULL;
allocate_host(this->nrow_ + 1, &row_offset);

int*       col = NULL;
ValueType* val = NULL;

row_offset[0] = 0;
for(int i = 0; i < this->nrow_; ++i)
{
if(cast_agg->vec_[i] >= 0)
{
row_offset[i + 1] = row_offset[i] + 1;
}
else
{
row_offset[i + 1] = row_offset[i];
}
}

allocate_host(row_offset[this->nrow_], &col);
allocate_host(row_offset[this->nrow_], &val);

for(int i = 0, j = 0; i < this->nrow_; ++i)
{
if(cast_agg->vec_[i] >= 0)
{
col[j] = cast_agg->vec_[i];
val[j] = 1.0;
++j;
}
}

cast_prolong->Clear();
cast_prolong->SetDataPtrCSR(
&row_offset, &col, &val, row_offset[this->nrow_], this->nrow_, ncol);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::FSAI(int power, const BaseMatrix<ValueType>* pattern)
{
HostMatrixCSR<ValueType> L(this->local_backend_);

const HostMatrixCSR<ValueType>* cast_pattern = NULL;
if(pattern != NULL)
{
cast_pattern = dynamic_cast<const HostMatrixCSR<ValueType>*>(pattern);
assert(cast_pattern != NULL);

cast_pattern->ExtractLDiagonal(&L);
}
else if(power > 1)
{
HostMatrixCSR<ValueType> structure(this->local_backend_);
structure.CopyFrom(*this);
structure.SymbolicPower(power);
structure.ExtractLDiagonal(&L);
}
else
{
this->ExtractLDiagonal(&L);
}

int64_t    nnz        = L.nnz_;
int        nrow       = L.nrow_;
int        ncol       = L.ncol_;
PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

L.LeaveDataPtrCSR(&row_offset, &col, &val);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
PtrType nnz_row = row_offset[ai + 1] - row_offset[ai];

if(nnz_row == 1)
{
PtrType aj = this->mat_.row_offset[ai];
if(this->mat_.col[aj] == ai)
{
val[row_offset[ai]] = static_cast<ValueType>(1) / this->mat_.val[aj];
}
}
else
{
std::vector<ValueType> Asub(nnz_row * nnz_row, static_cast<ValueType>(0));

for(PtrType k = 0; k < nnz_row; ++k)
{
PtrType row_begin = this->mat_.row_offset[col[row_offset[ai] + k]];
PtrType row_end   = this->mat_.row_offset[col[row_offset[ai] + k] + 1];

for(PtrType aj = row_begin; aj < row_end; ++aj)
{
for(PtrType j = 0; j < nnz_row; ++j)
{
int Asub_col = col[row_offset[ai] + j];

if(this->mat_.col[aj] < Asub_col)
{
break;
}

if(this->mat_.col[aj] == Asub_col)
{
Asub[j + k * nnz_row] = this->mat_.val[aj];
break;
}
}

if(this->mat_.col[aj] == ai)
{
break;
}
}
}

std::vector<ValueType> mk(nnz_row, static_cast<ValueType>(0));
mk[nnz_row - 1] = static_cast<ValueType>(1);

for(PtrType i = 0; i < nnz_row - 1; ++i)
{
for(PtrType k = i + 1; k < nnz_row; ++k)
{
Asub[i + k * nnz_row] /= Asub[i + i * nnz_row];

for(PtrType j = i + 1; j < nnz_row; ++j)
{
Asub[j + k * nnz_row] -= Asub[i + k * nnz_row] * Asub[j + i * nnz_row];
}
}
}

for(PtrType i = nnz_row - 1; i >= 0; --i)
{
mk[i] /= Asub[i + i * nnz_row];

for(PtrType j = 0; j < i; ++j)
{
mk[j] -= mk[i] * Asub[i + j * nnz_row];
}
}

for(PtrType aj = row_offset[ai], k = 0; aj < row_offset[ai + 1]; ++aj, ++k)
{
val[aj] = mk[k];
}
}
}

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < nrow; ++ai)
{
ValueType fac = sqrt(static_cast<ValueType>(1) / std::abs(val[row_offset[ai + 1] - 1]));

for(PtrType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
{
val[aj] *= fac;
}
}

this->Clear();
this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::SPAI(void)
{
int     nrow = this->nrow_;
int64_t nnz  = this->nnz_;

ValueType* val = NULL;
allocate_host(nnz, &val);

HostMatrixCSR<ValueType> T(this->local_backend_);
T.CopyFrom(*this);
this->Transpose();

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < nrow; ++i)
{
int* J     = NULL;
int  Jsize = static_cast<int>(this->mat_.row_offset[i + 1] - this->mat_.row_offset[i]);
allocate_host(Jsize, &J);
std::vector<int> I;

for(PtrType j = this->mat_.row_offset[i], idx = 0; j < this->mat_.row_offset[i + 1];
++j, ++idx)
{
J[idx] = this->mat_.col[j];
}

for(int idx = 0; idx < Jsize; ++idx)
{
for(PtrType j = this->mat_.row_offset[J[idx]];
j < this->mat_.row_offset[J[idx] + 1];
++j)
{
if(std::find(I.begin(), I.end(), this->mat_.col[j]) == I.end())
{
I.push_back(this->mat_.col[j]);
}
}
}

HostMatrixDENSE<ValueType> Asub(this->local_backend_);
Asub.AllocateDENSE(int(I.size()), Jsize);

for(int k = 0; k < Asub.nrow_; ++k)
{
for(PtrType aj = T.mat_.row_offset[I[k]]; aj < T.mat_.row_offset[I[k] + 1]; ++aj)
{
for(int j = 0; j < Jsize; ++j)
{
if(T.mat_.col[aj] == J[j])
{
Asub.mat_.val[DENSE_IND(k, j, Asub.nrow_, Asub.ncol_)] = T.mat_.val[aj];
}
}
}
}

Asub.QRDecompose();

HostVector<ValueType> ek(this->local_backend_);
HostVector<ValueType> mk(this->local_backend_);

ek.Allocate(Asub.nrow_);
mk.Allocate(Asub.ncol_);

for(int j = 0; j < ek.GetSize(); ++j)
{
if(I[j] == i)
{
ek.vec_[j] = 1.0;
}
}

Asub.QRSolve(ek, &mk);

for(int j = 0; j < Jsize; ++j)
{
val[this->mat_.row_offset[i] + j] = mk.vec_[j];
}

I.clear();
ek.Clear();
mk.Clear();
Asub.Clear();
free_host(&J);
}

free_host(&this->mat_.val);
this->mat_.val = val;

this->Transpose();

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSCoarsening(float             eps,
BaseVector<int>*  CFmap,
BaseVector<bool>* S) const
{
assert(CFmap != NULL);
assert(S != NULL);

HostVector<int>*  cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
HostVector<bool>* cast_S  = dynamic_cast<HostVector<bool>*>(S);

assert(cast_cf != NULL);
assert(cast_S != NULL);

cast_cf->Clear();
cast_cf->Allocate(this->nrow_);

cast_cf->Zeros();


cast_S->Clear();
cast_S->Allocate(this->nnz_);

cast_S->Zeros();

PtrType* S_row_offset = NULL;
int*     S_col        = NULL;

allocate_host(this->nrow_ + 1, &S_row_offset);
set_to_zero_host(this->nrow_ + 1, S_row_offset);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
ValueType min_a_ik = static_cast<ValueType>(0);
ValueType max_a_ik = static_cast<ValueType>(0);

PtrType row_begin = this->mat_.row_offset[i];
PtrType row_end   = this->mat_.row_offset[i + 1];

bool sign = false;

for(PtrType j = row_begin; j < row_end; ++j)
{
int       col = this->mat_.col[j];
ValueType val = this->mat_.val[j];

if(col == i)
{
sign = val < static_cast<ValueType>(0);
}
else
{
min_a_ik = (min_a_ik < val) ? min_a_ik : val;
max_a_ik = (max_a_ik > val) ? max_a_ik : val;
}
}

ValueType cond = (sign ? max_a_ik : min_a_ik) * static_cast<ValueType>(eps);

for(PtrType j = row_begin; j < row_end; ++j)
{
int       col = this->mat_.col[j];
ValueType val = this->mat_.val[j];

cast_S->vec_[j] = (col != i) && (val < cond);
}

if(cond == static_cast<ValueType>(0))
{
cast_cf->vec_[i] = 2;
}
}

for(int64_t i = 0; i < this->nnz_; ++i)
{
if(cast_S->vec_[i])
{
S_row_offset[this->mat_.col[i] + 1]++;
}
}

for(int i = 0; i < this->nrow_; ++i)
{
S_row_offset[i + 1] += S_row_offset[i];
}

allocate_host(S_row_offset[this->nrow_], &S_col);

for(int i = 0; i < this->nrow_; ++i)
{
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if(cast_S->vec_[j])
{
S_col[S_row_offset[this->mat_.col[j]]++] = i;
}
}
}

for(int i = this->nrow_; i > 0; --i)
{
S_row_offset[i] = S_row_offset[i - 1];
}

S_row_offset[0] = 0;

std::vector<int> lambda(this->nrow_);

for(int i = 0; i < this->nrow_; ++i)
{
int temp = 0;
for(PtrType j = S_row_offset[i]; j < S_row_offset[i + 1]; ++j)
{
temp += (cast_cf->vec_[S_col[j]] == 0 ? 1 : 2);
}

lambda[i] = temp;
}

std::vector<int> ptr(this->nrow_ + 1, static_cast<int>(0));
std::vector<int> cnt(this->nrow_, static_cast<int>(0));
std::vector<int> i2n(this->nrow_);
std::vector<int> n2i(this->nrow_);

for(int i = 0; i < this->nrow_; ++i)
{
ptr[lambda[i] + 1]++;
}

for(unsigned int i = 1; i < ptr.size(); ++i)
{
ptr[i] += ptr[i - 1];
}

for(int i = 0; i < this->nrow_; ++i)
{
int lam  = lambda[i];
int idx  = ptr[lam] + cnt[lam]++;
i2n[idx] = i;
n2i[i]   = idx;
}

for(int top = this->nrow_ - 1; top >= 0; --top)
{
int i   = i2n[top];
int lam = lambda[i];

if(lam == 0)
{
for(int ai = 0; ai < this->nrow_; ++ai)
{
if(cast_cf->vec_[ai] == 0)
{
cast_cf->vec_[ai] = 1;
}
}

break;
}

cnt[lam]--;

if(cast_cf->vec_[i] == 2)
{
continue;
}

assert(cast_cf->vec_[i] == 0);

cast_cf->vec_[i] = 1;

for(PtrType j = S_row_offset[i]; j < S_row_offset[i + 1]; ++j)
{
int c = S_col[j];

if(cast_cf->vec_[c] != 0)
{
continue;
}

cast_cf->vec_[c] = 2;

for(PtrType jj = this->mat_.row_offset[c]; jj < this->mat_.row_offset[c + 1]; ++jj)
{
if(!cast_S->vec_[jj])
{
continue;
}

int cc     = this->mat_.col[jj];
int lam_cc = lambda[cc];

if(cast_cf->vec_[cc] != 0 || lam_cc >= this->nrow_ - 1)
{
continue;
}

int old_pos = n2i[cc];
int new_pos = ptr[lam_cc] + cnt[lam_cc] - 1;

n2i[i2n[old_pos]] = new_pos;
n2i[i2n[new_pos]] = old_pos;

std::swap(i2n[old_pos], i2n[new_pos]);

--cnt[lam_cc];
++cnt[lam_cc + 1];
ptr[lam_cc + 1] = ptr[lam_cc] + cnt[lam_cc];

++lambda[cc];
}
}

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if(!cast_S->vec_[j])
{
continue;
}

int c   = this->mat_.col[j];
int lam = lambda[c];

if(cast_cf->vec_[c] != 0 || lam == 0)
{
continue;
}

int old_pos = n2i[c];
int new_pos = ptr[lam];

n2i[i2n[old_pos]] = new_pos;
n2i[i2n[new_pos]] = old_pos;

std::swap(i2n[old_pos], i2n[new_pos]);

--cnt[lam];
++cnt[lam - 1];
++ptr[lam];
--lambda[c];

assert(ptr[lam - 1] == ptr[lam] - cnt[lam - 1]);
}
}

free_host(&S_row_offset);
free_host(&S_col);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSPMISStrongInfluences(float                        eps,
BaseVector<bool>*            S,
BaseVector<float>*           omega,
unsigned long long           seed,
const BaseMatrix<ValueType>& ghost) const
{
assert(S != NULL);
assert(omega != NULL);

HostVector<bool>*               cast_S = dynamic_cast<HostVector<bool>*>(S);
HostVector<float>*              cast_w = dynamic_cast<HostVector<float>*>(omega);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

assert(cast_S != NULL);
assert(cast_w != NULL);
assert(cast_gst != NULL);

bool global = cast_gst->nrow_ > 0;

cast_S->Zeros();

omega->SetRandomUniform(seed, 0, 1); 

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
ValueType min_a_ik = static_cast<ValueType>(0);
ValueType max_a_ik = static_cast<ValueType>(0);

PtrType row_begin = this->mat_.row_offset[i];
PtrType row_end   = this->mat_.row_offset[i + 1];

bool sign = false;

for(PtrType j = row_begin; j < row_end; ++j)
{
int       col = this->mat_.col[j];
ValueType val = this->mat_.val[j];

if(col == i)
{
sign = val < static_cast<ValueType>(0);
}
else
{
min_a_ik = (min_a_ik < val) ? min_a_ik : val;
max_a_ik = (max_a_ik > val) ? max_a_ik : val;
}
}

if(global == true)
{
PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
{
ValueType val = cast_gst->mat_.val[j];

min_a_ik = (min_a_ik < val) ? min_a_ik : val;
max_a_ik = (max_a_ik > val) ? max_a_ik : val;
}
}

ValueType cond = (sign ? max_a_ik : min_a_ik) * static_cast<ValueType>(eps);

for(PtrType j = row_begin; j < row_end; ++j)
{
int       col = this->mat_.col[j];
ValueType val = this->mat_.val[j];

if(col != i && val < cond)
{
cast_S->vec_[j] = true;

#ifdef _OPENMP
#pragma omp atomic
#endif
cast_w->vec_[col] += 1.0f;
}
}

if(global == true)
{
PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
{
int       col = cast_gst->mat_.col[j];
ValueType val = cast_gst->mat_.val[j];

if(val < cond)
{
cast_S->vec_[j + this->nnz_] = true;

#ifdef _OPENMP
#pragma omp atomic
#endif
cast_w->vec_[col + this->nrow_] += 1.0f;
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSPMISUnassignedToCoarse(BaseVector<int>*         CFmap,
BaseVector<bool>*        marked,
const BaseVector<float>& omega) const
{
assert(CFmap != NULL);
assert(marked != NULL);

HostVector<int>*         cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
HostVector<bool>*        cast_m  = dynamic_cast<HostVector<bool>*>(marked);
const HostVector<float>* cast_w  = dynamic_cast<const HostVector<float>*>(&omega);

assert(cast_cf != NULL);
assert(cast_m != NULL);
assert(cast_w != NULL);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < cast_cf->size_; ++i)
{
cast_m->vec_[i] = false;

if(cast_cf->vec_[i] == 0)
{
if(cast_w->vec_[i] >= 1.0f)
{
cast_cf->vec_[i] = 1;

cast_m->vec_[i] = true;
}
else
{
cast_cf->vec_[i] = 2;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSPMISCorrectCoarse(BaseVector<int>*             CFmap,
const BaseVector<bool>&      S,
const BaseVector<bool>&      marked,
const BaseVector<float>&     omega,
const BaseMatrix<ValueType>& ghost) const
{
assert(CFmap != NULL);

HostVector<int>*                cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
const HostVector<bool>*         cast_S  = dynamic_cast<const HostVector<bool>*>(&S);
const HostVector<bool>*         cast_m  = dynamic_cast<const HostVector<bool>*>(&marked);
const HostVector<float>*        cast_w  = dynamic_cast<const HostVector<float>*>(&omega);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_m != NULL);
assert(cast_w != NULL);
assert(cast_gst != NULL);

bool global = cast_gst->nrow_ > 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
PtrType row_begin = this->mat_.row_offset[i];
PtrType row_end   = this->mat_.row_offset[i + 1];

if(cast_m->vec_[i])
{
float omega_row = cast_w->vec_[i];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j])
{
int col = this->mat_.col[j];

if(cast_m->vec_[col] == true)
{
float omega_col = cast_w->vec_[col];

if(omega_row > omega_col)
{
cast_cf->vec_[col] = 0;
}
else if(omega_row < omega_col)
{
cast_cf->vec_[i] = 0;
}
}
}
}

if(global == true)
{
PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
{
if(cast_S->vec_[j + this->nnz_])
{
int col = cast_gst->mat_.col[j];

if(cast_m->vec_[col + this->nrow_])
{
float omega_col = cast_w->vec_[col + this->nrow_];

if(omega_row > omega_col)
{
cast_cf->vec_[col + this->nrow_] = 0;
}
else if(omega_row < omega_col)
{
cast_cf->vec_[i] = 0;
}
}
}
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSPMISCoarseEdgesToFine(BaseVector<int>*             CFmap,
const BaseVector<bool>&      S,
const BaseMatrix<ValueType>& ghost) const
{
assert(CFmap != NULL);

HostVector<int>*                cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
const HostVector<bool>*         cast_S  = dynamic_cast<const HostVector<bool>*>(&S);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_gst != NULL);

bool global = cast_gst->nrow_ > 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
if(cast_cf->vec_[i] == 0)
{
PtrType row_begin = this->mat_.row_offset[i];
PtrType row_end   = this->mat_.row_offset[i + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j])
{
int col = this->mat_.col[j];

if(cast_cf->vec_[col] == 1)
{
cast_cf->vec_[i] = 2;
break;
}
}
}

if(global == true)
{
PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
{
if(cast_S->vec_[j + this->nnz_])
{
int col = cast_gst->mat_.col[j];

if(cast_cf->vec_[col + this->nrow_] == 1)
{
cast_cf->vec_[i] = 2;
break;
}
}
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSPMISCheckUndecided(bool&                  undecided,
const BaseVector<int>& CFmap) const
{
const HostVector<int>* cast_cf = dynamic_cast<const HostVector<int>*>(&CFmap);

assert(cast_cf != NULL);

undecided = false;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int i = 0; i < this->nrow_; ++i)
{
if(cast_cf->vec_[i] == 0)
{
undecided = true;
i         = this->nrow_;
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSDirectProlongNnz(const BaseVector<int>&       CFmap,
const BaseVector<bool>&      S,
const BaseMatrix<ValueType>& ghost,
BaseVector<ValueType>*       Amin,
BaseVector<ValueType>*       Amax,
BaseVector<int>*             f2c,
BaseMatrix<ValueType>*       prolong_int,
BaseMatrix<ValueType>* prolong_gst) const
{
const HostVector<int>*          cast_cf = dynamic_cast<const HostVector<int>*>(&CFmap);
const HostVector<bool>*         cast_S  = dynamic_cast<const HostVector<bool>*>(&S);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
HostVector<ValueType>*    cast_Amin = dynamic_cast<HostVector<ValueType>*>(Amin);
HostVector<ValueType>*    cast_Amax = dynamic_cast<HostVector<ValueType>*>(Amax);
HostVector<int>*          cast_f2c  = dynamic_cast<HostVector<int>*>(f2c);
HostMatrixCSR<ValueType>* cast_pi   = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
HostMatrixCSR<ValueType>* cast_pg   = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);

assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_f2c != NULL);
assert(cast_pi != NULL);
assert(cast_Amin != NULL);
assert(cast_Amax != NULL);
assert(cast_Amin->size_ == this->nrow_);
assert(cast_Amax->size_ == this->nrow_);

bool global = prolong_gst != NULL;

cast_pi->Clear();

allocate_host(this->nrow_ + 1, &cast_pi->mat_.row_offset);

cast_pi->nrow_ = this->nrow_;

if(global == true)
{
assert(cast_gst != NULL);
assert(cast_pg != NULL);

cast_pg->Clear();

allocate_host(this->nrow_ + 1, &cast_pg->mat_.row_offset);

cast_pg->nrow_ = this->nrow_;
}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int row = 0; row < this->nrow_; ++row)
{
if(cast_cf->vec_[row] == 1)
{
cast_f2c->vec_[row]           = 1;
cast_pi->mat_.row_offset[row] = 1;

if(global == true)
{
cast_pg->mat_.row_offset[row] = 0;
}
}
else
{
cast_f2c->vec_[row] = 0;

PtrType nnz = 0;

ValueType amin = static_cast<ValueType>(0);
ValueType amax = static_cast<ValueType>(0);

PtrType row_begin = this->mat_.row_offset[row];
PtrType row_end   = this->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(!cast_S->vec_[j] || cast_cf->vec_[this->mat_.col[j]] != 1)
{
continue;
}

amin = (amin < this->mat_.val[j]) ? amin : this->mat_.val[j];
amax = (amax > this->mat_.val[j]) ? amax : this->mat_.val[j];
}

if(global == true)
{
for(PtrType j = cast_gst->mat_.row_offset[row];
j < cast_gst->mat_.row_offset[row + 1];
++j)
{
if(!cast_S->vec_[j + this->nnz_]
|| cast_cf->vec_[cast_gst->mat_.col[j] + this->nrow_] != 1)
{
continue;
}

amin = (amin < cast_gst->mat_.val[j]) ? amin : cast_gst->mat_.val[j];
amax = (amax > cast_gst->mat_.val[j]) ? amax : cast_gst->mat_.val[j];
}
}

cast_Amin->vec_[row] = amin = amin * static_cast<ValueType>(0.2f);
cast_Amax->vec_[row] = amax = amax * static_cast<ValueType>(0.2f);

for(PtrType j = row_begin; j < row_end; ++j)
{
int col = this->mat_.col[j];

if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
{
if(this->mat_.val[j] <= amin || this->mat_.val[j] >= amax)
{
++nnz;
}
}
}

cast_pi->mat_.row_offset[row] = nnz;

if(global == true)
{
PtrType gst_nnz = 0;

row_begin = cast_gst->mat_.row_offset[row];
row_end   = cast_gst->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
int col = cast_gst->mat_.col[j];

if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
{
if(cast_gst->mat_.val[j] <= amin || cast_gst->mat_.val[j] >= amax)
{
++gst_nnz;
}
}
}

cast_pg->mat_.row_offset[row] = gst_nnz;
}
}
}

cast_f2c->ExclusiveSum(*cast_f2c);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSDirectProlongFill(const BaseVector<int64_t>&   l2g,
const BaseVector<int>&       f2c,
const BaseVector<int>&       CFmap,
const BaseVector<bool>&      S,
const BaseMatrix<ValueType>& ghost,
const BaseVector<ValueType>& Amin,
const BaseVector<ValueType>& Amax,
BaseMatrix<ValueType>*       prolong_int,
BaseMatrix<ValueType>*       prolong_gst,
BaseVector<int64_t>* global_ghost_col) const
{
const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
const HostVector<int>*          cast_f2c = dynamic_cast<const HostVector<int>*>(&f2c);
const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
const HostVector<ValueType>* cast_Amin = dynamic_cast<const HostVector<ValueType>*>(&Amin);
const HostVector<ValueType>* cast_Amax = dynamic_cast<const HostVector<ValueType>*>(&Amax);
HostMatrixCSR<ValueType>*    cast_pi = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
HostMatrixCSR<ValueType>*    cast_pg = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);
HostVector<int64_t>* cast_glo        = dynamic_cast<HostVector<int64_t>*>(global_ghost_col);

assert(cast_f2c != NULL);
assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_pi != NULL);
assert(cast_Amin != NULL);
assert(cast_Amax != NULL);
assert(cast_Amin->size_ == this->nrow_);
assert(cast_Amax->size_ == this->nrow_);

bool global = prolong_gst != NULL;

if(global == true)
{
assert(cast_l2g != NULL);
assert(cast_gst != NULL);
assert(cast_pg != NULL);
assert(cast_glo != NULL);
}

for(int i = this->nrow_; i > 0; --i)
{
cast_pi->mat_.row_offset[i] = cast_pi->mat_.row_offset[i - 1];
}

cast_pi->mat_.row_offset[0] = 0;

for(int i = 0; i < this->nrow_; ++i)
{
cast_pi->mat_.row_offset[i + 1] += cast_pi->mat_.row_offset[i];
}

cast_pi->nnz_ = cast_pi->mat_.row_offset[this->nrow_];

cast_pi->ncol_ = cast_f2c->vec_[this->nrow_];

allocate_host(cast_pi->nnz_, &cast_pi->mat_.col);
allocate_host(cast_pi->nnz_, &cast_pi->mat_.val);

if(global == true)
{
for(int i = this->nrow_; i > 0; --i)
{
cast_pg->mat_.row_offset[i] = cast_pg->mat_.row_offset[i - 1];
}

cast_pg->mat_.row_offset[0] = 0;

for(int i = 0; i < this->nrow_; ++i)
{
cast_pg->mat_.row_offset[i + 1] += cast_pg->mat_.row_offset[i];
}

cast_pg->nnz_  = cast_pg->mat_.row_offset[this->nrow_];
cast_pg->ncol_ = this->nrow_;

allocate_host(cast_pg->nnz_, &cast_pg->mat_.col);
allocate_host(cast_pg->nnz_, &cast_pg->mat_.val);

cast_glo->Allocate(cast_pg->nnz_);
}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int row = 0; row < this->nrow_; ++row)
{
PtrType row_P = cast_pi->mat_.row_offset[row];

if(cast_cf->vec_[row] == 1)
{
cast_pi->mat_.col[row_P] = cast_f2c->vec_[row];
cast_pi->mat_.val[row_P] = static_cast<ValueType>(1);

continue;
}

ValueType diag  = static_cast<ValueType>(0);
ValueType a_num = static_cast<ValueType>(0), a_den = static_cast<ValueType>(0);
ValueType b_num = static_cast<ValueType>(0), b_den = static_cast<ValueType>(0);
ValueType d_neg = static_cast<ValueType>(0), d_pos = static_cast<ValueType>(0);

PtrType row_begin = this->mat_.row_offset[row];
PtrType row_end   = this->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
int       col = this->mat_.col[j];
ValueType val = this->mat_.val[j];

if(col == row)
{
diag = val;
continue;
}

if(val < static_cast<ValueType>(0))
{
a_num += val;

if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
{
a_den += val;

if(val > cast_Amin->vec_[row])
{
d_neg += val;
}
}
}
else
{
b_num += val;

if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
{
b_den += val;

if(val < cast_Amax->vec_[row])
{
d_pos += val;
}
}
}
}

if(global == true)
{
PtrType ghost_row_begin = cast_gst->mat_.row_offset[row];
PtrType ghost_row_end   = cast_gst->mat_.row_offset[row + 1];

for(PtrType j = ghost_row_begin; j < ghost_row_end; ++j)
{
int       col = cast_gst->mat_.col[j];
ValueType val = cast_gst->mat_.val[j];

if(val < static_cast<ValueType>(0))
{
a_num += val;

if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
{
a_den += val;

if(val > cast_Amin->vec_[row])
{
d_neg += val;
}
}
}
else
{
b_num += val;

if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
{
b_den += val;

if(val < cast_Amax->vec_[row])
{
d_pos += val;
}
}
}
}
}

ValueType cf_neg = static_cast<ValueType>(1);
ValueType cf_pos = static_cast<ValueType>(1);

if(std::abs(a_den - d_neg) > 1e-32)
{
cf_neg = a_den / (a_den - d_neg);
}

if(std::abs(b_den - d_pos) > 1e-32)
{
cf_pos = b_den / (b_den - d_pos);
}

if(b_num > static_cast<ValueType>(0) && std::abs(b_den) < 1e-32)
{
diag += b_num;
}

ValueType alpha = std::abs(a_den) > 1e-32 ? -cf_neg * a_num / (diag * a_den)
: static_cast<ValueType>(0);
ValueType beta  = std::abs(b_den) > 1e-32 ? -cf_pos * b_num / (diag * b_den)
: static_cast<ValueType>(0);

for(PtrType j = row_begin; j < row_end; ++j)
{
int       col = this->mat_.col[j];
ValueType val = this->mat_.val[j];

if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
{
if(val > cast_Amin->vec_[row] && val < cast_Amax->vec_[row])
{
continue;
}

cast_pi->mat_.col[row_P] = cast_f2c->vec_[col];
cast_pi->mat_.val[row_P]
= (val < static_cast<ValueType>(0) ? alpha : beta) * val;
++row_P;
}
}

if(global)
{
PtrType ghost_row_P     = cast_pg->mat_.row_offset[row];
PtrType ghost_row_begin = cast_gst->mat_.row_offset[row];
PtrType ghost_row_end   = cast_gst->mat_.row_offset[row + 1];

for(PtrType j = ghost_row_begin; j < ghost_row_end; ++j)
{
int       col = cast_gst->mat_.col[j];
ValueType val = cast_gst->mat_.val[j];

if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
{
if(val > cast_Amin->vec_[row] && val < cast_Amax->vec_[row])
{
continue;
}

cast_glo->vec_[ghost_row_P] = cast_l2g->vec_[col];
cast_pg->mat_.val[ghost_row_P]
= (val < static_cast<ValueType>(0) ? alpha : beta) * val;
++ghost_row_P;
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSExtPIBoundaryNnz(const BaseVector<int>&       boundary,
const BaseVector<int>&       CFmap,
const BaseVector<bool>&      S,
const BaseMatrix<ValueType>& ghost,
BaseVector<PtrType>*         row_nnz) const
{
const HostVector<int>*          cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary);
const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
HostVector<PtrType>* cast_nnz = dynamic_cast<HostVector<PtrType>*>(row_nnz);

assert(cast_bnd != NULL);
assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_gst != NULL);
assert(cast_nnz != NULL);

assert(cast_nnz->size_ >= cast_bnd->size_);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < cast_bnd->size_; ++i)
{
int row = cast_bnd->vec_[i];

PtrType ext_nnz = 0;

PtrType row_begin = this->mat_.row_offset[row];
PtrType row_end   = this->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j] == false)
{
continue;
}

int col = this->mat_.col[j];

if(cast_cf->vec_[col] != 2)
{
++ext_nnz;
}
}

row_begin = cast_gst->mat_.row_offset[row];
row_end   = cast_gst->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j + this->nnz_] == false)
{
continue;
}

int col = cast_gst->mat_.col[j];

if(cast_cf->vec_[col + this->nrow_] != 2)
{
++ext_nnz;
}
}

cast_nnz->vec_[i] = ext_nnz;
}

return true;
}

template <typename ValueType>
bool
HostMatrixCSR<ValueType>::RSExtPIExtractBoundary(int64_t                global_column_begin,
const BaseVector<int>& boundary,
const BaseVector<int64_t>&   l2g,
const BaseVector<int>&       CFmap,
const BaseVector<bool>&      S,
const BaseMatrix<ValueType>& ghost,
const BaseVector<PtrType>& bnd_csr_row_ptr,
BaseVector<int64_t>* bnd_csr_col_ind) const
{
const HostVector<int>*          cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary);
const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
const HostVector<PtrType>* cast_ptr
= dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
HostVector<int64_t>* cast_col = dynamic_cast<HostVector<int64_t>*>(bnd_csr_col_ind);

assert(cast_bnd != NULL);
assert(cast_l2g != NULL);
assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_gst != NULL);
assert(cast_ptr != NULL);
assert(cast_col != NULL);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int64_t i = 0; i < cast_bnd->size_; ++i)
{
int     row = cast_bnd->vec_[i];
PtrType idx = cast_ptr->vec_[i];

PtrType row_begin = this->mat_.row_offset[row];
PtrType row_end   = this->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j] == false)
{
continue;
}

int col = this->mat_.col[j];

if(cast_cf->vec_[col] != 2)
{
cast_col->vec_[idx++] = col + global_column_begin;
}
}

row_begin = cast_gst->mat_.row_offset[row];
row_end   = cast_gst->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j + this->nnz_] == false)
{
continue;
}

int col = cast_gst->mat_.col[j];

if(cast_cf->vec_[col + this->nrow_] != 2)
{
cast_col->vec_[idx++] = cast_l2g->vec_[col];
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSExtPIProlongNnz(int64_t                    global_column_begin,
int64_t                    global_column_end,
bool                       FF1,
const BaseVector<int64_t>& l2g,
const BaseVector<int>&     CFmap,
const BaseVector<bool>&    S,
const BaseMatrix<ValueType>& ghost,
const BaseVector<PtrType>&   bnd_csr_row_ptr,
const BaseVector<int64_t>&   bnd_csr_col_ind,
BaseVector<int>*             f2c,
BaseMatrix<ValueType>*       prolong_int,
BaseMatrix<ValueType>*       prolong_gst) const
{
const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
const HostVector<PtrType>* cast_ptr
= dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
const HostVector<int64_t>* cast_col
= dynamic_cast<const HostVector<int64_t>*>(&bnd_csr_col_ind);
HostVector<int>*          cast_f2c = dynamic_cast<HostVector<int>*>(f2c);
HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);

assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_f2c != NULL);
assert(cast_pi != NULL);

bool global = prolong_gst != NULL;

cast_pi->Clear();

allocate_host(this->nrow_ + 1, &cast_pi->mat_.row_offset);

cast_pi->nrow_ = this->nrow_;

if(global == true)
{
assert(cast_l2g != NULL);
assert(cast_gst != NULL);
assert(cast_ptr != NULL);
assert(cast_col != NULL);
assert(cast_pg != NULL);

cast_pg->Clear();

allocate_host(this->nrow_ + 1, &cast_pg->mat_.row_offset);

cast_pg->nrow_ = this->nrow_;
}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int row = 0; row < this->nrow_; ++row)
{
if(cast_cf->vec_[row] == 1)
{
cast_f2c->vec_[row] = 1;

cast_pi->mat_.row_offset[row] = 1;

if(global == true)
{
cast_pg->mat_.row_offset[row] = 0;
}

continue;
}

std::unordered_set<int>     int_set;
std::unordered_set<int64_t> gst_set;

PtrType row_begin = this->mat_.row_offset[row];
PtrType row_end   = this->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j] == false)
{
continue;
}

int col_j = this->mat_.col[j];

if(col_j == row)
{
continue;
}

if(cast_cf->vec_[col_j] == 1)
{
int_set.insert(col_j);
}
else
{

bool skip_ghost = false;

PtrType row_begin_j = this->mat_.row_offset[col_j];
PtrType row_end_j   = this->mat_.row_offset[col_j + 1];

for(PtrType k = row_begin_j; k < row_end_j; ++k)
{
if(cast_S->vec_[k] == false)
{
continue;
}

int col_k = this->mat_.col[k];

if(col_k == col_j)
{
continue;
}

if(cast_cf->vec_[col_k] == 1)
{
int_set.insert(col_k);

if(FF1 == true)
{
skip_ghost = true;
break;
}
}
}

if(skip_ghost == false && global == true)
{
row_begin_j = cast_gst->mat_.row_offset[col_j];
row_end_j   = cast_gst->mat_.row_offset[col_j + 1];

for(PtrType k = row_begin_j; k < row_end_j; ++k)
{
if(cast_S->vec_[k + this->nnz_] == false)
{
continue;
}

int col_k = cast_gst->mat_.col[k];

if(cast_cf->vec_[col_k + this->nrow_] == 1)
{
int64_t gcol_k = cast_l2g->vec_[col_k] + global_column_end
- global_column_begin;

gst_set.insert(gcol_k);
}
}
}
}
}

if(global == true)
{
row_begin = cast_gst->mat_.row_offset[row];
row_end   = cast_gst->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
if(cast_S->vec_[j + this->nnz_] == false)
{
continue;
}

int col_j = cast_gst->mat_.col[j];

if(cast_cf->vec_[col_j + this->nrow_] == 1)
{
int64_t gcol_j
= cast_l2g->vec_[col_j] + global_column_end - global_column_begin;

gst_set.insert(gcol_j);
}
else
{

PtrType row_begin_j = cast_ptr->vec_[col_j];
PtrType row_end_j   = cast_ptr->vec_[col_j + 1];

for(PtrType k = row_begin_j; k < row_end_j; ++k)
{
int64_t gcol_k = cast_col->vec_[k];

if(gcol_k >= global_column_begin && gcol_k < global_column_end)
{
int col_k = static_cast<int>(gcol_k - global_column_begin);

int_set.insert(col_k);

if(FF1 == true)
{
break;
}
}
else
{
gst_set.insert(gcol_k + global_column_end - global_column_begin);

if(FF1 == true)
{
break;
}
}
}
}
}
}

cast_pi->mat_.row_offset[row] = int_set.size();

if(global == true)
{
cast_pg->mat_.row_offset[row] = gst_set.size();
}

cast_f2c->vec_[row] = 0;
}

cast_f2c->ExclusiveSum(*cast_f2c);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RSExtPIProlongFill(int64_t global_column_begin,
int64_t global_column_end,
bool    FF1,
const BaseVector<int64_t>&   l2g,
const BaseVector<int>&       f2c,
const BaseVector<int>&       CFmap,
const BaseVector<bool>&      S,
const BaseMatrix<ValueType>& ghost,
const BaseVector<PtrType>&   bnd_csr_row_ptr,
const BaseVector<int64_t>&   bnd_csr_col_ind,
const BaseVector<PtrType>&   ext_csr_row_ptr,
const BaseVector<int64_t>&   ext_csr_col_ind,
const BaseVector<ValueType>& ext_csr_val,
BaseMatrix<ValueType>*       prolong_int,
BaseMatrix<ValueType>*       prolong_gst,
BaseVector<int64_t>* global_ghost_col) const
{
const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
const HostVector<int>*          cast_f2c = dynamic_cast<const HostVector<int>*>(&f2c);
const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
const HostVector<PtrType>* cast_ptr
= dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
const HostVector<int64_t>* cast_col
= dynamic_cast<const HostVector<int64_t>*>(&bnd_csr_col_ind);
const HostVector<PtrType>* cast_ext_ptr
= dynamic_cast<const HostVector<PtrType>*>(&ext_csr_row_ptr);
const HostVector<int64_t>* cast_ext_col
= dynamic_cast<const HostVector<int64_t>*>(&ext_csr_col_ind);
const HostVector<ValueType>* cast_ext_val
= dynamic_cast<const HostVector<ValueType>*>(&ext_csr_val);
HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);
HostVector<int64_t>*      cast_glo = dynamic_cast<HostVector<int64_t>*>(global_ghost_col);

assert(cast_f2c != NULL);
assert(cast_cf != NULL);
assert(cast_S != NULL);
assert(cast_pi != NULL);

bool global = prolong_gst != NULL;

if(global == true)
{
assert(cast_l2g != NULL);
assert(cast_gst != NULL);
assert(cast_ptr != NULL);
assert(cast_col != NULL);
assert(cast_ext_ptr != NULL);
assert(cast_ext_col != NULL);
assert(cast_ext_val != NULL);
assert(cast_pg != NULL);
assert(cast_glo != NULL);
}

for(int i = this->nrow_; i > 0; --i)
{
cast_pi->mat_.row_offset[i] = cast_pi->mat_.row_offset[i - 1];
}

cast_pi->mat_.row_offset[0] = 0;

for(int i = 0; i < this->nrow_; ++i)
{
cast_pi->mat_.row_offset[i + 1] += cast_pi->mat_.row_offset[i];
}

cast_pi->nnz_ = cast_pi->mat_.row_offset[this->nrow_];

cast_pi->ncol_ = cast_f2c->vec_[this->nrow_];

allocate_host(cast_pi->nnz_, &cast_pi->mat_.col);
allocate_host(cast_pi->nnz_, &cast_pi->mat_.val);

if(global == true)
{
for(int i = this->nrow_; i > 0; --i)
{
cast_pg->mat_.row_offset[i] = cast_pg->mat_.row_offset[i - 1];
}

cast_pg->mat_.row_offset[0] = 0;

for(int i = 0; i < this->nrow_; ++i)
{
cast_pg->mat_.row_offset[i + 1] += cast_pg->mat_.row_offset[i];
}

cast_pg->nnz_  = cast_pg->mat_.row_offset[this->nrow_];
cast_pg->ncol_ = this->nrow_;

allocate_host(cast_pg->nnz_, &cast_pg->mat_.col);
allocate_host(cast_pg->nnz_, &cast_pg->mat_.val);

cast_glo->Allocate(cast_pg->nnz_);
}

HostVector<ValueType> diag(this->local_backend_);
diag.Allocate(this->nrow_);

this->ExtractDiagonal(&diag);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
for(int row = 0; row < this->nrow_; ++row)
{
constexpr ValueType zero = static_cast<ValueType>(0);

if(cast_cf->vec_[row] == 1)
{
PtrType idx = cast_pi->mat_.row_offset[row];

cast_pi->mat_.col[idx] = cast_f2c->vec_[row];
cast_pi->mat_.val[idx] = static_cast<ValueType>(1);

continue;
}

std::map<int, ValueType>     int_table;
std::map<int64_t, ValueType> gst_table;


PtrType row_begin = this->mat_.row_offset[row];
PtrType row_end   = this->mat_.row_offset[row + 1];

for(PtrType k = row_begin; k < row_end; ++k)
{
if(cast_S->vec_[k] == false)
{
continue;
}

int col_ik = this->mat_.col[k];

if(col_ik == row)
{
continue;
}

if(cast_cf->vec_[col_ik] == 1)
{
int_table[col_ik] = zero;
}
else
{

bool skip_ghost = false;

PtrType row_begin_k = this->mat_.row_offset[col_ik];
PtrType row_end_k   = this->mat_.row_offset[col_ik + 1];

for(PtrType l = row_begin_k; l < row_end_k; ++l)
{
if(cast_S->vec_[l] == false)
{
continue;
}

int col_kl = this->mat_.col[l];

if(col_kl == col_ik)
{
continue;
}

if(cast_cf->vec_[col_kl] == 1)
{
int_table[col_kl] = zero;

if(FF1 == true)
{
skip_ghost = true;
break;
}
}
}

if(skip_ghost == false && global == true)
{
for(PtrType l = cast_gst->mat_.row_offset[col_ik];
l < cast_gst->mat_.row_offset[col_ik + 1];
++l)
{
if(cast_S->vec_[l + this->nnz_] == false)
{
continue;
}

int col_kl = cast_gst->mat_.col[l];

if(cast_cf->vec_[col_kl + this->nrow_] == 1)
{

int64_t gcol_kl = cast_l2g->vec_[col_kl] + global_column_end
- global_column_begin;
gst_table[gcol_kl] = zero;
}
}
}
}
}

if(global == true)
{
for(PtrType k = cast_gst->mat_.row_offset[row];
k < cast_gst->mat_.row_offset[row + 1];
++k)
{
if(cast_S->vec_[k + this->nnz_] == false)
{
continue;
}

int col_ik = cast_gst->mat_.col[k];

if(cast_cf->vec_[col_ik + this->nrow_] == 1)
{
gst_table[cast_l2g->vec_[col_ik] + global_column_end - global_column_begin]
= zero;
}
else
{
for(PtrType l = cast_ptr->vec_[col_ik]; l < cast_ptr->vec_[col_ik + 1]; ++l)
{
int64_t gcol_kl = cast_col->vec_[l];

if(gcol_kl >= global_column_begin && gcol_kl < global_column_end)
{
int col_kl = static_cast<int>(gcol_kl - global_column_begin);

int_table[col_kl] = zero;

if(FF1 == true)
{
break;
}
}
else
{
gst_table[gcol_kl + global_column_end - global_column_begin] = zero;

if(FF1 == true)
{
break;
}
}
}
}
}
}


ValueType val_ii = diag.vec_[row];

bool pos_ii = val_ii >= zero;

ValueType sum_k = zero;
ValueType sum_n = zero;

for(PtrType k = row_begin; k < row_end; ++k)
{
int col_ik = this->mat_.col[k];

if(col_ik == row)
{
continue;
}

ValueType val_ik = this->mat_.val[k];

if(cast_S->vec_[k] == true && cast_cf->vec_[col_ik] == 2)
{
ValueType sum_l = zero;

ValueType val_kk = diag.vec_[col_ik];

ValueType val_ki = zero;

PtrType row_begin_k = this->mat_.row_offset[col_ik];
PtrType row_end_k   = this->mat_.row_offset[col_ik + 1];

for(PtrType l = row_begin_k; l < row_end_k; ++l)
{
int col_kl = this->mat_.col[l];

ValueType val_kl = this->mat_.val[l];

bool pos_kl = val_kl >= zero;

if(col_kl == row)
{
if(pos_ii != pos_kl)
{
sum_l += val_kl;
}

val_ki = val_kl;
}
else if(cast_cf->vec_[col_kl] == 1)
{
if(pos_ii != pos_kl)
{
if(int_table.find(col_kl) != int_table.end())
{
sum_l += val_kl;
}
}
}
}

if(global == true)
{
for(PtrType l = cast_gst->mat_.row_offset[col_ik];
l < cast_gst->mat_.row_offset[col_ik + 1];
++l)
{
int col_kl = cast_gst->mat_.col[l];

int64_t gcol_kl
= cast_l2g->vec_[col_kl] + global_column_end - global_column_begin;

ValueType val_kl = cast_gst->mat_.val[l];

bool pos_kl = val_kl >= zero;

if(cast_cf->vec_[col_kl + this->nrow_] == 1)
{
if(pos_ii != pos_kl)
{
if(gst_table.find(gcol_kl) != gst_table.end())
{
sum_l += val_kl;
}
}
}
}
}

sum_l = val_ik / sum_l;

bool pos_kk = val_kk >= zero;
bool pos_ki = val_ki >= zero;

for(PtrType l = row_begin_k; l < row_end_k; ++l)
{
int col_kl = this->mat_.col[l];

if(cast_cf->vec_[col_kl] != 1)
{
continue;
}

ValueType val_kl = this->mat_.val[l];

bool pos_kl = val_kl >= zero;

if(pos_kk != pos_kl)
{
if(int_table.find(col_kl) != int_table.end())
{
int_table[col_kl] += val_kl * sum_l;
}
}
}

if(global == true)
{
row_begin_k = cast_gst->mat_.row_offset[col_ik];
row_end_k   = cast_gst->mat_.row_offset[col_ik + 1];

for(PtrType l = row_begin_k; l < row_end_k; ++l)
{
int col_kl = cast_gst->mat_.col[l];

int64_t gcol_kl
= cast_l2g->vec_[col_kl] + global_column_end - global_column_begin;

ValueType val_kl = cast_gst->mat_.val[l];

bool pos_kl = val_kl >= zero;

if(pos_kk != pos_kl)
{
if(gst_table.find(gcol_kl) != gst_table.end())
{
gst_table[gcol_kl] += val_kl * sum_l;
}
}
}
}

if(pos_kk != pos_ki)
{
sum_k += val_ki * sum_l;
}
}

bool in_C_hat = false;

if(cast_cf->vec_[col_ik] == 1)
{
if(int_table.find(col_ik) != int_table.end())
{
int_table[col_ik] += val_ik;

in_C_hat = true;
}
}

if(in_C_hat == false && cast_S->vec_[k] == false)
{
sum_n += val_ik;
}
}

if(global == true)
{
for(PtrType k = cast_gst->mat_.row_offset[row];
k < cast_gst->mat_.row_offset[row + 1];
++k)
{
int col_ik = cast_gst->mat_.col[k];

ValueType val_ik = cast_gst->mat_.val[k];

if(cast_S->vec_[k + this->nnz_] == true
&& cast_cf->vec_[col_ik + this->nrow_] == 2)
{
ValueType sum_l = zero;

ValueType val_kk = zero;

int64_t grow_k = cast_l2g->vec_[col_ik];

PtrType row_begin_k = cast_ext_ptr->vec_[col_ik];
PtrType row_end_k   = cast_ext_ptr->vec_[col_ik + 1];

for(PtrType l = row_begin_k; l < row_end_k; ++l)
{
int64_t gcol_kl = cast_ext_col->vec_[l];

ValueType val_kl = cast_ext_val->vec_[l];

bool pos_kl = val_kl >= zero;

if(grow_k == gcol_kl)
{
val_kk = val_kl;
}

if(gcol_kl >= global_column_begin && gcol_kl < global_column_end)
{
int col_kl = static_cast<int>(gcol_kl - global_column_begin);

if(col_kl == row)
{
if(pos_ii != pos_kl)
{
sum_l += val_kl;
}
}
else
{
if(pos_ii != pos_kl)
{
if(int_table.find(col_kl) != int_table.end())
{
sum_l += val_kl;
}
}
}
}
else
{
if(pos_ii != pos_kl)
{
if(gst_table.find(gcol_kl + global_column_end
- global_column_begin)
!= gst_table.end())
{
sum_l += val_kl;
}
}
}
}

ValueType val_ki = zero;

sum_l = val_ik / sum_l;

for(PtrType l = row_begin_k; l < row_end_k; ++l)
{
int64_t gcol_kl = cast_ext_col->vec_[l];

ValueType val_kl = cast_ext_val->vec_[l];

bool pos_kl = val_kl >= zero;

if((val_kk >= zero) == pos_kl)
{
val_kl = zero;
}

if(gcol_kl >= global_column_begin && gcol_kl < global_column_end)
{
int col_kl = static_cast<int>(gcol_kl - global_column_begin);

if(row == col_kl)
{
val_ki = val_kl;
}

if(int_table.find(col_kl) != int_table.end())
{
int_table[col_kl] += val_kl * sum_l;
}
}
else
{
if(gst_table.find(gcol_kl + global_column_end - global_column_begin)
!= gst_table.end())
{
gst_table[gcol_kl + global_column_end - global_column_begin]
+= val_kl * sum_l;
}
}
}

sum_n += val_ki * sum_l;
}

int64_t gcol_ik
= cast_l2g->vec_[col_ik] + global_column_end - global_column_begin;

bool in_C_hat = false;

if(gst_table.find(gcol_ik) != gst_table.end())
{
gst_table[gcol_ik] += val_ik;

in_C_hat = true;
}

if(cast_S->vec_[k + this->nnz_] == false && in_C_hat == false)
{
sum_k += val_ik;
}
}
}

ValueType a_ii_tilde = static_cast<ValueType>(-1) / (sum_n + sum_k + val_ii);

PtrType int_idx = cast_pi->mat_.row_offset[row];

for(auto it = int_table.begin(); it != int_table.end(); ++it)
{
cast_pi->mat_.col[int_idx] = cast_f2c->vec_[it->first];
cast_pi->mat_.val[int_idx] = a_ii_tilde * it->second;
++int_idx;
}

if(global == true)
{
PtrType gst_idx = cast_pg->mat_.row_offset[row];

for(auto it = gst_table.begin(); it != gst_table.end(); ++it)
{
cast_glo->vec_[gst_idx] = it->first - global_column_end + global_column_begin;
cast_pg->mat_.val[gst_idx] = a_ii_tilde * it->second;
++gst_idx;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::InitialPairwiseAggregation(ValueType        beta,
int&             nc,
BaseVector<int>* G,
int&             Gsize,
int**            rG,
int&             rGsize,
int              ordering) const
{
assert(G != NULL);

HostVector<int>* cast_G = dynamic_cast<HostVector<int>*>(G);

assert(cast_G != NULL);

for(int i = 0; i < cast_G->size_; ++i)
{
cast_G->vec_[i] = -2;
}

int      Usize    = 0;
PtrType* ind_diag = NULL;
allocate_host(this->nrow_, &ind_diag);

for(int i = 0; i < this->nrow_; ++i)
{
ValueType sum = static_cast<ValueType>(0);

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if(i != this->mat_.col[j])
{
sum += std::abs(this->mat_.val[j]);
}
else
{
ind_diag[i] = j;
}
}

sum *= static_cast<ValueType>(5);

if(this->mat_.val[ind_diag[i]] > sum)
{
cast_G->vec_[i] = -1;
++Usize;
}
}

Gsize  = 2;
rGsize = this->nrow_ - Usize;
allocate_host(Gsize * rGsize, rG);

for(int i = 0; i < Gsize * rGsize; ++i)
{
(*rG)[i] = -1;
}

nc              = 0;
ValueType betam = -beta;

HostVector<int> perm(this->local_backend_);

switch(ordering)
{
case 0: 
break;

case 1: 
this->ConnectivityOrder(&perm);
break;

case 2: 
this->CMK(&perm);
break;

case 3: 
this->RCMK(&perm);
break;

case 4: 
int size;
this->MaximalIndependentSet(size, &perm);
break;

case 5: 
int  num_colors;
int* size_colors = NULL;
this->MultiColoring(num_colors, &size_colors, &perm);
free_host(&size_colors);
break;
}


for(int k = 0; k < this->nrow_; ++k)
{
int i;
if(ordering == 0)
{
i = k;
}
else
{
i = perm.vec_[k];
}

if(cast_G->vec_[i] != -2)
{
continue;
}

cast_G->vec_[i] = nc;
(*rG)[nc]       = i;

ValueType min_a_ij = static_cast<ValueType>(0);
ValueType max_a_ij = static_cast<ValueType>(0);
PtrType   min_j    = -1;
bool      neg      = false;
if(this->mat_.val[ind_diag[i]] < static_cast<ValueType>(0))
{
neg = true;
}

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int       col_j = this->mat_.col[j];
ValueType val_j = this->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(i == col_j)
{
continue;
}

if(min_j == -1)
{
max_a_ij = val_j;
if(cast_G->vec_[col_j] == -2)
{
min_j    = j;
min_a_ij = val_j;
}
}

if(val_j < min_a_ij && cast_G->vec_[col_j] == -2)
{
min_a_ij = val_j;
min_j    = j;
}

if(val_j > max_a_ij)
{
max_a_ij = val_j;
}
}

if(min_j != -1)
{
max_a_ij *= betam;

int       col_j = this->mat_.col[min_j];
ValueType val_j = this->mat_.val[min_j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(val_j < max_a_ij)
{
cast_G->vec_[col_j] = nc;
(*rG)[rGsize + nc]  = col_j;
}
}

++nc;
}

free_host(&ind_diag);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
ValueType                    beta,
int&                         nc,
BaseVector<int>*             G,
int&                         Gsize,
int**                        rG,
int&                         rGsize,
int ordering) const
{
assert(G != NULL);

HostVector<int>*                cast_G = dynamic_cast<HostVector<int>*>(G);
const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

assert(cast_G != NULL);
assert(cast_mat != NULL);

for(int i = 0; i < cast_G->size_; ++i)
{
cast_G->vec_[i] = -2;
}

int      Usize    = 0;
PtrType* ind_diag = NULL;
allocate_host(this->nrow_, &ind_diag);

for(int i = 0; i < this->nrow_; ++i)
{
ValueType sum = static_cast<ValueType>(0);

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if(i != this->mat_.col[j])
{
sum += std::abs(this->mat_.val[j]);
}
else
{
ind_diag[i] = j;
}
}

if(cast_mat->nnz_ > 0)
{
for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1];
++j)
{
sum += std::abs(cast_mat->mat_.val[j]);
}
}

sum *= static_cast<ValueType>(5);

if(this->mat_.val[ind_diag[i]] > sum)
{
cast_G->vec_[i] = -1;
++Usize;
}
}

Gsize  = 2;
rGsize = this->nrow_ - Usize;
allocate_host(Gsize * rGsize, rG);

for(int i = 0; i < Gsize * rGsize; ++i)
{
(*rG)[i] = -1;
}

nc              = 0;
ValueType betam = -beta;

HostVector<int> perm(this->local_backend_);

switch(ordering)
{
case 0: 
break;

case 1: 
this->ConnectivityOrder(&perm);
break;

case 2: 
this->CMK(&perm);
break;

case 3: 
this->RCMK(&perm);
break;

case 4: 
int size;
this->MaximalIndependentSet(size, &perm);
break;

case 5: 
int  num_colors;
int* size_colors = NULL;
this->MultiColoring(num_colors, &size_colors, &perm);
free_host(&size_colors);
break;
}


for(int k = 0; k < this->nrow_; ++k)
{
int i;
if(ordering == 0)
{
i = k;
}
else
{
i = perm.vec_[k];
}

if(cast_G->vec_[i] != -2)
{
continue;
}

cast_G->vec_[i] = nc;
(*rG)[nc]       = i;

ValueType min_a_ij = static_cast<ValueType>(0);
ValueType max_a_ij = static_cast<ValueType>(0);
int       min_j    = -1;
bool      neg      = false;
if(this->mat_.val[ind_diag[i]] < static_cast<ValueType>(0))
{
neg = true;
}

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int       col_j = this->mat_.col[j];
ValueType val_j = this->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(i == col_j)
{
continue;
}

if(min_j == -1)
{
max_a_ij = val_j;
if(cast_G->vec_[col_j] == -2)
{
min_j    = col_j;
min_a_ij = val_j;
}
}

if(val_j < min_a_ij && cast_G->vec_[col_j] == -2)
{
min_a_ij = val_j;
min_j    = col_j;
}

if(val_j > max_a_ij)
{
max_a_ij = val_j;
}
}

if(cast_mat->nnz_ > 0)
{
for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1];
++j)
{
ValueType val_j = cast_mat->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(val_j > max_a_ij)
{
max_a_ij = val_j;
}
}
}

if(min_j != -1)
{
max_a_ij *= betam;

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int       col_j = this->mat_.col[j];
ValueType val_j = this->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(i == col_j)
{
continue;
}

if(cast_G->vec_[col_j] != -2)
{
continue;
}

if(val_j < max_a_ij)
{
if(min_j == col_j)
{
cast_G->vec_[min_j] = nc;
(*rG)[rGsize + nc]  = min_j;
break;
}
}
}
}

++nc;
}

free_host(&ind_diag);

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::FurtherPairwiseAggregation(ValueType        beta,
int&             nc,
BaseVector<int>* G,
int&             Gsize,
int**            rG,
int&             rGsize,
int              ordering) const
{
assert(G != NULL);
HostVector<int>* cast_G = dynamic_cast<HostVector<int>*>(G);
assert(cast_G != NULL);

Gsize *= 2;
int  rGsizec = this->nrow_;
int* rGc     = NULL;
allocate_host(Gsize * rGsizec, &rGc);

for(int i = 0; i < Gsize * rGsizec; ++i)
{
rGc[i] = -1;
}

for(int i = 0; i < cast_G->size_; ++i)
{
cast_G->vec_[i] = -1;
}

int* U = NULL;
allocate_host(this->nrow_, &U);
set_to_zero_host(this->nrow_, U);

nc              = 0;
ValueType betam = -beta;

HostVector<int> perm(this->local_backend_);

switch(ordering)
{
case 0: 
break;

case 1: 
this->ConnectivityOrder(&perm);
break;

case 2: 
this->CMK(&perm);
break;

case 3: 
this->RCMK(&perm);
break;

case 4: 
int size;
this->MaximalIndependentSet(size, &perm);
break;

case 5: 
int  num_colors;
int* size_colors = NULL;
this->MultiColoring(num_colors, &size_colors, &perm);
free_host(&size_colors);
break;
}

for(int k = 0; k < this->nrow_; ++k)
{
int i;
if(ordering == 0)
{
i = k;
}
else
{
i = perm.vec_[k];
}

if(U[i] == 1)
{
continue;
}

U[i] = 1;

for(int r = 0; r < Gsize / 2; ++r)
{
rGc[r * rGsizec + nc] = (*rG)[r * rGsize + i];

if((*rG)[r * rGsize + i] >= 0)
{
cast_G->vec_[(*rG)[r * rGsize + i]] = nc;
}
}

bool neg = false;
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if(i == this->mat_.col[j])
{
if(this->mat_.val[j] < static_cast<ValueType>(0))
{
neg = true;
}

break;
}
}

ValueType min_a_ij = static_cast<ValueType>(0);
ValueType max_a_ij = static_cast<ValueType>(0);
PtrType   min_j    = -1;

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int       col_j = this->mat_.col[j];
ValueType val_j = this->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(i == col_j)
{
continue;
}

if(min_j == -1)
{
max_a_ij = val_j;

if(U[col_j] == 0)
{
min_j    = j;
min_a_ij = max_a_ij;
}
}

if(val_j < min_a_ij && U[col_j] == 0)
{
min_a_ij = val_j;
min_j    = j;
}

if(val_j < max_a_ij)
{
max_a_ij = val_j;
}
}

if(min_j != -1)
{
max_a_ij *= betam;

int       col_j = this->mat_.col[min_j];
ValueType val_j = this->mat_.val[min_j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(val_j < max_a_ij)
{
for(int r = 0; r < Gsize / 2; ++r)
{
rGc[(r + Gsize / 2) * rGsizec + nc] = (*rG)[r * rGsize + col_j];

if((*rG)[r * rGsize + col_j] >= 0)
{
cast_G->vec_[(*rG)[r * rGsize + col_j]] = nc;
}
}

U[col_j] = 1;
}
}

++nc;
}

free_host(&U);
free_host(rG);

(*rG)  = rGc;
rGsize = rGsizec;

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
ValueType                    beta,
int&                         nc,
BaseVector<int>*             G,
int&                         Gsize,
int**                        rG,
int&                         rGsize,
int ordering) const
{
assert(G != NULL);

HostVector<int>*                cast_G = dynamic_cast<HostVector<int>*>(G);
const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

assert(cast_G != NULL);
assert(cast_mat != NULL);

Gsize *= 2;
int  rGsizec = this->nrow_;
int* rGc     = NULL;
allocate_host(Gsize * rGsizec, &rGc);

for(int i = 0; i < Gsize * rGsizec; ++i)
{
rGc[i] = -1;
}

for(int i = 0; i < cast_G->size_; ++i)
{
cast_G->vec_[i] = -1;
}

int* U = NULL;
allocate_host(this->nrow_, &U);
set_to_zero_host(this->nrow_, U);

nc              = 0;
ValueType betam = -beta;

HostVector<int> perm(this->local_backend_);

switch(ordering)
{
case 0: 
break;

case 1: 
this->ConnectivityOrder(&perm);
break;

case 2: 
this->CMK(&perm);
break;

case 3: 
this->RCMK(&perm);
break;

case 4: 
int size;
this->MaximalIndependentSet(size, &perm);
break;

case 5: 
int  num_colors;
int* size_colors = NULL;
this->MultiColoring(num_colors, &size_colors, &perm);
free_host(&size_colors);
break;
}

for(int k = 0; k < this->nrow_; ++k)
{
int i;
if(ordering == 0)
{
i = k;
}
else
{
i = perm.vec_[k];
}

if(U[i] == 1)
{
continue;
}

U[i] = 1;

for(int r = 0; r < Gsize / 2; ++r)
{
rGc[r * rGsizec + nc] = (*rG)[r * rGsize + i];

if((*rG)[r * rGsize + i] >= 0)
{
cast_G->vec_[(*rG)[r * rGsize + i]] = nc;
}
}

bool neg = false;
for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if(i == this->mat_.col[j])
{
if(this->mat_.val[j] < static_cast<ValueType>(0))
{
neg = true;
}

break;
}
}

ValueType min_a_ij = static_cast<ValueType>(0);
ValueType max_a_ij = static_cast<ValueType>(0);
int       min_j    = -1;

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int       col_j = this->mat_.col[j];
ValueType val_j = this->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(i == col_j)
{
continue;
}

if(min_j == -1)
{
max_a_ij = val_j;
if(U[col_j] == 0)
{
min_j    = col_j;
min_a_ij = max_a_ij;
}
}

if(val_j < min_a_ij && U[col_j] == 0)
{
min_a_ij = val_j;
min_j    = col_j;
}

if(val_j < max_a_ij)
{
max_a_ij = val_j;
}
}

if(cast_mat->nnz_ > 0)
{
for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1];
++j)
{
ValueType val_j = cast_mat->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(val_j > max_a_ij)
{
max_a_ij = val_j;
}
}
}

if(min_j != -1)
{
max_a_ij *= betam;

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int       col_j = this->mat_.col[j];
ValueType val_j = this->mat_.val[j];

if(neg == true)
{
val_j *= static_cast<ValueType>(-1);
}

if(i == col_j)
{
continue;
}

if(U[col_j] == 1)
{
continue;
}

if(val_j < max_a_ij)
{
if(min_j == col_j)
{
for(int r = 0; r < Gsize / 2; ++r)
{
rGc[(r + Gsize / 2) * rGsizec + nc] = (*rG)[r * rGsize + min_j];

if((*rG)[r * rGsize + min_j] >= 0)
{
cast_G->vec_[(*rG)[r * rGsize + min_j]] = nc;
}
}

U[min_j] = 1;
break;
}
}
}
}

++nc;
}

free_host(&U);
free_host(rG);

(*rG)  = rGc;
rGsize = rGsizec;

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::CoarsenOperator(BaseMatrix<ValueType>* Ac,
int                    nrow,
int                    ncol,
const BaseVector<int>& G,
int                    Gsize,
const int*             rG,
int                    rGsize) const
{
assert(Ac != NULL);

HostMatrixCSR<ValueType>* cast_Ac = dynamic_cast<HostMatrixCSR<ValueType>*>(Ac);
const HostVector<int>*    cast_G  = dynamic_cast<const HostVector<int>*>(&G);

assert(cast_Ac != NULL);
assert(cast_G != NULL);

cast_Ac->Clear();

PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

allocate_host(nrow + 1, &row_offset);
allocate_host(this->nnz_, &col);
allocate_host(this->nnz_, &val);


PtrType* reverse_col = NULL;
int*     Gl          = NULL;
int*     erase       = NULL;

int size = (nrow > ncol) ? nrow : ncol;

allocate_host(size, &reverse_col);
allocate_host(size, &Gl);
allocate_host(size, &erase);

for(int i = 0; i < size; ++i)
{
reverse_col[i] = -1;
}

set_to_zero_host(size, Gl);

row_offset[0] = 0;

for(int k = 0; k < nrow; ++k)
{
row_offset[k + 1] = row_offset[k];

int m = 0;

for(int r = 0; r < Gsize; ++r)
{
int i = rG[r * rGsize + k];

if(i < 0)
{
continue;
}

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
int l = cast_G->vec_[this->mat_.col[j]];

if(l < 0)
{
continue;
}

if(Gl[l] == 0)
{
Gl[l]      = 1;
erase[m++] = l;

col[row_offset[k + 1]] = l;
val[row_offset[k + 1]] = this->mat_.val[j];
reverse_col[l]         = row_offset[k + 1];

++row_offset[k + 1];
}
else
{
val[reverse_col[l]] += this->mat_.val[j];
}
}
}

for(int j = 0; j < m; ++j)
{
Gl[erase[j]] = 0;
}
}

free_host(&reverse_col);
free_host(&Gl);
free_host(&erase);

PtrType nnz = row_offset[nrow];

int*       col_resized = NULL;
ValueType* val_resized = NULL;

allocate_host(nnz, &col_resized);
allocate_host(nnz, &val_resized);

copy_h2h(nnz, col, col_resized);
copy_h2h(nnz, val, val_resized);

free_host(&col);
free_host(&val);

cast_Ac->Clear();
cast_Ac->SetDataPtrCSR(&row_offset, &col_resized, &val_resized, nnz, nrow, nrow);

return true;
}

template <typename T>
int sgn(T val)
{
return (T(0) < val) - (val < T(0));
}

template <typename ValueType>
bool
HostMatrixCSR<ValueType>::Key(long int& row_key, long int& col_key, long int& val_key) const
{
row_key = 0;
col_key = 0;
val_key = 0;

int row_sign = 1;
int col_sign = 1;
int val_sign = 1;

int row_tmp = 0x12345678;
int col_tmp = 0x23456789;
int val_tmp = 0x34567890;

int row_mask = 0x09876543;
int col_mask = 0x98765432;
int val_mask = 0x87654321;

for(int ai = 0; ai < this->nrow_; ++ai)
{
row_key += row_sign * row_tmp * (row_mask & this->mat_.row_offset[ai]);
row_key  = row_key ^ (row_key >> 16);
row_sign = sgn(row_tmp - (row_mask & this->mat_.row_offset[ai]));
row_tmp  = row_mask & this->mat_.row_offset[ai];

PtrType row_beg = this->mat_.row_offset[ai];
PtrType row_end = this->mat_.row_offset[ai + 1];

for(PtrType aj = row_beg; aj < row_end; ++aj)
{
col_key += col_sign * col_tmp * (col_mask | this->mat_.col[aj]);
col_key  = col_key ^ (col_key >> 16);
col_sign = sgn(row_tmp - (col_mask | this->mat_.col[aj]));
col_tmp  = col_mask | this->mat_.col[aj];

double   double_val = std::abs(this->mat_.val[aj]);
long int val        = 0;

assert(sizeof(long int) == sizeof(double));

memcpy(&val, &double_val, sizeof(long int));

val_key += val_sign * val_tmp * (long int)(val_mask | val);
val_key = val_key ^ (val_key >> 16);

if(sgn(this->mat_.val[aj]) > 0)
{
val_key = val_key ^ val;
}
else
{
val_key = val_key | val;
}

val_sign = sgn(val_tmp - (long int)(val_mask | val));
val_tmp  = val_mask | val;
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec)
{
assert(vec.GetSize() == this->nrow_);

if(this->nnz_ > 0)
{
const HostVector<ValueType>* cast_vec
= dynamic_cast<const HostVector<ValueType>*>(&vec);
assert(cast_vec != NULL);

PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

int nrow = this->nrow_;
int ncol = this->ncol_;

allocate_host(nrow + 1, &row_offset);
row_offset[0] = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < nrow; ++i)
{
bool add = true;

row_offset[i + 1] = this->mat_.row_offset[i + 1] - this->mat_.row_offset[i];

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
if(this->mat_.col[j] == idx)
{
add = false;
break;
}
}

if(add == true && cast_vec->vec_[i] != static_cast<ValueType>(0))
{
++row_offset[i + 1];
}

if(add == false && cast_vec->vec_[i] == static_cast<ValueType>(0))
{
--row_offset[i + 1];
}
}

for(int i = 0; i < nrow; ++i)
{
row_offset[i + 1] += row_offset[i];
}

PtrType nnz = row_offset[nrow];

allocate_host(nnz, &col);
allocate_host(nnz, &val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < nrow; ++i)
{
PtrType k = row_offset[i];
PtrType j = this->mat_.row_offset[i];

for(; j < this->mat_.row_offset[i + 1]; ++j)
{
if(this->mat_.col[j] < idx)
{
col[k] = this->mat_.col[j];
val[k] = this->mat_.val[j];
++k;
}
else
{
break;
}
}

if(cast_vec->vec_[i] != static_cast<ValueType>(0))
{
col[k] = idx;
val[k] = cast_vec->vec_[i];
++k;
++j;
}

for(; j < this->mat_.row_offset[i + 1]; ++j)
{
if(this->mat_.col[j] > idx)
{
col[k] = this->mat_.col[j];
val[k] = this->mat_.val[j];
++k;
}
}
}

this->Clear();
this->SetDataPtrCSR(&row_offset, &col, &val, row_offset[nrow], nrow, ncol);
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const
{
assert(vec != NULL);
assert(vec->GetSize() == this->nrow_);

if(this->nnz_ > 0)
{
HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
assert(cast_vec != NULL);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->nrow_; ++ai)
{
cast_vec->vec_[ai] = static_cast<ValueType>(0);

for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
++aj)
{
if(idx == this->mat_.col[aj])
{
cast_vec->vec_[ai] = this->mat_.val[aj];
break;
}
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ReplaceRowVector(int idx, const BaseVector<ValueType>& vec)
{
assert(vec.GetSize() == this->ncol_);

if(this->nnz_ > 0)
{
const HostVector<ValueType>* cast_vec
= dynamic_cast<const HostVector<ValueType>*>(&vec);
assert(cast_vec != NULL);

PtrType*   row_offset = NULL;
int*       col        = NULL;
ValueType* val        = NULL;

int nrow = this->nrow_;
int ncol = this->ncol_;

allocate_host(nrow + 1, &row_offset);
row_offset[0] = 0;

int nnz_idx = 0;

for(int i = 0; i < ncol; ++i)
{
if(cast_vec->vec_[i] != static_cast<ValueType>(0))
{
++nnz_idx;
}
}

PtrType shift = nnz_idx - this->mat_.row_offset[idx + 1] + this->mat_.row_offset[idx];

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < nrow + 1; ++i)
{
if(i < idx + 1)
{
row_offset[i] = this->mat_.row_offset[i];
}
else
{
row_offset[i] = this->mat_.row_offset[i] + shift;
}
}

PtrType nnz = row_offset[nrow];

allocate_host(nnz, &col);
allocate_host(nnz, &val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int i = 0; i < nrow; ++i)
{
if(i < idx)
{
for(PtrType j = row_offset[i]; j < row_offset[i + 1]; ++j)
{
col[j] = this->mat_.col[j];
val[j] = this->mat_.val[j];
}

}
else if(i == idx)
{
PtrType k = row_offset[i];

for(int j = 0; j < ncol; ++j)
{
if(cast_vec->vec_[j] != static_cast<ValueType>(0))
{
col[k] = j;
val[k] = cast_vec->vec_[j];
++k;
}
}

}
else if(i > idx)
{
PtrType k = row_offset[i];

for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
{
col[k] = this->mat_.col[j];
val[k] = this->mat_.val[j];
++k;
}
}
}

this->Clear();
this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractRowVector(int idx, BaseVector<ValueType>* vec) const
{
assert(vec != NULL);
assert(vec->GetSize() == this->ncol_);

if(this->nnz_ > 0)
{
HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
assert(cast_vec != NULL);

_set_omp_backend_threads(this->local_backend_, this->nrow_);

cast_vec->Zeros();

for(PtrType aj = this->mat_.row_offset[idx]; aj < this->mat_.row_offset[idx + 1]; ++aj)
{
cast_vec->vec_[this->mat_.col[aj]] = this->mat_.val[aj];
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractGlobalColumnIndices(int     ncol,
int64_t global_offset,
const BaseVector<int64_t>& l2g,
BaseVector<int64_t>* global_col) const
{
if(this->nnz_ > 0)
{
const HostVector<int64_t>* cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
HostVector<int64_t>*       cast_col = dynamic_cast<HostVector<int64_t>*>(global_col);

assert(cast_col != NULL);
assert(this->nnz_ == cast_col->size_);

for(int64_t i = 0; i < this->nnz_; ++i)
{
int local_col = this->mat_.col[i];

if(local_col >= ncol)
{
cast_col->vec_[i] = cast_l2g->vec_[local_col - ncol];
}
else
{
cast_col->vec_[i] = local_col + global_offset;
}
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractExtRowNnz(int offset, BaseVector<PtrType>* row_nnz) const
{
assert(row_nnz != NULL);

if(this->GetNnz() > 0)
{
HostVector<PtrType>* cast_vec = dynamic_cast<HostVector<PtrType>*>(row_nnz);

assert(cast_vec != NULL);

int size = this->nrow_ - offset;

for(int i = 0; i < size; ++i)
{
cast_vec->vec_[i]
= this->mat_.row_offset[i + 1 + offset] - this->mat_.row_offset[i + offset];
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractBoundaryRowNnz(BaseVector<PtrType>*   row_nnz,
const BaseVector<int>& boundary_index,
const BaseMatrix<ValueType>& gst) const
{
assert(row_nnz != NULL);

HostVector<PtrType>*   cast_vec = dynamic_cast<HostVector<PtrType>*>(row_nnz);
const HostVector<int>* cast_idx = dynamic_cast<const HostVector<int>*>(&boundary_index);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&gst);

assert(cast_vec != NULL);
assert(cast_idx != NULL);
assert(cast_gst != NULL);

for(int64_t i = 0; i < cast_idx->size_; ++i)
{
int row = cast_idx->vec_[i];

cast_vec->vec_[i] = this->mat_.row_offset[row + 1] - this->mat_.row_offset[row]
+ cast_gst->mat_.row_offset[row + 1]
- cast_gst->mat_.row_offset[row];
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::ExtractBoundaryRows(const BaseVector<PtrType>& bnd_csr_row_ptr,
BaseVector<int64_t>*       bnd_csr_col_ind,
BaseVector<ValueType>*     bnd_csr_val,
int64_t                global_column_offset,
const BaseVector<int>& boundary_index,
const BaseVector<int64_t>&   ghost_mapping,
const BaseMatrix<ValueType>& gst) const
{
assert(bnd_csr_col_ind != NULL);
assert(bnd_csr_val != NULL);

const HostVector<PtrType>* cast_ptr
= dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
HostVector<int64_t>*       cast_col = dynamic_cast<HostVector<int64_t>*>(bnd_csr_col_ind);
HostVector<ValueType>*     cast_val = dynamic_cast<HostVector<ValueType>*>(bnd_csr_val);
const HostVector<int>*     cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary_index);
const HostVector<int64_t>* cast_l2g
= dynamic_cast<const HostVector<int64_t>*>(&ghost_mapping);
const HostMatrixCSR<ValueType>* cast_gst
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&gst);

assert(cast_ptr != NULL);
assert(cast_col != NULL);
assert(cast_val != NULL);
assert(cast_bnd != NULL);
assert(cast_l2g != NULL);
assert(cast_gst != NULL);

for(int64_t i = 0; i < cast_bnd->size_; ++i)
{
int row = cast_bnd->vec_[i];

PtrType idx = cast_ptr->vec_[i];

PtrType row_begin = this->mat_.row_offset[row];
PtrType row_end   = this->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
cast_col->vec_[idx] = this->mat_.col[j] + global_column_offset;
cast_val->vec_[idx] = this->mat_.val[j];

++idx;
}

row_begin = cast_gst->mat_.row_offset[row];
row_end   = cast_gst->mat_.row_offset[row + 1];

for(PtrType j = row_begin; j < row_end; ++j)
{
cast_col->vec_[idx] = cast_l2g->vec_[cast_gst->mat_.col[j]];
cast_val->vec_[idx] = cast_gst->mat_.val[j];

++idx;
}
}

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::CopyGhostFromGlobalReceive(
const BaseVector<int>&       boundary,
const BaseVector<PtrType>&   recv_csr_row_ptr,
const BaseVector<int64_t>&   recv_csr_col_ind,
const BaseVector<ValueType>& recv_csr_val,
BaseVector<int64_t>*         global_col)
{
const HostVector<int>*     cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary);
const HostVector<PtrType>* cast_ptr
= dynamic_cast<const HostVector<PtrType>*>(&recv_csr_row_ptr);
const HostVector<int64_t>* cast_col
= dynamic_cast<const HostVector<int64_t>*>(&recv_csr_col_ind);
const HostVector<ValueType>* cast_val
= dynamic_cast<const HostVector<ValueType>*>(&recv_csr_val);
HostVector<int64_t>* cast_glo = dynamic_cast<HostVector<int64_t>*>(global_col);

assert(cast_bnd != NULL);
assert(cast_ptr != NULL);
assert(cast_col != NULL);
assert(cast_val != NULL);

for(int i = 0; i < cast_bnd->size_; ++i)
{
int     row = cast_bnd->vec_[i];
PtrType nnz = cast_ptr->vec_[i + 1] - cast_ptr->vec_[i];

this->mat_.row_offset[row + 1] += nnz;
}

this->mat_.row_offset[0] = 0;
for(int i = 0; i < this->nrow_; ++i)
{
this->mat_.row_offset[i + 1] += this->mat_.row_offset[i];
}

assert(this->mat_.row_offset[this->nrow_] == this->nnz_);

cast_glo->Allocate(this->nnz_);

for(int i = 0; i < cast_bnd->size_; ++i)
{
int row = cast_bnd->vec_[i];

PtrType row_begin = cast_ptr->vec_[i];
PtrType row_end   = cast_ptr->vec_[i + 1];

PtrType idx = this->mat_.row_offset[row];

for(PtrType j = row_begin; j < row_end; ++j)
{
cast_glo->vec_[idx] = cast_col->vec_[j];
this->mat_.val[idx] = cast_val->vec_[j];

++idx;
}

this->mat_.row_offset[row] = idx;
}

for(int i = this->nrow_; i > 0; --i)
{
this->mat_.row_offset[i] = this->mat_.row_offset[i - 1];
}

this->mat_.row_offset[0] = 0;

return true;
}

template <typename ValueType>
bool HostMatrixCSR<ValueType>::RenumberGlobalToLocal(const BaseVector<int64_t>& column_indices)
{
if(this->nnz_ > 0)
{
const HostVector<int64_t>* cast_col
= dynamic_cast<const HostVector<int64_t>*>(&column_indices);

assert(cast_col != NULL);

HostVector<int>     perm(this->local_backend_);
HostVector<int64_t> sorted(this->local_backend_);
HostVector<int>     workspace(this->local_backend_);

perm.Allocate(this->nnz_);
sorted.Allocate(this->nnz_);
workspace.Allocate(this->nnz_);

cast_col->Sort(&sorted, &perm);


for(int64_t i = 0; i < this->nnz_; ++i)
{
if(i == 0)
{
workspace.vec_[0] = 1;

continue;
}

int64_t global_col = sorted.vec_[i];

if(global_col == sorted.vec_[i - 1])
{
workspace.vec_[i] = 0;
}
else
{
workspace.vec_[i] = 1;
}
}

this->ncol_ = workspace.InclusiveSum(workspace);

for(int64_t i = 0; i < this->nnz_; ++i)
{
this->mat_.col[perm.vec_[i]] = workspace.vec_[i] - 1;
}
}

return true;
}

template class HostMatrixCSR<double>;
template class HostMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HostMatrixCSR<std::complex<double>>;
template class HostMatrixCSR<std::complex<float>>;
#endif

} 
