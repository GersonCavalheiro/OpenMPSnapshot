

#include "host_matrix_bcsr.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../matrix_formats_ind.hpp"
#include "host_conversion.hpp"
#include "host_matrix_csr.hpp"
#include "host_vector.hpp"

#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution
{

template <typename ValueType>
HostMatrixBCSR<ValueType>::HostMatrixBCSR()
{
LOG_INFO("no default constructor");
FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
HostMatrixBCSR<ValueType>::HostMatrixBCSR(const Rocalution_Backend_Descriptor& local_backend,
int                                  blockdim)
{
log_debug(this, "HostMatrixBCSR::HostMatrixBCSR()", "constructor with local_backend");

this->mat_.row_offset = NULL;
this->mat_.col        = NULL;
this->mat_.val        = NULL;
this->mat_.blockdim   = blockdim;

this->set_backend(local_backend);
}

template <typename ValueType>
HostMatrixBCSR<ValueType>::~HostMatrixBCSR()
{
log_debug(this, "HostMatrixBCSR::~HostMatrixBCSR()", "destructor");

this->Clear();
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::Info(void) const
{
LOG_INFO("HostMatrixBCSR<ValueType>");
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::Clear()
{
if(this->nnz_ > 0)
{
free_host(&this->mat_.row_offset);
free_host(&this->mat_.col);
free_host(&this->mat_.val);

this->nrow_ = 0;
this->ncol_ = 0;
this->nnz_  = 0;
}
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::AllocateBCSR(int64_t nnzb, int nrowb, int ncolb, int blockdim)
{
assert(nnzb >= 0);
assert(ncolb >= 0);
assert(nrowb >= 0);
assert(blockdim > 1);

if(this->nnz_ > 0)
{
this->Clear();
}

if(nnzb > 0)
{
allocate_host(nrowb + 1, &this->mat_.row_offset);
allocate_host(nnzb, &this->mat_.col);
allocate_host(nnzb * blockdim * blockdim, &this->mat_.val);

set_to_zero_host(nrowb + 1, this->mat_.row_offset);
set_to_zero_host(nnzb, this->mat_.col);
set_to_zero_host(nnzb * blockdim * blockdim, this->mat_.val);

this->nrow_ = nrowb * blockdim;
this->ncol_ = ncolb * blockdim;
this->nnz_  = nnzb * blockdim * blockdim;

this->mat_.nrowb = nrowb;
this->mat_.ncolb = ncolb;
this->mat_.nnzb  = nnzb;

this->mat_.blockdim = blockdim;
}
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::SetDataPtrBCSR(int**       row_offset,
int**       col,
ValueType** val,
int64_t     nnzb,
int         nrowb,
int         ncolb,
int         blockdim)
{
assert(*row_offset != NULL);
assert(*col != NULL);
assert(*val != NULL);
assert(nnzb > 0);
assert(nrowb > 0);
assert(ncolb > 0);
assert(blockdim > 1);

this->Clear();

this->nrow_ = nrowb * blockdim;
this->ncol_ = ncolb * blockdim;
this->nnz_  = nnzb * blockdim * blockdim;

this->mat_.nrowb    = nrowb;
this->mat_.ncolb    = ncolb;
this->mat_.nnzb     = nnzb;
this->mat_.blockdim = blockdim;

this->mat_.row_offset = *row_offset;
this->mat_.col        = *col;
this->mat_.val        = *val;
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::LeaveDataPtrBCSR(int**       row_offset,
int**       col,
ValueType** val,
int&        blockdim)
{
assert(this->nrow_ > 0);
assert(this->ncol_ > 0);
assert(this->nnz_ > 0);
assert(this->mat_.blockdim > 1);

*row_offset = this->mat_.row_offset;
*col        = this->mat_.col;
*val        = this->mat_.val;

this->mat_.row_offset = NULL;
this->mat_.col        = NULL;
this->mat_.val        = NULL;

blockdim = this->mat_.blockdim;

this->mat_.blockdim = 0;

this->nrow_ = 0;
this->ncol_ = 0;
this->nnz_  = 0;
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
{
assert(this->GetMatFormat() == mat.GetMatFormat());
assert(this->GetMatBlockDimension() == mat.GetMatBlockDimension());

if(const HostMatrixBCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
{
this->AllocateBCSR(cast_mat->mat_.nnzb,
cast_mat->mat_.nrowb,
cast_mat->mat_.ncolb,
cast_mat->mat_.blockdim);

assert((this->nnz_ == cast_mat->nnz_) && (this->nrow_ == cast_mat->nrow_)
&& (this->ncol_ == cast_mat->ncol_));

if(this->nnz_ > 0)
{
copy_h2h(this->mat_.nrowb + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
copy_h2h(this->mat_.nnzb, cast_mat->mat_.col, this->mat_.col);
copy_h2h(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
cast_mat->mat_.val,
this->mat_.val);
}
}
else
{
mat.CopyTo(this);
}
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
{
mat->CopyFrom(*this);
}

template <typename ValueType>
bool HostMatrixBCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
{
this->Clear();

if(mat.GetNnz() == 0)
{
return true;
}

if(const HostMatrixBCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
{
this->CopyFrom(*cast_mat);
return true;
}

if(const HostMatrixCSR<ValueType>* cast_mat
= dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
{
this->Clear();

if(csr_to_bcsr(this->local_backend_.OpenMP_threads,
cast_mat->nnz_,
cast_mat->nrow_,
cast_mat->ncol_,
cast_mat->mat_,
&this->mat_)
== true)
{
this->nrow_ = this->mat_.nrowb * this->mat_.blockdim;
this->ncol_ = this->mat_.ncolb * this->mat_.blockdim;
this->nnz_  = this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim;

return true;
}
}

return false;
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
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

_set_omp_backend_threads(this->local_backend_, this->mat_.nrowb);

int bsrdim = this->mat_.blockdim;

#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->mat_.nrowb; ++ai)
{
for(int bi = 0; bi < bsrdim; ++bi)
{
int row_begin = this->mat_.row_offset[ai];
int row_end   = this->mat_.row_offset[ai + 1];

ValueType sum = static_cast<ValueType>(0);

for(int aj = row_begin; aj < row_end; ++aj)
{
int col = this->mat_.col[aj];

for(int bj = 0; bj < bsrdim; ++bj)
{
sum += this->mat_.val[BCSR_IND(bsrdim * bsrdim * aj, bi, bj, bsrdim)]
* cast_in->vec_[bsrdim * col + bj];
}
}

cast_out->vec_[ai * bsrdim + bi] = sum;
}
}
}
}

template <typename ValueType>
void HostMatrixBCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

assert(this->nrow_ == this->ncol_);

int bsrdim = this->mat_.blockdim;
#ifdef _OPENMP
#pragma omp parallel for
#endif
for(int ai = 0; ai < this->mat_.nrowb; ++ai)
{
for(int bi = 0; bi < bsrdim; ++bi)
{
int row_begin = this->mat_.row_offset[ai];
int row_end   = this->mat_.row_offset[ai + 1];

ValueType sum = static_cast<ValueType>(0);

for(int aj = row_begin; aj < row_end; ++aj)
{
int col = this->mat_.col[aj];

for(int bj = 0; bj < bsrdim; ++bj)
{
sum += this->mat_.val[BCSR_IND(bsrdim * bsrdim * aj, bi, bj, bsrdim)]
* cast_in->vec_[bsrdim * col + bj];
}
}

cast_out->vec_[ai * bsrdim + bi] += scalar * sum;
}
}
}
}

template class HostMatrixBCSR<double>;
template class HostMatrixBCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HostMatrixBCSR<std::complex<double>>;
template class HostMatrixBCSR<std::complex<float>>;
#endif

} 