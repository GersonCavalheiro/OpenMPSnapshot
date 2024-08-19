#include "tl_dense_symmetric_matrix_impl_lapack.h"

#include <cassert>
#include <iostream>

#include "TlUtils.h"
#include "lapack.h"
#include "tl_dense_general_matrix_lapack.h"
#include "tl_dense_vector_impl_lapack.h"

TlDenseSymmetricMatrix_ImplLapack::TlDenseSymmetricMatrix_ImplLapack(
const TlMatrixObject::index_type dim, double const* const pBuf)
: TlDenseGeneralMatrix_ImplLapack(dim, dim) {
if (pBuf != NULL) {
this->vtr2mat(pBuf);
}
}

TlDenseSymmetricMatrix_ImplLapack::TlDenseSymmetricMatrix_ImplLapack(
const TlDenseSymmetricMatrix_ImplLapack& rhs)
: TlDenseGeneralMatrix_ImplLapack(rhs.getNumOfRows(), rhs.getNumOfCols()) {
std::copy(rhs.matrix_, rhs.matrix_ + rhs.getNumOfElements(), this->matrix_);
}

TlDenseSymmetricMatrix_ImplLapack::TlDenseSymmetricMatrix_ImplLapack(
const TlDenseGeneralMatrix_ImplLapack& rhs)
: TlDenseGeneralMatrix_ImplLapack(rhs.getNumOfRows(), rhs.getNumOfRows()) {
const TlMatrixObject::index_type dim = rhs.getNumOfRows();

const char UPLO = 'U';
const int N = dim;
const int LDA = std::max(1, N);
int INFO;
dtrttp_(&UPLO, &N, rhs.matrix_, &LDA, this->matrix_, &INFO);
if (INFO != 0) {
this->log_.critical(
TlUtils::format("program error: %d@%s", __LINE__, __FILE__));
}
}

TlDenseSymmetricMatrix_ImplLapack::~TlDenseSymmetricMatrix_ImplLapack() {}

TlDenseSymmetricMatrix_ImplLapack::operator std::vector<double>() const {
std::vector<double> answer(this->matrix_,
this->matrix_ + this->getNumOfElements());
return answer;
}

void TlDenseSymmetricMatrix_ImplLapack::resize(TlMatrixObject::index_type row,
TlMatrixObject::index_type col) {
const TlMatrixObject::index_type& dim = row;
assert(row == col);
assert(dim > 0);

TlDenseSymmetricMatrix_ImplLapack oldMatrix(*this);

this->row_ = row;
this->col_ = col;
this->initialize(true);

const TlMatrixObject::index_type dimForCopy =
std::min<TlMatrixObject::index_type>(oldMatrix.getNumOfRows(), dim);

#pragma omp parallel for
for (TlMatrixObject::index_type i = 0; i < dimForCopy; ++i) {
for (TlMatrixObject::index_type j = 0; j <= i; ++j) {
this->set(i, j, oldMatrix.get(i, j));
}
}
}

TlDenseSymmetricMatrix_ImplLapack& TlDenseSymmetricMatrix_ImplLapack::
operator*=(const double coef) {
const int n = this->getNumOfElements();
const int incx = 1;
dscal_(&n, &coef, this->matrix_, &incx);

return *this;
}



TlDenseSymmetricMatrix_ImplLapack TlDenseSymmetricMatrix_ImplLapack::transpose()
const {
return *this;
}

TlDenseSymmetricMatrix_ImplLapack TlDenseSymmetricMatrix_ImplLapack::inverse()
const {
TlDenseSymmetricMatrix_ImplLapack answer = *this;

char UPLO = 'U';

const int N = this->getNumOfRows();

double* AP = answer.matrix_;

int* IPIV = new int[N];

double* WORK = new double[N];

int INFO = 0;

dsptrf_(&UPLO, &N, AP, IPIV, &INFO);
if (INFO == 0) {
dsptri_(&UPLO, &N, AP, IPIV, WORK, &INFO);
if (INFO != 0) {
this->log_.critical(
TlUtils::format("dsptri() return code = %d. (%d@%s)", INFO,
__LINE__, __FILE__));
}
} else {
this->log_.critical(TlUtils::format(
"dsptrf() return code = %d. (%d@%s)", INFO, __LINE__, __FILE__));
}

delete[] IPIV;
IPIV = NULL;
delete[] WORK;
WORK = NULL;

return answer;
}

bool TlDenseSymmetricMatrix_ImplLapack::eig(
TlDenseVector_ImplLapack* pEigVal,
TlDenseGeneralMatrix_ImplLapack* pEigVec) const {
bool answer = false;
const char JOBZ = 'V';  
const char UPLO = 'U';
const int N = this->getNumOfRows();  

assert(this->getNumOfElements() == N * (N + 1) / 2);
double* AP = new double[this->getNumOfElements()];
std::copy(this->matrix_, this->matrix_ + this->getNumOfElements(), AP);

pEigVal->resize(N);
double* W = pEigVal->vector_;

const int LDZ = N;
pEigVec->resize(LDZ, N);
double* Z = pEigVec->matrix_;

double* WORK = new double[3 * N];  
int INFO = 0;                      

dspev_(&JOBZ, &UPLO, &N, AP, W, Z, &LDZ, WORK, &INFO);

if (INFO == 0) {
answer = true;
} else {
this->log_.critical(
TlUtils::format("dspev calculation faild: INFO=%d (%d@%s)", INFO,
__LINE__, __FILE__));
answer = false;
}

delete[] WORK;
WORK = NULL;
delete[] AP;
AP = NULL;

return answer;
}


TlMatrixObject::size_type TlDenseSymmetricMatrix_ImplLapack::getNumOfElements() const {
const TlMatrixObject::index_type dim = this->getNumOfRows();
assert(dim == this->getNumOfCols());

const std::size_t elements = dim * (dim + 1) / 2;

return elements;
}

TlMatrixObject::size_type TlDenseSymmetricMatrix_ImplLapack::index(
TlMatrixObject::index_type row, TlMatrixObject::index_type col) const {
assert((0 <= row) && (row < this->getNumOfRows()));
assert((0 <= col) && (col < this->getNumOfCols()));

if (row < col) {
std::swap(row, col);
}

const std::size_t r = row;
const std::size_t c = col;
const std::size_t index = r * (r + 1) / 2 + c;

return index;
}

void TlDenseSymmetricMatrix_ImplLapack::vtr2mat(double const* const pBuf) {
std::copy(pBuf, pBuf + this->getNumOfElements(), this->matrix_);
}


TlDenseGeneralMatrix_ImplLapack operator*(
const TlDenseSymmetricMatrix_ImplLapack& rhs1,
const TlDenseGeneralMatrix_ImplLapack& rhs2) {
TlDenseGeneralMatrix_ImplLapack gen_rhs1 = rhs1;

return gen_rhs1 * rhs2;
}

TlDenseGeneralMatrix_ImplLapack operator*(
const TlDenseGeneralMatrix_ImplLapack& rhs1,
const TlDenseSymmetricMatrix_ImplLapack& rhs2) {
TlDenseGeneralMatrix_ImplLapack gen_rhs2 = rhs2;

return rhs1 * gen_rhs2;
}

TlDenseVector_ImplLapack operator*(const TlDenseSymmetricMatrix_ImplLapack& mat,
const TlDenseVector_ImplLapack& vec) {
TlLogging& logger = TlLogging::getInstance();
if (mat.getNumOfCols() != vec.getSize()) {
logger.critical(TlUtils::format("size mismatch: %d != %d (%d@%s)",
mat.getNumOfCols(), vec.getSize(),
__LINE__, __FILE__));
}

TlDenseVector_ImplLapack answer(vec.getSize());

const char UPLO =
'U';  
const int N = mat.getNumOfRows();
const double ALPHA = 1.0;  
const double* AP = mat.matrix_;
const double* pX = vec.vector_;
const int INCX = 1;
double BETA = 1.0;
double* Y = answer.vector_;
const int INCY = 1;

dspmv_(&UPLO, &N, &ALPHA, AP, pX, &INCX, &BETA, Y, &INCY);

return answer;
}
