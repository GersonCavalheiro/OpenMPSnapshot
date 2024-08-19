
#ifndef DFXMATRIX_H
#define DFXMATRIX_H

#include <cassert>
#include <cmath>

#include "DfObject.h"
#include "tl_dense_general_matrix_lapack.h"
#include "tl_dense_vector_lapack.h"

class DfXMatrix : public DfObject {
public:
DfXMatrix(TlSerializeData* pPdfParam);
virtual ~DfXMatrix();

public:
virtual void buildX();

virtual void canonicalOrthogonalize(const TlDenseSymmetricMatrix_Lapack& S,
TlDenseGeneralMatrix_Lapack* pX,
TlDenseGeneralMatrix_Lapack* pXinv,
const std::string& eigvalFilePath = "");

virtual void lowdinOrthogonalize(const TlDenseSymmetricMatrix_Lapack& S,
TlDenseGeneralMatrix_Lapack* pX,
TlDenseGeneralMatrix_Lapack* pXinv,
const std::string& eigvalFilePath = "");

protected:
template <typename SymmetricMatrixType, typename MatrixType, typename VectorTYpe>
void buildX_templ();

template <typename SymmetricMatrixType, typename MatrixType, typename VectorType>
void canonicalOrthogonalizeTmpl(const SymmetricMatrixType& S,
MatrixType* pX, MatrixType* pXinv,
const std::string& eigvalFilePath = "");

template <typename SymmetricMatrixType, typename MatrixType>
void lowdinOrthogonalizeTmpl(const SymmetricMatrixType& S, MatrixType* pX,
MatrixType* pXinv,
const std::string& eigvalFilePath = "");

template <typename MatrixType>
void check_X(const MatrixType& X, const MatrixType& Xinv,
const std::string& savePathPrefix);

protected:
double threshold_trancation_canonical_;

double threshold_trancation_lowdin_;


bool debugSaveEigval_;

bool debugSaveMatrix_;

bool debugCheckX_;
};

template <typename SymmetricMatrixType, typename MatrixType, typename VectorType>
void DfXMatrix::buildX_templ() {
SymmetricMatrixType S = this->getSpqMatrix<SymmetricMatrixType>();
MatrixType X;
MatrixType Xinv;

std::string eigvalFilePath = "";
if (this->debugSaveEigval_) {
eigvalFilePath = DfObject::getXEigvalVtrPath();
}
this->canonicalOrthogonalizeTmpl<SymmetricMatrixType, MatrixType, VectorType>(S, &X, &Xinv, eigvalFilePath);

DfObject::saveXMatrix(X);
DfObject::saveXInvMatrix(Xinv);
(*(this->pPdfParam_))["num_of_MOs"] = X.getNumOfCols();
}

template <typename SymmetricMatrixType, typename MatrixType, typename VectorType>
void DfXMatrix::canonicalOrthogonalizeTmpl(const SymmetricMatrixType& S,
MatrixType* pX, MatrixType* pXinv,
const std::string& eigvalFilePath) {
this->log_.info("orthogonalize by canonical method");
this->log_.info(
TlUtils::format("S: %d x %d", S.getNumOfRows(), S.getNumOfCols()));

const TlMatrixObject::index_type dim = S.getNumOfRows();
TlMatrixObject::index_type rest = 0;

TlDenseVector_Lapack sqrt_s;  
MatrixType U;                 
{
this->loggerTime("diagonalization of S matrix");
VectorType EigVal;
MatrixType EigVec;
S.eig(&EigVal, &EigVec);
assert(EigVal.getSize() == dim);

if (!eigvalFilePath.empty()) {
this->log_.info(
TlUtils::format("save eigvals to %s", eigvalFilePath.c_str()));
EigVal.save(eigvalFilePath);
}

this->loggerTime("truncation of linear dependent");
{
const double threshold = this->threshold_trancation_canonical_;
this->log_.info(TlUtils::format("threshold: %f", threshold));
int cutoffCount = 0;
for (index_type k = 0; k < dim; ++k) {
if (EigVal.get(k) < threshold) {
++cutoffCount;
} else {
break;
}
}
rest = dim - cutoffCount;
}

this->loggerTime(" generation of U matrix");
const TlMatrixObject::index_type cutoffBasis = dim - rest;

{
MatrixType trans(dim, rest);
#pragma omp parallel for
for (index_type i = 0; i < rest; ++i) {
trans.set(cutoffBasis + i, i, 1.0);
}
U = EigVec * trans;
}
{
sqrt_s.resize(rest);
#pragma omp parallel for
for (index_type k = 0; k < rest; ++k) {
const index_type index = cutoffBasis + k;
sqrt_s.set(k, std::sqrt(EigVal.get(index)));
}
}
}
if (this->debugSaveMatrix_) {
U.save("U.mat");
sqrt_s.save("sqrt_s.vct");
}

if (pX != NULL) {
this->loggerTime("generate X matrix");
SymmetricMatrixType S12(rest);
for (index_type i = 0; i < rest; ++i) {
S12.set(i, i, (1.0 / sqrt_s.get(i)));
}

*pX = U * S12;
}

if (pXinv != NULL) {
this->loggerTime("generate X^-1 matrix");

SymmetricMatrixType S12(rest);
for (TlMatrixObject::index_type i = 0; i < rest; ++i) {
S12.set(i, i, sqrt_s.get(i));
}

this->loggerTime("transpose U matrix");
U.transposeInPlace();

*pXinv = S12 * U;
}

this->loggerTime(" finalize");

if (this->debugCheckX_) {
this->check_X(*pX, *pXinv,
TlUtils::format("%s/S_", this->m_sWorkDirPath.c_str()));
}
}

template <typename SymmetricMatrixType, typename MatrixType>
void DfXMatrix::lowdinOrthogonalizeTmpl(const SymmetricMatrixType& S,
MatrixType* pX, MatrixType* pXinv,
const std::string& eigvalFilePath) {
this->log_.info("orthogonalize by lowdin method");
const index_type dim = S.getNumOfRows();
index_type rest = 0;

TlDenseVector_Lapack sqrt_s;  
MatrixType U;                 
{
this->loggerTime("diagonalization of S matrix");
TlDenseVector_Lapack EigVal;
MatrixType EigVec;
S.eig(&EigVal, &EigVec);
assert(EigVal.getSize() == dim);

if (!eigvalFilePath.empty()) {
this->log_.info(
TlUtils::format("save eigvals to %s", eigvalFilePath.c_str()));
EigVal.save(eigvalFilePath);
}

this->loggerTime("truncation of linear dependent");
{
const double threshold = this->threshold_trancation_lowdin_;
this->log_.info(TlUtils::format("threshold: %f", threshold));
int cutoffCount = 0;
for (index_type k = 0; k < dim; ++k) {
if (EigVal.get(k) < threshold) {
++cutoffCount;
} else {
break;
}
}
rest = dim - cutoffCount;
}

this->loggerTime(" generation of U matrix");
const index_type cutoffBasis = dim - rest;

{
MatrixType trans(dim, rest);
#pragma omp parallel for
for (index_type i = 0; i < rest; ++i) {
trans.set(cutoffBasis + i, i, 1.0);
}
U = EigVec * trans;
}
{
sqrt_s.resize(rest);
#pragma omp parallel for
for (index_type k = 0; k < rest; ++k) {
const index_type index = cutoffBasis + k;
sqrt_s.set(k, std::sqrt(EigVal.get(index)));
}
}
}
if (this->debugSaveMatrix_) {
U.save("U.mat");
sqrt_s.save("sqrt_s.vct");
}

if (pX != NULL) {
this->loggerTime("generate X matrix");
SymmetricMatrixType S12(rest);
for (index_type i = 0; i < rest; ++i) {
S12.set(i, i, (1.0 / sqrt_s.get(i)));
}

*pX = U * S12;

MatrixType Ut = U;
Ut.transposeInPlace();

*pX *= Ut;
}

if (pXinv != NULL) {
this->loggerTime("generate X^-1 matrix");
*pXinv = *pX;
pXinv->inverse();
}

this->loggerTime(" finalize");

if (this->debugCheckX_) {
this->check_X(*pX, *pXinv,
TlUtils::format("%s/S_", this->m_sWorkDirPath.c_str()));
}
}

template <typename MatrixType>
void DfXMatrix::check_X(const MatrixType& X, const MatrixType& Xinv,
const std::string& savePathPrefix) {
this->log_.info("check X");
{
const std::string path =
TlUtils::format("XXinv.mat", savePathPrefix.c_str());
this->log_.info(
TlUtils::format("calc X * Xinv to save %s.", path.c_str()));
const MatrixType XinvX = X * Xinv;
XinvX.save(path);
}

{
const std::string path =
TlUtils::format("XinvX.mat", savePathPrefix.c_str());
this->log_.info(
TlUtils::format("calc invX * X to save %s.", path.c_str()));
const MatrixType invXX = Xinv * X;
invXX.save(path);
}
}

#endif  
