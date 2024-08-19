
#pragma once

#include <cmath>
#include <type_traits>


#include "input_output/logger.h"
#include "includes/ublas_interface.h"

namespace Kratos
{







template<class TDataType = double>
class KRATOS_API(KRATOS_CORE) MathUtils
{
public:


using MatrixType = Matrix;

using VectorType = Vector;

using SizeType = std::size_t;

using IndexType = std::size_t;

using IndirectArrayType = boost::numeric::ublas::indirect_array<DenseVector<std::size_t>>;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();





static inline double GetZeroTolerance()
{
return ZeroTolerance;
}


template<bool TCheck>
static inline double Heron(
double a,
double b,
double c
)
{
const double s = 0.5 * (a + b + c);
const double A2 = s * (s - a) * (s - b) * (s - c);
if constexpr(TCheck) {
if(A2 < 0.0) {
KRATOS_ERROR << "The square of area is negative, probably the triangle is in bad shape:" << A2 << std::endl;
} else {
return std::sqrt(A2);
}
} else {
return std::sqrt(std::abs(A2));
}
}


template<class TMatrixType>
static double Cofactor(const TMatrixType& rMat, IndexType i, IndexType j)
{
static_assert(std::is_same<typename TMatrixType::value_type, double>::value, "Bad value type.");

KRATOS_ERROR_IF(rMat.size1() != rMat.size2() || rMat.size1() == 0) << "Bad matrix dimensions." << std::endl;

if (rMat.size1() == 1)
return 1;

IndirectArrayType ia1(rMat.size1() - 1), ia2(rMat.size2() - 1);

unsigned i_sub = 0;
for (unsigned k = 0; k < rMat.size1(); ++k)
if (k != i)
ia1(i_sub++) = k;

unsigned j_sub = 0;
for (unsigned k = 0; k < rMat.size2(); ++k)
if (k != j)
ia2(j_sub++) = k;

boost::numeric::ublas::matrix_indirect<const TMatrixType, IndirectArrayType> sub_mat(rMat, ia1, ia2);
const double first_minor = Det(sub_mat);
return ((i + j) % 2) ? -first_minor : first_minor;
}


template<class TMatrixType>
static MatrixType CofactorMatrix(const TMatrixType& rMat)
{
static_assert(std::is_same<typename TMatrixType::value_type, double>::value, "Bad value type.");

MatrixType cofactor_matrix(rMat.size1(), rMat.size2());

for (IndexType i = 0; i < rMat.size1(); ++i)
for (IndexType j = 0; j < rMat.size2(); ++j)
cofactor_matrix(i, j) = Cofactor(rMat, i, j);

return cofactor_matrix;
}


template<SizeType TDim>
KRATOS_DEPRECATED_MESSAGE("Please use InvertMatrix() instead")
static inline BoundedMatrix<double, TDim, TDim> InvertMatrix(
const BoundedMatrix<double, TDim, TDim>& rInputMatrix,
double& rInputMatrixDet,
const double Tolerance = ZeroTolerance
)
{
BoundedMatrix<double, TDim, TDim> inverted_matrix;


rInputMatrixDet = Det(rInputMatrix);

if constexpr (TDim == 1) {
inverted_matrix(0,0) = 1.0/rInputMatrix(0,0);
rInputMatrixDet = rInputMatrix(0,0);
} else if constexpr (TDim == 2) {
InvertMatrix2(rInputMatrix, inverted_matrix, rInputMatrixDet);
} else if constexpr (TDim == 3) {
InvertMatrix3(rInputMatrix, inverted_matrix, rInputMatrixDet);
} else if constexpr (TDim == 4) {
InvertMatrix4(rInputMatrix, inverted_matrix, rInputMatrixDet);
} else {
KRATOS_ERROR << "Size not implemented. Size: " << TDim << std::endl;
}

if (Tolerance > 0.0) { 
CheckConditionNumber(rInputMatrix, inverted_matrix, Tolerance);
}

return inverted_matrix;
}


template<class TMatrix1, class TMatrix2>
static inline bool CheckConditionNumber(
const TMatrix1& rInputMatrix,
TMatrix2& rInvertedMatrix,
const double Tolerance = std::numeric_limits<double>::epsilon(),
const bool ThrowError = true
)
{
const double max_condition_number = (1.0/Tolerance) * 1.0e-4;

const double input_matrix_norm = norm_frobenius(rInputMatrix);
const double inverted_matrix_norm = norm_frobenius(rInvertedMatrix);

const double cond_number = input_matrix_norm * inverted_matrix_norm ;
if (cond_number > max_condition_number) {
if (ThrowError) {
KRATOS_WATCH(rInputMatrix);
KRATOS_ERROR << " Condition number of the matrix is too high!, cond_number = " << cond_number << std::endl;
}
return false;
}

return true;
}


template<class TMatrix1, class TMatrix2>
static void GeneralizedInvertMatrix(
const TMatrix1& rInputMatrix,
TMatrix2& rInvertedMatrix,
double& rInputMatrixDet,
const double Tolerance = ZeroTolerance
)
{
const SizeType size_1 = rInputMatrix.size1();
const SizeType size_2 = rInputMatrix.size2();

if (size_1 == size_2) {
InvertMatrix(rInputMatrix, rInvertedMatrix, rInputMatrixDet, Tolerance);
} else if (size_1 < size_2) { 
if (rInvertedMatrix.size1() != size_2 || rInvertedMatrix.size2() != size_1) {
rInvertedMatrix.resize(size_2, size_1, false);
}
const Matrix aux = prod(rInputMatrix, trans(rInputMatrix));
Matrix auxInv;
InvertMatrix(aux, auxInv, rInputMatrixDet, Tolerance);
rInputMatrixDet = std::sqrt(rInputMatrixDet);
noalias(rInvertedMatrix) = prod(trans(rInputMatrix), auxInv);
} else { 
if (rInvertedMatrix.size1() != size_2 || rInvertedMatrix.size2() != size_1) {
rInvertedMatrix.resize(size_2, size_1, false);
}
const Matrix aux = prod(trans(rInputMatrix), rInputMatrix);
Matrix auxInv;
InvertMatrix(aux, auxInv, rInputMatrixDet, Tolerance);
rInputMatrixDet = std::sqrt(rInputMatrixDet);
noalias(rInvertedMatrix) = prod(auxInv, trans(rInputMatrix));
}
}


static void Solve(
MatrixType A,
VectorType& rX,
const VectorType& rB
);


template<class TMatrix1, class TMatrix2>
static void InvertMatrix(
const TMatrix1& rInputMatrix,
TMatrix2& rInvertedMatrix,
double& rInputMatrixDet,
const double Tolerance = ZeroTolerance
)
{
KRATOS_DEBUG_ERROR_IF_NOT(rInputMatrix.size1() == rInputMatrix.size2()) << "Matrix provided is non-square" << std::endl;

const SizeType size = rInputMatrix.size2();

if(size == 1) {
if(rInvertedMatrix.size1() != 1 || rInvertedMatrix.size2() != 1) {
rInvertedMatrix.resize(1,1,false);
}
rInvertedMatrix(0,0) = 1.0/rInputMatrix(0,0);
rInputMatrixDet = rInputMatrix(0,0);
} else if (size == 2) {
InvertMatrix2(rInputMatrix, rInvertedMatrix, rInputMatrixDet);
} else if (size == 3) {
InvertMatrix3(rInputMatrix, rInvertedMatrix, rInputMatrixDet);
} else if (size == 4) {
InvertMatrix4(rInputMatrix, rInvertedMatrix, rInputMatrixDet);
} else if (std::is_same<TMatrix1, Matrix>::value) {

const SizeType size1 = rInputMatrix.size1();
const SizeType size2 = rInputMatrix.size2();
if(rInvertedMatrix.size1() != size1 || rInvertedMatrix.size2() != size2) {
rInvertedMatrix.resize(size1, size2,false);
}

Matrix A(rInputMatrix);
typedef permutation_matrix<SizeType> pmatrix;
pmatrix pm(A.size1());
const int singular = lu_factorize(A,pm);
rInvertedMatrix.assign( IdentityMatrix(A.size1()));
KRATOS_ERROR_IF(singular == 1) << "Matrix is singular: " << rInputMatrix << std::endl;
lu_substitute(A, pm, rInvertedMatrix);

rInputMatrixDet = 1.0;

for (IndexType i = 0; i < size1;++i) {
IndexType ki = pm[i] == i ? 0 : 1;
rInputMatrixDet *= (ki == 0) ? A(i,i) : -A(i,i);
}
} else { 
const SizeType size1 = rInputMatrix.size1();
const SizeType size2 = rInputMatrix.size2();

Matrix A(rInputMatrix);
Matrix invA(rInvertedMatrix);

typedef permutation_matrix<SizeType> pmatrix;
pmatrix pm(size1);
const int singular = lu_factorize(A,pm);
invA.assign( IdentityMatrix(size1));
KRATOS_ERROR_IF(singular == 1) << "Matrix is singular: " << rInputMatrix << std::endl;
lu_substitute(A, pm, invA);

rInputMatrixDet = 1.0;

for (IndexType i = 0; i < size1;++i) {
IndexType ki = pm[i] == i ? 0 : 1;
rInputMatrixDet *= (ki == 0) ? A(i,i) : -A(i,i);
}

for (IndexType i = 0; i < size1;++i) {
for (IndexType j = 0; j < size2;++j) {
rInvertedMatrix(i,j) = invA(i,j);
}
}
}

if (Tolerance > 0.0) { 
CheckConditionNumber(rInputMatrix, rInvertedMatrix, Tolerance);
}
}


template<class TMatrix1, class TMatrix2>
static void InvertMatrix2(
const TMatrix1& rInputMatrix,
TMatrix2& rInvertedMatrix,
double& rInputMatrixDet
)
{
KRATOS_TRY;

KRATOS_DEBUG_ERROR_IF_NOT(rInputMatrix.size1() == rInputMatrix.size2()) << "Matrix provided is non-square" << std::endl;

if(rInvertedMatrix.size1() != 2 || rInvertedMatrix.size2() != 2) {
rInvertedMatrix.resize(2,2,false);
}

rInputMatrixDet = rInputMatrix(0,0)*rInputMatrix(1,1)-rInputMatrix(0,1)*rInputMatrix(1,0);

rInvertedMatrix(0,0) =  rInputMatrix(1,1);
rInvertedMatrix(0,1) = -rInputMatrix(0,1);
rInvertedMatrix(1,0) = -rInputMatrix(1,0);
rInvertedMatrix(1,1) =  rInputMatrix(0,0);

rInvertedMatrix/=rInputMatrixDet;

KRATOS_CATCH("");
}


template<class TMatrix1, class TMatrix2>
static void InvertMatrix3(
const TMatrix1& rInputMatrix,
TMatrix2& rInvertedMatrix,
double& rInputMatrixDet
)
{
KRATOS_TRY;

KRATOS_DEBUG_ERROR_IF_NOT(rInputMatrix.size1() == rInputMatrix.size2()) << "Matrix provided is non-square" << std::endl;

if(rInvertedMatrix.size1() != 3 || rInvertedMatrix.size2() != 3) {
rInvertedMatrix.resize(3,3,false);
}

rInvertedMatrix(0,0) = rInputMatrix(1,1)*rInputMatrix(2,2) - rInputMatrix(1,2)*rInputMatrix(2,1);
rInvertedMatrix(1,0) = -rInputMatrix(1,0)*rInputMatrix(2,2) + rInputMatrix(1,2)*rInputMatrix(2,0);
rInvertedMatrix(2,0) = rInputMatrix(1,0)*rInputMatrix(2,1) - rInputMatrix(1,1)*rInputMatrix(2,0);

rInvertedMatrix(0,1) = -rInputMatrix(0,1)*rInputMatrix(2,2) + rInputMatrix(0,2)*rInputMatrix(2,1);
rInvertedMatrix(1,1) = rInputMatrix(0,0)*rInputMatrix(2,2) - rInputMatrix(0,2)*rInputMatrix(2,0);
rInvertedMatrix(2,1) = -rInputMatrix(0,0)*rInputMatrix(2,1) + rInputMatrix(0,1)*rInputMatrix(2,0);

rInvertedMatrix(0,2) = rInputMatrix(0,1)*rInputMatrix(1,2) - rInputMatrix(0,2)*rInputMatrix(1,1);
rInvertedMatrix(1,2) = -rInputMatrix(0,0)*rInputMatrix(1,2) + rInputMatrix(0,2)*rInputMatrix(1,0);
rInvertedMatrix(2,2) = rInputMatrix(0,0)*rInputMatrix(1,1) - rInputMatrix(0,1)*rInputMatrix(1,0);

rInputMatrixDet = rInputMatrix(0,0)*rInvertedMatrix(0,0) + rInputMatrix(0,1)*rInvertedMatrix(1,0) + rInputMatrix(0,2)*rInvertedMatrix(2,0);

rInvertedMatrix /= rInputMatrixDet;

KRATOS_CATCH("")
}


template<class TMatrix1, class TMatrix2>
static void InvertMatrix4(
const TMatrix1& rInputMatrix,
TMatrix2& rInvertedMatrix,
double& rInputMatrixDet
)
{
KRATOS_TRY;

KRATOS_DEBUG_ERROR_IF_NOT(rInputMatrix.size1() == rInputMatrix.size2()) << "Matrix provided is non-square" << std::endl;

if (rInvertedMatrix.size1() != 4 || rInvertedMatrix.size2() != 4) {
rInvertedMatrix.resize(4, 4, false);
}


rInvertedMatrix(0, 0) = -(rInputMatrix(1, 3) * rInputMatrix(2, 2) * rInputMatrix(3, 1)) + rInputMatrix(1, 2) * rInputMatrix(2, 3) * rInputMatrix(3, 1) + rInputMatrix(1, 3) * rInputMatrix(2, 1) * rInputMatrix(3, 2) - rInputMatrix(1, 1) * rInputMatrix(2, 3) * rInputMatrix(3, 2) - rInputMatrix(1, 2) * rInputMatrix(2, 1) * rInputMatrix(3, 3) + rInputMatrix(1, 1) * rInputMatrix(2, 2) * rInputMatrix(3, 3);
rInvertedMatrix(0, 1) = rInputMatrix(0, 3) * rInputMatrix(2, 2) * rInputMatrix(3, 1) - rInputMatrix(0, 2) * rInputMatrix(2, 3) * rInputMatrix(3, 1) - rInputMatrix(0, 3) * rInputMatrix(2, 1) * rInputMatrix(3, 2) + rInputMatrix(0, 1) * rInputMatrix(2, 3) * rInputMatrix(3, 2) + rInputMatrix(0, 2) * rInputMatrix(2, 1) * rInputMatrix(3, 3) - rInputMatrix(0, 1) * rInputMatrix(2, 2) * rInputMatrix(3, 3);
rInvertedMatrix(0, 2) = -(rInputMatrix(0, 3) * rInputMatrix(1, 2) * rInputMatrix(3, 1)) + rInputMatrix(0, 2) * rInputMatrix(1, 3) * rInputMatrix(3, 1) + rInputMatrix(0, 3) * rInputMatrix(1, 1) * rInputMatrix(3, 2) - rInputMatrix(0, 1) * rInputMatrix(1, 3) * rInputMatrix(3, 2) - rInputMatrix(0, 2) * rInputMatrix(1, 1) * rInputMatrix(3, 3) + rInputMatrix(0, 1) * rInputMatrix(1, 2) * rInputMatrix(3, 3);
rInvertedMatrix(0, 3) = rInputMatrix(0, 3) * rInputMatrix(1, 2) * rInputMatrix(2, 1) - rInputMatrix(0, 2) * rInputMatrix(1, 3) * rInputMatrix(2, 1) - rInputMatrix(0, 3) * rInputMatrix(1, 1) * rInputMatrix(2, 2) + rInputMatrix(0, 1) * rInputMatrix(1, 3) * rInputMatrix(2, 2) + rInputMatrix(0, 2) * rInputMatrix(1, 1) * rInputMatrix(2, 3) - rInputMatrix(0, 1) * rInputMatrix(1, 2) * rInputMatrix(2, 3);

rInvertedMatrix(1, 0) = rInputMatrix(1, 3) * rInputMatrix(2, 2) * rInputMatrix(3, 0) - rInputMatrix(1, 2) * rInputMatrix(2, 3) * rInputMatrix(3, 0) - rInputMatrix(1, 3) * rInputMatrix(2, 0) * rInputMatrix(3, 2) + rInputMatrix(1, 0) * rInputMatrix(2, 3) * rInputMatrix(3, 2) + rInputMatrix(1, 2) * rInputMatrix(2, 0) * rInputMatrix(3, 3) - rInputMatrix(1, 0) * rInputMatrix(2, 2) * rInputMatrix(3, 3);
rInvertedMatrix(1, 1) = -(rInputMatrix(0, 3) * rInputMatrix(2, 2) * rInputMatrix(3, 0)) + rInputMatrix(0, 2) * rInputMatrix(2, 3) * rInputMatrix(3, 0) + rInputMatrix(0, 3) * rInputMatrix(2, 0) * rInputMatrix(3, 2) - rInputMatrix(0, 0) * rInputMatrix(2, 3) * rInputMatrix(3, 2) - rInputMatrix(0, 2) * rInputMatrix(2, 0) * rInputMatrix(3, 3) + rInputMatrix(0, 0) * rInputMatrix(2, 2) * rInputMatrix(3, 3);
rInvertedMatrix(1, 2) = rInputMatrix(0, 3) * rInputMatrix(1, 2) * rInputMatrix(3, 0) - rInputMatrix(0, 2) * rInputMatrix(1, 3) * rInputMatrix(3, 0) - rInputMatrix(0, 3) * rInputMatrix(1, 0) * rInputMatrix(3, 2) + rInputMatrix(0, 0) * rInputMatrix(1, 3) * rInputMatrix(3, 2) + rInputMatrix(0, 2) * rInputMatrix(1, 0) * rInputMatrix(3, 3) - rInputMatrix(0, 0) * rInputMatrix(1, 2) * rInputMatrix(3, 3);
rInvertedMatrix(1, 3) = -(rInputMatrix(0, 3) * rInputMatrix(1, 2) * rInputMatrix(2, 0)) + rInputMatrix(0, 2) * rInputMatrix(1, 3) * rInputMatrix(2, 0) + rInputMatrix(0, 3) * rInputMatrix(1, 0) * rInputMatrix(2, 2) - rInputMatrix(0, 0) * rInputMatrix(1, 3) * rInputMatrix(2, 2) - rInputMatrix(0, 2) * rInputMatrix(1, 0) * rInputMatrix(2, 3) + rInputMatrix(0, 0) * rInputMatrix(1, 2) * rInputMatrix(2, 3);

rInvertedMatrix(2, 0) = -(rInputMatrix(1, 3) * rInputMatrix(2, 1) * rInputMatrix(3, 0)) + rInputMatrix(1, 1) * rInputMatrix(2, 3) * rInputMatrix(3, 0) + rInputMatrix(1, 3) * rInputMatrix(2, 0) * rInputMatrix(3, 1) - rInputMatrix(1, 0) * rInputMatrix(2, 3) * rInputMatrix(3, 1) - rInputMatrix(1, 1) * rInputMatrix(2, 0) * rInputMatrix(3, 3) + rInputMatrix(1, 0) * rInputMatrix(2, 1) * rInputMatrix(3, 3);
rInvertedMatrix(2, 1) = rInputMatrix(0, 3) * rInputMatrix(2, 1) * rInputMatrix(3, 0) - rInputMatrix(0, 1) * rInputMatrix(2, 3) * rInputMatrix(3, 0) - rInputMatrix(0, 3) * rInputMatrix(2, 0) * rInputMatrix(3, 1) + rInputMatrix(0, 0) * rInputMatrix(2, 3) * rInputMatrix(3, 1) + rInputMatrix(0, 1) * rInputMatrix(2, 0) * rInputMatrix(3, 3) - rInputMatrix(0, 0) * rInputMatrix(2, 1) * rInputMatrix(3, 3);
rInvertedMatrix(2, 2) = -(rInputMatrix(0, 3) * rInputMatrix(1, 1) * rInputMatrix(3, 0)) + rInputMatrix(0, 1) * rInputMatrix(1, 3) * rInputMatrix(3, 0) + rInputMatrix(0, 3) * rInputMatrix(1, 0) * rInputMatrix(3, 1) - rInputMatrix(0, 0) * rInputMatrix(1, 3) * rInputMatrix(3, 1) - rInputMatrix(0, 1) * rInputMatrix(1, 0) * rInputMatrix(3, 3) + rInputMatrix(0, 0) * rInputMatrix(1, 1) * rInputMatrix(3, 3);
rInvertedMatrix(2, 3) = rInputMatrix(0, 3) * rInputMatrix(1, 1) * rInputMatrix(2, 0) - rInputMatrix(0, 1) * rInputMatrix(1, 3) * rInputMatrix(2, 0) - rInputMatrix(0, 3) * rInputMatrix(1, 0) * rInputMatrix(2, 1) + rInputMatrix(0, 0) * rInputMatrix(1, 3) * rInputMatrix(2, 1) + rInputMatrix(0, 1) * rInputMatrix(1, 0) * rInputMatrix(2, 3) - rInputMatrix(0, 0) * rInputMatrix(1, 1) * rInputMatrix(2, 3);

rInvertedMatrix(3, 0) = rInputMatrix(1, 2) * rInputMatrix(2, 1) * rInputMatrix(3, 0) - rInputMatrix(1, 1) * rInputMatrix(2, 2) * rInputMatrix(3, 0) - rInputMatrix(1, 2) * rInputMatrix(2, 0) * rInputMatrix(3, 1) + rInputMatrix(1, 0) * rInputMatrix(2, 2) * rInputMatrix(3, 1) + rInputMatrix(1, 1) * rInputMatrix(2, 0) * rInputMatrix(3, 2) - rInputMatrix(1, 0) * rInputMatrix(2, 1) * rInputMatrix(3, 2);
rInvertedMatrix(3, 1) = -(rInputMatrix(0, 2) * rInputMatrix(2, 1) * rInputMatrix(3, 0)) + rInputMatrix(0, 1) * rInputMatrix(2, 2) * rInputMatrix(3, 0) + rInputMatrix(0, 2) * rInputMatrix(2, 0) * rInputMatrix(3, 1) - rInputMatrix(0, 0) * rInputMatrix(2, 2) * rInputMatrix(3, 1) - rInputMatrix(0, 1) * rInputMatrix(2, 0) * rInputMatrix(3, 2) + rInputMatrix(0, 0) * rInputMatrix(2, 1) * rInputMatrix(3, 2);
rInvertedMatrix(3, 2) = rInputMatrix(0, 2) * rInputMatrix(1, 1) * rInputMatrix(3, 0) - rInputMatrix(0, 1) * rInputMatrix(1, 2) * rInputMatrix(3, 0) - rInputMatrix(0, 2) * rInputMatrix(1, 0) * rInputMatrix(3, 1) + rInputMatrix(0, 0) * rInputMatrix(1, 2) * rInputMatrix(3, 1) + rInputMatrix(0, 1) * rInputMatrix(1, 0) * rInputMatrix(3, 2) - rInputMatrix(0, 0) * rInputMatrix(1, 1) * rInputMatrix(3, 2);
rInvertedMatrix(3, 3) = -(rInputMatrix(0, 2) * rInputMatrix(1, 1) * rInputMatrix(2, 0)) + rInputMatrix(0, 1) * rInputMatrix(1, 2) * rInputMatrix(2, 0) + rInputMatrix(0, 2) * rInputMatrix(1, 0) * rInputMatrix(2, 1) - rInputMatrix(0, 0) * rInputMatrix(1, 2) * rInputMatrix(2, 1) - rInputMatrix(0, 1) * rInputMatrix(1, 0) * rInputMatrix(2, 2) + rInputMatrix(0, 0) * rInputMatrix(1, 1) * rInputMatrix(2, 2);

rInputMatrixDet = rInputMatrix(0, 1) * rInputMatrix(1, 3) * rInputMatrix(2, 2) * rInputMatrix(3, 0) - rInputMatrix(0, 1) * rInputMatrix(1, 2) * rInputMatrix(2, 3) * rInputMatrix(3, 0) - rInputMatrix(0, 0) * rInputMatrix(1, 3) * rInputMatrix(2, 2) * rInputMatrix(3, 1) + rInputMatrix(0, 0) * rInputMatrix(1, 2) * rInputMatrix(2, 3) * rInputMatrix(3, 1) - rInputMatrix(0, 1) * rInputMatrix(1, 3) * rInputMatrix(2, 0) * rInputMatrix(3, 2) + rInputMatrix(0, 0) * rInputMatrix(1, 3) * rInputMatrix(2, 1) * rInputMatrix(3, 2) + rInputMatrix(0, 1) * rInputMatrix(1, 0) * rInputMatrix(2, 3) * rInputMatrix(3, 2) - rInputMatrix(0, 0) * rInputMatrix(1, 1) * rInputMatrix(2, 3) * rInputMatrix(3, 2) + rInputMatrix(0, 3) * (rInputMatrix(1, 2) * rInputMatrix(2, 1) * rInputMatrix(3, 0) - rInputMatrix(1, 1) * rInputMatrix(2, 2) * rInputMatrix(3, 0) - rInputMatrix(1, 2) * rInputMatrix(2, 0) * rInputMatrix(3, 1) + rInputMatrix(1, 0) * rInputMatrix(2, 2) * rInputMatrix(3, 1) + rInputMatrix(1, 1) * rInputMatrix(2, 0) * rInputMatrix(3, 2) - rInputMatrix(1, 0) * rInputMatrix(2, 1) * rInputMatrix(3, 2)) + (rInputMatrix(0, 1) * rInputMatrix(1, 2) * rInputMatrix(2, 0) - rInputMatrix(0, 0) * rInputMatrix(1, 2) * rInputMatrix(2, 1) - rInputMatrix(0, 1) * rInputMatrix(1, 0) * rInputMatrix(2, 2) + rInputMatrix(0, 0) * rInputMatrix(1, 1) * rInputMatrix(2, 2)) * rInputMatrix(3, 3) + rInputMatrix(0, 2) * (-(rInputMatrix(1, 3) * rInputMatrix(2, 1) * rInputMatrix(3, 0)) + rInputMatrix(1, 1) * rInputMatrix(2, 3) * rInputMatrix(3, 0) + rInputMatrix(1, 3) * rInputMatrix(2, 0) * rInputMatrix(3, 1) - rInputMatrix(1, 0) * rInputMatrix(2, 3) * rInputMatrix(3, 1) - rInputMatrix(1, 1) * rInputMatrix(2, 0) * rInputMatrix(3, 3) + rInputMatrix(1, 0) * rInputMatrix(2, 1) * rInputMatrix(3, 3));

rInvertedMatrix /= rInputMatrixDet;

KRATOS_CATCH("");
}


template<class TMatrixType>
static inline double Det2(const TMatrixType& rA)
{
KRATOS_DEBUG_ERROR_IF_NOT(rA.size1() == rA.size2()) << "Matrix provided is non-square" << std::endl;

return (rA(0,0)*rA(1,1)-rA(0,1)*rA(1,0));
}


template<class TMatrixType>
static inline double Det3(const TMatrixType& rA)
{
KRATOS_DEBUG_ERROR_IF_NOT(rA.size1() == rA.size2()) << "Matrix provided is non-square" << std::endl;

const double a = rA(1,1)*rA(2,2) - rA(1,2)*rA(2,1);
const double b = rA(1,0)*rA(2,2) - rA(1,2)*rA(2,0);
const double c = rA(1,0)*rA(2,1) - rA(1,1)*rA(2,0);

return rA(0,0)*a - rA(0,1)*b + rA(0,2)*c;
}


template<class TMatrixType>
static inline double Det4(const TMatrixType& rA)
{
KRATOS_DEBUG_ERROR_IF_NOT(rA.size1() == rA.size2()) << "Matrix provided is non-square" << std::endl;

const double det = rA(0,1)*rA(1,3)*rA(2,2)*rA(3,0)-rA(0,1)*rA(1,2)*rA(2,3)*rA(3,0)-rA(0,0)*rA(1,3)*rA(2,2)*rA(3,1)+rA(0,0)*rA(1,2)*rA(2,3)*rA(3,1)
-rA(0,1)*rA(1,3)*rA(2,0)*rA(3,2)+rA(0,0)*rA(1,3)*rA(2,1)*rA(3,2)+rA(0,1)*rA(1,0)*rA(2,3)*rA(3,2)-rA(0,0)*rA(1,1)*rA(2,3)*rA(3,2)+rA(0,3)*(rA(1,2)*rA(2,1)*rA(3,0)-rA(1,1)*rA(2,2)*rA(3,0)-rA(1,2)*rA(2,0)*rA(3,1)+rA(1,0)*rA(2,2)*rA(3,1)+rA(1,1)*rA(2,0)*rA(3,2)
-rA(1,0)*rA(2,1)*rA(3,2))+(rA(0,1)*rA(1,2)*rA(2,0)-rA(0,0)*rA(1,2)*rA(2,1)-rA(0,1)*rA(1,0)*rA(2,2)+rA(0,0)*rA(1,1)*rA(2,2))*rA(3,3)+rA(0,2)*(-(rA(1,3)*rA(2,1)*rA(3,0))+rA(1,1)*rA(2,3)*rA(3,0)+rA(1,3)*rA(2,0)*rA(3,1)-rA(1,0)*rA(2,3)*rA(3,1)-rA(1,1)*rA(2,0)*rA(3,3)+rA(1,0)*rA(2,1)*rA(3,3));
return det;
}

public:

template<class TMatrixType>
static inline double Det(const TMatrixType& rA)
{
KRATOS_DEBUG_ERROR_IF_NOT(rA.size1() == rA.size2()) << "Matrix provided is non-square" << std::endl;

switch (rA.size1()) {
case 2:
return Det2(rA);
case 3:
return Det3(rA);
case 4:
return Det4(rA);
default:
double det = 1.0;
using namespace boost::numeric::ublas;
typedef permutation_matrix<SizeType> pmatrix;
Matrix Aux(rA);
pmatrix pm(Aux.size1());
bool singular = lu_factorize(Aux,pm);

if (singular) {
return 0.0;
}

for (IndexType i = 0; i < Aux.size1();++i) {
IndexType ki = pm[i] == i ? 0 : 1;
det *= std::pow(-1.0, ki) * Aux(i,i);
}
return det;
}
}


template<class TMatrixType>
static inline double GeneralizedDet(const TMatrixType& rA)
{
if (rA.size1() == rA.size2()) {
return Det(rA);
} else if (rA.size1() < rA.size2()) { 
const Matrix AAT = prod( rA, trans(rA) );
return std::sqrt(Det(AAT));
} else { 
const Matrix ATA = prod( trans(rA), rA );
return std::sqrt(Det(ATA));
}
}


static inline double Dot3(
const Vector& a,
const Vector& b
)
{
return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}


static inline double Dot(
const Vector& rFirstVector,
const Vector& rSecondVector
)
{
Vector::const_iterator i = rFirstVector.begin();
Vector::const_iterator j = rSecondVector.begin();
double temp = 0.0;
while(i != rFirstVector.end()) {
temp += *i++ * *j++;
}
return temp;
}


template<class TVectorType>
static inline double Norm3(const TVectorType& a)
{
double temp = std::pow(a[0],2) + std::pow(a[1],2) + std::pow(a[2],2);
temp = std::sqrt(temp);
return temp;
}


static inline double Norm(const Vector& a)
{
Vector::const_iterator i = a.begin();
double temp = 0.0;
while(i != a.end()) {
temp += (*i) * (*i);
i++;
}
return std::sqrt(temp);
}


static inline double StableNorm(const Vector& a)
{
if (a.size() == 0) {
return 0;
}

if (a.size() == 1) {
return a[0];
}

double scale {0};

double sqr_sum_scaled {1};

for (auto it = a.begin(); it != a.end(); ++it) {
double x = *it;

if (x != 0) {
const double abs_x = std::abs(x);

if (scale < abs_x) {
const double f = scale / abs_x;
sqr_sum_scaled = sqr_sum_scaled * (f * f) + 1.0;
scale = abs_x;
} else {
x = abs_x / scale;
sqr_sum_scaled += x * x;
}
}
}

return scale * std::sqrt(sqr_sum_scaled);
}


template<class T>
static inline T CrossProduct(
const T& a,
const T& b
)
{
T c(a);

c[0] = a[1]*b[2] - a[2]*b[1];
c[1] = a[2]*b[0] - a[0]*b[2];
c[2] = a[0]*b[1] - a[1]*b[0];

return c;
}


template< class T1, class T2>
static inline typename std::enable_if<std::is_same<T1, T2>::value, bool>::type CheckIsAlias(T1& value1, T2& value2)
{
return value1 == value2;
}


template< class T1, class T2>
static inline typename std::enable_if<!std::is_same<T1, T2>::value, bool>::type CheckIsAlias(T1& value1, T2& value2)
{
return false;
}


template< class T1, class T2 , class T3>
static inline void CrossProduct(T1& c, const T2& a, const T3& b ){
if (c.size() != 3) c.resize(3);

KRATOS_DEBUG_ERROR_IF(a.size() != 3 || b.size() != 3 || c.size() != 3)
<< "The size of the vectors is different of 3: "
<< a << ", " << b << " and " << c << std::endl;
KRATOS_DEBUG_ERROR_IF(CheckIsAlias(c, a))
<< "Aliasing between the output parameter and the first "
<< "input parameter" << std::endl;
KRATOS_DEBUG_ERROR_IF(CheckIsAlias(c, b))  << "Aliasing between "
<< "the output parameter and the second input parameter"  << std::endl;

c[0] = a[1]*b[2] - a[2]*b[1];
c[1] = a[2]*b[0] - a[0]*b[2];
c[2] = a[0]*b[1] - a[1]*b[0];
}


template< class T1, class T2 , class T3>
static inline void UnitCrossProduct(T1& c, const T2& a, const T3& b ){
CrossProduct(c,a,b);
const double norm = norm_2(c);
KRATOS_DEBUG_ERROR_IF(norm < 1000.0*ZeroTolerance)
<< "norm is 0 when making the UnitCrossProduct of the vectors "
<< a << " and " << b << std::endl;
c/=norm;
}


template< class T1, class T2 , class T3>
static inline void OrthonormalBasis(const T1& c,T2& a,T3& b, const IndexType Type = 0 ){
if (Type == 0)
OrthonormalBasisHughesMoeller(c,a,b);
else if (Type == 1)
OrthonormalBasisFrisvad(c,a,b);
else
OrthonormalBasisNaive(c,a,b);
}


template< class T1, class T2 , class T3>
static inline void OrthonormalBasisHughesMoeller(const T1& c,T2& a,T3& b ){
KRATOS_DEBUG_ERROR_IF(norm_2(c) < (1.0 - 1.0e-6) || norm_2(c) > (1.0 + 1.0e-6)) << "Input should be a normal vector" << std::endl;
if(std::abs(c[0]) > std::abs(c[2])) {
b[0] =  c[1];
b[1] = -c[0];
b[2] =  0.0;
} else {
b[0] =   0.0;
b[1] =   c[2];
b[2]  = -c[1];
}
b /=  norm_2(b); 
UnitCrossProduct(a, b , c); 
}


template< class T1, class T2 , class T3>
static inline void OrthonormalBasisFrisvad(const T1& c,T2& a,T3& b ){
KRATOS_DEBUG_ERROR_IF(norm_2(c) < (1.0 - 1.0e-3) || norm_2(c) > (1.0 + 1.0e-3)) << "Input should be a normal vector" << std::endl;
if ((c[2] + 1.0) > 1.0e4 * ZeroTolerance) {
a[0] = 1.0 - std::pow(c[0], 2)/(1.0 + c[2]);
a[1] = - (c[0] * c[1])/(1.0 + c[2]);
a[2] = - c[0];
const double norm_a = norm_2(a);
a /= norm_a;
b[0] = - (c[0] * c[1])/(1.0 + c[2]);
b[1] = 1.0 - std::pow(c[1], 2)/(1.0 + c[2]);
b[2] = -c[1];
const double norm_b = norm_2(b);
b /= norm_b;
} else { 
a[0] = 1.0;
a[1] = 0.0;
a[2] = 0.0;
b[0] = 0.0;
b[1] = -1.0;
b[2] = 0.0;
}
}


template< class T1, class T2 , class T3>
static inline void OrthonormalBasisNaive(const T1& c,T2& a,T3& b ){
KRATOS_DEBUG_ERROR_IF(norm_2(c) < (1.0 - 1.0e-3) || norm_2(c) > (1.0 + 1.0e-3)) << "Input should be a normal vector" << std::endl;
if(c[0] > 0.9f) {
a[0] = 0.0;
a[1] = 1.0;
a[2] = 0.0;
} else {
a[0] = 1.0;
a[1] = 0.0;
a[2] = 0.0;
}
a  -= c * inner_prod(a, c); 
a /=  norm_2(a);            
UnitCrossProduct(b, c, a);  
}


template< class T1, class T2>
static inline double VectorsAngle(const T1& rV1, const T2& rV2 ){
const T1 aux_1 = rV1 * norm_2(rV2);
const T2 aux_2 = norm_2(rV1) * rV2;
const double num = norm_2(aux_1 - aux_2);
const double denom = norm_2(aux_1 + aux_2);
return 2.0 * std::atan2( num , denom);
}


static inline MatrixType TensorProduct3(
const Vector& a,
const Vector& b
)
{
MatrixType A(3,3);

A(0,0) = a[0]*b[0];
A(0,1) = a[0]*b[1];
A(0,2) = a[0]*b[2];
A(1,0) = a[1]*b[0];
A(1,1) = a[1]*b[1];
A(1,2) = a[1]*b[2];
A(2,0) = a[2]*b[0];
A(2,1) = a[2]*b[1];
A(2,2) = a[2]*b[2];

return A;
}


template<class TMatrixType1, class TMatrixType2>
static inline void AddMatrix(
TMatrixType1& rDestination,
const TMatrixType2& rInputMatrix,
const IndexType InitialRow,
const IndexType InitialCol
)
{
KRATOS_TRY

for(IndexType i = 0; i < rInputMatrix.size1(); ++i) {
for(IndexType j = 0; j < rInputMatrix.size2(); ++j) {
rDestination(InitialRow+i, InitialCol+j) += rInputMatrix(i,j);
}
}
KRATOS_CATCH("")
}


template<class TVectorType1, class TVectorType2>
static inline void AddVector(
TVectorType1& rDestination,
const TVectorType2& rInputVector,
const IndexType InitialIndex
)
{
KRATOS_TRY

for(IndexType i = 0; i < rInputVector.size(); ++i) {
rDestination[InitialIndex+i] += rInputVector[i];
}
KRATOS_CATCH("")
}


static inline void  SubtractMatrix(
MatrixType& rDestination,
const MatrixType& rInputMatrix,
const IndexType InitialRow,
const IndexType InitialCol
)
{
KRATOS_TRY;

for(IndexType i = 0; i<rInputMatrix.size1(); ++i) {
for(IndexType j = 0; j<rInputMatrix.size2(); ++j) {
rDestination(InitialRow+i, InitialCol+j) -= rInputMatrix(i,j);
}
}

KRATOS_CATCH("");
}


static inline void  WriteMatrix(
MatrixType& rDestination,
const MatrixType& rInputMatrix,
const IndexType InitialRow,
const IndexType InitialCol
)
{
KRATOS_TRY;

for(IndexType i = 0; i < rInputMatrix.size1(); ++i) {
for(IndexType j = 0; j < rInputMatrix.size2(); ++j) {
rDestination(InitialRow+i, InitialCol+j) = rInputMatrix(i,j);
}
}

KRATOS_CATCH("");
}


static inline void ExpandReducedMatrix(
MatrixType& rDestination,
const MatrixType& rReducedMatrix,
const SizeType Dimension
)
{
KRATOS_TRY;

const SizeType size = rReducedMatrix.size2();
IndexType rowindex = 0;
IndexType colindex = 0;

for (IndexType i = 0; i < size; ++i) {
rowindex = i * Dimension;
for (IndexType j = 0; j < size; ++j) {
colindex = j * Dimension;
for(IndexType ii = 0; ii < Dimension; ++ii) {
rDestination(rowindex+ii, colindex+ii) = rReducedMatrix(i, j);
}
}
}

KRATOS_CATCH("");
}


static inline void  ExpandAndAddReducedMatrix(
MatrixType& rDestination,
const MatrixType& rReducedMatrix,
const SizeType Dimension
)
{
KRATOS_TRY;

const SizeType size = rReducedMatrix.size2();
IndexType rowindex = 0;
IndexType colindex = 0;

for (IndexType i = 0; i < size; ++i) {
rowindex = i * Dimension;
for (IndexType j = 0; j < size; ++j) {
colindex = j * Dimension;
for(IndexType ii = 0; ii < Dimension; ++ii) {
rDestination(rowindex+ii, colindex+ii) += rReducedMatrix(i, j);
}
}
}

KRATOS_CATCH("");
}


static inline void  VecAdd(
Vector& rX,
const double coeff,
Vector& rY
)
{
KRATOS_TRY
SizeType size=rX.size();

for (IndexType i=0; i<size; ++i) {
rX[i] += coeff * rY[i];
}
KRATOS_CATCH("")
}


template<class TVector, class TMatrixType = MatrixType>
static inline TMatrixType StressVectorToTensor(const TVector& rStressVector)
{
KRATOS_TRY;

const SizeType matrix_size = rStressVector.size() == 3 ? 2 : 3;
TMatrixType stress_tensor(matrix_size, matrix_size);

if (rStressVector.size()==3) {
stress_tensor(0,0) = rStressVector[0];
stress_tensor(0,1) = rStressVector[2];
stress_tensor(1,0) = rStressVector[2];
stress_tensor(1,1) = rStressVector[1];
} else if (rStressVector.size()==4) {
stress_tensor(0,0) = rStressVector[0];
stress_tensor(0,1) = rStressVector[3];
stress_tensor(0,2) = 0.0;
stress_tensor(1,0) = rStressVector[3];
stress_tensor(1,1) = rStressVector[1];
stress_tensor(1,2) = 0.0;
stress_tensor(2,0) = 0.0;
stress_tensor(2,1) = 0.0;
stress_tensor(2,2) = rStressVector[2];
} else if (rStressVector.size()==6) {
stress_tensor(0,0) = rStressVector[0];
stress_tensor(0,1) = rStressVector[3];
stress_tensor(0,2) = rStressVector[5];
stress_tensor(1,0) = rStressVector[3];
stress_tensor(1,1) = rStressVector[1];
stress_tensor(1,2) = rStressVector[4];
stress_tensor(2,0) = rStressVector[5];
stress_tensor(2,1) = rStressVector[4];
stress_tensor(2,2) = rStressVector[2];
}

return stress_tensor;

KRATOS_CATCH("");
}


template<class TVector, class TMatrixType = MatrixType>
static inline TMatrixType VectorToSymmetricTensor(const TVector& rVector)
{
KRATOS_TRY;

const SizeType matrix_size = rVector.size() == 3 ? 2 : 3;
TMatrixType tensor(matrix_size, matrix_size);

if (rVector.size() == 3) {
tensor(0,0) = rVector[0];
tensor(0,1) = rVector[2];
tensor(1,0) = rVector[2];
tensor(1,1) = rVector[1];
} else if (rVector.size() == 4) {
tensor(0,0) = rVector[0];
tensor(0,1) = rVector[3];
tensor(0,2) = 0.0;
tensor(1,0) = rVector[3];
tensor(1,1) = rVector[1];
tensor(1,2) = 0.0;
tensor(2,0) = 0.0;
tensor(2,1) = 0.0;
tensor(2,2) = rVector[2];
} else if (rVector.size() == 6) {
tensor(0,0) = rVector[0];
tensor(0,1) = rVector[3];
tensor(0,2) = rVector[5];
tensor(1,0) = rVector[3];
tensor(1,1) = rVector[1];
tensor(1,2) = rVector[4];
tensor(2,0) = rVector[5];
tensor(2,1) = rVector[4];
tensor(2,2) = rVector[2];
}

return tensor;

KRATOS_CATCH("");
}


static inline int Sign(const double& ThisDataType)
{
KRATOS_TRY;
const double& x = ThisDataType;
return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
KRATOS_CATCH("");
}



template<class TVector, class TMatrixType = MatrixType>
static inline TMatrixType StrainVectorToTensor( const TVector& rStrainVector)
{
KRATOS_TRY

const SizeType matrix_size = rStrainVector.size() == 3 ? 2 : 3;
TMatrixType strain_tensor(matrix_size, matrix_size);

if (rStrainVector.size()==3) {
strain_tensor(0,0) = rStrainVector[0];
strain_tensor(0,1) = 0.5*rStrainVector[2];
strain_tensor(1,0) = 0.5*rStrainVector[2];
strain_tensor(1,1) = rStrainVector[1];
} else if (rStrainVector.size()==4) {
strain_tensor(0,0) = rStrainVector[0];
strain_tensor(0,1) = 0.5*rStrainVector[3];
strain_tensor(0,2) = 0;
strain_tensor(1,0) = 0.5*rStrainVector[3];
strain_tensor(1,1) = rStrainVector[1];
strain_tensor(1,2) = 0;
strain_tensor(2,0) = 0;
strain_tensor(2,1) = 0;
strain_tensor(2,2) = rStrainVector[2];
} else if (rStrainVector.size()==6) {
strain_tensor(0,0) = rStrainVector[0];
strain_tensor(0,1) = 0.5*rStrainVector[3];
strain_tensor(0,2) = 0.5*rStrainVector[5];
strain_tensor(1,0) = 0.5*rStrainVector[3];
strain_tensor(1,1) = rStrainVector[1];
strain_tensor(1,2) = 0.5*rStrainVector[4];
strain_tensor(2,0) = 0.5*rStrainVector[5];
strain_tensor(2,1) = 0.5*rStrainVector[4];
strain_tensor(2,2) = rStrainVector[2];
}

return strain_tensor;

KRATOS_CATCH("");
}


template<class TMatrixType, class TVector = Vector>
static inline Vector StrainTensorToVector(
const TMatrixType& rStrainTensor,
SizeType rSize = 0
)
{
KRATOS_TRY;

if(rSize == 0) {
if(rStrainTensor.size1() == 2) {
rSize = 3;
} else if(rStrainTensor.size1() == 3) {
rSize = 6;
}
}

Vector strain_vector(rSize);

if (rSize == 3) {
strain_vector[0] = rStrainTensor(0,0);
strain_vector[1] = rStrainTensor(1,1);
strain_vector[2] = 2.0*rStrainTensor(0,1);
} else if (rSize == 4) {
strain_vector[0] = rStrainTensor(0,0);
strain_vector[1] = rStrainTensor(1,1);
strain_vector[2] = rStrainTensor(2,2);
strain_vector[3] = 2.0*rStrainTensor(0,1);
} else if (rSize == 6) {
strain_vector[0] = rStrainTensor(0,0);
strain_vector[1] = rStrainTensor(1,1);
strain_vector[2] = rStrainTensor(2,2);
strain_vector[3] = 2.0*rStrainTensor(0,1);
strain_vector[4] = 2.0*rStrainTensor(1,2);
strain_vector[5] = 2.0*rStrainTensor(0,2);
}

return strain_vector;

KRATOS_CATCH("");
}


template<class TMatrixType, class TVector = Vector>
static inline TVector StressTensorToVector(
const TMatrixType& rStressTensor,
SizeType rSize = 0
)
{
KRATOS_TRY;

if(rSize == 0) {
if(rStressTensor.size1() == 2) {
rSize = 3;
} else if(rStressTensor.size1() == 3) {
rSize = 6;
}
}

TVector stress_vector(rSize);

if (rSize == 3) {
stress_vector[0] = rStressTensor(0,0);
stress_vector[1] = rStressTensor(1,1);
stress_vector[2] = rStressTensor(0,1);
} else if (rSize == 4) {
stress_vector[0] = rStressTensor(0,0);
stress_vector[1] = rStressTensor(1,1);
stress_vector[2] = rStressTensor(2,2);
stress_vector[3] = rStressTensor(0,1);
} else if (rSize == 6) {
stress_vector[0] = rStressTensor(0,0);
stress_vector[1] = rStressTensor(1,1);
stress_vector[2] = rStressTensor(2,2);
stress_vector[3] = rStressTensor(0,1);
stress_vector[4] = rStressTensor(1,2);
stress_vector[5] = rStressTensor(0,2);
}

return stress_vector;

KRATOS_CATCH("");
}


template<class TMatrixType, class TVector = Vector>
static inline TVector SymmetricTensorToVector(
const TMatrixType& rTensor,
SizeType rSize = 0
)
{
KRATOS_TRY;

if(rSize == 0) {
if(rTensor.size1() == 2) {
rSize = 3;
} else if(rTensor.size1() == 3) {
rSize = 6;
}
}

Vector vector(rSize);

if (rSize == 3) {
vector[0]= rTensor(0,0);
vector[1]= rTensor(1,1);
vector[2]= rTensor(0,1);

} else if (rSize==4) {
vector[0]= rTensor(0,0);
vector[1]= rTensor(1,1);
vector[2]= rTensor(2,2);
vector[3]= rTensor(0,1);
} else if (rSize==6) {
vector[0]= rTensor(0,0);
vector[1]= rTensor(1,1);
vector[2]= rTensor(2,2);
vector[3]= rTensor(0,1);
vector[4]= rTensor(1,2);
vector[5]= rTensor(0,2);
}

return vector;

KRATOS_CATCH("");
}


template<class TMatrixType1, class TMatrixType2, class TMatrixType3>
static inline void BtDBProductOperation(
TMatrixType1& rA,
const TMatrixType2& rD,
const TMatrixType3& rB
)
{
const SizeType size1 = rB.size2();
const SizeType size2 = rB.size2();

if (rA.size1() != size1 || rA.size2() != size2)
rA.resize(size1, size2, false);


rA.clear();
for(IndexType k = 0; k< rD.size1(); ++k) {
for(IndexType l = 0; l < rD.size2(); ++l) {
const double Dkl = rD(k, l);
for(IndexType j = 0; j < rB.size2(); ++j) {
const double DklBlj = Dkl * rB(l, j);
for(IndexType i = 0; i< rB.size2(); ++i) {
rA(i, j) += rB(k, i) * DklBlj;
}
}
}
}
}


template<class TMatrixType1, class TMatrixType2, class TMatrixType3>
static inline void BDBtProductOperation(
TMatrixType1& rA,
const TMatrixType2& rD,
const TMatrixType3& rB
)
{
const SizeType size1 = rB.size1();
const SizeType size2 = rB.size1();

if (rA.size1() != size1 || rA.size2() != size2)
rA.resize(size1, size2, false);


rA.clear();
for(IndexType k = 0; k< rD.size1(); ++k) {
for(IndexType l = 0; l < rD.size2(); ++l) {
const double Dkl = rD(k,l);
for(IndexType j = 0; j < rB.size1(); ++j) {
const double DklBjl = Dkl * rB(j,l);
for(IndexType i = 0; i< rB.size1(); ++i) {
rA(i, j) += rB(i, k) * DklBjl;
}
}
}
}
}


template<class TMatrixType1, class TMatrixType2>
static inline bool GaussSeidelEigenSystem(
const TMatrixType1& rA,
TMatrixType2& rEigenVectorsMatrix,
TMatrixType2& rEigenValuesMatrix,
const double Tolerance = 1.0e-18,
const SizeType MaxIterations = 20
)
{
bool is_converged = false;

const SizeType size = rA.size1();

if (rEigenVectorsMatrix.size1() != size || rEigenVectorsMatrix.size2() != size)
rEigenVectorsMatrix.resize(size, size, false);
if (rEigenValuesMatrix.size1() != size || rEigenValuesMatrix.size2() != size)
rEigenValuesMatrix.resize(size, size, false);

const TMatrixType2 identity_matrix = IdentityMatrix(size);
noalias(rEigenVectorsMatrix) = identity_matrix;
noalias(rEigenValuesMatrix) = rA;

TMatrixType2 aux_A, aux_V_matrix, rotation_matrix;
double a, u, c, s, gamma, teta;
IndexType index1, index2;

aux_A.resize(size,size,false);
aux_V_matrix.resize(size,size,false);
rotation_matrix.resize(size,size,false);

for(IndexType iterations = 0; iterations < MaxIterations; ++iterations) {
is_converged = true;

a = 0.0;
index1 = 0;
index2 = 1;

for(IndexType i = 0; i < size; ++i) {
for(IndexType j = (i + 1); j < size; ++j) {
if((std::abs(rEigenValuesMatrix(i, j)) > a ) && (std::abs(rEigenValuesMatrix(i, j)) > Tolerance)) {
a = std::abs(rEigenValuesMatrix(i,j));
index1 = i;
index2 = j;
is_converged = false;
}
}
}

if(is_converged) {
break;
}

gamma = (rEigenValuesMatrix(index2, index2)-rEigenValuesMatrix(index1, index1)) / (2 * rEigenValuesMatrix(index1, index2));
u = 1.0;

if(std::abs(gamma) > Tolerance && std::abs(gamma)< (1.0/Tolerance)) {
u = gamma / std::abs(gamma) * 1.0 / (std::abs(gamma) + std::sqrt(1.0 + gamma * gamma));
} else {
if  (std::abs(gamma) >= (1.0/Tolerance)) {
u = 0.5 / gamma;
}
}

c = 1.0 / (std::sqrt(1.0 + u * u));
s = c * u;
teta = s / (1.0 + c);

noalias(aux_A) = rEigenValuesMatrix;
aux_A(index2, index2) = rEigenValuesMatrix(index2,index2) + u * rEigenValuesMatrix(index1, index2);
aux_A(index1, index1) = rEigenValuesMatrix(index1,index1) - u * rEigenValuesMatrix(index1, index2);
aux_A(index1, index2) = 0.0;
aux_A(index2, index1) = 0.0;

for(IndexType i = 0; i < size; ++i) {
if((i!= index1) && (i!= index2)) {
aux_A(index2, i) = rEigenValuesMatrix(index2, i) + s * (rEigenValuesMatrix(index1, i)- teta * rEigenValuesMatrix(index2, i));
aux_A(i, index2) = rEigenValuesMatrix(index2, i) + s * (rEigenValuesMatrix(index1, i)- teta * rEigenValuesMatrix(index2, i));
aux_A(index1, i) = rEigenValuesMatrix(index1, i) - s * (rEigenValuesMatrix(index2, i) + teta * rEigenValuesMatrix(index1, i));
aux_A(i, index1) = rEigenValuesMatrix(index1, i) - s * (rEigenValuesMatrix(index2, i) + teta * rEigenValuesMatrix(index1, i));
}
}

noalias(rEigenValuesMatrix) = aux_A;

noalias(rotation_matrix) = identity_matrix;
rotation_matrix(index2, index1) = -s;
rotation_matrix(index1, index2) =  s;
rotation_matrix(index1, index1) =  c;
rotation_matrix(index2, index2) =  c;

noalias(aux_V_matrix) = ZeroMatrix(size, size);

for(IndexType i = 0; i < size; ++i) {
for(IndexType j = 0; j < size; ++j) {
for(IndexType k = 0; k < size; ++k) {
aux_V_matrix(i, j) += rEigenVectorsMatrix(i, k) * rotation_matrix(k, j);
}
}
}
noalias(rEigenVectorsMatrix) = aux_V_matrix;
}

KRATOS_WARNING_IF("MathUtils::EigenSystem", !is_converged) << "Spectral decomposition not converged " << std::endl;

return is_converged;
}


template<SizeType TDim>
KRATOS_DEPRECATED_MESSAGE("Please use GaussSeidelEigenSystem() instead. Note the resulting EigenVectors matrix is transposed respect GaussSeidelEigenSystem()")
static inline bool EigenSystem(
const BoundedMatrix<double, TDim, TDim>& rA,
BoundedMatrix<double, TDim, TDim>& rEigenVectorsMatrix,
BoundedMatrix<double, TDim, TDim>& rEigenValuesMatrix,
const double Tolerance = 1.0e-18,
const SizeType MaxIterations = 20
)
{
const bool is_converged = GaussSeidelEigenSystem(rA, rEigenVectorsMatrix, rEigenValuesMatrix, Tolerance, MaxIterations);

const BoundedMatrix<double, TDim, TDim> V_matrix = rEigenVectorsMatrix;
for(IndexType i = 0; i < TDim; ++i) {
for(IndexType j = 0; j < TDim; ++j) {
rEigenVectorsMatrix(i, j) = V_matrix(j, i);
}
}

return is_converged;
}


template<class TMatrixType1, class TMatrixType2>
static inline bool MatrixSquareRoot(
const TMatrixType1 &rA,
TMatrixType2 &rMatrixSquareRoot,
const double Tolerance = 1.0e-18,
const SizeType MaxIterations = 20
)
{
TMatrixType2 eigenvectors_matrix, eigenvalues_matrix;
const bool is_converged = GaussSeidelEigenSystem(rA, eigenvectors_matrix, eigenvalues_matrix, Tolerance, MaxIterations);
KRATOS_WARNING_IF("MatrixSquareRoot", !is_converged) << "GaussSeidelEigenSystem did not converge.\n";

SizeType size = eigenvalues_matrix.size1();
for (SizeType i = 0; i < size; ++i) {
KRATOS_ERROR_IF(eigenvalues_matrix(i,i) < 0) << "Eigenvalue " << i << " is negative. Square root matrix cannot be computed" << std::endl;
eigenvalues_matrix(i,i) = std::sqrt(eigenvalues_matrix(i,i));
}

BDBtProductOperation(rMatrixSquareRoot, eigenvalues_matrix, eigenvectors_matrix);

return is_converged;
}


template<class TIntegerType>
static inline TIntegerType Factorial(const TIntegerType Number)
{
if (Number == 0) {
return 1;
}
TIntegerType k = Number;
for (TIntegerType i = Number - 1; i > 0; --i){
k *= i;
}
return k;
}


template<class TMatrixType>
static inline void CalculateExponentialOfMatrix(
const TMatrixType& rMatrix,
TMatrixType& rExponentialMatrix,
const double Tolerance = 1000.0*ZeroTolerance,
const SizeType MaxTerms = 200
)
{
SizeType series_term = 2;
SizeType factorial = 1;
const SizeType dimension = rMatrix.size1();

noalias(rExponentialMatrix) = IdentityMatrix(dimension) + rMatrix;
TMatrixType exponent_matrix = rMatrix;
TMatrixType aux_matrix;

while (series_term < MaxTerms) {
noalias(aux_matrix) = prod(exponent_matrix, rMatrix);
noalias(exponent_matrix) = aux_matrix;
factorial = Factorial(series_term);
noalias(rExponentialMatrix) += exponent_matrix / factorial;
const double norm_series_term = std::abs(norm_frobenius(exponent_matrix) / factorial);
if (norm_series_term < Tolerance)
break;
series_term++;
}
}


private:


MathUtils(void);

MathUtils(MathUtils& rSource);

}; 



}  
