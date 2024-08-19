
#if !defined(KRATOS_SPARSE_MATRIX_MULTIPLICATION_UTILITY_H_INCLUDED )
#define  KRATOS_SPARSE_MATRIX_MULTIPLICATION_UTILITY_H_INCLUDED

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "amgcl/value_type/interface.hpp"

#include "includes/define.h"
#include "utilities/atomic_utilities.h"

namespace Kratos
{






class SparseMatrixMultiplicationUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION( SparseMatrixMultiplicationUtility );

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef std::ptrdiff_t  SignedIndexType;

typedef DenseVector<IndexType> IndexVectorType;

typedef DenseVector<SignedIndexType> SignedIndexVectorType;


SparseMatrixMultiplicationUtility(){};

virtual ~SparseMatrixMultiplicationUtility()= default;



template <class T, class Enable = void>
struct value_type {
typedef typename T::value_type type;
};


template <class AMatrix, class BMatrix, class CMatrix>
static void MatrixMultiplication(
const AMatrix& rA,
const BMatrix& rB,
CMatrix& rC
)
{
#ifdef _OPENMP
const int nt = omp_get_max_threads();
#else
const int nt = 1;
#endif

if (nt > 16) {
MatrixMultiplicationRMerge(rA, rB, rC);
} else {
MatrixMultiplicationSaad(rA, rB, rC);
}
}


template <class AMatrix, class BMatrix, class CMatrix>
static void MatrixMultiplicationSaad(
const AMatrix& A,
const BMatrix& B,
CMatrix& C
)
{
typedef typename value_type<CMatrix>::type ValueType;

const SizeType nrows = A.size1();
const SizeType ncols = B.size2();

if ((nrows == 0) || (ncols == 0))
return void();

const IndexType* index1_a = A.index1_data().begin();
const IndexType* index2_a = A.index2_data().begin();
const double* values_a = A.value_data().begin();
const IndexType* index1_b = B.index1_data().begin();
const IndexType* index2_b = B.index2_data().begin();
const double* values_b = B.value_data().begin();
IndexType* c_ptr = new IndexType[nrows + 1];

c_ptr[0] = 0;

#pragma omp parallel
{
SignedIndexVectorType marker(ncols);
for (int i_fill = 0; i_fill < static_cast<int>(ncols); ++i_fill)
marker[i_fill] = -1;

#pragma omp for
for(int ia = 0; ia < static_cast<int>(nrows); ++ia) {
const IndexType row_begin_a = index1_a[ia];
const IndexType row_end_a   = index1_a[ia+1];

IndexType C_cols = 0;
for(IndexType ja = row_begin_a; ja < row_end_a; ++ja) {
const IndexType ca = index2_a[ja];
const IndexType row_begin_b = index1_b[ca];
const IndexType row_end_b   = index1_b[ca+1];

for(IndexType jb = row_begin_b; jb < row_end_b; ++jb) {
const IndexType cb = index2_b[jb];
if (marker[cb] != ia) {
marker[cb]  = ia;
++C_cols;
}
}
}
c_ptr[ia + 1] = C_cols;
}
}

std::partial_sum(c_ptr, c_ptr + nrows + 1, c_ptr);
const SizeType nonzero_values = c_ptr[nrows];
IndexType* aux_index2_c = new IndexType[nonzero_values];
ValueType* aux_val_c = new ValueType[nonzero_values];

#pragma omp parallel
{
SignedIndexVectorType marker(ncols);
for (int i_fill = 0; i_fill < static_cast<int>(ncols); ++i_fill)
marker[i_fill] = -1;

#pragma omp for
for(int ia = 0; ia < static_cast<int>(nrows); ++ia) {
const IndexType row_begin_a = index1_a[ia];
const IndexType row_end_a   = index1_a[ia+1];

const IndexType row_beg = c_ptr[ia];
IndexType row_end = row_beg;

for(IndexType ja = row_begin_a; ja < row_end_a; ++ja) {
const IndexType ca = index2_a[ja];
const ValueType va = values_a[ja];

const IndexType row_begin_b = index1_b[ca];
const IndexType row_end_b   = index1_b[ca+1];

for(IndexType jb = row_begin_b; jb < row_end_b; ++jb) {
const IndexType cb = index2_b[jb];
const ValueType vb = values_b[jb];

if (marker[cb] < static_cast<SignedIndexType>(row_beg)) {
marker[cb] = row_end;
aux_index2_c[row_end] = cb;
aux_val_c[row_end] = va * vb;
++row_end;
} else {
aux_val_c[marker[cb]] += va * vb;
}
}
}
}
}

SortRows(c_ptr, nrows, ncols, aux_index2_c, aux_val_c);

CreateSolutionMatrix(C, nrows, ncols, c_ptr, aux_index2_c, aux_val_c);

delete[] c_ptr;
delete[] aux_index2_c;
delete[] aux_val_c;
}


template <class AMatrix, class BMatrix, class CMatrix>
static void MatrixMultiplicationRMerge(
const AMatrix &A,
const BMatrix &B,
CMatrix &C
)
{
typedef typename value_type<CMatrix>::type ValueType;

const SizeType nrows = A.size1();
const SizeType ncols = B.size2();

if ((nrows == 0) || (ncols == 0))
return void();

const IndexType* index1_a = A.index1_data().begin();
const IndexType* index2_a = A.index2_data().begin();
const double* values_a = A.value_data().begin();
const IndexType* index1_b = B.index1_data().begin();
const IndexType* index2_b = B.index2_data().begin();
const double* values_b = B.value_data().begin();

IndexType max_row_width = 0;

#pragma omp parallel
{
IndexType my_max = 0;

#pragma omp for
for(int i = 0; i < static_cast<int>(nrows); ++i) {
const IndexType row_beg = index1_a[i];
const IndexType row_end = index1_a[i+1];

IndexType row_width = 0;
for(IndexType j = row_beg; j < row_end; ++j) {
const IndexType a_col = index2_a[j];
row_width += index1_b[a_col + 1] - index1_b[a_col];
}
my_max = std::max(my_max, row_width);
}

#pragma omp critical
max_row_width = std::max(max_row_width, my_max);
}

#ifdef _OPENMP
const int nthreads = omp_get_max_threads();
#else
const int nthreads = 1;
#endif

std::vector< std::vector<IndexType> > tmp_col(nthreads);
std::vector< std::vector<ValueType> > tmp_val(nthreads);

for(int i = 0; i < nthreads; ++i) {
tmp_col[i].resize(3 * max_row_width);
tmp_val[i].resize(2 * max_row_width);
}

IndexType* c_ptr = new IndexType[nrows + 1];
c_ptr[0] = 0;

#pragma omp parallel
{
#ifdef _OPENMP
const int tid = omp_get_thread_num();
#else
const int tid = 0;
#endif

IndexType* t_col = &tmp_col[tid][0];

#pragma omp for
for(int i = 0; i < static_cast<int>(nrows); ++i) {
const IndexType row_beg = index1_a[i];
const IndexType row_end = index1_a[i+1];

c_ptr[i+1] = ProdRowWidth( index2_a + row_beg, index2_a + row_end, index1_b, index2_b, t_col, t_col + max_row_width, t_col + 2 * max_row_width );
}
}

std::partial_sum(c_ptr, c_ptr + nrows + 1, c_ptr);
const SizeType nonzero_values = c_ptr[nrows];
IndexType* aux_index2_c = new IndexType[nonzero_values];
ValueType* aux_val_c = new ValueType[nonzero_values];

#pragma omp parallel
{
#ifdef _OPENMP
const int tid = omp_get_thread_num();
#else
const int tid = 0;
#endif

IndexType* t_col = tmp_col[tid].data();
ValueType *t_val = tmp_val[tid].data();

#pragma omp for
for(int i = 0; i < static_cast<int>(nrows); ++i) {
const IndexType row_beg = index1_a[i];
const IndexType row_end = index1_a[i+1];

ProdRow(index2_a + row_beg, index2_a + row_end, values_a + row_beg,
index1_b, index2_b, values_b, aux_index2_c + c_ptr[i], aux_val_c + c_ptr[i], t_col, t_val, t_col + max_row_width, t_val + max_row_width );
}
}

CreateSolutionMatrix(C, nrows, ncols, c_ptr, aux_index2_c, aux_val_c);

delete[] c_ptr;
delete[] aux_index2_c;
delete[] aux_val_c;
}


template <class AMatrix, class BMatrix>
static void MatrixAdd(
AMatrix& A,
const BMatrix& B,
const double Factor = 1.0
)
{
typedef typename value_type<AMatrix>::type ValueType;

const SizeType nrows = A.size1();
const SizeType ncols = A.size2();


if ((nrows == 0) || (ncols == 0))
return void();

KRATOS_ERROR_IF_NOT(nrows == B.size1()) << "The second matrix has a wrong number of rows" << std::endl;
KRATOS_ERROR_IF_NOT(ncols == B.size2()) << "The second matrix has a wrong number of columns" << std::endl;

const IndexType* index1_a = A.index1_data().begin();
const IndexType* index2_a = A.index2_data().begin();
const double* values_a = A.value_data().begin();
const IndexType* index1_b = B.index1_data().begin();
const IndexType* index2_b = B.index2_data().begin();
const double* values_b = B.value_data().begin();

IndexType* new_a_ptr = new IndexType[nrows + 1];
new_a_ptr[0] = 0;

#pragma omp parallel
{
#pragma omp for
for(int ia = 0; ia < static_cast<int>(nrows); ++ia) {

SignedIndexVectorType marker(ncols);
for (int i = 0; i < static_cast<int>(ncols); ++i)
marker[i] = -1;

IndexType new_A_cols = 0;

const IndexType row_begin_a = index1_a[ia];
const IndexType row_end_a   = index1_a[ia+1];
for(IndexType ja = row_begin_a; ja < row_end_a; ++ja) {
const IndexType ca = index2_a[ja];
marker[ca] = 1;
++new_A_cols;
}

const IndexType row_begin_b = index1_b[ia];
const IndexType row_end_b   = index1_b[ia+1];
for(IndexType jb = row_begin_b; jb < row_end_b; ++jb) {
const IndexType cb = index2_b[jb];
if (marker[cb] < 0) {
marker[cb] = 1;
++new_A_cols;
}
}
new_a_ptr[ia + 1] = new_A_cols;
}
}

std::partial_sum(new_a_ptr, new_a_ptr + nrows + 1, new_a_ptr);
const SizeType nonzero_values = new_a_ptr[nrows];
IndexType* aux_index2_new_a = new IndexType[nonzero_values];
ValueType* aux_val_new_a = new ValueType[nonzero_values];

#pragma omp parallel
{
#pragma omp for
for(int ia = 0; ia < static_cast<int>(nrows); ++ia) {

SignedIndexVectorType marker(ncols);
for (int i = 0; i < static_cast<int>(ncols); ++i)
marker[i] = -1;

const IndexType row_beg = new_a_ptr[ia];
IndexType row_end = row_beg;

const IndexType row_begin_a = index1_a[ia];
const IndexType row_end_a   = index1_a[ia+1];
for(IndexType ja = row_begin_a; ja < row_end_a; ++ja) {
const IndexType ca = index2_a[ja];
const ValueType va = values_a[ja];

marker[ca] = row_end;
aux_index2_new_a[row_end] = ca;
aux_val_new_a[row_end] = va;
++row_end;
}

const IndexType row_begin_b = index1_b[ia];
const IndexType row_end_b   = index1_b[ia+1];
for(IndexType jb = row_begin_b; jb < row_end_b; ++jb) {
const IndexType cb = index2_b[jb];
const ValueType vb = values_b[jb];

if (marker[cb] < 0) {
marker[cb] = row_end;
aux_index2_new_a[row_end] = cb;
aux_val_new_a[row_end] = Factor * vb;
++row_end;
} else {
aux_val_new_a[marker[cb]] += Factor * vb;
}
}
}
}

SortRows(new_a_ptr, nrows, ncols, aux_index2_new_a, aux_val_new_a);

CreateSolutionMatrix(A, nrows, ncols, new_a_ptr, aux_index2_new_a, aux_val_new_a);

delete[] new_a_ptr;
delete[] aux_index2_new_a;
delete[] aux_val_new_a;
}


template <class AMatrix, class BMatrix>
static void TransposeMatrix(
AMatrix& rA,
const BMatrix& rB,
const double Factor = 1.0
)
{
typedef typename value_type<AMatrix>::type ValueType;

const IndexType* index1 = rB.index1_data().begin();
const IndexType* index2 = rB.index2_data().begin();
const ValueType* data = rB.value_data().begin();
const SizeType transpose_nonzero_values = rB.value_data().end() - rB.value_data().begin();

const SizeType size_system_1 = rB.size1();
const SizeType size_system_2 = rB.size2();

if (rA.size1() != size_system_2 || rA.size2() != size_system_1 ) {
rA.resize(size_system_2, size_system_1, false);
}

IndexVectorType new_a_ptr(size_system_2 + 1);
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(size_system_2 + 1); ++i)
new_a_ptr[i] = 0;
IndexVectorType aux_index2_new_a(transpose_nonzero_values);
DenseVector<ValueType> aux_val_new_a(transpose_nonzero_values);

const IndexType aux_1_index = 1;

#pragma omp parallel for
for (int i=0; i<static_cast<int>(size_system_1); ++i) {
IndexType row_begin = index1[i];
IndexType row_end   = index1[i+1];

for (IndexType j=row_begin; j<row_end; j++) {
AtomicAdd(new_a_ptr[index2[j] + 1], aux_1_index);
}
}

std::partial_sum(new_a_ptr.begin(), new_a_ptr.end(), &new_a_ptr[0]);

IndexVectorType aux_indexes(size_system_2);
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(size_system_2); ++i)
aux_indexes[i] = 0;

for (int i=0; i<static_cast<int>(size_system_1); ++i) {
IndexType row_begin = index1[i];
IndexType row_end   = index1[i+1];

for (IndexType j=row_begin; j<row_end; j++) {
const IndexType current_row = index2[j];
const IndexType initial_position = new_a_ptr[current_row];

const IndexType current_index = initial_position + aux_indexes[current_row];
aux_index2_new_a[current_index] = i;
aux_val_new_a[current_index] = Factor * data[j];

aux_indexes[current_row] += 1;
}

}

SortRows(&new_a_ptr[0], size_system_2, size_system_1, &aux_index2_new_a[0], &aux_val_new_a[0]);

CreateSolutionMatrix(rA, size_system_2, size_system_1, &new_a_ptr[0], &aux_index2_new_a[0], &aux_val_new_a[0]);
}


template <class CMatrix, typename TSize, typename Ptr, typename IndexType, typename ValueType>
static inline void CreateSolutionMatrix(
CMatrix& C,
const TSize NRows,
const TSize NCols,
const Ptr* CPtr,
const IndexType* AuxIndex2C,
const ValueType* AuxValC
)
{
if ((NRows == 0) || (NCols == 0))
return void();

const TSize nonzero_values = CPtr[NRows];

C = CMatrix(NRows, NCols, nonzero_values);
IndexType* index1_c = C.index1_data().begin();
IndexType* index2_c = C.index2_data().begin();
double* values_c = C.value_data().begin();

index1_c[0] = 0;
for (TSize i = 0; i < NRows; i++)
index1_c[i+1] = index1_c[i] + (CPtr[i+1] - CPtr[i]);

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(nonzero_values); i++) {
KRATOS_DEBUG_ERROR_IF(AuxIndex2C[i] > static_cast<IndexType>(NCols)) << "Index " << AuxIndex2C[i] <<" is greater than the number of columns " << NCols << std::endl;
index2_c[i] = AuxIndex2C[i];
values_c[i] = AuxValC[i];
}

C.set_filled(NRows+1, nonzero_values);
}


template <typename TSize, typename Col, typename TIndexType, typename ValueType>
static inline void SortRows(
const TIndexType* CPtr,
const TSize NRows,
const TSize NCols,
Col* Columns,
ValueType* Values
)
{
#pragma omp parallel
{
#pragma omp for
for (int i_row=0; i_row<static_cast<int>(NRows); i_row++) {
const TIndexType row_beg = CPtr[i_row];
const TIndexType row_end = CPtr[i_row + 1];

for(IndexType j = 1; j < row_end - row_beg; ++j) {
const IndexType c = Columns[j + row_beg];
const double v = Values[j + row_beg];

SignedIndexType i = j - 1;

while(i >= 0 && Columns[i + row_beg] > c) {
KRATOS_DEBUG_ERROR_IF(Columns[i + row_beg] > static_cast<Col>(NCols)) << " Index for column: " << i + row_beg << ". Index " << Columns[i + row_beg] <<" is greater than the number of columns " << NCols << std::endl;
Columns[i + 1 + row_beg] = Columns[i + row_beg];
Values[i + 1 + row_beg] = Values[i + row_beg];
i--;
}

Columns[i + 1 + row_beg] = c;
Values[i + 1 + row_beg] = v;
}
}
}
}


static inline void AssembleSparseMatrixByBlocks(
CompressedMatrix& rMatrix,
const DenseMatrix<CompressedMatrix*>& rMatricespBlocks,
DenseMatrix<double> ContributionCoefficients = DenseMatrix<double>(0,0),
DenseMatrix<bool> TransposeBlocks = DenseMatrix<bool>(0,0)
)
{
const SizeType number_of_rows_blocks = rMatricespBlocks.size1();
const SizeType number_of_columns_blocks = rMatricespBlocks.size2();

if (ContributionCoefficients.size1() == 0 && ContributionCoefficients.size2() == 0) {
ContributionCoefficients.resize(number_of_rows_blocks, number_of_columns_blocks);
for (IndexType i = 0; i < number_of_rows_blocks; ++i) {
for (IndexType j = 0; j < number_of_columns_blocks; ++j) {
ContributionCoefficients(i, j) = 1.0;
}
}
} else {
KRATOS_ERROR_IF(ContributionCoefficients.size1() != number_of_rows_blocks || ContributionCoefficients.size2() != number_of_columns_blocks) << "The ContributionCoefficients dimensions" << ContributionCoefficients.size1() << " and " << ContributionCoefficients.size2() << "do not coincide with the dimensions of rMatricespBlocks" << number_of_rows_blocks << "and " << number_of_columns_blocks << std::endl;
}
if (TransposeBlocks.size1() == 0 && TransposeBlocks.size2() == 0) {
TransposeBlocks.resize(number_of_rows_blocks, number_of_columns_blocks);
for (IndexType i = 0; i < number_of_rows_blocks; ++i) {
for (IndexType j = 0; j < number_of_rows_blocks; ++j) {
TransposeBlocks(i, j) = false;
}
}
} else {
KRATOS_ERROR_IF(TransposeBlocks.size1() != number_of_rows_blocks || TransposeBlocks.size2() != number_of_columns_blocks) << "The TransposeBlocks dimensions" << TransposeBlocks.size1() << " and " << TransposeBlocks.size2() << "do not coincide with the dimensions of rMatricespBlocks" << number_of_rows_blocks << "and " << number_of_columns_blocks << std::endl;
}

SizeType nrows = 0, ncols = 0;
std::vector<SizeType> row_sizes(number_of_rows_blocks);
std::vector<SizeType> column_sizes(number_of_columns_blocks);
for (int i=0; i<static_cast<int>(number_of_rows_blocks); ++i) {
if (TransposeBlocks(i, 0)) {
row_sizes[i] = (*rMatricespBlocks(i, 0)).size2();
} else {
row_sizes[i] = (*rMatricespBlocks(i, 0)).size1();
}
nrows += row_sizes[i];
}
for (int j=0; j<static_cast<int>(number_of_columns_blocks); ++j) {
if (TransposeBlocks(0, j)) {
column_sizes[j] = (*rMatricespBlocks(0, j)).size1();
} else {
column_sizes[j] = (*rMatricespBlocks(0, j)).size2();
}
ncols += column_sizes[j];
}

for (int i=0; i<static_cast<int>(number_of_rows_blocks); ++i) {
for (int j=0; j<static_cast<int>(number_of_columns_blocks); ++j) {
if (TransposeBlocks(i, j)) {
KRATOS_ERROR_IF((*rMatricespBlocks(i, j)).size2() != row_sizes[i] || (*rMatricespBlocks(i, j)).size1() != column_sizes[j]) << " Not consistent size in block " << i << ", " << j << ".\t" << (*rMatricespBlocks(i, j)).size2() << ", " << (*rMatricespBlocks(i, j)).size1() << " vs " <<  row_sizes[i] << ", " << row_sizes[j] << std::endl;
} else {
KRATOS_ERROR_IF((*rMatricespBlocks(i, j)).size1() != row_sizes[i] || (*rMatricespBlocks(i, j)).size2() != column_sizes[j]) << " Not consistent size in block " << i << ", " << j << ".\t" << (*rMatricespBlocks(i, j)).size1() << ", " << (*rMatricespBlocks(i, j)).size2() << " vs " <<  row_sizes[i] << ", " << row_sizes[j] << std::endl;
}
}
}
if ((nrows == 0) || (ncols == 0))
return void();

IndexType* matrix_ptr = new IndexType[nrows + 1];
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(nrows + 1); ++i)
matrix_ptr[i] = 0;

#ifdef KRATOS_DEBUG
IndexType check_non_zero = 0;
DenseMatrix<IndexType> check_non_zero_blocks(number_of_rows_blocks, number_of_columns_blocks);
for (int i=0; i<static_cast<int>(number_of_rows_blocks); ++i) {
for (int j=0; j<static_cast<int>(number_of_columns_blocks); ++j) {
check_non_zero_blocks(i, j) = 0;
}
}
#endif

#pragma omp parallel
{
#pragma omp for
for (int i=0; i<static_cast<int>(number_of_rows_blocks); ++i) {
for (int k=0; k<static_cast<int>(row_sizes[i]); ++k) {
IndexType matrix_cols_aux = 0;
for (int j=0; j<static_cast<int>(number_of_columns_blocks); ++j) {
#ifdef KRATOS_DEBUG
IndexType partial_matrix_cols_aux = 0;
#endif
CompressedMatrix& r_matrix = *rMatricespBlocks(i, j);
if (r_matrix.nnz() > 0) {
if (TransposeBlocks(i, j)) {
const SizeType size_system_1 = r_matrix.size1();
const SizeType size_system_2 = r_matrix.size2();
CompressedMatrix transpose(size_system_2, size_system_1);
TransposeMatrix<CompressedMatrix, CompressedMatrix>(transpose, r_matrix);
ComputeNonZeroBlocks(transpose, k, matrix_cols_aux);
#ifdef KRATOS_DEBUG
ComputeNonZeroBlocks(transpose, k, partial_matrix_cols_aux);
#endif
} else {
ComputeNonZeroBlocks(r_matrix, k, matrix_cols_aux);
#ifdef KRATOS_DEBUG
ComputeNonZeroBlocks(r_matrix, k, partial_matrix_cols_aux);
#endif
}
}
#ifdef KRATOS_DEBUG
check_non_zero_blocks(i, j) += partial_matrix_cols_aux;
#endif
}
IndexType& r_matrix_ptr_value = matrix_ptr[std::accumulate(row_sizes.begin(), row_sizes.begin() + i, 0) + k + 1];
AtomicAdd(r_matrix_ptr_value, matrix_cols_aux);

#ifdef KRATOS_DEBUG
AtomicAdd(check_non_zero, matrix_cols_aux);
#endif
}
}
}

std::partial_sum(matrix_ptr, matrix_ptr + nrows + 1, matrix_ptr);
const SizeType nonzero_values = matrix_ptr[nrows];

#ifdef KRATOS_DEBUG
SizeType total_nnz = 0;
for (int i=0; i<static_cast<int>(number_of_rows_blocks); ++i) {
for (int j=0; j<static_cast<int>(number_of_columns_blocks); ++j) {
const SizeType block_nnz = rMatricespBlocks(i, j)->nnz();
KRATOS_ERROR_IF_NOT(check_non_zero_blocks(i, j) == block_nnz) << "Inconsistent number of non-zero values. Check 0: " << block_nnz << " vs " << check_non_zero_blocks(i, j) << ". Block: " << i << ", " << j << std::endl;
total_nnz += block_nnz;
}
}
KRATOS_ERROR_IF_NOT(check_non_zero == total_nnz) << "Inconsistent number of non-zero values. Check 1: " << total_nnz << " vs " << check_non_zero << std::endl;
KRATOS_ERROR_IF_NOT(nonzero_values == total_nnz) << "Inconsistent number of non-zero values. Check 2: " << total_nnz << " vs " << nonzero_values << std::endl;
#endif

rMatrix = CompressedMatrix(nrows, ncols, nonzero_values);

double* Matrix_values = rMatrix.value_data().begin();
IndexType* Matrix_index1 = rMatrix.index1_data().begin();
IndexType* Matrix_index2 = rMatrix.index2_data().begin();

Matrix_index1[0] = 0;
for (IndexType i = 0; i < nrows; ++i)
Matrix_index1[i+1] = Matrix_index1[i] + (matrix_ptr[i + 1] - matrix_ptr[i]);

#pragma omp parallel
{
#pragma omp for
for (int i=0; i<static_cast<int>(number_of_rows_blocks); ++i) {
for (int k=0; k<static_cast<int>(row_sizes[i]); ++k) {
const IndexType row_beg = matrix_ptr[std::accumulate(row_sizes.begin(), row_sizes.begin() + i, 0) + k];
IndexType row_end = row_beg;
for (int j=0; j<static_cast<int>(number_of_columns_blocks); ++j) {
const SizeType initial_index_column = std::accumulate(column_sizes.begin(), column_sizes.begin() + j, 0);

CompressedMatrix& r_matrix = *rMatricespBlocks(i, j);
if (r_matrix.nnz() > 0) {
if (TransposeBlocks(i, j)) {
const SizeType size_system_1 = r_matrix.size1();
const SizeType size_system_2 = r_matrix.size2();
CompressedMatrix transpose(size_system_2, size_system_1);
TransposeMatrix<CompressedMatrix, CompressedMatrix>(transpose, r_matrix);
ComputeAuxiliarValuesBlocks(transpose, Matrix_index2, Matrix_values, k, row_end, initial_index_column, ContributionCoefficients(i, j));
} else {
ComputeAuxiliarValuesBlocks(r_matrix, Matrix_index2, Matrix_values, k, row_end, initial_index_column, ContributionCoefficients(i, j));
}
}
}
}
}
}

rMatrix.set_filled(nrows+1, nonzero_values);

delete[] matrix_ptr;
}


static inline void ComputeNonZeroBlocks(
const CompressedMatrix& rMatrix,
const int CurrentRow,
IndexType& rNonZeroColsAux2
)
{
const IndexType* aux_matrix_index1 = rMatrix.index1_data().begin();

const IndexType row_begin = aux_matrix_index1[CurrentRow];
const IndexType row_end   = aux_matrix_index1[CurrentRow + 1];

for (IndexType j=row_begin; j<row_end; j++) {
++rNonZeroColsAux2;
}
}


static inline void ComputeAuxiliarValuesBlocks(
const CompressedMatrix& rMatrix,
IndexType* AuxIndex2,
double* AuxVals,
const int CurrentRow,
IndexType& RowEnd,
const SizeType InitialIndexColumn,
const double ContributionCoefficient = 1.0
)
{
const double* aux_values = rMatrix.value_data().begin();
const IndexType* aux_Matrix_index1 = rMatrix.index1_data().begin();
const IndexType* aux_Matrix_index2 = rMatrix.index2_data().begin();

const IndexType aux_Matrix_row_begin = aux_Matrix_index1[CurrentRow];
const IndexType aux_Matrix_row_end   = aux_Matrix_index1[CurrentRow + 1];

for (IndexType j=aux_Matrix_row_begin; j<aux_Matrix_row_end; j++) {
const IndexType col_index = InitialIndexColumn + aux_Matrix_index2[j];
AuxIndex2[RowEnd] = col_index;
AuxVals[RowEnd] = ContributionCoefficient * aux_values[j];
++RowEnd;
}
}




std::string Info() const
{
return "SparseMatrixMultiplicationUtility";
}

void PrintInfo (std::ostream& rOStream) const
{
rOStream << "SparseMatrixMultiplicationUtility";
}

void PrintData (std::ostream& rOStream) const
{
}


protected:







private:





template <bool TNeedOut, class TIndex>
static TIndex* MergeRows(
const TIndex* Column1,
const TIndex* Column1End,
const TIndex* Column2,
const TIndex* Column2End,
TIndex* Column3
)
{
while(Column1 != Column1End && Column2 != Column2End) {
TIndex c1 = *Column1;
TIndex c2 = *Column2;

if (c1 < c2) {
if constexpr (TNeedOut) *Column3 = c1;
++Column1;
} else if (c1 == c2) {
if constexpr (TNeedOut) *Column3 = c1;
++Column1;
++Column2;
} else {
if constexpr (TNeedOut) *Column3 = c2;
++Column2;
}
++Column3;
}

if constexpr (TNeedOut) {
if (Column1 < Column1End) {
return std::copy(Column1, Column1End, Column3);
} else if (Column2 < Column2End) {
return std::copy(Column2, Column2End, Column3);
} else {
return Column3;
}
} else {
return Column3 + (Column1End - Column1) + (Column2End - Column2);
}
}


template <class TIndex, class TValueType>
static TIndex* MergeRows(
const TValueType &rAlpha1,
const TIndex* Column1,
const TIndex* Column1End,
const TValueType *Value1,
const TValueType &rAlpha2,
const TIndex* Column2,
const TIndex* Column2End,
const TValueType *Value2,
TIndex* Column3,
TValueType *Value3
)
{
while(Column1 != Column1End && Column2 != Column2End) {
TIndex c1 = *Column1;
TIndex c2 = *Column2;

if (c1 < c2) {
++Column1;

*Column3 = c1;
*Value3 = rAlpha1 * (*Value1++);
} else if (c1 == c2) {
++Column1;
++Column2;

*Column3 = c1;
*Value3 = rAlpha1 * (*Value1++) + rAlpha2 * (*Value2++);
} else {
++Column2;

*Column3 = c2;
*Value3 = rAlpha2 * (*Value2++);
}

++Column3;
++Value3;
}

while(Column1 < Column1End) {
*Column3++ = *Column1++;
*Value3++ = rAlpha1 * (*Value1++);
}

while(Column2 < Column2End) {
*Column3++ = *Column2++;
*Value3++ = rAlpha2 * (*Value2++);
}

return Column3;
}


template <class TIndex>
static TIndex ProdRowWidth(
const TIndex* AColumn,
const TIndex* AColumnEnd,
const TIndex* BPtr,
const TIndex* BColumn,
TIndex* Tmp1Column,
TIndex* Tmp2Column,
TIndex* Tmp3Column
)
{
const TIndex nrow = AColumnEnd - AColumn;


if (nrow == 0) return 0;


if (nrow == 1) return BPtr[*AColumn + 1] - BPtr[*AColumn];


if (nrow == 2) {
int a1 = AColumn[0];
int a2 = AColumn[1];

return MergeRows<false>( BColumn + BPtr[a1], BColumn + BPtr[a1+1], BColumn + BPtr[a2], BColumn + BPtr[a2+1], Tmp1Column) - Tmp1Column;
}


TIndex a1 = *AColumn++;
TIndex a2 = *AColumn++;
TIndex c_col1 = MergeRows<true>(  BColumn + BPtr[a1], BColumn + BPtr[a1+1], BColumn + BPtr[a2], BColumn + BPtr[a2+1], Tmp1Column ) - Tmp1Column;

while(AColumn + 1 < AColumnEnd) {
a1 = *AColumn++;
a2 = *AColumn++;

TIndex c_col2 = MergeRows<true>(  BColumn + BPtr[a1], BColumn + BPtr[a1+1], BColumn + BPtr[a2], BColumn + BPtr[a2+1], Tmp2Column ) - Tmp2Column;

if (AColumn == AColumnEnd) {
return MergeRows<false>( Tmp1Column, Tmp1Column + c_col1, Tmp2Column, Tmp2Column + c_col2, Tmp3Column ) - Tmp3Column;
} else {
c_col1 = MergeRows<true>( Tmp1Column, Tmp1Column + c_col1, Tmp2Column, Tmp2Column + c_col2, Tmp3Column ) - Tmp3Column;

std::swap(Tmp1Column, Tmp3Column);
}
}

a2 = *AColumn;
return MergeRows<false>( Tmp1Column, Tmp1Column + c_col1, BColumn + BPtr[a2], BColumn + BPtr[a2+1], Tmp2Column ) - Tmp2Column;
}


template <class TIndex, class TValueType>
static void ProdRow(
const TIndex* AColumn,
const TIndex* AColumnEnd,
const TValueType *AValue,
const TIndex* BPtr,
const TIndex* BColumn,
const TValueType *BValue,
TIndex* OutColumn,
TValueType *OutValue,
TIndex* Tmp2Column,
TValueType *Tmp2Value,
TIndex* Tmp3Column,
TValueType *Tmp3Value
)
{
const TIndex nrow = AColumnEnd - AColumn;


if (nrow == 0) return;


if (nrow == 1) {
TIndex ac = *AColumn;
TValueType av = *AValue;

const TValueType *bv = BValue + BPtr[ac];
const TIndex* bc = BColumn + BPtr[ac];
const TIndex* be = BColumn + BPtr[ac+1];

while(bc != be) {
*OutColumn++ = *bc++;
*OutValue++ = av * (*bv++);
}

return;
}


if (nrow == 2) {
TIndex ac1 = AColumn[0];
TIndex ac2 = AColumn[1];

TValueType av1 = AValue[0];
TValueType av2 = AValue[1];

MergeRows( av1, BColumn + BPtr[ac1], BColumn + BPtr[ac1+1], BValue + BPtr[ac1], av2, BColumn + BPtr[ac2], BColumn + BPtr[ac2+1], BValue + BPtr[ac2], OutColumn, OutValue );
}


TIndex ac1 = *AColumn++;
TIndex ac2 = *AColumn++;

TValueType av1 = *AValue++;
TValueType av2 = *AValue++;

TIndex* tm1_col = OutColumn;
TValueType *tm1_val = OutValue;

TIndex c_col1 = MergeRows( av1, BColumn + BPtr[ac1], BColumn + BPtr[ac1+1], BValue + BPtr[ac1], av2, BColumn + BPtr[ac2], BColumn + BPtr[ac2+1], BValue + BPtr[ac2], tm1_col, tm1_val ) - tm1_col;

while(AColumn + 1 < AColumnEnd) {
ac1 = *AColumn++;
ac2 = *AColumn++;

av1 = *AValue++;
av2 = *AValue++;

TIndex c_col2 = MergeRows( av1, BColumn + BPtr[ac1], BColumn + BPtr[ac1+1], BValue + BPtr[ac1], av2, BColumn + BPtr[ac2], BColumn + BPtr[ac2+1], BValue + BPtr[ac2], Tmp2Column, Tmp2Value ) - Tmp2Column;

c_col1 = MergeRows( amgcl::math::identity<TValueType>(), tm1_col, tm1_col + c_col1, tm1_val, amgcl::math::identity<TValueType>(), Tmp2Column, Tmp2Column + c_col2, Tmp2Value, Tmp3Column, Tmp3Value ) - Tmp3Column;

std::swap(Tmp3Column, tm1_col);
std::swap(Tmp3Value, tm1_val);
}

if (AColumn < AColumnEnd) {
ac2 = *AColumn++;
av2 = *AValue++;

c_col1 = MergeRows( amgcl::math::identity<TValueType>(), tm1_col, tm1_col + c_col1, tm1_val, av2, BColumn + BPtr[ac2], BColumn + BPtr[ac2+1], BValue + BPtr[ac2], Tmp3Column, Tmp3Value ) - Tmp3Column;

std::swap(Tmp3Column, tm1_col);
std::swap(Tmp3Value, tm1_val);
}

if (tm1_col != OutColumn) {
std::copy(tm1_col, tm1_col + c_col1, OutColumn);
std::copy(tm1_val, tm1_val + c_col1, OutValue);
}

return;
}





}; 







}  

#endif 
