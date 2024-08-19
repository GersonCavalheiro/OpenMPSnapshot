
#pragma once

#include <numeric>


#include "includes/define.h"
#include "includes/process_info.h"
#include "includes/ublas_interface.h"
#include "includes/matrix_market_interface.h"
#include "utilities/dof_updater.h"
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"

namespace Kratos
{
#ifdef _OPENMP

template <class Type>
class MultValueNoAdd
{
private:
Type Factor; 
public:

MultValueNoAdd(const Type& _Val) : Factor(_Val)
{
}


inline Type operator () (const Type& elem) const
{
return elem * Factor;
}
};

template <class Type>
class MultAndAddValue
{
private:
Type Factor; 
public:

MultAndAddValue(const Type& _Val) : Factor(_Val)
{
}


inline Type operator () (const Type& elem1, const Type& elem2) const
{
return elem1 * Factor + elem2;
}
};
#endif



template <class TDataType, class TMatrixType, class TVectorType>
class UblasSpace;

template <class TDataType>
using TUblasSparseSpace =
UblasSpace<TDataType, boost::numeric::ublas::compressed_matrix<TDataType>, boost::numeric::ublas::vector<TDataType>>;
template <class TDataType>
using TUblasDenseSpace =
UblasSpace<TDataType, DenseMatrix<TDataType>, DenseVector<TDataType>>;


enum class SCALING_DIAGONAL {NO_SCALING = 0, CONSIDER_NORM_DIAGONAL = 1, CONSIDER_MAX_DIAGONAL = 2, CONSIDER_PRESCRIBED_DIAGONAL = 3};





template<class TDataType, class TMatrixType, class TVectorType>
class UblasSpace
{
public:

KRATOS_CLASS_POINTER_DEFINITION(UblasSpace);

typedef TDataType DataType;

typedef TMatrixType MatrixType;

typedef TVectorType VectorType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef typename Kratos::shared_ptr< TMatrixType > MatrixPointerType;
typedef typename Kratos::shared_ptr< TVectorType > VectorPointerType;

typedef DofUpdater< UblasSpace<TDataType,TMatrixType,TVectorType> > DofUpdaterType;
typedef typename DofUpdaterType::UniquePointer DofUpdaterPointerType;


UblasSpace()
{
}

virtual ~UblasSpace()
{
}



static MatrixPointerType CreateEmptyMatrixPointer()
{
return MatrixPointerType(new TMatrixType(0, 0));
}

static VectorPointerType CreateEmptyVectorPointer()
{
return VectorPointerType(new TVectorType(0));
}


static IndexType Size(VectorType const& rV)
{
return rV.size();
}


static IndexType Size1(MatrixType const& rM)
{
return rM.size1();
}


static IndexType Size2(MatrixType const& rM)
{
return rM.size2();
}

template<typename TColumnType>
static void GetColumn(unsigned int j, Matrix& rM, TColumnType& rX)
{
if (rX.size() != rM.size1())
rX.resize(rM.size1(), false);

for (std::size_t i = 0; i < rM.size1(); i++) {
rX[i] = rM(i, j);
}
}

template<typename TColumnType>
static void SetColumn(unsigned int j, Matrix& rM, TColumnType& rX)
{
for (std::size_t i = 0; i < rM.size1(); i++) {
rM(i,j) = rX[i];
}
}



static void Copy(MatrixType const& rX, MatrixType& rY)
{
rY.assign(rX);
}


static void Copy(VectorType const& rX, VectorType& rY)
{
#ifndef _OPENMP
rY.assign(rX);
#else

const int size = rX.size();
if (rY.size() != static_cast<unsigned int>(size))
rY.resize(size, false);

#pragma omp parallel for
for (int i = 0; i < size; i++)
rY[i] = rX[i];
#endif
}


static TDataType Dot(VectorType const& rX, VectorType const& rY)
{
#ifndef _OPENMP
return inner_prod(rX, rY);
#else
const int size = static_cast<int>(rX.size());

TDataType total = TDataType();
#pragma omp parallel for reduction( +: total), firstprivate(size)
for(int i =0; i<size; ++i)
total += rX[i]*rY[i];

return total;
#endif
}



static TDataType TwoNorm(VectorType const& rX)
{
return std::sqrt(Dot(rX, rX));
}

static TDataType TwoNorm(const Matrix& rA) 
{
TDataType aux_sum = TDataType();
#pragma omp parallel for reduction(+:aux_sum)
for (int i=0; i<static_cast<int>(rA.size1()); ++i) {
for (int j=0; j<static_cast<int>(rA.size2()); ++j) {
aux_sum += std::pow(rA(i,j),2);
}
}
return std::sqrt(aux_sum);
}

static TDataType TwoNorm(const compressed_matrix<TDataType> & rA) 
{
TDataType aux_sum = TDataType();

const auto& r_values = rA.value_data();

#pragma omp parallel for reduction(+:aux_sum)
for (int i=0; i<static_cast<int>(r_values.size()); ++i) {
aux_sum += std::pow(r_values[i] , 2);
}
return std::sqrt(aux_sum);
}


static TDataType JacobiNorm(const Matrix& rA)
{
TDataType aux_sum = TDataType();
#pragma omp parallel for reduction(+:aux_sum)
for (int i=0; i<static_cast<int>(rA.size1()); ++i) {
for (int j=0; j<static_cast<int>(rA.size2()); ++j) {
if (i != j) {
aux_sum += std::abs(rA(i,j));
}
}
}
return aux_sum;
}

static TDataType JacobiNorm(const compressed_matrix<TDataType>& rA)
{
TDataType aux_sum = TDataType();

typedef typename compressed_matrix<TDataType>::const_iterator1 t_it_1;
typedef typename compressed_matrix<TDataType>::const_iterator2 t_it_2;

for (t_it_1 it_1 = rA.begin1(); it_1 != rA.end1(); ++it_1) {
for (t_it_2 it_2 = it_1.begin(); it_2 != it_1.end(); ++it_2) {
if (it_2.index1() != it_2.index2()) {
aux_sum += std::abs(*it_2);
}
}
}
return aux_sum;
}

static void Mult(const Matrix& rA, VectorType& rX, VectorType& rY)
{
axpy_prod(rA, rX, rY, true);
}

static void Mult(const compressed_matrix<TDataType>& rA, VectorType& rX, VectorType& rY)
{
#ifndef _OPENMP
axpy_prod(rA, rX, rY, true);
#else
ParallelProductNoAdd(rA, rX, rY);
#endif
}

template< class TOtherMatrixType >
static void TransposeMult(TOtherMatrixType& rA, VectorType& rX, VectorType& rY)
{
boost::numeric::ublas::axpy_prod(rX, rA, rY, true);
} 

static inline SizeType GraphDegree(IndexType i, TMatrixType& A)
{
typename MatrixType::iterator1 a_iterator = A.begin1();
std::advance(a_iterator, i);
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
return ( std::distance(a_iterator.begin(), a_iterator.end()));
#else
return ( std::distance(begin(a_iterator, boost::numeric::ublas::iterator1_tag()),
end(a_iterator, boost::numeric::ublas::iterator1_tag())));
#endif
}

static inline void GraphNeighbors(IndexType i, TMatrixType& A, std::vector<IndexType>& neighbors)
{
neighbors.clear();
typename MatrixType::iterator1 a_iterator = A.begin1();
std::advance(a_iterator, i);
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
for (typename MatrixType::iterator2 row_iterator = a_iterator.begin();
row_iterator != a_iterator.end(); ++row_iterator)
{
#else
for (typename MatrixType::iterator2 row_iterator = begin(a_iterator,
boost::numeric::ublas::iterator1_tag());
row_iterator != end(a_iterator,
boost::numeric::ublas::iterator1_tag()); ++row_iterator)
{
#endif
neighbors.push_back(row_iterator.index2());
}
}


/
static double CheckAndCorrectZeroDiagonalValues(
const ProcessInfo& rProcessInfo,
MatrixType& rA,
VectorType& rb,
const SCALING_DIAGONAL ScalingDiagonal = SCALING_DIAGONAL::NO_SCALING
)
{
const std::size_t system_size = rA.size1();

const double* Avalues = rA.value_data().begin();
const std::size_t* Arow_indices = rA.index1_data().begin();

const double zero_tolerance = std::numeric_limits<double>::epsilon();

const double scale_factor = GetScaleNorm(rProcessInfo, rA, ScalingDiagonal);

IndexPartition(system_size).for_each([&](std::size_t Index){
bool empty = true;

const std::size_t col_begin = Arow_indices[Index];
const std::size_t col_end = Arow_indices[Index + 1];

for (std::size_t j = col_begin; j < col_end; ++j) {
if(std::abs(Avalues[j]) > zero_tolerance) {
empty = false;
break;
}
}

if(empty) {
rA(Index, Index) = scale_factor;
rb[Index] = 0.0;
}
});

return scale_factor;
}


static double GetScaleNorm(
const ProcessInfo& rProcessInfo,
const MatrixType& rA,
const SCALING_DIAGONAL ScalingDiagonal = SCALING_DIAGONAL::NO_SCALING
)
{
switch (ScalingDiagonal) {
case SCALING_DIAGONAL::NO_SCALING:
return 1.0;
case SCALING_DIAGONAL::CONSIDER_PRESCRIBED_DIAGONAL: {
KRATOS_ERROR_IF_NOT(rProcessInfo.Has(BUILD_SCALE_FACTOR)) << "Scale factor not defined at process info" << std::endl;
return rProcessInfo.GetValue(BUILD_SCALE_FACTOR);
}
case SCALING_DIAGONAL::CONSIDER_NORM_DIAGONAL:
return GetDiagonalNorm(rA)/static_cast<double>(rA.size1());
case SCALING_DIAGONAL::CONSIDER_MAX_DIAGONAL:
return GetMaxDiagonal(rA);
default:
return GetMaxDiagonal(rA);
}
}


static double GetDiagonalNorm(const MatrixType& rA)
{
const double* Avalues = rA.value_data().begin();
const std::size_t* Arow_indices = rA.index1_data().begin();
const std::size_t* Acol_indices = rA.index2_data().begin();

const double diagonal_norm = IndexPartition<std::size_t>(Size1(rA)).for_each<SumReduction<double>>([&](std::size_t Index){
const std::size_t col_begin = Arow_indices[Index];
const std::size_t col_end = Arow_indices[Index+1];
for (std::size_t j = col_begin; j < col_end; ++j) {
if (Acol_indices[j] == Index ) {
return std::pow(Avalues[j], 2);
}
}
return 0.0;
});

return std::sqrt(diagonal_norm);
}


static double GetAveragevalueDiagonal(const MatrixType& rA)
{
return 0.5 * (GetMaxDiagonal(rA) + GetMinDiagonal(rA));
}


static double GetMaxDiagonal(const MatrixType& rA)
{
const double* Avalues = rA.value_data().begin();
const std::size_t* Arow_indices = rA.index1_data().begin();
const std::size_t* Acol_indices = rA.index2_data().begin();

return IndexPartition<std::size_t>(Size1(rA)).for_each<MaxReduction<double>>([&](std::size_t Index){
const std::size_t col_begin = Arow_indices[Index];
const std::size_t col_end = Arow_indices[Index+1];
for (std::size_t j = col_begin; j < col_end; ++j) {
if (Acol_indices[j] == Index ) {
return std::abs(Avalues[j]);
}
}
return std::numeric_limits<double>::lowest();
});
}


static double GetMinDiagonal(const MatrixType& rA)
{
const double* Avalues = rA.value_data().begin();
const std::size_t* Arow_indices = rA.index1_data().begin();
const std::size_t* Acol_indices = rA.index2_data().begin();

return IndexPartition<std::size_t>(Size1(rA)).for_each<MinReduction<double>>([&](std::size_t Index){
const std::size_t col_begin = Arow_indices[Index];
const std::size_t col_end = Arow_indices[Index+1];
for (std::size_t j = col_begin; j < col_end; ++j) {
if (Acol_indices[j] == Index ) {
return std::abs(Avalues[j]);
}
}
return std::numeric_limits<double>::max();
});
}





virtual std::string Info() const
{
return "UBlasSpace";
}


virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "UBlasSpace";
}


virtual void PrintData(std::ostream& rOStream) const
{
}

/ TOtherMatrixType& rM, const bool Symmetric)
{
return Kratos::WriteMatrixMarketMatrix(pFileName, rM, Symmetric);
}

template< class VectorType >
static bool WriteMatrixMarketVector(const char* pFileName, const VectorType& rV)
{
return Kratos::WriteMatrixMarketVector(pFileName, rV);
}

static DofUpdaterPointerType CreateDofUpdater()
{
DofUpdaterType tmp;
return tmp.Create();
}


protected:







private:



#ifdef _OPENMP

static void ParallelProductNoAdd(const MatrixType& A, const VectorType& in, VectorType& out)
{
DenseVector<unsigned int> partition;
unsigned int number_of_threads = omp_get_max_threads();
unsigned int number_of_initialized_rows = A.filled1() - 1;
CreatePartition(number_of_threads, number_of_initialized_rows, partition);
#pragma omp parallel
{
int thread_id = omp_get_thread_num();
int number_of_rows = partition[thread_id + 1] - partition[thread_id];
typename compressed_matrix<TDataType>::index_array_type::const_iterator row_iter_begin = A.index1_data().begin() + partition[thread_id];
typename compressed_matrix<TDataType>::index_array_type::const_iterator index_2_begin = A.index2_data().begin()+*row_iter_begin;
typename compressed_matrix<TDataType>::value_array_type::const_iterator value_begin = A.value_data().begin()+*row_iter_begin;


partial_product_no_add(number_of_rows,
row_iter_begin,
index_2_begin,
value_begin,
in,
partition[thread_id],
out
);
}
}

static void CreatePartition(unsigned int number_of_threads, const int number_of_rows, DenseVector<unsigned int>& partitions)
{
partitions.resize(number_of_threads + 1);
int partition_size = number_of_rows / number_of_threads;
partitions[0] = 0;
partitions[number_of_threads] = number_of_rows;
for (unsigned int i = 1; i < number_of_threads; i++)
partitions[i] = partitions[i - 1] + partition_size;
}



static void partial_product_no_add(
int number_of_rows,
typename compressed_matrix<TDataType>::index_array_type::const_iterator row_begin,
typename compressed_matrix<TDataType>::index_array_type::const_iterator index2_begin,
typename compressed_matrix<TDataType>::value_array_type::const_iterator value_begin,
const VectorType& input_vec,
unsigned int output_begin_index,
VectorType& output_vec
)
{
int row_size;
int kkk = output_begin_index;
typename MatrixType::index_array_type::const_iterator row_it = row_begin;
for (int k = 0; k < number_of_rows; k++)
{
row_size = *(row_it + 1)-*row_it;
row_it++;
TDataType t = TDataType();

for (int i = 0; i < row_size; i++)
t += *value_begin++ * (input_vec[*index2_begin++]);

output_vec[kkk++] = t;

}
}
#endif





UblasSpace & operator=(UblasSpace const& rOther);

UblasSpace(UblasSpace const& rOther);

}; 







} 
