
#if !defined(KRATOS_SCALING_SOLVER_H_INCLUDED )
#define  KRATOS_SCALING_SOLVER_H_INCLUDED

#include <cmath>
#include <complex>


#include "includes/define.h"
#include "factories/linear_solver_factory.h"
#include "linear_solvers/linear_solver.h"
#include "utilities/openmp_utils.h"

namespace Kratos
{







template<class TSparseSpaceType, class TDenseSpaceType,
class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType> >
class ScalingSolver
: public LinearSolver<TSparseSpaceType, TDenseSpaceType,  TReordererType>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ScalingSolver);

typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType> BaseType;

typedef typename TSparseSpaceType::MatrixType SparseMatrixType;

typedef typename TSparseSpaceType::VectorType VectorType;

typedef typename TDenseSpaceType::MatrixType DenseMatrixType;

typedef LinearSolverFactory<TSparseSpaceType,TDenseSpaceType> LinearSolverFactoryType;

typedef typename TSparseSpaceType::IndexType IndexType;


ScalingSolver()
{
}


ScalingSolver(
typename BaseType::Pointer pLinearSolver,
const bool SymmetricScaling = true
) : BaseType (),
mpLinearSolver(pLinearSolver),
mSymmetricScaling(SymmetricScaling)
{
}


ScalingSolver(Parameters ThisParameters)
: BaseType ()
{
KRATOS_TRY

KRATOS_ERROR_IF_NOT(ThisParameters.Has("solver_type")) << "Solver_type must be specified to construct the ScalingSolver" << std::endl;

mpLinearSolver = LinearSolverFactoryType().Create(ThisParameters);

mSymmetricScaling = ThisParameters.Has("symmetric_scaling") ? ThisParameters["symmetric_scaling"].GetBool() : true;

KRATOS_CATCH("")
}

ScalingSolver(const ScalingSolver& Other) : BaseType(Other) {}


~ScalingSolver() override {}



ScalingSolver& operator=(const ScalingSolver& Other)
{
BaseType::operator=(Other);
return *this;
}


bool AdditionalPhysicalDataIsNeeded() override
{
return mpLinearSolver->AdditionalPhysicalDataIsNeeded();
}


void ProvideAdditionalData(
SparseMatrixType& rA,
VectorType& rX,
VectorType& rB,
typename ModelPart::DofsArrayType& rdof_set,
ModelPart& r_model_part
) override
{
mpLinearSolver->ProvideAdditionalData(rA,rX,rB,rdof_set,r_model_part);
}

void InitializeSolutionStep (SparseMatrixType& rA, VectorType& rX, VectorType& rB) override
{
mpLinearSolver->InitializeSolutionStep(rA,rX,rB);
}


void FinalizeSolutionStep (SparseMatrixType& rA, VectorType& rX, VectorType& rB) override
{
mpLinearSolver->FinalizeSolutionStep(rA,rX,rB);
}


void Clear() override
{
mpLinearSolver->Clear();
}


bool Solve(SparseMatrixType& rA, VectorType& rX, VectorType& rB) override
{
if(this->IsNotConsistent(rA, rX, rB))
return false;

VectorType scaling_vector(rX.size());

GetScalingWeights(rA,scaling_vector);

if(mSymmetricScaling == false)
{
KRATOS_THROW_ERROR(std::logic_error,"not yet implemented","")
}
else
{
IndexPartition<std::size_t>(scaling_vector.size()).for_each([&](std::size_t Index){
scaling_vector[Index] = sqrt(std::abs(scaling_vector[Index]));
});

SymmetricScaling(rA,scaling_vector);


}

IndexPartition<std::size_t>(scaling_vector.size()).for_each([&](std::size_t Index){
rB[Index] /= scaling_vector[Index];
});


bool is_solved = mpLinearSolver->Solve(rA,rX,rB);

if(mSymmetricScaling == true)
{
IndexPartition<std::size_t>(scaling_vector.size()).for_each([&](std::size_t Index){
rX[Index] /= scaling_vector[Index];
});
}

return is_solved;
}




IndexType GetIterationsNumber() override
{
return mpLinearSolver->GetIterationsNumber();
}





std::string Info() const override
{
std::stringstream buffer;
buffer << "Composite Linear Solver. Uses internally the following linear solver " << mpLinearSolver->Info();
return  buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
BaseType::PrintData(rOStream);
}





protected:















private:


typename LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType>::Pointer mpLinearSolver;
bool mSymmetricScaling;

static void SymmetricScaling( SparseMatrixType& A, const VectorType& aux)
{

OpenMPUtils::PartitionVector partition;
int number_of_threads = ParallelUtilities::GetNumThreads();
OpenMPUtils::DivideInPartitions(A.size1(),number_of_threads,  partition);

#pragma omp parallel
{
int thread_id = OpenMPUtils::ThisThread();
int number_of_rows = partition[thread_id+1] - partition[thread_id];
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::iterator row_iter_begin = A.index1_data().begin()+partition[thread_id];
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::iterator index_2_begin = A.index2_data().begin()+*row_iter_begin;
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::value_array_type::iterator value_begin = A.value_data().begin()+*row_iter_begin;

perform_matrix_scaling(    number_of_rows,
row_iter_begin,
index_2_begin,
value_begin,
partition[thread_id],
aux
);
}
}


static void perform_matrix_scaling(
int number_of_rows,
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::iterator row_begin,
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::iterator index2_begin,
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::value_array_type::iterator value_begin,
unsigned int output_begin_index,
const VectorType& weights
)
{
int row_size;
typename SparseMatrixType::index_array_type::const_iterator row_it = row_begin;
int kkk = output_begin_index;
for(int k = 0; k < number_of_rows; k++)
{
row_size= *(row_it+1)-*row_it;
row_it++;
const typename TDenseSpaceType::DataType row_weight = weights[kkk++];

for(int i = 0; i<row_size; i++)
{
const typename TDenseSpaceType::DataType col_weight = weights[*index2_begin];
typename TDenseSpaceType::DataType t = (*value_begin);
t /= (row_weight*col_weight);
(*value_begin) = t; 
value_begin++;
index2_begin++;
}

}
}


static void GetScalingWeights( const SparseMatrixType& A, VectorType& aux)
{

OpenMPUtils::PartitionVector partition;
int number_of_threads = ParallelUtilities::GetNumThreads();
OpenMPUtils::DivideInPartitions(A.size1(),number_of_threads,  partition);

#pragma omp parallel
{
int thread_id = OpenMPUtils::ThisThread();
int number_of_rows = partition[thread_id+1] - partition[thread_id];
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::const_iterator row_iter_begin = A.index1_data().begin()+partition[thread_id];
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::const_iterator index_2_begin = A.index2_data().begin()+*row_iter_begin;
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::value_array_type::const_iterator value_begin = A.value_data().begin()+*row_iter_begin;

GS2weights(    number_of_rows,
row_iter_begin,
index_2_begin,
value_begin,
partition[thread_id],
aux
);
}
}


static void GS2weights(
int number_of_rows,
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::const_iterator row_begin,
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::index_array_type::const_iterator index2_begin,
typename boost::numeric::ublas::compressed_matrix<typename TDenseSpaceType::DataType>::value_array_type::const_iterator value_begin,
unsigned int output_begin_index,
VectorType& weights
)
{
int row_size;
typename SparseMatrixType::index_array_type::const_iterator row_it = row_begin;
int kkk = output_begin_index;
for(int k = 0; k < number_of_rows; k++)
{
row_size= *(row_it+1)-*row_it;
row_it++;
double t = 0.0;

for(int i = 0; i<row_size; i++)
{
double tmp = std::abs(*value_begin);
t += tmp*tmp;
value_begin++;
}
t = sqrt(t);
weights[kkk++] = t;
}
}











}; 






template<class TSparseSpaceType, class TDenseSpaceType,
class TPreconditionerType,
class TReordererType>
inline std::istream& operator >> (std::istream& IStream,
ScalingSolver<TSparseSpaceType, TDenseSpaceType,
TReordererType>& rThis)
{
return IStream;
}

template<class TSparseSpaceType, class TDenseSpaceType,
class TPreconditionerType,
class TReordererType>
inline std::ostream& operator << (std::ostream& OStream,
const ScalingSolver<TSparseSpaceType, TDenseSpaceType,
TReordererType>& rThis)
{
rThis.PrintInfo(OStream);
OStream << std::endl;
rThis.PrintData(OStream);

return OStream;
}


}  

#endif 


