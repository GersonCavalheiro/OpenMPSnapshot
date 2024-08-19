
#if !defined(KRATOS_FEAST_CONDITION_NUMBER_UTILITY )
#define  KRATOS_FEAST_CONDITION_NUMBER_UTILITY



#include "spaces/ublas_space.h"
#include "linear_solvers/linear_solver.h"
#ifdef USE_EIGEN_FEAST
#include "custom_solvers/feast_eigensystem_solver.h"
#endif

namespace Kratos
{







template<class TSparseSpace = UblasSpace<double, CompressedMatrix, Vector>,
class TDenseSpace = UblasSpace<double, Matrix, Vector>
>
class FEASTConditionNumberUtility
{
public:


KRATOS_CLASS_POINTER_DEFINITION(FEASTConditionNumberUtility);

typedef std::size_t                                          SizeType;
typedef std::size_t                                         IndexType;

typedef typename TSparseSpace::MatrixType            SparseMatrixType;
typedef typename TSparseSpace::VectorType            SparseVectorType;

typedef typename TDenseSpace::MatrixType              DenseMatrixType;
typedef typename TDenseSpace::VectorType              DenseVectorType;






static inline double GetConditionNumber(const SparseMatrixType& InputMatrix)
{
#ifdef USE_EIGEN_FEAST
typedef FEASTEigensystemSolver<true, double, double> FEASTSolverType;

Parameters this_params(R"(
{
"solver_type"                : "feast",
"symmetric"                  : true,
"number_of_eigenvalues"      : 3,
"search_lowest_eigenvalues"  : true,
"search_highest_eigenvalues" : false,
"e_min"                      : 0.0,
"e_max"                      : 1.0,
"echo_level"                 : 0
})");

const std::size_t size_matrix = InputMatrix.size1();

const double normA = TSparseSpace::TwoNorm(InputMatrix);

this_params["e_max"].SetDouble(normA);
this_params["e_min"].SetDouble(-normA);
SparseMatrixType copy_matrix = InputMatrix;
SparseMatrixType identity_matrix(size_matrix, size_matrix);
for (IndexType i = 0; i < size_matrix; ++i)
identity_matrix.push_back(i, i, 1.0);

DenseMatrixType eigen_vectors(size_matrix, 1);
DenseVectorType eigen_values(size_matrix);

FEASTSolverType feast_solver_lowest(this_params);

feast_solver_lowest.Solve(copy_matrix, identity_matrix, eigen_values, eigen_vectors);

int dim_eigen_values = eigen_values.size();

#pragma omp parallel for
for (int i = 0; i < dim_eigen_values; i++) {
eigen_values[i] = std::abs(eigen_values[i]);
}

std::sort(eigen_values.begin(), eigen_values.end());

const double lowest_eigen_value = eigen_values[0];

this_params["search_lowest_eigenvalues"].SetBool(false);
this_params["search_highest_eigenvalues"].SetBool(true);
FEASTSolverType feast_solver_highest(this_params);

copy_matrix = InputMatrix;
feast_solver_highest.Solve(copy_matrix, identity_matrix, eigen_values, eigen_vectors);

dim_eigen_values = eigen_values.size();

#pragma omp parallel for
for (int i = 0; i < dim_eigen_values; i++) {
eigen_values[i] = std::abs(eigen_values[i]);
}

std::sort(eigen_values.begin(), eigen_values.end());

const double highest_eigen_value = eigen_values[dim_eigen_values - 1];

const double condition_number = highest_eigen_value/lowest_eigen_value;
#else
const double condition_number = 0.0;
KRATOS_ERROR << "YOU MUST COMPILE FEAST IN ORDER TO USE THIS UTILITY" << std::endl;
#endif

return condition_number;
}







private:









FEASTConditionNumberUtility(void);

FEASTConditionNumberUtility(FEASTConditionNumberUtility& rSource);

}; 




}  

#endif 

