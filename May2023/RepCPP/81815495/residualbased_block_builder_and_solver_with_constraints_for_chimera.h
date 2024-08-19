#if !defined(RESIDUAL_BASED_BLOCK_BUILDER_AND_SOLVER_WITH_CONSTRAINTS_FOR_CHIMERA)
#define RESIDUAL_BASED_BLOCK_BUILDER_AND_SOLVER_WITH_CONSTRAINTS_FOR_CHIMERA


#include <unordered_set>
#include <unordered_map>




#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"
#include "includes/master_slave_constraint.h"

namespace Kratos
{







template <class TSparseSpace,
class TDenseSpace,
class TLinearSolver>
class ResidualBasedBlockBuilderAndSolverWithConstraintsForChimera
: public ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:

typedef ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::IndexType IndexType;

typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;


KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedBlockBuilderAndSolverWithConstraintsForChimera);



explicit ResidualBasedBlockBuilderAndSolverWithConstraintsForChimera(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pNewLinearSystemSolver)
{
}


~ResidualBasedBlockBuilderAndSolverWithConstraintsForChimera() = default;





void Clear() override
{
BaseType::Clear();
mL.resize(0,0,false);
}




std::string Info() const override
{
return "ResidualBasedBlockBuilderAndSolverWithConstraintsForChimera";
}



protected:
TSystemMatrixType mL; 


void ConstructMasterSlaveConstraintsStructure(ModelPart &rModelPart) override
{
BaseType::ConstructMasterSlaveConstraintsStructure(rModelPart);
if (rModelPart.MasterSlaveConstraints().size() > 0)
{
mL = BaseType::mT;
}
}

void BuildMasterSlaveConstraints(ModelPart &rModelPart) override
{

KRATOS_TRY

BaseType::BuildMasterSlaveConstraints(rModelPart);

for (auto eq_id : BaseType::mMasterIds)
{
mL(eq_id, eq_id) = 1.0;
}

for (auto eq_id : BaseType::mInactiveSlaveDofs)
{
mL(eq_id, eq_id) = 1.0;
}

KRATOS_CATCH("")
}

void ApplyConstraints(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rb) override
{
KRATOS_TRY

if (rModelPart.MasterSlaveConstraints().size() != 0)
{
double start_constraints = OpenMPUtils::GetCurrentTime();
BuildMasterSlaveConstraints(rModelPart);
TSystemMatrixType L_transpose_matrix(mL.size2(), mL.size1());
SparseMatrixMultiplicationUtility::TransposeMatrix<TSystemMatrixType, TSystemMatrixType>(L_transpose_matrix, mL, 1.0);

TSystemVectorType b_modified(rb.size());
TSparseSpace::Mult(L_transpose_matrix, rb, b_modified);
TSparseSpace::Copy(b_modified, rb);
b_modified.resize(0, false); 

TSystemMatrixType auxiliar_A_matrix(BaseType::mT.size2(), rA.size2());
SparseMatrixMultiplicationUtility::MatrixMultiplication(L_transpose_matrix, rA, auxiliar_A_matrix); 
L_transpose_matrix.resize(0, 0, false);                                                             

SparseMatrixMultiplicationUtility::MatrixMultiplication(auxiliar_A_matrix, BaseType::mT, rA); 
auxiliar_A_matrix.resize(0, 0, false);                                                        

double max_diag = 0.0;
for (IndexType i = 0; i < rA.size1(); ++i)
{
max_diag = std::max(std::abs(rA(i, i)), max_diag);
}
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(BaseType::mSlaveIds.size()); ++i)
{
const IndexType slave_equation_id = BaseType::mSlaveIds[i];
if (BaseType::mInactiveSlaveDofs.find(slave_equation_id) == BaseType::mInactiveSlaveDofs.end())
{
rA(slave_equation_id, slave_equation_id) = max_diag;
rb[slave_equation_id] = 0.0;
}
}
const double stop_constraints = OpenMPUtils::GetCurrentTime();
KRATOS_INFO_IF("ResidualBasedBlockBuilderAndSolverWithConstraintsForChimera", this->GetEchoLevel() >= 1 )<< "Applying constraints time: " << stop_constraints - start_constraints << std::endl;
}

KRATOS_CATCH("")
}





}; 




} 

#endif 
