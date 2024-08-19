
#if !defined(KRATOS_TRILINOS_RESIDUAL_CRITERIA_H_INCLUDED)
#define  KRATOS_TRILINOS_RESIDUAL_CRITERIA_H_INCLUDED



#include "includes/define.h"
#include "solving_strategies/convergencecriterias/residual_criteria.h"

namespace Kratos
{



template< class TSparseSpace, class TDenseSpace >
class TrilinosResidualCriteria : public ResidualCriteria< TSparseSpace, TDenseSpace >
{
public:


KRATOS_CLASS_POINTER_DEFINITION(TrilinosResidualCriteria);

typedef ResidualCriteria< TSparseSpace, TDenseSpace > BaseType;

typedef typename BaseType::TDataType TDataType;


explicit TrilinosResidualCriteria(TDataType NewRatioTolerance,TDataType AlwaysConvergedNorm):
ResidualCriteria<TSparseSpace,TDenseSpace>(NewRatioTolerance, AlwaysConvergedNorm)
{}

explicit TrilinosResidualCriteria(const TrilinosResidualCriteria& rOther):
ResidualCriteria<TSparseSpace,TDenseSpace>(rOther)
{}

~TrilinosResidualCriteria() override {}


TrilinosResidualCriteria& operator=(TrilinosResidualCriteria const& rOther) = delete;


protected:



void CalculateResidualNorm(
ModelPart& rModelPart,
TDataType& rResidualSolutionNorm,
typename BaseType::SizeType& rDofNum,
typename BaseType::DofsArrayType& rDofSet,
const typename BaseType::TSystemVectorType& rB) override
{
TDataType residual_solution_norm = TDataType();
long int local_dof_num = 0;

const int rank = rB.Comm().MyPID();

#pragma omp parallel for reduction(+:residual_solution_norm,local_dof_num)
for (int i = 0; i < static_cast<int>(rDofSet.size()); i++) {
auto it_dof = rDofSet.begin() + i;

typename BaseType::IndexType dof_id;
TDataType residual_dof_value;

if (it_dof->IsFree() && (it_dof->GetSolutionStepValue(PARTITION_INDEX) == rank)) {
dof_id = it_dof->EquationId();
residual_dof_value = TSparseSpace::GetValue(rB,dof_id);
residual_solution_norm += residual_dof_value * residual_dof_value;
local_dof_num++;
}
}

rB.Comm().SumAll(&residual_solution_norm,&rResidualSolutionNorm,1);
long int global_dof_num = 0;
rB.Comm().SumAll(&local_dof_num,&global_dof_num,1);
rDofNum = static_cast<typename BaseType::SizeType>(global_dof_num);

rResidualSolutionNorm = std::sqrt(rResidualSolutionNorm);
}


private:





}; 



}  

#endif 
