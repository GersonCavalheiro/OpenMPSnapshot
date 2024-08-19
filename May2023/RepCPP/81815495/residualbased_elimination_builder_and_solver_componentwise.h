#if !defined(KRATOS_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVERCOMPONENTWISE )
#define  KRATOS_RESIDUAL_BASED_ELIMINATION_BUILDER_AND_SOLVERCOMPONENTWISE


#include <set>


#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#endif


#include "includes/define.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver.h"
#include "includes/global_pointer_variables.h"
#include "utilities/builtin_timer.h"

namespace Kratos
{



























template<class TSparseSpace,
class TDenseSpace ,
class TLinearSolver,
class TVariableType
>
class ResidualBasedEliminationBuilderAndSolverComponentwise
: public ResidualBasedEliminationBuilderAndSolver< TSparseSpace,TDenseSpace,TLinearSolver >
{
public:


KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedEliminationBuilderAndSolverComponentwise );


typedef BuilderAndSolver<TSparseSpace,TDenseSpace, TLinearSolver> BaseType;
typedef ResidualBasedEliminationBuilderAndSolver<TSparseSpace,TDenseSpace, TLinearSolver> ResidualBasedEliminationBuilderAndSolverType;

typedef typename BaseType::TSchemeType TSchemeType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;


typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

typedef typename BaseType::ElementsContainerType ElementsContainerType;



explicit ResidualBasedEliminationBuilderAndSolverComponentwise(
typename TLinearSolver::Pointer pNewLinearSystemSolver,
Parameters ThisParameters
) : ResidualBasedEliminationBuilderAndSolverType(pNewLinearSystemSolver)
{
Parameters default_parameters = Parameters(R"(
{
"name"                     : "ResidualBasedEliminationBuilderAndSolverComponentwise",
"components_wise_variable" : "SCALAR_VARIABLE_OR_COMPONENT"
})" );

ThisParameters.ValidateAndAssignDefaults(default_parameters);

rVar = KratosComponents<TVariableType>::Get(ThisParameters["components_wise_variable"].GetString());
}


explicit ResidualBasedEliminationBuilderAndSolverComponentwise(
typename TLinearSolver::Pointer pNewLinearSystemSolver,TVariableType const& Var)
: ResidualBasedEliminationBuilderAndSolverType(pNewLinearSystemSolver)
, rVar(Var)
{



}



~ResidualBasedEliminationBuilderAndSolverComponentwise() override {}








/














std::string Info() const override
{
return "ResidualBasedEliminationBuilderAndSolverComponentwise";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info();
}








protected:












/






















private:







TVariableType const & rVar;
GlobalPointersVector<Node > mActiveNodes;




/

























}; 









}  

#endif 
