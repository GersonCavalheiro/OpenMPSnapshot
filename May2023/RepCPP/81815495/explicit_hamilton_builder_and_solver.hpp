
#if !defined(KRATOS_EXPLICIT_HAMILTON_BUILDER_AND_SOLVER )
#define  KRATOS_EXPLICIT_HAMILTON_BUILDER_AND_SOLVER



#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif


#include "utilities/timer.h"


#include "includes/define.h"
#include "includes/kratos_flags.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "includes/model_part.h"
#include "utilities/beam_math_utilities.hpp"

#include "solid_mechanics_application_variables.h"

namespace Kratos
{
















template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ExplicitHamiltonBuilderAndSolver : public BuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:



KRATOS_CLASS_POINTER_DEFINITION( ExplicitHamiltonBuilderAndSolver );

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

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

typedef BeamMathUtils<double> BeamMathUtilsType;

typedef Quaternion<double> QuaternionType;

typedef GlobalPointersVector<Element>     ElementWeakPtrVectorType;






ExplicitHamiltonBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >(pNewLinearSystemSolver)
{
}


virtual ~ExplicitHamiltonBuilderAndSolver()
{
}







/
void Clear()
{
this->mDofSet = DofsArrayType();

if (this->mpReactionsVector != NULL)
TSparseSpace::Clear((this->mpReactionsVector));

this->mpLinearSystemSolver->Clear();

if (this->GetEchoLevel() > 1)
{
std::cout << "ExplicitHamiltonBuilderAndSolver Clear Function called" << std::endl;
}
}


virtual int Check(ModelPart& r_model_part)
{
KRATOS_TRY

return 0;

KRATOS_CATCH( "" )
}
















protected:









/













private:






















}; 







} 

#endif 
