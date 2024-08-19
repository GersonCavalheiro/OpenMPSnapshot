
#pragma once



#include "includes/define.h"
#include "trilinos_space.h"
#include "factories/linear_solver_factory.h"

namespace Kratos
{






template <typename TSparseSpace, typename TLocalSpace, typename TLinearSolverType>
class KRATOS_API(TRILINOS_APPLICATION) TrilinosLinearSolverFactory
: public LinearSolverFactory<TSparseSpace,TLocalSpace>
{

typedef LinearSolver<TSparseSpace,TLocalSpace> LinearSolverType;

protected:


typename LinearSolverType::Pointer CreateSolver(Kratos::Parameters settings) const override
{
return typename LinearSolverType::Pointer(new TLinearSolverType(settings));
}
};




template <typename TSparseSpace, typename TLocalSpace, typename TLinearSolverType>
inline std::ostream& operator << (std::ostream& rOStream,
const TrilinosLinearSolverFactory<TSparseSpace,TLocalSpace,TLinearSolverType>& rThis)
{
rOStream << "TrilinosLinearSolverFactory" << std::endl;
return rOStream;
}



void RegisterTrilinosLinearSolvers();

typedef TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector> TrilinosSparseSpaceType;
typedef UblasSpace<double, Matrix, Vector> TrilinosLocalSpaceType;

typedef LinearSolverFactory<TrilinosSparseSpaceType,  TrilinosLocalSpaceType> TrilinosLinearSolverFactoryType;

#ifdef KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER
#undef KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER
#endif
#define KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER(name, reference) ; \
KratosComponents<TrilinosLinearSolverFactoryType>::Add(name, reference);

KRATOS_API_EXTERN template class KRATOS_API(TRILINOS_APPLICATION) KratosComponents<TrilinosLinearSolverFactoryType>;

}  
