

#if !defined(KRATOS_SPALART_ALLMARAS_TURBULENCE_H_INCLUDED )
#define  KRATOS_SPALART_ALLMARAS_TURBULENCE_H_INCLUDED



#include <string>
#include <iostream>




#include "includes/define.h"
#include "containers/model.h"
#include "processes/process.h"
#include "includes/cfd_variables.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "solving_strategies/strategies/residualbased_newton_raphson_strategy.h"
#include "solving_strategies/schemes/residualbased_incremental_aitken_static_scheme.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver_componentwise.h"
#include "solving_strategies/convergencecriterias/residual_criteria.h"

#include "custom_utilities/periodic_condition_utilities.h"
#include "fluid_dynamics_application_variables.h"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
class SpalartAllmarasTurbulenceModel : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SpalartAllmarasTurbulenceModel);



SpalartAllmarasTurbulenceModel(
ModelPart& rModelPart,
typename TLinearSolver::Pointer pLinearSolver,
unsigned int DomainSize,
double NonLinearTol,
unsigned int MaxIter,
bool ReformDofSet,
unsigned int TimeOrder)
: mr_model_part(rModelPart),
mrSpalartModelPart(rModelPart.GetModel().CreateModelPart("SpalartModelPart")),
mdomain_size(DomainSize),
mtol(NonLinearTol),
mmax_it(MaxIter),
mtime_order(TimeOrder),
madapt_for_fractional_step(false)
{
/

KRATOS_CATCH("");
}








std::string Info() const override
{
std::stringstream buffer;
buffer << "SpalartAllmarasTurbulenceModel";
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SpalartAllmarasTurbulenceModel";
}


void PrintData(std::ostream& rOStream) const override
{
}





protected:



ModelPart& mr_model_part;
ModelPart& mrSpalartModelPart;
unsigned int mdomain_size;
double mtol;
unsigned int mmax_it;
unsigned int mtime_order;
bool madapt_for_fractional_step;
typename ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::Pointer mpSolutionStrategy;










SpalartAllmarasTurbulenceModel(ModelPart& rModelPart)
:
Process(),
mr_model_part(rModelPart),
mrSpalartModelPart(rModelPart.GetModel().CreateModelPart("SpalartModelPart"))
{}


private:






/
void AuxSolve()
{
KRATOS_TRY

ProcessInfo& rCurrentProcessInfo = mrSpalartModelPart.GetProcessInfo();
double Dt = rCurrentProcessInfo[DELTA_TIME];

if (mtime_order == 2)
{
double dt_old = rCurrentProcessInfo.GetPreviousTimeStepInfo(1)[DELTA_TIME];

double rho = dt_old / Dt;
double coeff = 1.0 / (Dt * rho * rho + Dt * rho);

Vector& BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(3);
BDFcoeffs[0] = coeff * (rho * rho + 2.0 * rho); 
BDFcoeffs[1] = -coeff * (rho * rho + 2.0 * rho + 1.0); 
BDFcoeffs[2] = coeff;
}
else
{
Vector& BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(2);
BDFcoeffs[0] = 1.0 / Dt; 
BDFcoeffs[1] = -1.0 / Dt; 
}




int current_fract_step = rCurrentProcessInfo[FRACTIONAL_STEP];
rCurrentProcessInfo[FRACTIONAL_STEP] = 2;

CalculateProjection();

rCurrentProcessInfo[FRACTIONAL_STEP] = 1;
mpSolutionStrategy->Solve();

rCurrentProcessInfo[FRACTIONAL_STEP] = current_fract_step;








KRATOS_CATCH("")
}



double CalculateVarNorm()
{
KRATOS_TRY;

double norm = 0.00;



for (ModelPart::NodeIterator i = mrSpalartModelPart.NodesBegin();
i != mrSpalartModelPart.NodesEnd(); ++i)
{
norm += pow(i->FastGetSolutionStepValue(TURBULENT_VISCOSITY), 2);
}

return sqrt(norm);

KRATOS_CATCH("")
}


void CalculateProjection()
{
KRATOS_TRY;

const ProcessInfo& rCurrentProcessInfo = mrSpalartModelPart.GetProcessInfo();

for (ModelPart::NodeIterator i = mrSpalartModelPart.NodesBegin();
i != mrSpalartModelPart.NodesEnd(); ++i)
{
(i)->FastGetSolutionStepValue(TEMP_CONV_PROJ) = 0.00;
(i)->FastGetSolutionStepValue(NODAL_AREA) = 0.00;
}



for (ModelPart::ElementIterator i = mrSpalartModelPart.ElementsBegin();
i != mrSpalartModelPart.ElementsEnd(); ++i)
{
(i)->InitializeSolutionStep(rCurrentProcessInfo);
}

Communicator& rComm = mrSpalartModelPart.GetCommunicator();

rComm.AssembleCurrentData(NODAL_AREA);
rComm.AssembleCurrentData(TEMP_CONV_PROJ);

for (ModelPart::NodeIterator i = mrSpalartModelPart.NodesBegin();
i != mrSpalartModelPart.NodesEnd(); ++i)
{
const double NodalArea = i->FastGetSolutionStepValue(NODAL_AREA);
if(NodalArea > 0.0)
{
double& rConvProj = i->FastGetSolutionStepValue(TEMP_CONV_PROJ);
rConvProj /= NodalArea;
}
}

KRATOS_CATCH("")
}








SpalartAllmarasTurbulenceModel & operator=(SpalartAllmarasTurbulenceModel const& rOther)
{
return *this;
}


SpalartAllmarasTurbulenceModel(SpalartAllmarasTurbulenceModel const& rOther)
: mr_model_part(rOther.mr_model_part), mdomain_size(rOther.mdomain_size)
{
}



}; 







template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
inline std::istream & operator >>(std::istream& rIStream,
SpalartAllmarasTurbulenceModel<TSparseSpace, TDenseSpace, TLinearSolver>& rThis)
{
return rIStream;
}


template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
inline std::ostream & operator <<(std::ostream& rOStream,
const SpalartAllmarasTurbulenceModel<TSparseSpace, TDenseSpace, TLinearSolver>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


} 

#endif 


