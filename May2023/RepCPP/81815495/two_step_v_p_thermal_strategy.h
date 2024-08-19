
#ifndef KRATOS_TWO_STEP_V_P_THERMAL_STRATEGY_H
#define KRATOS_TWO_STEP_V_P_THERMAL_STRATEGY_H

#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "includes/cfd_variables.h"
#include "utilities/openmp_utils.h"
#include "processes/process.h"
#include "solving_strategies/schemes/scheme.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "custom_utilities/mesher_utilities.hpp"
#include "custom_utilities/boundary_normals_calculation_utilities.hpp"

#include "solving_strategies/schemes/residualbased_incrementalupdate_static_scheme.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver_componentwise.h"
#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"

#include "custom_utilities/solver_settings.h"

#include "custom_strategies/strategies/gauss_seidel_linear_strategy.h"

#include "pfem_fluid_dynamics_application_variables.h"

#include "processes/integration_values_extrapolation_to_nodes_process.h"

#include "two_step_v_p_thermal_strategy.h"

#include <stdio.h>
#include <cmath>

namespace Kratos
{








template <class TSparseSpace,
class TDenseSpace,
class TLinearSolver>
class TwoStepVPThermalStrategy : public TwoStepVPStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
KRATOS_CLASS_POINTER_DEFINITION(TwoStepVPThermalStrategy);

typedef TwoStepVPStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef typename ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::Pointer StrategyPointerType;

typedef TwoStepVPSolverSettings<TSparseSpace, TDenseSpace, TLinearSolver> SolverSettingsType;

using TwoStepVPStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::mVelocityTolerance;
using TwoStepVPStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::mPressureTolerance;
using TwoStepVPStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::mMaxPressureIter;
using TwoStepVPStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::mDomainSize;
using TwoStepVPStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::mReformDofSet;

TwoStepVPThermalStrategy(ModelPart &rModelPart,
typename TLinearSolver::Pointer pVelocityLinearSolver,
typename TLinearSolver::Pointer pPressureLinearSolver,
bool ReformDofSet = true,
double VelTol = 0.0001,
double PresTol = 0.0001,
int MaxPressureIterations = 1, 
unsigned int TimeOrder = 2,
unsigned int DomainSize = 2) : BaseType(rModelPart,
pVelocityLinearSolver,
pPressureLinearSolver,
ReformDofSet,
VelTol,
PresTol,
MaxPressureIterations,
TimeOrder,
DomainSize)
{
KRATOS_TRY;

BaseType::SetEchoLevel(1);

this->Check();

bool CalculateNormDxFlag = true;

bool ReformDofAtEachIteration = false; 

typedef typename BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>::Pointer BuilderSolverTypePointer;
typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef Scheme<TSparseSpace, TDenseSpace> SchemeType;
typename SchemeType::Pointer pScheme;

typename SchemeType::Pointer Temp = typename SchemeType::Pointer(new ResidualBasedIncrementalUpdateStaticScheme<TSparseSpace, TDenseSpace>());
pScheme.swap(Temp);

BuilderSolverTypePointer vel_build = BuilderSolverTypePointer(new ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pVelocityLinearSolver));

this->mpMomentumStrategy = typename BaseType::Pointer(new GaussSeidelLinearStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart, pScheme, pVelocityLinearSolver, vel_build, ReformDofAtEachIteration, CalculateNormDxFlag));

vel_build->SetCalculateReactionsFlag(false);

BuilderSolverTypePointer pressure_build = BuilderSolverTypePointer(new ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pPressureLinearSolver));

this->mpPressureStrategy = typename BaseType::Pointer(new GaussSeidelLinearStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart, pScheme, pPressureLinearSolver, pressure_build, ReformDofAtEachIteration, CalculateNormDxFlag));

pressure_build->SetCalculateReactionsFlag(false);

KRATOS_CATCH("");
}

virtual ~TwoStepVPThermalStrategy() {}



std::string Info() const override
{
std::stringstream buffer;
buffer << "TwoStepVPThermalStrategy";
return buffer.str();
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "TwoStepVPThermalStrategy";
}

void PrintData(std::ostream &rOStream) const override
{
}



protected:







bool SolveSolutionStep() override
{
ModelPart &rModelPart = BaseType::GetModelPart();
ProcessInfo &rCurrentProcessInfo = rModelPart.GetProcessInfo();
double currentTime = rCurrentProcessInfo[TIME];
double timeInterval = rCurrentProcessInfo[DELTA_TIME];
bool timeIntervalChanged = rCurrentProcessInfo[TIME_INTERVAL_CHANGED];
unsigned int stepsWithChangedDt = rCurrentProcessInfo[STEPS_WITH_CHANGED_DT];
bool converged = false;

unsigned int maxNonLinearIterations = mMaxPressureIter;

KRATOS_INFO("\nSolution with two_step_vp_thermal_strategy at t=") << currentTime << "s" << std::endl;

if ((timeIntervalChanged == true && currentTime > 10 * timeInterval) || stepsWithChangedDt > 0)
{
maxNonLinearIterations *= 2;
}
if (currentTime < 10 * timeInterval)
{
if (BaseType::GetEchoLevel() > 1)
std::cout << "within the first 10 time steps, I consider the given iteration number x3" << std::endl;
maxNonLinearIterations *= 3;
}
if (currentTime < 20 * timeInterval && currentTime >= 10 * timeInterval)
{
if (BaseType::GetEchoLevel() > 1)
std::cout << "within the second 10 time steps, I consider the given iteration number x2" << std::endl;
maxNonLinearIterations *= 2;
}
bool momentumConverged = true;
bool continuityConverged = false;
bool fixedTimeStep = false;

double pressureNorm = 0;
double velocityNorm = 0;

this->SetBlockedAndIsolatedFlags();

for (unsigned int it = 0; it < maxNonLinearIterations; ++it)
{
momentumConverged = this->SolveMomentumIteration(it, maxNonLinearIterations, fixedTimeStep, velocityNorm);

this->UpdateTopology(rModelPart, BaseType::GetEchoLevel());

if (fixedTimeStep == false)
{
continuityConverged = this->SolveContinuityIteration(it, maxNonLinearIterations, pressureNorm);
}
if (it == maxNonLinearIterations - 1 || ((continuityConverged && momentumConverged) && it > 2))
{
this->UpdateThermalStressStrain();
}

if ((continuityConverged && momentumConverged) && it > 2)
{
rCurrentProcessInfo.SetValue(BAD_VELOCITY_CONVERGENCE, false);
rCurrentProcessInfo.SetValue(BAD_PRESSURE_CONVERGENCE, false);
converged = true;

KRATOS_INFO("TwoStepVPThermalStrategy") << "V-P thermal strategy converged in " << it + 1 << " iterations." << std::endl;

break;
}
if (fixedTimeStep == true)
{
break;
}
}

if (!continuityConverged && !momentumConverged && BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
std::cout << "Convergence tolerance not reached." << std::endl;

if (mReformDofSet)
this->Clear();

return converged;
}

void UpdateThermalStressStrain()
{
ModelPart &rModelPart = BaseType::GetModelPart();
const ProcessInfo &rCurrentProcessInfo = rModelPart.GetProcessInfo();

#pragma omp parallel
{
ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Elements(), ElemBegin, ElemEnd);
for (ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
itElem->InitializeSolutionStep(rCurrentProcessInfo);
}
}

this->CalculateTemporalVariables();

Parameters extrapolation_parameters(R"(
{
"list_of_variables": ["MECHANICAL_DISSIPATION"]
})");
auto extrapolation_process = IntegrationValuesExtrapolationToNodesProcess(rModelPart, extrapolation_parameters);
extrapolation_process.Execute();

const auto &this_var = KratosComponents<Variable<double>>::Get("MECHANICAL_DISSIPATION");
for (auto &node : rModelPart.Nodes())
{
node.FastGetSolutionStepValue(HEAT_FLUX) = node.GetValue(this_var);
}

}

void SetEchoLevel(int Level) override
{
BaseType::SetEchoLevel(Level);
int StrategyLevel = Level > 0 ? Level - 1 : 0;
mpMomentumStrategy->SetEchoLevel(StrategyLevel);
mpPressureStrategy->SetEchoLevel(StrategyLevel);
}









StrategyPointerType mpMomentumStrategy;

StrategyPointerType mpPressureStrategy;






TwoStepVPThermalStrategy &operator=(TwoStepVPThermalStrategy const &rOther) {}

TwoStepVPThermalStrategy(TwoStepVPThermalStrategy const &rOther) {}


}; 




} 

#endif 
