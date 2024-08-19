
#ifndef KRATOS_THREE_STEP_V_P_STRATEGY_H
#define KRATOS_THREE_STEP_V_P_STRATEGY_H

#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "includes/cfd_variables.h"
#include "utilities/openmp_utils.h"
#include "processes/process.h"
#include "solving_strategies/schemes/scheme.h"
#include "solving_strategies/strategies/solving_strategy.h"
#include "custom_utilities/mesher_utilities.hpp"
#include "custom_utilities/boundary_normals_calculation_utilities.hpp"

#include "solving_strategies/schemes/residualbased_incrementalupdate_static_scheme.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver_componentwise.h"
#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"

#include "custom_utilities/solver_settings.h"

#include "custom_strategies/strategies/gauss_seidel_linear_strategy.h"

#include "pfem_fluid_dynamics_application_variables.h"

#include "v_p_strategy.h"

#include <stdio.h>
#include <cmath>

namespace Kratos
{








template <class TSparseSpace,
class TDenseSpace,
class TLinearSolver>
class ThreeStepVPStrategy : public VPStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
KRATOS_CLASS_POINTER_DEFINITION(ThreeStepVPStrategy);

typedef VPStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef typename SolvingStrategy<TSparseSpace, TDenseSpace>::Pointer StrategyPointerType;

typedef TwoStepVPSolverSettings<TSparseSpace, TDenseSpace, TLinearSolver> SolverSettingsType;


ThreeStepVPStrategy(ModelPart &rModelPart,
typename TLinearSolver::Pointer pVelocityLinearSolver,
typename TLinearSolver::Pointer pPressureLinearSolver,
bool ReformDofSet = true,
double VelTol = 0.0001,
double PresTol = 0.0001,
int MaxPressureIterations = 1, 
unsigned int TimeOrder = 2,
unsigned int DomainSize = 2) : BaseType(rModelPart, pVelocityLinearSolver, pPressureLinearSolver, ReformDofSet, DomainSize),
mVelocityTolerance(VelTol),
mPressureTolerance(PresTol),
mMaxPressureIter(MaxPressureIterations),
mDomainSize(DomainSize),
mTimeOrder(TimeOrder),
mReformDofSet(ReformDofSet)
{
KRATOS_TRY;

BaseType::SetEchoLevel(1);

this->Check();

bool CalculateNormDxFlag = true;

bool ReformDofAtEachIteration = false; 

typedef typename BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>::Pointer BuilderSolverTypePointer;
typedef SolvingStrategy<TSparseSpace, TDenseSpace> BaseType;

typedef Scheme<TSparseSpace, TDenseSpace> SchemeType;
typename SchemeType::Pointer pScheme;

typename SchemeType::Pointer Temp = typename SchemeType::Pointer(new ResidualBasedIncrementalUpdateStaticScheme<TSparseSpace, TDenseSpace>());
pScheme.swap(Temp);

BuilderSolverTypePointer vel_build = BuilderSolverTypePointer(new ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver > (pVelocityLinearSolver));

this->mpMomentumStrategy = typename BaseType::Pointer(new GaussSeidelLinearStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart, pScheme, pVelocityLinearSolver, vel_build, ReformDofAtEachIteration, CalculateNormDxFlag));

this->mpMomentumStrategy->SetEchoLevel(BaseType::GetEchoLevel());

vel_build->SetCalculateReactionsFlag(false);


BuilderSolverTypePointer pressure_build = BuilderSolverTypePointer(new ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>(pPressureLinearSolver));

this->mpPressureStrategy = typename BaseType::Pointer(new GaussSeidelLinearStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart, pScheme, pPressureLinearSolver, pressure_build, ReformDofAtEachIteration, CalculateNormDxFlag));

this->mpPressureStrategy->SetEchoLevel(BaseType::GetEchoLevel());

pressure_build->SetCalculateReactionsFlag(false);

KRATOS_CATCH("");
}

virtual ~ThreeStepVPStrategy() {}

int Check() override
{
KRATOS_TRY;

int ierr = BaseType::Check();
if (ierr != 0)
return ierr;

if (DELTA_TIME.Key() == 0)
KRATOS_THROW_ERROR(std::runtime_error, "DELTA_TIME Key is 0. Check that the application was correctly registered.", "");

ModelPart &rModelPart = BaseType::GetModelPart();

const auto &r_current_process_info = rModelPart.GetProcessInfo();
for (const auto &r_element : rModelPart.Elements())
{
ierr = r_element.Check(r_current_process_info);
if (ierr != 0)
{
break;
}
}

return ierr;

KRATOS_CATCH("");
}

void SetTimeCoefficients(ProcessInfo &rCurrentProcessInfo)
{
KRATOS_TRY;

if (mTimeOrder == 2)
{
double Dt = rCurrentProcessInfo[DELTA_TIME];
double OldDt = rCurrentProcessInfo.GetPreviousTimeStepInfo(1)[DELTA_TIME];

double Rho = OldDt / Dt;
double TimeCoeff = 1.0 / (Dt * Rho * Rho + Dt * Rho);

Vector &BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(3, false);

BDFcoeffs[0] = TimeCoeff * (Rho * Rho + 2.0 * Rho);        
BDFcoeffs[1] = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0); 
BDFcoeffs[2] = TimeCoeff;                                  
}
else if (mTimeOrder == 1)
{
double Dt = rCurrentProcessInfo[DELTA_TIME];
double TimeCoeff = 1.0 / Dt;

Vector &BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(2, false);

BDFcoeffs[0] = TimeCoeff;  
BDFcoeffs[1] = -TimeCoeff; 
}

KRATOS_CATCH("");
}

bool SolveSolutionStep() override
{
ModelPart &rModelPart = BaseType::GetModelPart();
ProcessInfo &rCurrentProcessInfo = rModelPart.GetProcessInfo();
double currentTime = rCurrentProcessInfo[TIME];
bool converged = false;
double NormV = 0;

unsigned int maxNonLinearIterations = mMaxPressureIter;

KRATOS_INFO("\nSolution with three_step_vp_strategy at t=") << currentTime << "s" << std::endl;

bool momentumConverged = true;
bool continuityConverged = false;

this->SetBlockedAndIsolatedFlags();


for (unsigned int it = 0; it < maxNonLinearIterations; ++it)
{
KRATOS_INFO("\n ------------------- ITERATION ") << it << " ------------------- " << std::endl;

rModelPart.GetProcessInfo().SetValue(FRACTIONAL_STEP, 1);

if (it == 0)
{
mpMomentumStrategy->InitializeSolutionStep();
}
else
{
this->RecoverFractionalVelocity();
}

momentumConverged = this->SolveFirstVelocitySystem(NormV);

rModelPart.GetProcessInfo().SetValue(FRACTIONAL_STEP, 5);

if (it == 0)
{
mpPressureStrategy->InitializeSolutionStep();
}
continuityConverged = this->SolveContinuityIteration();

this->CalculateEndOfStepVelocity(NormV);

this->UpdateTopology(rModelPart, BaseType::GetEchoLevel());

if (continuityConverged && momentumConverged)
{
rCurrentProcessInfo.SetValue(BAD_VELOCITY_CONVERGENCE, false);
rCurrentProcessInfo.SetValue(BAD_PRESSURE_CONVERGENCE, false);
converged = true;
this->UpdateStressStrain();
KRATOS_INFO("ThreeStepVPStrategy") << "V-P strategy converged in " << it + 1 << " iterations." << std::endl;
break;
}
else if (it == (maxNonLinearIterations - 1) && it != 0)
{
this->UpdateStressStrain();
}
}

if (!continuityConverged && !momentumConverged && BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
std::cout << "Convergence tolerance not reached." << std::endl;

if (mReformDofSet)
this->Clear();

return converged;
}

void FinalizeSolutionStep() override
{
}

void InitializeSolutionStep() override
{
}

void UpdateStressStrain() override
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
}

void CalculateTemporalVariables() override
{
ModelPart &rModelPart = BaseType::GetModelPart();
ProcessInfo &rCurrentProcessInfo = rModelPart.GetProcessInfo();

for (ModelPart::NodeIterator i = rModelPart.NodesBegin();
i != rModelPart.NodesEnd(); ++i)
{

array_1d<double, 3> &CurrentVelocity = (i)->FastGetSolutionStepValue(VELOCITY, 0);
array_1d<double, 3> &PreviousVelocity = (i)->FastGetSolutionStepValue(VELOCITY, 1);

array_1d<double, 3> &CurrentAcceleration = (i)->FastGetSolutionStepValue(ACCELERATION, 0);
array_1d<double, 3> &PreviousAcceleration = (i)->FastGetSolutionStepValue(ACCELERATION, 1);


if ((i)->IsNot(ISOLATED) && ((i)->IsNot(RIGID) || (i)->Is(SOLID)))
{
UpdateAccelerations(CurrentAcceleration, CurrentVelocity, PreviousAcceleration, PreviousVelocity);
}
else if ((i)->Is(RIGID))
{
array_1d<double, 3> Zeros(3, 0.0);
(i)->FastGetSolutionStepValue(ACCELERATION, 0) = Zeros;
(i)->FastGetSolutionStepValue(ACCELERATION, 1) = Zeros;
}
else
{
(i)->FastGetSolutionStepValue(PRESSURE, 0) = 0.0;
(i)->FastGetSolutionStepValue(PRESSURE, 1) = 0.0;
(i)->FastGetSolutionStepValue(PRESSURE_VELOCITY, 0) = 0.0;
(i)->FastGetSolutionStepValue(PRESSURE_VELOCITY, 1) = 0.0;
(i)->FastGetSolutionStepValue(PRESSURE_ACCELERATION, 0) = 0.0;
(i)->FastGetSolutionStepValue(PRESSURE_ACCELERATION, 1) = 0.0;
if ((i)->SolutionStepsDataHas(VOLUME_ACCELERATION))
{
array_1d<double, 3> &VolumeAcceleration = (i)->FastGetSolutionStepValue(VOLUME_ACCELERATION);
(i)->FastGetSolutionStepValue(ACCELERATION, 0) = VolumeAcceleration;
(i)->FastGetSolutionStepValue(VELOCITY, 0) += VolumeAcceleration * rCurrentProcessInfo[DELTA_TIME];
}
}

const double timeInterval = rCurrentProcessInfo[DELTA_TIME];
unsigned int timeStep = rCurrentProcessInfo[STEP];
if (timeStep == 1)
{
(i)->FastGetSolutionStepValue(PRESSURE_VELOCITY, 0) = 0;
(i)->FastGetSolutionStepValue(PRESSURE_VELOCITY, 1) = 0;
(i)->FastGetSolutionStepValue(PRESSURE_ACCELERATION, 0) = 0;
(i)->FastGetSolutionStepValue(PRESSURE_ACCELERATION, 1) = 0;
}
else
{
double &CurrentPressure = (i)->FastGetSolutionStepValue(PRESSURE, 0);
double &PreviousPressure = (i)->FastGetSolutionStepValue(PRESSURE, 1);
double &CurrentPressureVelocity = (i)->FastGetSolutionStepValue(PRESSURE_VELOCITY, 0);
double &CurrentPressureAcceleration = (i)->FastGetSolutionStepValue(PRESSURE_ACCELERATION, 0);

CurrentPressureAcceleration = CurrentPressureVelocity / timeInterval;

CurrentPressureVelocity = (CurrentPressure - PreviousPressure) / timeInterval;

CurrentPressureAcceleration += -CurrentPressureVelocity / timeInterval;
}
}
}

inline void UpdateAccelerations(array_1d<double, 3> &CurrentAcceleration,
const array_1d<double, 3> &CurrentVelocity,
array_1d<double, 3> &PreviousAcceleration,
const array_1d<double, 3> &PreviousVelocity)
{
ModelPart &rModelPart = BaseType::GetModelPart();
ProcessInfo &rCurrentProcessInfo = rModelPart.GetProcessInfo();
double Dt = rCurrentProcessInfo[DELTA_TIME];
noalias(CurrentAcceleration) = (CurrentVelocity - PreviousVelocity) / Dt; 
}

void Clear() override
{
mpMomentumStrategy->Clear();
mpPressureStrategy->Clear();
}


void SetEchoLevel(int Level) override
{
BaseType::SetEchoLevel(Level);
int StrategyLevel = Level > 0 ? Level - 1 : 0;
mpMomentumStrategy->SetEchoLevel(StrategyLevel);
mpPressureStrategy->SetEchoLevel(StrategyLevel);
}



std::string Info() const override
{
std::stringstream buffer;
buffer << "ThreeStepVPStrategy";
return buffer.str();
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "ThreeStepVPStrategy";
}

void PrintData(std::ostream &rOStream) const override
{
}



protected:







bool SolveFirstVelocitySystem(double &NormV)
{
std::cout << "1. SolveFirstVelocitySystem " << std::endl;

bool momentumConvergence = false;
double NormDv = 0;

NormDv = mpMomentumStrategy->Solve();

momentumConvergence = this->CheckVelocityIncrementConvergence(NormDv, NormV);

if (!momentumConvergence && BaseType::GetEchoLevel() > 0)
std::cout << "Momentum equations did not reach the convergence tolerance." << std::endl;

return momentumConvergence;
}

bool SolveContinuityIteration()
{
std::cout << "2. SolveContinuityIteration " << std::endl;
bool continuityConvergence = false;
double NormDp = 0;

NormDp = mpPressureStrategy->Solve();

continuityConvergence = CheckPressureIncrementConvergence(NormDp);

if (!continuityConvergence && BaseType::GetEchoLevel() > 0)
std::cout << "Continuity equation did not reach the convergence tolerance." << std::endl;

return continuityConvergence;
}

void CalculateEndOfStepVelocity(const double NormV)
{

std::cout << "3. CalculateEndOfStepVelocity()" << std::endl;
ModelPart &rModelPart = BaseType::GetModelPart();
const int n_nodes = rModelPart.NumberOfNodes();
const int n_elems = rModelPart.NumberOfElements();

array_1d<double, 3> Out = ZeroVector(3);
VariableUtils().SetHistoricalVariableToZero(FRACT_VEL, rModelPart.Nodes());
VariableUtils().SetHistoricalVariableToZero(NODAL_VOLUME, rModelPart.Nodes());

#pragma omp parallel for
for (int i_elem = 0; i_elem < n_elems; ++i_elem)
{
const auto it_elem = rModelPart.ElementsBegin() + i_elem;
it_elem->Calculate(VELOCITY, Out, rModelPart.GetProcessInfo());
Element::GeometryType &geometry = it_elem->GetGeometry();
double elementalVolume = 0;

if (mDomainSize == 2)
{
elementalVolume = geometry.Area() / 3.0;
}
else if (mDomainSize == 3)
{
elementalVolume = geometry.Volume() * 0.25;
}
unsigned int numNodes = geometry.size();
for (unsigned int i = 0; i < numNodes; i++)
{
double &nodalVolume = geometry(i)->FastGetSolutionStepValue(NODAL_VOLUME);
nodalVolume += elementalVolume;
}
}

rModelPart.GetCommunicator().AssembleCurrentData(FRACT_VEL);

if (mDomainSize > 2)
{
#pragma omp parallel for
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
auto it_node = rModelPart.NodesBegin() + i_node;
const double NodalVolume = it_node->FastGetSolutionStepValue(NODAL_VOLUME);
double fractionalVelocity = 0;
if (it_node->IsNot(ISOLATED))
{
if (!it_node->IsFixed(VELOCITY_X))
{
fractionalVelocity = it_node->FastGetSolutionStepValue(VELOCITY_X);                                            
it_node->FastGetSolutionStepValue(VELOCITY_X) += it_node->FastGetSolutionStepValue(FRACT_VEL_X) / NodalVolume; 
it_node->FastGetSolutionStepValue(FRACT_VEL_X) = fractionalVelocity;                                           
}
if (!it_node->IsFixed(VELOCITY_Y))
{
fractionalVelocity = it_node->FastGetSolutionStepValue(VELOCITY_Y);                                            
it_node->FastGetSolutionStepValue(VELOCITY_Y) += it_node->FastGetSolutionStepValue(FRACT_VEL_Y) / NodalVolume; 
it_node->FastGetSolutionStepValue(FRACT_VEL_Y) = fractionalVelocity;                                           
}
if (!it_node->IsFixed(VELOCITY_Z))
{
fractionalVelocity = it_node->FastGetSolutionStepValue(VELOCITY_Z);                                            
it_node->FastGetSolutionStepValue(VELOCITY_Z) += it_node->FastGetSolutionStepValue(FRACT_VEL_Z) / NodalVolume; 
it_node->FastGetSolutionStepValue(FRACT_VEL_Z) = fractionalVelocity;                                           
}
}
}
}
else
{
#pragma omp parallel for
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
auto it_node = rModelPart.NodesBegin() + i_node;
const double NodalArea = it_node->FastGetSolutionStepValue(NODAL_VOLUME);
double fractionalVelocity = 0;
if (it_node->IsNot(ISOLATED))
{
if (!it_node->IsFixed(VELOCITY_X))
{
fractionalVelocity = it_node->FastGetSolutionStepValue(VELOCITY_X);                                          
it_node->FastGetSolutionStepValue(VELOCITY_X) += it_node->FastGetSolutionStepValue(FRACT_VEL_X) / NodalArea; 
it_node->FastGetSolutionStepValue(FRACT_VEL_X) = fractionalVelocity;                                         
}
if (!it_node->IsFixed(VELOCITY_Y))
{
fractionalVelocity = it_node->FastGetSolutionStepValue(VELOCITY_Y);                                          
it_node->FastGetSolutionStepValue(VELOCITY_Y) += it_node->FastGetSolutionStepValue(FRACT_VEL_Y) / NodalArea; 
it_node->FastGetSolutionStepValue(FRACT_VEL_Y) = fractionalVelocity;                                         
}
}
}
}
this->CheckVelocityConvergence(NormV);
}

void RecoverFractionalVelocity()
{

ModelPart &rModelPart = BaseType::GetModelPart();
const int n_nodes = rModelPart.NumberOfNodes();

rModelPart.GetCommunicator().AssembleCurrentData(FRACT_VEL);

if (mDomainSize > 2)
{
#pragma omp parallel for
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
auto it_node = rModelPart.NodesBegin() + i_node;
if (it_node->IsNot(ISOLATED))
{
if (!it_node->IsFixed(VELOCITY_X))
{
it_node->FastGetSolutionStepValue(VELOCITY_X) = it_node->FastGetSolutionStepValue(FRACT_VEL_X);
}
if (!it_node->IsFixed(VELOCITY_Y))
{
it_node->FastGetSolutionStepValue(VELOCITY_Y) = it_node->FastGetSolutionStepValue(FRACT_VEL_Y);
}
if (!it_node->IsFixed(VELOCITY_Z))
{
it_node->FastGetSolutionStepValue(VELOCITY_Z) = it_node->FastGetSolutionStepValue(FRACT_VEL_Z);
}
}
}
}
else
{
#pragma omp parallel for
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
auto it_node = rModelPart.NodesBegin() + i_node;
if (it_node->IsNot(ISOLATED))
{
if (!it_node->IsFixed(VELOCITY_X))
{
it_node->FastGetSolutionStepValue(VELOCITY_X) = it_node->FastGetSolutionStepValue(FRACT_VEL_X);
}
if (!it_node->IsFixed(VELOCITY_Y))
{
it_node->FastGetSolutionStepValue(VELOCITY_Y) = it_node->FastGetSolutionStepValue(FRACT_VEL_Y);
}
}
}
}
}

bool CheckVelocityIncrementConvergence(const double NormDv, double &NormV)
{
ModelPart &rModelPart = BaseType::GetModelPart();
const int n_nodes = rModelPart.NumberOfNodes();

NormV = 0.00;
double errorNormDv = 0;
double temp_norm = NormV;

#pragma omp parallel for reduction(+ : temp_norm)
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
const auto it_node = rModelPart.NodesBegin() + i_node;
const auto &r_vel = it_node->FastGetSolutionStepValue(VELOCITY);
for (unsigned int d = 0; d < 3; ++d)
{
temp_norm += r_vel[d] * r_vel[d];
}
}
NormV = temp_norm;
NormV = BaseType::GetModelPart().GetCommunicator().GetDataCommunicator().SumAll(NormV);
NormV = sqrt(NormV);

const double zero_tol = 1.0e-12;
errorNormDv = (NormV < zero_tol) ? NormDv : NormDv / NormV;

if (BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
{
std::cout << "The norm of velocity increment is: " << NormDv << std::endl;
std::cout << "The norm of velocity is: " << NormV << std::endl;
std::cout << "Velocity error: " << errorNormDv << "mVelocityTolerance: " << mVelocityTolerance << std::endl;
}

if (errorNormDv < mVelocityTolerance)
{
std::cout << "The norm of velocity is: " << NormV << " The norm of velocity increment is: " << NormDv << " Velocity error: " << errorNormDv << " Converged!" << std::endl;
return true;
}
else
{
std::cout << "The norm of velocity is: " << NormV << " The norm of velocity increment is: " << NormDv << " Velocity error: " << errorNormDv << " Not converged!" << std::endl;
return false;
}
}

void CheckVelocityConvergence(const double NormOldV)
{
ModelPart &rModelPart = BaseType::GetModelPart();
const int n_nodes = rModelPart.NumberOfNodes();

double NormV = 0.00;
#pragma omp parallel for reduction(+ \
: NormV)
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
const auto it_node = rModelPart.NodesBegin() + i_node;
const auto &r_vel = it_node->FastGetSolutionStepValue(VELOCITY);
for (unsigned int d = 0; d < 3; ++d)
{
NormV += r_vel[d] * r_vel[d];
}
}
NormV = BaseType::GetModelPart().GetCommunicator().GetDataCommunicator().SumAll(NormV);
NormV = sqrt(NormV);

std::cout << "The norm of velocity is: " << NormV << " Old velocity norm was: " << NormOldV << std::endl;
}

bool CheckPressureIncrementConvergence(const double NormDp)
{
ModelPart &rModelPart = BaseType::GetModelPart();
const int n_nodes = rModelPart.NumberOfNodes();

double NormP = 0.00;
double errorNormDp = 0;

#pragma omp parallel for reduction(+ \
: NormP)
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
const auto it_node = rModelPart.NodesBegin() + i_node;
const double Pr = it_node->FastGetSolutionStepValue(PRESSURE);
NormP += Pr * Pr;
}
NormP = BaseType::GetModelPart().GetCommunicator().GetDataCommunicator().SumAll(NormP);
NormP = sqrt(NormP);

const double zero_tol = 1.0e-12;
errorNormDp = (NormP < zero_tol) ? NormDp : NormDp / NormP;

if (BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
{
std::cout << "         The norm of pressure increment is: " << NormDp << std::endl;
std::cout << "         The norm of pressure is: " << NormP << std::endl;
std::cout << "         The norm of pressure increment is: " << NormDp << "         Pressure error: " << errorNormDp << std::endl;
}

if (errorNormDp < mPressureTolerance)
{
std::cout << "         The norm of pressure is: " << NormP << "  The norm of pressure increment is: " << NormDp << " Pressure error: " << errorNormDp << " Converged!" << std::endl;
return true;
}
else
{
std::cout << "         The norm of pressure is: " << NormP << "  The norm of pressure increment is: " << NormDp << " Pressure error: " << errorNormDp << " Not converged!" << std::endl;
return false;
}
}

void FixPressure()
{
ModelPart &rModelPart = BaseType::GetModelPart();
const int n_nodes = rModelPart.NumberOfNodes();
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
const auto it_node = rModelPart.NodesBegin() + i_node;
if (it_node->Is(FREE_SURFACE))
{
it_node->FastGetSolutionStepValue(PRESSURE) = 0;
it_node->Fix(PRESSURE);
}
}
}

void FreePressure()
{
ModelPart &rModelPart = BaseType::GetModelPart();
const int n_nodes = rModelPart.NumberOfNodes();
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
const auto it_node = rModelPart.NodesBegin() + i_node;
it_node->Free(PRESSURE);
}
}







double mVelocityTolerance;

double mPressureTolerance;

unsigned int mMaxPressureIter;

unsigned int mDomainSize;

unsigned int mTimeOrder;

bool mReformDofSet;



StrategyPointerType mpMomentumStrategy;

StrategyPointerType mpPressureStrategy;






ThreeStepVPStrategy &operator=(ThreeStepVPStrategy const &rOther) {}

ThreeStepVPStrategy(ThreeStepVPStrategy const &rOther) {}


}; 




} 

#endif 
