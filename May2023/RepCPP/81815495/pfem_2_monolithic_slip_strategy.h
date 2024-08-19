#ifndef KRATOS_PFEM2_MONOLITHIC_SLIP_STRATEGY_H
#define KRATOS_PFEM2_MONOLITHIC_SLIP_STRATEGY_H

#include "includes/define.h"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "processes/process.h"
#include "solving_strategies/schemes/scheme.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"

#include "solving_strategies/schemes/residualbased_incrementalupdate_static_scheme.h"
#include "solving_strategies/schemes/residualbased_incrementalupdate_static_scheme_slip.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver.h"
#include "solving_strategies/builder_and_solvers/residualbased_elimination_builder_and_solver_componentwise.h"
#include "solving_strategies/strategies/residualbased_linear_strategy.h"

#include "custom_utilities/solver_settings.h"

namespace Kratos {













template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
class PFEM2MonolithicSlipStrategy : public ImplicitSolvingStrategy<TSparseSpace,TDenseSpace,TLinearSolver>
{
public:

typedef boost::shared_ptr< FSStrategy<TSparseSpace, TDenseSpace, TLinearSolver> > Pointer;

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TDataType TDataType;


typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef typename ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>::Pointer StrategyPointerType;

typedef SolverSettings<TSparseSpace,TDenseSpace,TLinearSolver> SolverSettingsType;


PFEM2MonolithicSlipStrategy(ModelPart& rModelPart,
SolverSettingsType& rSolverConfig,
bool PredictorCorrector):
BaseType(rModelPart,false),
mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{
InitializeStrategy(rSolverConfig,PredictorCorrector);
}

PFEM2MonolithicSlipStrategy(ModelPart& rModelPart,
SolverSettingsType& rSolverConfig,
bool PredictorCorrector,
const Kratos::Variable<int>& PeriodicVar):
BaseType(rModelPart,false),
mrPeriodicIdVar(PeriodicVar)
{
InitializeStrategy(rSolverConfig,PredictorCorrector);
}


SolvingStrategyPython(self.model_part,
self.time_scheme,
self.monolithic_linear_solver,
self.conv_criteria,
CalculateReactionFlag,
ReformDofSetAtEachStep,
MoveMeshFlag)
self.monolithic_solver.SetMaximumIterations(self.maximum_nonlin_iterations)

PFEM2MonolithicSlipStrategy(ModelPart& rModelPart,

typename TLinearSolver::Pointer pLinearSolver,
bool ReformDofSet = true,
double Tol = 0.01,
int MaxIterations = 3,
unsigned int DomainSize = 2):
BaseType(rModelPart,MoveMeshFlag), 
mVelocityTolerance(VelTol),
mPressureTolerance(PresTol),
mMaxVelocityIter(MaxVelocityIterations),
mMaxPressureIter(MaxPressureIterations),
mDomainSize(DomainSize),
mTimeOrder(TimeOrder),
mPredictorCorrector(PredictorCorrector),
mUseSlipConditions(true), 
mReformDofSet(ReformDofSet),
mExtraIterationSteps(),
mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{
KRATOS_TRY;

BaseType::SetEchoLevel(1);

this->Check();

bool CalculateReactions = false;
bool CalculateNormDxFlag = true;

bool ReformDofAtEachIteration = false; 

typedef typename Kratos::VariableComponent<Kratos::VectorComponentAdaptor<Kratos::array_1d<double, 3 > > > VarComponent;
typedef typename BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>::Pointer BuilderSolverTypePointer;
typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef Scheme< TSparseSpace, TDenseSpace > SchemeType;
typename SchemeType::Pointer pScheme;
if (mUseSlipConditions)
{
typename SchemeType::Pointer Temp = typename SchemeType::Pointer(new ResidualBasedIncrementalUpdateStaticSchemeSlip< TSparseSpace, TDenseSpace > (mDomainSize,mDomainSize));
pScheme.swap(Temp);
}
else
{
typename SchemeType::Pointer Temp = typename SchemeType::Pointer(new ResidualBasedIncrementalUpdateStaticScheme< TSparseSpace, TDenseSpace > ());
pScheme.swap(Temp);
}

this->mpMomentumStrategy = typename BaseType::Pointer(new ResidualBasedLinearStrategy<TSparseSpace, TDenseSpace, TLinearSolver > (rModelPart, pScheme, pVelocityLinearSolver, CalculateReactions, ReformDofAtEachIteration, CalculateNormDxFlag));
this->mpMomentumStrategy->SetEchoLevel( BaseType::GetEchoLevel() );


this->mpPressureStrategy = typename BaseType::Pointer(new ResidualBasedLinearStrategy<TSparseSpace, TDenseSpace, TLinearSolver > (rModelPart, pScheme, pPressureLinearSolver, CalculateReactions, ReformDofAtEachIteration, CalculateNormDxFlag));
this->mpPressureStrategy->SetEchoLevel( BaseType::GetEchoLevel() );

if (mUseSlipConditions)
{
#pragma omp parallel
{
ModelPart::ConditionIterator CondBegin;
ModelPart::ConditionIterator CondEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Conditions(),CondBegin,CondEnd);

for (ModelPart::ConditionIterator itCond = CondBegin; itCond != CondEnd; ++itCond)
{
const double FlagValue = itCond->GetValue(IS_STRUCTURE);
itCond->Set(SLIP);
if (FlagValue != 0.0)
{

Condition::GeometryType& rGeom = itCond->GetGeometry();
for (unsigned int i = 0; i < rGeom.PointsNumber(); ++i)
{
rGeom[i].SetLock();
rGeom[i].SetValue(IS_STRUCTURE,FlagValue);
rGeom[i].Set(SLIP);
rGeom[i].UnSetLock();
}
}
}
}
rModelPart.GetCommunicator().AssembleNonHistoricalData(IS_STRUCTURE);
rModelPart.GetCommunicator().SynchronizeOrNodalFlags(SLIP);
}


KRATOS_CATCH("");
}

virtual ~FSStrategy(){}




virtual int Check()
{
KRATOS_TRY;

int ierr = BaseType::Check();
if (ierr != 0) return ierr;

if(DELTA_TIME.Key() == 0)
KRATOS_THROW_ERROR(std::runtime_error,"DELTA_TIME Key is 0. Check that the application was correctly registered.","");
if(BDF_COEFFICIENTS.Key() == 0)
KRATOS_THROW_ERROR(std::runtime_error,"BDF_COEFFICIENTS Key is 0. Check that the application was correctly registered.","");

ModelPart& rModelPart = BaseType::GetModelPart();

if ( mTimeOrder == 2 && rModelPart.GetBufferSize() < 3 )
KRATOS_THROW_ERROR(std::invalid_argument,"Buffer size too small for fractional step strategy (BDF2), needed 3, got ",rModelPart.GetBufferSize());
if ( mTimeOrder == 1 && rModelPart.GetBufferSize() < 2 )
KRATOS_THROW_ERROR(std::invalid_argument,"Buffer size too small for fractional step strategy (Backward Euler), needed 2, got ",rModelPart.GetBufferSize());

const ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

for ( ModelPart::ElementIterator itEl = rModelPart.ElementsBegin(); itEl != rModelPart.ElementsEnd(); ++itEl )
{
ierr = itEl->Check(rCurrentProcessInfo);
if (ierr != 0) break;
}

for ( ModelPart::ConditionIterator itCond = rModelPart.ConditionsBegin(); itCond != rModelPart.ConditionsEnd(); ++itCond)
{
ierr = itCond->Check(rCurrentProcessInfo);
if (ierr != 0) break;
}

return ierr;

KRATOS_CATCH("");
}

virtual double Solve()
{
ModelPart& rModelPart = BaseType::GetModelPart();
this->SetTimeCoefficients(rModelPart.GetProcessInfo());

double NormDp = 0.0;

if (mPredictorCorrector)
{
bool Converged = false;

for(unsigned int it = 0; it < mMaxPressureIter; ++it)
{
if ( BaseType::GetEchoLevel() > 1 && rModelPart.GetCommunicator().MyPID() == 0)
std::cout << "Pressure iteration " << it << std::endl;

NormDp = this->SolveStep();

Converged = this->CheckPressureConvergence(NormDp);

if ( Converged )
{
if ( BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
std::cout << "Predictor-corrector converged in " << it+1 << " iterations." << std::endl;
break;
}
}
if (!Converged && BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
std::cout << "Predictor-correctior iterations did not converge." << std::endl;

}
else
{
NormDp = this->SolveStep();
}

if (mReformDofSet)
this->Clear();

return NormDp;
}


virtual void CalculateReactions()
{
ModelPart& rModelPart = BaseType::GetModelPart();
ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

int OriginalStep = rCurrentProcessInfo[FRACTIONAL_STEP];
rCurrentProcessInfo.SetValue(FRACTIONAL_STEP,1);

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

const array_1d<double,3> Zero(3,0.0);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
itNode->FastGetSolutionStepValue(REACTION) = Zero;
}
}

#pragma omp parallel
{
ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Elements(),ElemBegin,ElemEnd);

LocalSystemVectorType RHS_Contribution;
LocalSystemMatrixType LHS_Contribution;

for (ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{


itElem->CalculateLocalSystem(LHS_Contribution, RHS_Contribution, rCurrentProcessInfo);

Element::GeometryType& rGeom = itElem->GetGeometry();
unsigned int NumNodes = rGeom.PointsNumber();
unsigned int index = 0;

for (unsigned int i = 0; i < NumNodes; i++)
{
rGeom[i].SetLock();
array_1d<double,3>& rReaction = rGeom[i].FastGetSolutionStepValue(REACTION);
for (unsigned int d = 0; d < mDomainSize; ++d)
rReaction[d] -= RHS_Contribution[index++];
rGeom[i].UnSetLock();
}
}
}

rModelPart.GetCommunicator().AssembleCurrentData(REACTION);

rCurrentProcessInfo.SetValue(FRACTIONAL_STEP,OriginalStep);
}

virtual void AddIterationStep(Process::Pointer pNewStep)
{
mExtraIterationSteps.push_back(pNewStep);
}

virtual void ClearExtraIterationSteps()
{
mExtraIterationSteps.clear();
}

virtual void Clear()
{
mpMomentumStrategy->Clear();
mpPressureStrategy->Clear();
}



virtual void SetEchoLevel(int Level)
{
BaseType::SetEchoLevel(Level);
int StrategyLevel = Level > 0 ? Level - 1 : 0;
mpMomentumStrategy->SetEchoLevel(StrategyLevel);
mpPressureStrategy->SetEchoLevel(StrategyLevel);
}





virtual std::string Info() const
{
std::stringstream buffer;
buffer << "FSStrategy" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const {rOStream << "FSStrategy";}

virtual void PrintData(std::ostream& rOStream) const {}





protected:












void SetTimeCoefficients(ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY;

if (mTimeOrder == 2)
{
double Dt = rCurrentProcessInfo[DELTA_TIME];
double OldDt = rCurrentProcessInfo.GetPreviousTimeStepInfo(1)[DELTA_TIME];

double Rho = OldDt / Dt;
double TimeCoeff = 1.0 / (Dt * Rho * Rho + Dt * Rho);

Vector& BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(3, false);

BDFcoeffs[0] = TimeCoeff * (Rho * Rho + 2.0 * Rho); 
BDFcoeffs[1] = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0); 
BDFcoeffs[2] = TimeCoeff; 
}
else if (mTimeOrder == 1)
{
double Dt = rCurrentProcessInfo[DELTA_TIME];
double TimeCoeff = 1.0 / Dt;

Vector& BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(2, false);

BDFcoeffs[0] = TimeCoeff; 
BDFcoeffs[1] = -TimeCoeff; 
}

KRATOS_CATCH("");
}

double SolveStep()
{
ModelPart& rModelPart = BaseType::GetModelPart();

rModelPart.GetProcessInfo().SetValue(FRACTIONAL_STEP,1);

bool Converged = false;
int Rank = rModelPart.GetCommunicator().MyPID();

for(unsigned int it = 0; it < mMaxVelocityIter; ++it)
{
if ( BaseType::GetEchoLevel() > 1 && Rank == 0)
std::cout << "Momentum iteration " << it << std::endl;

rModelPart.GetProcessInfo().SetValue(FRACTIONAL_STEP,1);
double NormDv = mpMomentumStrategy->Solve();



Converged = this->CheckFractionalStepConvergence(NormDv);

if (Converged)
{
if ( BaseType::GetEchoLevel() > 0 && Rank == 0)
std::cout << "Fractional velocity converged in " << it+1 << " iterations." << std::endl;
break;
}
}

if (!Converged && BaseType::GetEchoLevel() > 0 && Rank == 0)
std::cout << "Fractional velocity iterations did not converge." << std::endl;

rModelPart.GetProcessInfo().SetValue(FRACTIONAL_STEP,4);
this->ComputeSplitOssProjections(rModelPart);

rModelPart.GetProcessInfo().SetValue(FRACTIONAL_STEP,5);

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
const double OldPress = itNode->FastGetSolutionStepValue(PRESSURE);
itNode->FastGetSolutionStepValue(PRESSURE_OLD_IT) = -OldPress;
}
}

if (BaseType::GetEchoLevel() > 0 && Rank == 0)
std::cout << "Calculating Pressure." << std::endl;
double NormDp = mpPressureStrategy->Solve();

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
itNode->FastGetSolutionStepValue(PRESSURE_OLD_IT) += itNode->FastGetSolutionStepValue(PRESSURE);
}

if (BaseType::GetEchoLevel() > 0 && Rank == 0)
std::cout << "Updating Velocity." << std::endl;
rModelPart.GetProcessInfo().SetValue(FRACTIONAL_STEP,6);

this->CalculateEndOfStepVelocity();

for (std::vector<Process::Pointer>::iterator iExtraSteps = mExtraIterationSteps.begin();
iExtraSteps != mExtraIterationSteps.end(); ++iExtraSteps)
(*iExtraSteps)->Execute();


return NormDp;
}

bool CheckFractionalStepConvergence(const double NormDv)
{
ModelPart& rModelPart = BaseType::GetModelPart();

double NormV = 0.00;

#pragma omp parallel reduction(+:NormV)
{
ModelPart::NodeIterator NodeBegin;
ModelPart::NodeIterator NodeEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodeBegin,NodeEnd);

for (ModelPart::NodeIterator itNode = NodeBegin; itNode != NodeEnd; ++itNode)
{
const array_1d<double,3> &Vel = itNode->FastGetSolutionStepValue(VELOCITY);

for (unsigned int d = 0; d < 3; ++d)
NormV += Vel[d] * Vel[d];
}
}

BaseType::GetModelPart().GetCommunicator().SumAll(NormV);

NormV = sqrt(NormV);

if (NormV == 0.0) NormV = 1.00;

double Ratio = NormDv / NormV;

if ( BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
std::cout << "Fractional velocity relative error: " << Ratio << std::endl;

if (Ratio < mVelocityTolerance)
{
return true;
}
else
return false;
}

bool CheckPressureConvergence(const double NormDp)
{
ModelPart& rModelPart = BaseType::GetModelPart();

double NormP = 0.00;

#pragma omp parallel reduction(+:NormP)
{
ModelPart::NodeIterator NodeBegin;
ModelPart::NodeIterator NodeEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodeBegin,NodeEnd);

for (ModelPart::NodeIterator itNode = NodeBegin; itNode != NodeEnd; ++itNode)
{
const double Pr = itNode->FastGetSolutionStepValue(PRESSURE);
NormP += Pr * Pr;
}
}

BaseType::GetModelPart().GetCommunicator().SumAll(NormP);

NormP = sqrt(NormP);

if (NormP == 0.0) NormP = 1.00;

double Ratio = NormDp / NormP;

if ( BaseType::GetEchoLevel() > 0 && rModelPart.GetCommunicator().MyPID() == 0)
std::cout << "Pressure relative error: " << Ratio << std::endl;

if (Ratio < mPressureTolerance)
{
return true;
}
else
return false;
}


void ComputeSplitOssProjections(ModelPart& rModelPart)
{
const array_1d<double,3> Zero(3,0.0);

array_1d<double,3> Out(3,0.0);

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for ( ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode )
{
itNode->FastGetSolutionStepValue(CONV_PROJ) = Zero;
itNode->FastGetSolutionStepValue(PRESS_PROJ) = Zero;
itNode->FastGetSolutionStepValue(DIVPROJ) = 0.0;
itNode->FastGetSolutionStepValue(NODAL_AREA) = 0.0;
}
}

#pragma omp parallel
{
ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Elements(),ElemBegin,ElemEnd);

for ( ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem )
{
itElem->Calculate(CONV_PROJ,Out,rModelPart.GetProcessInfo());
}
}

rModelPart.GetCommunicator().AssembleCurrentData(CONV_PROJ);
rModelPart.GetCommunicator().AssembleCurrentData(PRESS_PROJ);
rModelPart.GetCommunicator().AssembleCurrentData(DIVPROJ);
rModelPart.GetCommunicator().AssembleCurrentData(NODAL_AREA);

this->PeriodicConditionProjectionCorrection(rModelPart);

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for ( ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode )
{
const double NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
itNode->FastGetSolutionStepValue(CONV_PROJ) /= NodalArea;
itNode->FastGetSolutionStepValue(PRESS_PROJ) /= NodalArea;
itNode->FastGetSolutionStepValue(DIVPROJ) /= NodalArea;
}
}
}

void CalculateEndOfStepVelocity()
{
ModelPart& rModelPart = BaseType::GetModelPart();

const array_1d<double,3> Zero(3,0.0);
array_1d<double,3> Out(3,0.0);

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for ( ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode )
{
itNode->FastGetSolutionStepValue(FRACT_VEL) = Zero;
}
}

#pragma omp parallel
{
ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Elements(),ElemBegin,ElemEnd);

for ( ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem )
{
itElem->Calculate(VELOCITY,Out,rModelPart.GetProcessInfo());
}
}

rModelPart.GetCommunicator().AssembleCurrentData(FRACT_VEL);
this->PeriodicConditionVelocityCorrection(rModelPart);

if (mUseSlipConditions)
this->EnforceSlipCondition(SLIP);

if (mDomainSize > 2)
{
#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for ( ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode )
{
const double NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
if ( ! itNode->IsFixed(VELOCITY_X) )
itNode->FastGetSolutionStepValue(VELOCITY_X) += itNode->FastGetSolutionStepValue(FRACT_VEL_X) / NodalArea;
if ( ! itNode->IsFixed(VELOCITY_Y) )
itNode->FastGetSolutionStepValue(VELOCITY_Y) += itNode->FastGetSolutionStepValue(FRACT_VEL_Y) / NodalArea;
if ( ! itNode->IsFixed(VELOCITY_Z) )
itNode->FastGetSolutionStepValue(VELOCITY_Z) += itNode->FastGetSolutionStepValue(FRACT_VEL_Z) / NodalArea;
}
}
}
else
{
#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for ( ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode )
{
const double NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
if ( ! itNode->IsFixed(VELOCITY_X) )
itNode->FastGetSolutionStepValue(VELOCITY_X) += itNode->FastGetSolutionStepValue(FRACT_VEL_X) / NodalArea;
if ( ! itNode->IsFixed(VELOCITY_Y) )
itNode->FastGetSolutionStepValue(VELOCITY_Y) += itNode->FastGetSolutionStepValue(FRACT_VEL_Y) / NodalArea;
}
}
}
}


void EnforceSlipCondition(const Kratos::Flags& rSlipWallFlag)
{
ModelPart& rModelPart = BaseType::GetModelPart();

#pragma omp parallel
{
ModelPart::NodeIterator NodeBegin; 
ModelPart::NodeIterator NodeEnd; 
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodeBegin,NodeEnd);

for ( ModelPart::NodeIterator itNode = NodeBegin; itNode != NodeEnd; ++itNode )
{
if ( itNode->Is(rSlipWallFlag) )
{
const array_1d<double,3>& rNormal = itNode->FastGetSolutionStepValue(NORMAL);
array_1d<double,3>& rDeltaVelocity = itNode->FastGetSolutionStepValue(FRACT_VEL);

double Proj = rNormal[0] * rDeltaVelocity[0];
double Norm = rNormal[0] * rNormal[0];

for (unsigned int d = 1; d < mDomainSize; ++d)
{
Proj += rNormal[d] * rDeltaVelocity[d];
Norm += rNormal[d] * rNormal[d];
}

Proj /= Norm;
rDeltaVelocity -= Proj * rNormal;
}
}
}
}


void PeriodicConditionProjectionCorrection(ModelPart& rModelPart)
{
if (mrPeriodicIdVar.Key() != 0)
{
int GlobalNodesNum = rModelPart.GetCommunicator().LocalMesh().Nodes().size();
rModelPart.GetCommunicator().SumAll(GlobalNodesNum);

for (typename ModelPart::ConditionIterator itCond = rModelPart.ConditionsBegin(); itCond != rModelPart.ConditionsEnd(); itCond++ )
{
ModelPart::ConditionType::GeometryType& rGeom = itCond->GetGeometry();
if (rGeom.PointsNumber() == 2)
{
Node& rNode0 = rGeom[0];
int Node0Pair = rNode0.FastGetSolutionStepValue(mrPeriodicIdVar);

Node& rNode1 = rGeom[1];
int Node1Pair = rNode1.FastGetSolutionStepValue(mrPeriodicIdVar);

if ( ( static_cast<int>(rNode0.Id()) == Node1Pair ) && (static_cast<int>(rNode1.Id()) == Node0Pair ) )
{
double NodalArea = rNode0.FastGetSolutionStepValue(NODAL_AREA) + rNode1.FastGetSolutionStepValue(NODAL_AREA);
array_1d<double,3> ConvProj = rNode0.FastGetSolutionStepValue(CONV_PROJ) + rNode1.FastGetSolutionStepValue(CONV_PROJ);
array_1d<double,3> PressProj = rNode0.FastGetSolutionStepValue(PRESS_PROJ) + rNode1.FastGetSolutionStepValue(PRESS_PROJ);
double DivProj = rNode0.FastGetSolutionStepValue(DIVPROJ) + rNode1.FastGetSolutionStepValue(DIVPROJ);

rNode0.GetValue(NODAL_AREA) = NodalArea;
rNode0.GetValue(CONV_PROJ) = ConvProj;
rNode0.GetValue(PRESS_PROJ) = PressProj;
rNode0.GetValue(DIVPROJ) = DivProj;
rNode1.GetValue(NODAL_AREA) = NodalArea;
rNode1.GetValue(CONV_PROJ) = ConvProj;
rNode1.GetValue(PRESS_PROJ) = PressProj;
rNode1.GetValue(DIVPROJ) = DivProj;
}
}
else if (rGeom.PointsNumber() == 4 && rGeom[0].FastGetSolutionStepValue(mrPeriodicIdVar) > GlobalNodesNum)
{
double NodalArea = rGeom[0].FastGetSolutionStepValue(NODAL_AREA);
array_1d<double,3> ConvProj = rGeom[0].FastGetSolutionStepValue(CONV_PROJ);
array_1d<double,3> PressProj = rGeom[0].FastGetSolutionStepValue(PRESS_PROJ);
double DivProj = rGeom[0].FastGetSolutionStepValue(DIVPROJ);

for (unsigned int i = 1; i < 4; i++)
{
NodalArea += rGeom[i].FastGetSolutionStepValue(NODAL_AREA);
ConvProj += rGeom[i].FastGetSolutionStepValue(CONV_PROJ);
PressProj += rGeom[i].FastGetSolutionStepValue(PRESS_PROJ);
DivProj += rGeom[i].FastGetSolutionStepValue(DIVPROJ);
}

for (unsigned int i = 0; i < 4; i++)
{
rGeom[i].GetValue(NODAL_AREA) = NodalArea;
rGeom[i].GetValue(CONV_PROJ) = ConvProj;
rGeom[i].GetValue(PRESS_PROJ) = PressProj;
rGeom[i].GetValue(DIVPROJ) = DivProj;
}
}
}

rModelPart.GetCommunicator().AssembleNonHistoricalData(NODAL_AREA);
rModelPart.GetCommunicator().AssembleNonHistoricalData(CONV_PROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(PRESS_PROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(DIVPROJ);

for (typename ModelPart::NodeIterator itNode = rModelPart.NodesBegin(); itNode != rModelPart.NodesEnd(); itNode++)
{
if (itNode->GetValue(NODAL_AREA) != 0.0)
{
itNode->FastGetSolutionStepValue(NODAL_AREA) = itNode->GetValue(NODAL_AREA);
itNode->FastGetSolutionStepValue(CONV_PROJ) = itNode->GetValue(CONV_PROJ);
itNode->FastGetSolutionStepValue(PRESS_PROJ) = itNode->GetValue(PRESS_PROJ);
itNode->FastGetSolutionStepValue(DIVPROJ) = itNode->GetValue(DIVPROJ);

itNode->GetValue(NODAL_AREA) = 0.0;
itNode->GetValue(CONV_PROJ) = array_1d<double,3>(3,0.0);
itNode->GetValue(PRESS_PROJ) = array_1d<double,3>(3,0.0);
itNode->GetValue(DIVPROJ) = 0.0;
}
}
}
}

void PeriodicConditionVelocityCorrection(ModelPart& rModelPart)
{
if (mrPeriodicIdVar.Key() != 0)
{
int GlobalNodesNum = rModelPart.GetCommunicator().LocalMesh().Nodes().size();
rModelPart.GetCommunicator().SumAll(GlobalNodesNum);

for (typename ModelPart::ConditionIterator itCond = rModelPart.ConditionsBegin(); itCond != rModelPart.ConditionsEnd(); itCond++ )
{
ModelPart::ConditionType::GeometryType& rGeom = itCond->GetGeometry();
if (rGeom.PointsNumber() == 2)
{
Node& rNode0 = rGeom[0];
int Node0Pair = rNode0.FastGetSolutionStepValue(mrPeriodicIdVar);

Node& rNode1 = rGeom[1];
int Node1Pair = rNode1.FastGetSolutionStepValue(mrPeriodicIdVar);

if ( ( static_cast<int>(rNode0.Id()) == Node1Pair ) && (static_cast<int>(rNode1.Id()) == Node0Pair ) )
{
array_1d<double,3> DeltaVel = rNode0.FastGetSolutionStepValue(FRACT_VEL) + rNode1.FastGetSolutionStepValue(FRACT_VEL);

rNode0.GetValue(FRACT_VEL) = DeltaVel;
rNode1.GetValue(FRACT_VEL) = DeltaVel;
}
}
else if (rGeom.PointsNumber() == 4 && rGeom[0].FastGetSolutionStepValue(mrPeriodicIdVar) > GlobalNodesNum)
{
array_1d<double,3> DeltaVel = rGeom[0].FastGetSolutionStepValue(FRACT_VEL);
for (unsigned int i = 1; i < 4; i++)
{
DeltaVel += rGeom[i].FastGetSolutionStepValue(FRACT_VEL);
}

for (unsigned int i = 0; i < 4; i++)
{
rGeom[i].GetValue(FRACT_VEL) = DeltaVel;
}
}
}

rModelPart.GetCommunicator().AssembleNonHistoricalData(FRACT_VEL);

for (typename ModelPart::NodeIterator itNode = rModelPart.NodesBegin(); itNode != rModelPart.NodesEnd(); itNode++)
{
array_1d<double,3>& rDeltaVel = itNode->GetValue(FRACT_VEL);
if ( rDeltaVel[0]*rDeltaVel[0] + rDeltaVel[1]*rDeltaVel[1] + rDeltaVel[2]*rDeltaVel[2] != 0.0)
{
itNode->FastGetSolutionStepValue(FRACT_VEL) = itNode->GetValue(FRACT_VEL);
rDeltaVel = array_1d<double,3>(3,0.0);
}
}
}
}









private:



double mVelocityTolerance;

double mPressureTolerance;

unsigned int mMaxVelocityIter;

unsigned int mMaxPressureIter;

unsigned int mDomainSize;

unsigned int mTimeOrder;

bool mPredictorCorrector;

bool mUseSlipConditions;

bool mReformDofSet;



StrategyPointerType mpMomentumStrategy;

StrategyPointerType mpPressureStrategy;

std::vector< Process::Pointer > mExtraIterationSteps;

const Kratos::Variable<int>& mrPeriodicIdVar;





void InitializeStrategy(SolverSettingsType& rSolverConfig,
bool PredictorCorrector)
{
KRATOS_TRY;

mTimeOrder = rSolverConfig.GetTimeOrder();

this->Check();

ModelPart& rModelPart = this->GetModelPart();

mDomainSize = rSolverConfig.GetDomainSize();

mPredictorCorrector = PredictorCorrector;

mUseSlipConditions = rSolverConfig.UseSlipConditions();

mReformDofSet = rSolverConfig.GetReformDofSet();

BaseType::SetEchoLevel(rSolverConfig.GetEchoLevel());

bool HaveVelStrategy = rSolverConfig.FindStrategy(SolverSettingsType::Velocity,mpMomentumStrategy);

if (HaveVelStrategy)
{
rSolverConfig.FindTolerance(SolverSettingsType::Velocity,mVelocityTolerance);
rSolverConfig.FindMaxIter(SolverSettingsType::Velocity,mMaxVelocityIter);
}
else
{
KRATOS_THROW_ERROR(std::runtime_error,"FS_Strategy error: No Velocity strategy defined in FractionalStepSettings","");
}

bool HavePressStrategy = rSolverConfig.FindStrategy(SolverSettingsType::Pressure,mpPressureStrategy);

if (HavePressStrategy)
{
rSolverConfig.FindTolerance(SolverSettingsType::Pressure,mPressureTolerance);
rSolverConfig.FindMaxIter(SolverSettingsType::Pressure,mMaxPressureIter);
}
else
{
KRATOS_THROW_ERROR(std::runtime_error,"FS_Strategy error: No Pressure strategy defined in FractionalStepSettings","");
}

Process::Pointer pTurbulenceProcess;
bool HaveTurbulence = rSolverConfig.GetTurbulenceModel(pTurbulenceProcess);

if (HaveTurbulence)
mExtraIterationSteps.push_back(pTurbulenceProcess);

if (mUseSlipConditions)
{
#pragma omp parallel
{
ModelPart::ConditionIterator CondBegin;
ModelPart::ConditionIterator CondEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Conditions(),CondBegin,CondEnd);

for (ModelPart::ConditionIterator itCond = CondBegin; itCond != CondEnd; ++itCond)
{
const bool is_slip = itCond->Is(SLIP);
if (is_slip)
{

Condition::GeometryType& rGeom = itCond->GetGeometry();
for (unsigned int i = 0; i < rGeom.PointsNumber(); ++i)
{
rGeom[i].SetLock();
rGeom[i].Set(SLIP);
rGeom[i].UnSetLock();
}
}
}
}
rModelPart.GetCommunicator().SynchronizeOrNodalFlags(SLIP);
}

this->Check();

KRATOS_CATCH("");
}







FSStrategy& operator=(FSStrategy const& rOther){}

FSStrategy(FSStrategy const& rOther){}



}; 





} 

#endif 
