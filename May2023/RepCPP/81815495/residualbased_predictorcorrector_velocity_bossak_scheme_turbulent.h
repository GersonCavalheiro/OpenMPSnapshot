

#if !defined(KRATOS_RESIDUALBASED_PREDICTOR_CORRECTOR_VELOCITY_BOSSAK_TURBULENT_SCHEME )
#define  KRATOS_RESIDUALBASED_PREDICTOR_CORRECTOR_VELOCITY_BOSSAK_TURBULENT_SCHEME






#include "boost/smart_ptr.hpp"


#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "includes/cfd_variables.h"
#include "containers/array_1d.h"
#include "utilities/openmp_utils.h"
#include "utilities/dof_updater.h"
#include "utilities/coordinate_transformation_utilities.h"
#include "processes/process.h"

namespace Kratos {



























template<class TSparseSpace,
class TDenseSpace 
>
class ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent : public Scheme<TSparseSpace, TDenseSpace> {
public:



KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent);

typedef Scheme<TSparseSpace, TDenseSpace> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename Element::DofsVectorType DofsVectorType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef Element::GeometryType  GeometryType;







ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent(
double NewAlphaBossak,
double MoveMeshStrategy,
unsigned int DomainSize)
:
Scheme<TSparseSpace, TDenseSpace>(),
mRotationTool(DomainSize,DomainSize+1,SLIP), 
mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{
mAlphaBossak = NewAlphaBossak;
mBetaNewmark = 0.25 * pow((1.00 - mAlphaBossak), 2);
mGammaNewmark = 0.5 - mAlphaBossak;
mMeshVelocity = MoveMeshStrategy;


int NumThreads = ParallelUtilities::GetNumThreads();
mMass.resize(NumThreads);
mDamp.resize(NumThreads);
mvel.resize(NumThreads);
macc.resize(NumThreads);
maccold.resize(NumThreads);
}



ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent(
double NewAlphaBossak,
unsigned int DomainSize,
const Variable<int>& rPeriodicIdVar)
:
Scheme<TSparseSpace, TDenseSpace>(),
mRotationTool(DomainSize,DomainSize+1,SLIP), 
mrPeriodicIdVar(rPeriodicIdVar)
{
mAlphaBossak = NewAlphaBossak;
mBetaNewmark = 0.25 * pow((1.00 - mAlphaBossak), 2);
mGammaNewmark = 0.5 - mAlphaBossak;
mMeshVelocity = 0.0;


int NumThreads = ParallelUtilities::GetNumThreads();
mMass.resize(NumThreads);
mDamp.resize(NumThreads);
mvel.resize(NumThreads);
macc.resize(NumThreads);
maccold.resize(NumThreads);
}



ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent(
double NewAlphaBossak,
double MoveMeshStrategy,
unsigned int DomainSize,
Kratos::Flags& rSlipFlag)
:
Scheme<TSparseSpace, TDenseSpace>(),
mRotationTool(DomainSize,DomainSize+1,rSlipFlag), 
mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{
mAlphaBossak = NewAlphaBossak;
mBetaNewmark = 0.25 * pow((1.00 - mAlphaBossak), 2);
mGammaNewmark = 0.5 - mAlphaBossak;
mMeshVelocity = MoveMeshStrategy;


int NumThreads = ParallelUtilities::GetNumThreads();
mMass.resize(NumThreads);
mDamp.resize(NumThreads);
mvel.resize(NumThreads);
macc.resize(NumThreads);
maccold.resize(NumThreads);
}


ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent(
double NewAlphaBossak,
double MoveMeshStrategy,
unsigned int DomainSize,
Process::Pointer pTurbulenceModel)
:
Scheme<TSparseSpace, TDenseSpace>(),
mRotationTool(DomainSize,DomainSize+1,SLIP), 
mrPeriodicIdVar(Kratos::Variable<int>::StaticObject()),
mpTurbulenceModel(pTurbulenceModel)
{
mAlphaBossak = NewAlphaBossak;
mBetaNewmark = 0.25 * pow((1.00 - mAlphaBossak), 2);
mGammaNewmark = 0.5 - mAlphaBossak;
mMeshVelocity = MoveMeshStrategy;


int NumThreads = ParallelUtilities::GetNumThreads();
mMass.resize(NumThreads);
mDamp.resize(NumThreads);
mvel.resize(NumThreads);
macc.resize(NumThreads);
maccold.resize(NumThreads);
}


ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent(
double NewAlphaBossak,
double MoveMeshStrategy,
unsigned int DomainSize,
const double RelaxationFactor,
Process::Pointer pTurbulenceModel)
:
Scheme<TSparseSpace, TDenseSpace>(),
mRotationTool(DomainSize,DomainSize+1,SLIP), 
mrPeriodicIdVar(Kratos::Variable<int>::StaticObject()),
mpTurbulenceModel(pTurbulenceModel)
{
mAlphaBossak = NewAlphaBossak;
mBetaNewmark = 0.25 * pow((1.00 - mAlphaBossak), 2);
mGammaNewmark = 0.5 - mAlphaBossak;
mMeshVelocity = MoveMeshStrategy;
mRelaxationFactor = RelaxationFactor;


int NumThreads = ParallelUtilities::GetNumThreads();
mMass.resize(NumThreads);
mDamp.resize(NumThreads);
mvel.resize(NumThreads);
macc.resize(NumThreads);
maccold.resize(NumThreads);
}


~ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent() override {
}







/
void CalculateSystemContributions(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY
int k = OpenMPUtils::ThisThread();

rCurrentElement.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);

rCurrentElement.CalculateMassMatrix(mMass[k], CurrentProcessInfo);
rCurrentElement.CalculateLocalVelocityContribution(mDamp[k], RHS_Contribution, CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);


AddDynamicsToLHS(LHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);
AddDynamicsToRHS(rCurrentElement, RHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

mRotationTool.Rotate(LHS_Contribution,RHS_Contribution,rCurrentElement.GetGeometry());
mRotationTool.ApplySlipCondition(LHS_Contribution,RHS_Contribution,rCurrentElement.GetGeometry());

KRATOS_CATCH("")
}

void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
int k = OpenMPUtils::ThisThread();

rCurrentElement.CalculateRightHandSide(RHS_Contribution, CurrentProcessInfo);
rCurrentElement.CalculateMassMatrix(mMass[k], CurrentProcessInfo);

rCurrentElement.CalculateLocalVelocityContribution(mDamp[k], RHS_Contribution, CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);


AddDynamicsToRHS(rCurrentElement, RHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

mRotationTool.Rotate(RHS_Contribution,rCurrentElement.GetGeometry());
mRotationTool.ApplySlipCondition(RHS_Contribution,rCurrentElement.GetGeometry());
}


void CalculateSystemContributions(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY
int k = OpenMPUtils::ThisThread();

rCurrentCondition.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);
rCurrentCondition.CalculateMassMatrix(mMass[k], CurrentProcessInfo);
rCurrentCondition.CalculateLocalVelocityContribution(mDamp[k], RHS_Contribution, CurrentProcessInfo);
rCurrentCondition.EquationIdVector(EquationId, CurrentProcessInfo);


AddDynamicsToLHS(LHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

AddDynamicsToRHS(rCurrentCondition, RHS_Contribution, mDamp[k], mMass[k], CurrentProcessInfo);

mRotationTool.Rotate(LHS_Contribution,RHS_Contribution,rCurrentCondition.GetGeometry());
mRotationTool.ApplySlipCondition(LHS_Contribution,RHS_Contribution,rCurrentCondition.GetGeometry());

KRATOS_CATCH("")
}

void CalculateRHSContribution(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_TRY;

int k = OpenMPUtils::ThisThread();


rCurrentCondition.CalculateRightHandSide(RHS_Contribution,rCurrentProcessInfo);
rCurrentCondition.CalculateMassMatrix(mMass[k],rCurrentProcessInfo);
rCurrentCondition.CalculateLocalVelocityContribution(mDamp[k], RHS_Contribution,rCurrentProcessInfo);
rCurrentCondition.EquationIdVector(EquationId,rCurrentProcessInfo);

AddDynamicsToRHS(rCurrentCondition, RHS_Contribution, mDamp[k], mMass[k],rCurrentProcessInfo);

mRotationTool.Rotate(RHS_Contribution,rCurrentCondition.GetGeometry());
mRotationTool.ApplySlipCondition(RHS_Contribution,rCurrentCondition.GetGeometry());

KRATOS_CATCH("");
}
/





















protected:







double mAlphaBossak;
double mBetaNewmark;
double mGammaNewmark;
double mMeshVelocity;
double mRelaxationFactor = 1.0;

double ma0;
double ma1;
double ma2;
double ma3;
double ma4;
double ma5;
double mam;

std::vector< Matrix > mMass;
std::vector< Matrix > mDamp;
std::vector< Vector > mvel;
std::vector< Vector > macc;
std::vector< Vector > maccold;






void PeriodicConditionProjectionCorrection(ModelPart& rModelPart)
{
const int num_nodes = rModelPart.NumberOfNodes();
const int num_conditions = rModelPart.NumberOfConditions();

#pragma omp parallel for
for (int i = 0; i < num_nodes; i++) {
auto it_node = rModelPart.NodesBegin() + i;

it_node->SetValue(NODAL_AREA,0.0);
it_node->SetValue(ADVPROJ,ZeroVector(3));
it_node->SetValue(DIVPROJ,0.0);
}

#pragma omp parallel for
for (int i = 0; i < num_conditions; i++) {
auto it_cond = rModelPart.ConditionsBegin() + i;

if(it_cond->Is(PERIODIC)) {
this->AssemblePeriodicContributionToProjections(it_cond->GetGeometry());
}
}

rModelPart.GetCommunicator().AssembleNonHistoricalData(NODAL_AREA);
rModelPart.GetCommunicator().AssembleNonHistoricalData(ADVPROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(DIVPROJ);

#pragma omp parallel for
for (int i = 0; i < num_nodes; i++) {
auto it_node = rModelPart.NodesBegin() + i;
this->CorrectContributionsOnPeriodicNode(*it_node);
}
}

void AssemblePeriodicContributionToProjections(Geometry< Node >& rGeometry)
{
unsigned int nodes_in_cond = rGeometry.PointsNumber();

double nodal_area = 0.0;
array_1d<double,3> momentum_projection = ZeroVector(3);
double mass_projection = 0.0;
for ( unsigned int i = 0; i < nodes_in_cond; i++ )
{
auto& r_node = rGeometry[i];
nodal_area += r_node.FastGetSolutionStepValue(NODAL_AREA);
noalias(momentum_projection) += r_node.FastGetSolutionStepValue(ADVPROJ);
mass_projection += r_node.FastGetSolutionStepValue(DIVPROJ);
}

for ( unsigned int i = 0; i < nodes_in_cond; i++ )
{
auto& r_node = rGeometry[i];

r_node.SetLock();
r_node.GetValue(NODAL_AREA) = nodal_area;
noalias(r_node.GetValue(ADVPROJ)) = momentum_projection;
r_node.GetValue(DIVPROJ) = mass_projection;
r_node.UnSetLock();
}
}

void CorrectContributionsOnPeriodicNode(Node& rNode)
{
if (rNode.GetValue(NODAL_AREA) != 0.0) 
{
rNode.FastGetSolutionStepValue(NODAL_AREA) = rNode.GetValue(NODAL_AREA);
noalias(rNode.FastGetSolutionStepValue(ADVPROJ)) = rNode.GetValue(ADVPROJ);
rNode.FastGetSolutionStepValue(DIVPROJ) = rNode.GetValue(DIVPROJ);
}
}


/
void AddDynamicsToLHS(LocalSystemMatrixType& LHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
const ProcessInfo& CurrentProcessInfo)
{

LHS_Contribution *= ma1;

if (M.size1() != 0) 
{
noalias(LHS_Contribution) += mam*M;
}

if (D.size1() != 0) 
{
noalias(LHS_Contribution) += D;
}
}





/
void AddDynamicsToRHS(
Element& rCurrentElement,
LocalSystemVectorType& rRHS_Contribution,
LocalSystemMatrixType& rD,
LocalSystemMatrixType& rM,
const ProcessInfo& rCurrentProcessInfo)
{
if (rM.size1() != 0) {
const auto& r_const_elem_ref = rCurrentElement;
int k = OpenMPUtils::ThisThread();
r_const_elem_ref.GetSecondDerivativesVector(macc[k], 0);
(macc[k]) *= (1.00 - mAlphaBossak);
r_const_elem_ref.GetSecondDerivativesVector(maccold[k], 1);
noalias(macc[k]) += mAlphaBossak * maccold[k];
noalias(rRHS_Contribution) -= prod(rM, macc[k]);
}
}


void AddDynamicsToRHS(
Condition& rCurrentCondition,
LocalSystemVectorType& rRHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& rM,
const ProcessInfo& rCurrentProcessInfo)
{
if (rM.size1() != 0)
{
const auto& r_const_cond_ref = rCurrentCondition;
int k = OpenMPUtils::ThisThread();
r_const_cond_ref.GetSecondDerivativesVector(macc[k], 0);
(macc[k]) *= (1.00 - mAlphaBossak);
r_const_cond_ref.GetSecondDerivativesVector(maccold[k], 1);
noalias(macc[k]) += mAlphaBossak * maccold[k];

noalias(rRHS_Contribution) -= prod(rM, macc[k]);
}
}
























private:








CoordinateTransformationUtils<LocalSystemMatrixType,LocalSystemVectorType,double> mRotationTool;

const Variable<int>& mrPeriodicIdVar;

Process::Pointer mpTurbulenceModel;

typename TSparseSpace::DofUpdaterPointerType mpDofUpdater = TSparseSpace::CreateDofUpdater();



























}; 









} 

#endif 
