

#if !defined(KRATOS_MPM_RESIDUAL_BASED_BOSSAK_SCHEME )
#define      KRATOS_MPM_RESIDUAL_BASED_BOSSAK_SCHEME






#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/variables.h"
#include "includes/element.h"
#include "containers/array_1d.h"
#include "solving_strategies/schemes/scheme.h"
#include "solving_strategies/schemes/residual_based_implicit_time_scheme.h"
#include "solving_strategies/schemes/residual_based_bossak_displacement_scheme.hpp"
#include "custom_utilities/mpm_boundary_rotation_utility.h"

namespace Kratos
{


template<class TSparseSpace,  class TDenseSpace >
class MPMResidualBasedBossakScheme
: public ResidualBasedBossakDisplacementScheme<TSparseSpace,TDenseSpace>
{
public:
KRATOS_CLASS_POINTER_DEFINITION( MPMResidualBasedBossakScheme );

typedef Scheme<TSparseSpace,TDenseSpace>                                      BaseType;

typedef ResidualBasedImplicitTimeScheme<TSparseSpace,TDenseSpace>     ImplicitBaseType;

typedef ResidualBasedBossakDisplacementScheme<TSparseSpace,TDenseSpace> BossakBaseType;

typedef typename BossakBaseType::TDataType                                   TDataType;

typedef typename BossakBaseType::DofsArrayType                           DofsArrayType;

typedef typename Element::DofsVectorType                                DofsVectorType;

typedef typename BossakBaseType::TSystemMatrixType                   TSystemMatrixType;

typedef typename BossakBaseType::TSystemVectorType                   TSystemVectorType;

typedef typename BossakBaseType::LocalSystemVectorType           LocalSystemVectorType;

typedef typename BossakBaseType::LocalSystemMatrixType           LocalSystemMatrixType;

typedef ModelPart::ElementsContainerType                             ElementsArrayType;

typedef ModelPart::ConditionsContainerType                         ConditionsArrayType;

typedef typename BaseType::Pointer                                     BaseTypePointer;

using BossakBaseType::mpDofUpdater;

using BossakBaseType::mBossak;

using ImplicitBaseType::mMatrix;



MPMResidualBasedBossakScheme(ModelPart& rGridModelPart, unsigned int DomainSize,
unsigned int BlockSize, double Alpha = 0.0,
double NewmarkBeta = 0.25, bool IsDynamic = true)
:ResidualBasedBossakDisplacementScheme<TSparseSpace,TDenseSpace>(Alpha, NewmarkBeta),
mGridModelPart(rGridModelPart), mRotationTool(DomainSize, BlockSize, IS_STRUCTURE)
{
mIsDynamic = IsDynamic;

mDomainSize = DomainSize;
mBlockSize  = BlockSize;
}


MPMResidualBasedBossakScheme(MPMResidualBasedBossakScheme& rOther)
:BossakBaseType(rOther)
,mGridModelPart(rOther.mGridModelPart)
,mRotationTool(rOther.mDomainSize,rOther.mBlockSize,IS_STRUCTURE)
{
}


BaseTypePointer Clone() override
{
return BaseTypePointer( new MPMResidualBasedBossakScheme(*this) );
}


virtual ~MPMResidualBasedBossakScheme
() {}


/
void Update(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb ) override
{
KRATOS_TRY

mRotationTool.RotateDisplacements(rModelPart);

mpDofUpdater->UpdateDofs(rDofSet, rDx);

mRotationTool.RecoverDisplacements(rModelPart);

const int num_nodes = static_cast<int>( rModelPart.Nodes().size() );
const auto it_node_begin = rModelPart.Nodes().begin();

#pragma omp parallel for
for(int i = 0;  i < num_nodes; ++i) {
auto it_node = it_node_begin + i;

const array_1d<double, 3 > & r_delta_displacement = it_node->FastGetSolutionStepValue(DISPLACEMENT);

array_1d<double, 3>& r_current_velocity = it_node->FastGetSolutionStepValue(VELOCITY);
const array_1d<double, 3>& r_previous_velocity = it_node->FastGetSolutionStepValue(VELOCITY, 1);

array_1d<double, 3>& r_current_acceleration = it_node->FastGetSolutionStepValue(ACCELERATION);
const array_1d<double, 3>& r_previous_acceleration = it_node->FastGetSolutionStepValue(ACCELERATION, 1);

if (mIsDynamic){
BossakBaseType::UpdateVelocity(r_current_velocity, r_delta_displacement, r_previous_velocity, r_previous_acceleration);
BossakBaseType::UpdateAcceleration(r_current_acceleration, r_delta_displacement, r_previous_velocity, r_previous_acceleration);
}
}

KRATOS_CATCH( "" )
}


void Predict(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
KRATOS_TRY;

#pragma omp parallel for
for(int iter = 0; iter < static_cast<int>(rModelPart.Nodes().size()); ++iter)
{
auto i = rModelPart.NodesBegin() + iter;
const array_1d<double, 3 > & r_previous_displacement = (i)->FastGetSolutionStepValue(DISPLACEMENT, 1);
const array_1d<double, 3 > & r_previous_velocity     = (i)->FastGetSolutionStepValue(VELOCITY, 1);
const array_1d<double, 3 > & r_previous_acceleration = (i)->FastGetSolutionStepValue(ACCELERATION, 1);

array_1d<double, 3 > & r_current_displacement  = (i)->FastGetSolutionStepValue(DISPLACEMENT);

if (!(i->pGetDof(DISPLACEMENT_X)->IsFixed()))
r_current_displacement[0] = 0.0;
else
r_current_displacement[0]  = r_previous_displacement[0];

if (!(i->pGetDof(DISPLACEMENT_Y)->IsFixed()))
r_current_displacement[1] = 0.0;
else
r_current_displacement[1]  = r_previous_displacement[1];

if (i->HasDofFor(DISPLACEMENT_Z))
{
if (!(i->pGetDof(DISPLACEMENT_Z)->IsFixed()))
r_current_displacement[2] = 0.0;
else
r_current_displacement[2]  = r_previous_displacement[2];
}

if (i->HasDofFor(PRESSURE))
{
double& r_current_pressure        = (i)->FastGetSolutionStepValue(PRESSURE);
const double& r_previous_pressure = (i)->FastGetSolutionStepValue(PRESSURE, 1);

if (!(i->pGetDof(PRESSURE))->IsFixed())
r_current_pressure = r_previous_pressure;
}

array_1d<double, 3 > & current_velocity       = (i)->FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3 > & current_acceleration   = (i)->FastGetSolutionStepValue(ACCELERATION);

if (mIsDynamic){
BossakBaseType::UpdateVelocity(current_velocity, r_current_displacement, r_previous_velocity, r_previous_acceleration);
BossakBaseType::UpdateAcceleration (current_acceleration, r_current_displacement, r_previous_velocity, r_previous_acceleration);
}

}

KRATOS_CATCH( "" );
}


void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
KRATOS_TRY

#pragma omp parallel for
for(int iter = 0; iter < static_cast<int>(mGridModelPart.Nodes().size()); ++iter)
{
auto i = mGridModelPart.NodesBegin() + iter;

double & r_nodal_mass     = (i)->FastGetSolutionStepValue(NODAL_MASS);
array_1d<double, 3 > & r_nodal_momentum = (i)->FastGetSolutionStepValue(NODAL_MOMENTUM);
array_1d<double, 3 > & r_nodal_inertia  = (i)->FastGetSolutionStepValue(NODAL_INERTIA);

array_1d<double, 3 > & r_nodal_displacement = (i)->FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3 > & r_nodal_velocity     = (i)->FastGetSolutionStepValue(VELOCITY,1);
array_1d<double, 3 > & r_nodal_acceleration = (i)->FastGetSolutionStepValue(ACCELERATION,1);

double & r_nodal_old_pressure = (i)->FastGetSolutionStepValue(PRESSURE,1);
double & r_nodal_pressure = (i)->FastGetSolutionStepValue(PRESSURE);

r_nodal_mass = 0.0;
r_nodal_momentum.clear();
r_nodal_inertia.clear();

r_nodal_displacement.clear();
r_nodal_velocity.clear();
r_nodal_acceleration.clear();
r_nodal_old_pressure = 0.0;
r_nodal_pressure = 0.0;

if ((i)->SolutionStepsDataHas(NODAL_AREA)){
double & r_nodal_area = (i)->FastGetSolutionStepValue(NODAL_AREA);
r_nodal_area          = 0.0;
}
if(i->SolutionStepsDataHas(NODAL_MPRESSURE)) {
double & r_nodal_mpressure = (i)->FastGetSolutionStepValue(NODAL_MPRESSURE);
r_nodal_mpressure = 0.0;
}
}

ImplicitBaseType::InitializeSolutionStep(rModelPart,rA,rDx,rb);

#pragma omp parallel for
for(int iter = 0; iter < static_cast<int>(mGridModelPart.Nodes().size()); ++iter)
{
auto i = mGridModelPart.NodesBegin() + iter;
const double & r_nodal_mass = (i)->FastGetSolutionStepValue(NODAL_MASS);

if (r_nodal_mass > std::numeric_limits<double>::epsilon())
{
const array_1d<double, 3 > & r_nodal_momentum   = (i)->FastGetSolutionStepValue(NODAL_MOMENTUM);
const array_1d<double, 3 > & r_nodal_inertia    = (i)->FastGetSolutionStepValue(NODAL_INERTIA);

array_1d<double, 3 > & r_nodal_velocity     = (i)->FastGetSolutionStepValue(VELOCITY,1);
array_1d<double, 3 > & r_nodal_acceleration = (i)->FastGetSolutionStepValue(ACCELERATION,1);
double & r_nodal_pressure = (i)->FastGetSolutionStepValue(PRESSURE,1);

double delta_nodal_pressure = 0.0;

if (i->HasDofFor(PRESSURE) && i->SolutionStepsDataHas(NODAL_MPRESSURE))
{
double & nodal_mpressure = (i)->FastGetSolutionStepValue(NODAL_MPRESSURE);
delta_nodal_pressure = nodal_mpressure/r_nodal_mass;
}

const array_1d<double, 3 > delta_nodal_velocity = r_nodal_momentum/r_nodal_mass;
const array_1d<double, 3 > delta_nodal_acceleration = r_nodal_inertia/r_nodal_mass;

r_nodal_velocity += delta_nodal_velocity;
r_nodal_acceleration += delta_nodal_acceleration;

r_nodal_pressure += delta_nodal_pressure;
}
}

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
const double delta_time = r_current_process_info[DELTA_TIME];

mBossak.c0 = ( 1.0 / (mBossak.beta * delta_time * delta_time) );
mBossak.c1 = ( mBossak.gamma / (mBossak.beta * delta_time) );
mBossak.c2 = ( 1.0 / (mBossak.beta * delta_time) );
mBossak.c3 = ( 0.5 / (mBossak.beta) - 1.0 );
mBossak.c4 = ( (mBossak.gamma / mBossak.beta) - 1.0  );
mBossak.c5 = ( delta_time * 0.5 * ( ( mBossak.gamma / mBossak.beta ) - 2.0 ) );

KRATOS_CATCH( "" )
}


void CalculateSystemContributions(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_TRY

const IndexType this_thread = OpenMPUtils::ThisThread();
const auto& rConstElemRef = rCurrentElement;
rCurrentElement.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,rCurrentProcessInfo);
rConstElemRef.EquationIdVector(EquationId,rCurrentProcessInfo);

if(mIsDynamic)
{
rCurrentElement.CalculateMassMatrix(mMatrix.M[this_thread],rCurrentProcessInfo);
rCurrentElement.CalculateDampingMatrix(mMatrix.D[this_thread],rCurrentProcessInfo);
BossakBaseType::AddDynamicsToLHS(LHS_Contribution, mMatrix.D[this_thread], mMatrix.M[this_thread], rCurrentProcessInfo);
BossakBaseType::AddDynamicsToRHS(rCurrentElement, RHS_Contribution, mMatrix.D[this_thread], mMatrix.M[this_thread], rCurrentProcessInfo);
}

mRotationTool.Rotate(LHS_Contribution,RHS_Contribution,rCurrentElement.GetGeometry());
mRotationTool.ElementApplySlipCondition(LHS_Contribution,RHS_Contribution,rCurrentElement.GetGeometry());

KRATOS_CATCH( "" )
}


void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo) override
{

KRATOS_TRY

const IndexType this_thread = OpenMPUtils::ThisThread();
const auto& r_const_elem_ref = rCurrentElement;

rCurrentElement.CalculateRightHandSide(RHS_Contribution,rCurrentProcessInfo);
r_const_elem_ref.EquationIdVector(EquationId,rCurrentProcessInfo);

if(mIsDynamic)
{
rCurrentElement.CalculateMassMatrix(mMatrix.M[this_thread],rCurrentProcessInfo);
rCurrentElement.CalculateDampingMatrix(mMatrix.D[this_thread],rCurrentProcessInfo);
BossakBaseType::AddDynamicsToRHS(rCurrentElement, RHS_Contribution, mMatrix.D[this_thread], mMatrix.M[this_thread], rCurrentProcessInfo);
}

mRotationTool.RotateRHS(RHS_Contribution,rCurrentElement.GetGeometry());
mRotationTool.ElementApplySlipCondition(RHS_Contribution,rCurrentElement.GetGeometry());

KRATOS_CATCH( "" )
}


void CalculateSystemContributions(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo) override
{

KRATOS_TRY

const IndexType this_thread = OpenMPUtils::ThisThread();
const auto& r_const_cond_ref = rCurrentCondition;

rCurrentCondition.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,rCurrentProcessInfo);
r_const_cond_ref.EquationIdVector(EquationId,rCurrentProcessInfo);

if(mIsDynamic)
{
rCurrentCondition.CalculateMassMatrix(mMatrix.M[this_thread],rCurrentProcessInfo);
rCurrentCondition.CalculateDampingMatrix(mMatrix.D[this_thread],rCurrentProcessInfo);
BossakBaseType::AddDynamicsToLHS(LHS_Contribution, mMatrix.D[this_thread], mMatrix.M[this_thread], rCurrentProcessInfo);
BossakBaseType::AddDynamicsToRHS(rCurrentCondition, RHS_Contribution, mMatrix.D[this_thread], mMatrix.M[this_thread], rCurrentProcessInfo);
}

mRotationTool.Rotate(LHS_Contribution,RHS_Contribution,rCurrentCondition.GetGeometry());
mRotationTool.ConditionApplySlipCondition(LHS_Contribution,RHS_Contribution,rCurrentCondition.GetGeometry());

KRATOS_CATCH( "" )
}


void CalculateRHSContribution(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_TRY

const IndexType this_thread = OpenMPUtils::ThisThread();
const auto& r_const_cond_ref = rCurrentCondition;
rCurrentCondition.CalculateRightHandSide(RHS_Contribution,rCurrentProcessInfo);
r_const_cond_ref.EquationIdVector(EquationId,rCurrentProcessInfo);

if(mIsDynamic)
{
rCurrentCondition.CalculateMassMatrix(mMatrix.M[this_thread],rCurrentProcessInfo);
rCurrentCondition.CalculateDampingMatrix(mMatrix.D[this_thread],rCurrentProcessInfo);
BossakBaseType::AddDynamicsToRHS(rCurrentCondition, RHS_Contribution, mMatrix.D[this_thread], mMatrix.M[this_thread], rCurrentProcessInfo);
}

mRotationTool.RotateRHS(RHS_Contribution,rCurrentCondition.GetGeometry());
mRotationTool.ConditionApplySlipCondition(RHS_Contribution,rCurrentCondition.GetGeometry());

KRATOS_CATCH( "" )
}

protected:

ModelPart& mGridModelPart;

bool mIsDynamic;

unsigned int mDomainSize;
unsigned int mBlockSize;
MPMBoundaryRotationUtility<LocalSystemMatrixType,LocalSystemVectorType> mRotationTool;

}; 
}  

#endif 

