
#if !defined(KRATOS_PORO_EXPLICIT_CD_SCHEME_HPP_INCLUDED)
#define KRATOS_PORO_EXPLICIT_CD_SCHEME_HPP_INCLUDED




#include "solving_strategies/schemes/scheme.h"
#include "utilities/variable_utils.h"

#include "poromechanics_application_variables.h"

namespace Kratos {








template <class TSparseSpace,
class TDenseSpace 
>
class PoroExplicitCDScheme
: public Scheme<TSparseSpace, TDenseSpace> {

public:

typedef Scheme<TSparseSpace, TDenseSpace> BaseType;

typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef ModelPart::ElementsContainerType ElementsArrayType;
typedef ModelPart::ConditionsContainerType ConditionsArrayType;
typedef ModelPart::NodesContainerType NodesArrayType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef typename ModelPart::NodeIterator NodeIterator;

static constexpr double numerical_limit = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION(PoroExplicitCDScheme);



PoroExplicitCDScheme()
: Scheme<TSparseSpace, TDenseSpace>() {}


virtual ~PoroExplicitCDScheme() {}



int Check(const ModelPart& rModelPart) const override
{
KRATOS_TRY;

BaseType::Check(rModelPart);

KRATOS_ERROR_IF(rModelPart.GetBufferSize() < 2) << "Insufficient buffer size for CD Scheme. It has to be >= 2" << std::endl;

KRATOS_ERROR_IF_NOT(rModelPart.GetProcessInfo().Has(DOMAIN_SIZE)) << "DOMAIN_SIZE not defined on ProcessInfo. Please define" << std::endl;

return 0;

KRATOS_CATCH("");
}


void Initialize(ModelPart& rModelPart) override
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

mDeltaTime = r_current_process_info[DELTA_TIME];
mAlpha = r_current_process_info[RAYLEIGH_ALPHA];
mBeta = r_current_process_info[RAYLEIGH_BETA];
mTheta = r_current_process_info[THETA_FACTOR];
mGCoefficient = r_current_process_info[G_COEFFICIENT];

const SizeType dim = r_current_process_info[DOMAIN_SIZE];

if (!BaseType::SchemeIsInitialized())
InitializeExplicitScheme(rModelPart, dim);
else
SchemeCustomInitialization(rModelPart, dim);

BaseType::SetSchemeIsInitialized();

KRATOS_CATCH("")
}

void InitializeElements( ModelPart& rModelPart) override
{
KRATOS_TRY

const ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();

int NElems = static_cast<int>(rModelPart.Elements().size());
ModelPart::ElementsContainerType::iterator el_begin = rModelPart.ElementsBegin();

for(int i = 0; i < NElems; i++)
{
ModelPart::ElementsContainerType::iterator itElem = el_begin + i;
itElem -> Initialize(CurrentProcessInfo);
}

this->SetElementsAreInitialized();

KRATOS_CATCH("")
}


virtual void InitializeExplicitScheme(
ModelPart& rModelPart,
const SizeType DomainSize = 3
)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const auto it_node_begin = rModelPart.NodesBegin();

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
auto it_node = (it_node_begin + i);
it_node->SetValue(NODAL_MASS, 0.0);
array_1d<double, 3>& r_force_residual = it_node->FastGetSolutionStepValue(FORCE_RESIDUAL);
double& r_flux_residual = it_node->FastGetSolutionStepValue(FLUX_RESIDUAL);
array_1d<double, 3>& r_external_force = it_node->FastGetSolutionStepValue(EXTERNAL_FORCE);
array_1d<double, 3>& r_internal_force = it_node->FastGetSolutionStepValue(INTERNAL_FORCE);
noalias(r_force_residual) = ZeroVector(3);
r_flux_residual = 0.0;
noalias(r_external_force) = ZeroVector(3);
noalias(r_internal_force) = ZeroVector(3);
Matrix& rInitialStress = it_node->FastGetSolutionStepValue(INITIAL_STRESS_TENSOR);
if(rInitialStress.size1() != 3)
rInitialStress.resize(3,3,false);
noalias(rInitialStress) = ZeroMatrix(3,3);
}

KRATOS_CATCH("")
}


virtual void SchemeCustomInitialization(
ModelPart& rModelPart,
const SizeType DomainSize = 3
)
{
KRATOS_TRY

KRATOS_CATCH("")
}


void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY

BaseType::InitializeSolutionStep(rModelPart, rA, rDx, rb);

InitializeResidual(rModelPart);

KRATOS_CATCH("")
}


virtual void InitializeResidual(ModelPart& rModelPart)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const array_1d<double, 3> zero_array = ZeroVector(3);
VariableUtils().SetVariable(FORCE_RESIDUAL, zero_array,r_nodes);
VariableUtils().SetVariable(FLUX_RESIDUAL, 0.0,r_nodes);
VariableUtils().SetVariable(EXTERNAL_FORCE, zero_array,r_nodes);
VariableUtils().SetVariable(INTERNAL_FORCE, zero_array,r_nodes);

KRATOS_CATCH("")
}



void InitializeNonLinIteration(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY;

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

const auto it_elem_begin = rModelPart.ElementsBegin();
#pragma omp parallel for schedule(guided,512)
for(int i=0; i<static_cast<int>(rModelPart.Elements().size()); ++i) {
auto it_elem = it_elem_begin + i;
it_elem->InitializeNonLinearIteration(r_current_process_info);
}

const auto it_cond_begin = rModelPart.ConditionsBegin();
#pragma omp parallel for schedule(guided,512)
for(int i=0; i<static_cast<int>(rModelPart.Conditions().size()); ++i) {
auto it_elem = it_cond_begin + i;
it_elem->InitializeNonLinearIteration(r_current_process_info);
}

KRATOS_CATCH( "" );
}

void Predict(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
{
KRATOS_TRY;

this->CalculateAndAddRHS(rModelPart);

KRATOS_CATCH("")
}

virtual void CalculateAndAddRHS(ModelPart& rModelPart)
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
ConditionsArrayType& r_conditions = rModelPart.Conditions();
ElementsArrayType& r_elements = rModelPart.Elements();

LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);
Element::EquationIdVectorType equation_id_vector_dummy; 

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_conditions.size()); ++i) {
auto it_cond = r_conditions.begin() + i;
CalculateRHSContribution(*it_cond, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
}

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_elements.size()); ++i) {
auto it_elem = r_elements.begin() + i;
CalculateRHSContribution(*it_elem, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
}

KRATOS_CATCH("")
}


void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) override
{
KRATOS_TRY

rCurrentElement.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, FORCE_RESIDUAL, rCurrentProcessInfo);

KRATOS_CATCH("")
}


void CalculateRHSContribution(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) override
{
KRATOS_TRY

rCurrentCondition.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);
rCurrentCondition.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, FORCE_RESIDUAL, rCurrentProcessInfo);

KRATOS_CATCH("")
}


void Update(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY
const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

NodesArrayType& r_nodes = rModelPart.Nodes();

const SizeType dim = r_current_process_info[DOMAIN_SIZE];

mDeltaTime = r_current_process_info[DELTA_TIME];

const auto it_node_begin = rModelPart.NodesBegin();

const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
this->UpdateTranslationalDegreesOfFreedom(it_node_begin + i, disppos, dim);
} 

KRATOS_CATCH("")
}


virtual void UpdateTranslationalDegreesOfFreedom(
NodeIterator itCurrentNode,
const IndexType DisplacementPosition,
const SizeType DomainSize = 3
)
{
array_1d<double, 3>& r_displacement = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3> displacement_aux;
noalias(displacement_aux) = r_displacement;
array_1d<double, 3>& r_displacement_old = itCurrentNode->FastGetSolutionStepValue(DISPLACEMENT_OLD);
const double nodal_mass = itCurrentNode->GetValue(NODAL_MASS);

double& r_current_water_pressure = itCurrentNode->FastGetSolutionStepValue(WATER_PRESSURE);
double& r_current_dt_water_pressure = itCurrentNode->FastGetSolutionStepValue(DT_WATER_PRESSURE);      

const array_1d<double, 3>& r_external_force = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE);
const array_1d<double, 3>& r_external_force_old = itCurrentNode->FastGetSolutionStepValue(EXTERNAL_FORCE,1);
const array_1d<double, 3>& r_internal_force = itCurrentNode->FastGetSolutionStepValue(INTERNAL_FORCE);
const array_1d<double, 3>& r_internal_force_old = itCurrentNode->FastGetSolutionStepValue(INTERNAL_FORCE,1);

std::array<bool, 3> fix_displacements = {false, false, false};
fix_displacements[0] = (itCurrentNode->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
fix_displacements[1] = (itCurrentNode->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
if (DomainSize == 3)
fix_displacements[2] = (itCurrentNode->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

for (IndexType j = 0; j < DomainSize; j++) {
if (fix_displacements[j] == false) {
r_displacement[j] = ( (2.0*(1.0+mGCoefficient*mDeltaTime)-mAlpha*mDeltaTime)*nodal_mass*r_displacement[j]
+ (mAlpha*mDeltaTime-(1.0+mGCoefficient*mDeltaTime))*nodal_mass*r_displacement_old[j]
- mDeltaTime*(mBeta+mTheta*mDeltaTime)*r_internal_force[j]
+ mDeltaTime*(mBeta-mDeltaTime*(1.0-mTheta))*r_internal_force_old[j]
+ mDeltaTime*mDeltaTime*(mTheta*r_external_force[j]+(1.0-mTheta)*r_external_force_old[j])
) / ( nodal_mass*(1.0+mGCoefficient*mDeltaTime) );
}
}

if( itCurrentNode->IsFixed(WATER_PRESSURE) == false ) {
r_current_water_pressure = 0.0;
r_current_dt_water_pressure = 0.0;
}

noalias(r_displacement_old) = displacement_aux;
const array_1d<double, 3>& r_velocity_old = itCurrentNode->FastGetSolutionStepValue(VELOCITY,1);
array_1d<double, 3>& r_velocity = itCurrentNode->FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3>& r_acceleration = itCurrentNode->FastGetSolutionStepValue(ACCELERATION);

noalias(r_velocity) = (1.0/mDeltaTime) * (r_displacement - r_displacement_old);
noalias(r_acceleration) = (1.0/mDeltaTime) * (r_velocity - r_velocity_old);
}


void FinalizeNonLinIteration(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
{
KRATOS_TRY

BaseType::FinalizeNonLinIteration(rModelPart, A, Dx, b);

this->CalculateAndAddRHSFinal(rModelPart);

KRATOS_CATCH("")
}

virtual void CalculateAndAddRHSFinal(ModelPart& rModelPart)
{
KRATOS_TRY

NodesArrayType& r_nodes = rModelPart.Nodes();

const array_1d<double, 3> zero_array = ZeroVector(3);
VariableUtils().SetVariable(FORCE_RESIDUAL, zero_array, r_nodes);
VariableUtils().SetVariable(FLUX_RESIDUAL, 0.0, r_nodes);

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
ConditionsArrayType& r_conditions = rModelPart.Conditions();
ElementsArrayType& r_elements = rModelPart.Elements();

LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);
Element::EquationIdVectorType equation_id_vector_dummy; 

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_conditions.size()); ++i) {
auto it_cond = r_conditions.begin() + i;
CalculateRHSContributionResidual(*it_cond, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
}

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_elements.size()); ++i) {
auto it_elem = r_elements.begin() + i;
CalculateRHSContributionResidual(*it_elem, RHS_Contribution, equation_id_vector_dummy, r_current_process_info);
}

KRATOS_CATCH("")
}


virtual void CalculateRHSContributionResidual(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) 
{
KRATOS_TRY

rCurrentElement.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, REACTION, rCurrentProcessInfo);
KRATOS_CATCH("")
}


virtual void CalculateRHSContributionResidual(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) 
{
KRATOS_TRY

rCurrentCondition.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);
rCurrentCondition.AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, REACTION, rCurrentProcessInfo);

KRATOS_CATCH("")
}

void FinalizeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb) override
{
KRATOS_TRY

if(rModelPart.GetProcessInfo()[NODAL_SMOOTHING] == true)
{
const int NNodes = static_cast<int>(rModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator node_begin = rModelPart.NodesBegin();

#pragma omp parallel for
for(int i = 0; i < NNodes; i++)
{
ModelPart::NodesContainerType::iterator itNode = node_begin + i;

itNode->FastGetSolutionStepValue(NODAL_AREA) = 0.0;
Matrix& rNodalStress = itNode->FastGetSolutionStepValue(NODAL_EFFECTIVE_STRESS_TENSOR);
if(rNodalStress.size1() != 3)
rNodalStress.resize(3,3,false);
noalias(rNodalStress) = ZeroMatrix(3,3);
array_1d<double,3>& r_nodal_grad_pressure = itNode->FastGetSolutionStepValue(NODAL_WATER_PRESSURE_GRADIENT);
noalias(r_nodal_grad_pressure) = ZeroVector(3);
itNode->FastGetSolutionStepValue(NODAL_DAMAGE_VARIABLE) = 0.0;
itNode->FastGetSolutionStepValue(NODAL_JOINT_AREA) = 0.0;
itNode->FastGetSolutionStepValue(NODAL_JOINT_WIDTH) = 0.0;
itNode->FastGetSolutionStepValue(NODAL_JOINT_DAMAGE) = 0.0;
}

BaseType::FinalizeSolutionStep(rModelPart, rA, rDx, rb);

#pragma omp parallel for
for(int n = 0; n < NNodes; n++)
{
ModelPart::NodesContainerType::iterator itNode = node_begin + n;

const double& NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
if (NodalArea>1.0e-20)
{
const double InvNodalArea = 1.0/NodalArea;
Matrix& rNodalStress = itNode->FastGetSolutionStepValue(NODAL_EFFECTIVE_STRESS_TENSOR);
array_1d<double,3>& r_nodal_grad_pressure = itNode->FastGetSolutionStepValue(NODAL_WATER_PRESSURE_GRADIENT);
for(unsigned int i = 0; i<3; i++)
{
r_nodal_grad_pressure[i] *= InvNodalArea;
for(unsigned int j = 0; j<3; j++)
{
rNodalStress(i,j) *= InvNodalArea;
}
}
double& NodalDamage = itNode->FastGetSolutionStepValue(NODAL_DAMAGE_VARIABLE);
NodalDamage *= InvNodalArea;
}

const double& NodalJointArea = itNode->FastGetSolutionStepValue(NODAL_JOINT_AREA);
if (NodalJointArea>1.0e-20)
{
const double InvNodalJointArea = 1.0/NodalJointArea;
double& NodalJointWidth = itNode->FastGetSolutionStepValue(NODAL_JOINT_WIDTH);
NodalJointWidth *= InvNodalJointArea;
double& NodalJointDamage = itNode->FastGetSolutionStepValue(NODAL_JOINT_DAMAGE);
NodalJointDamage *= InvNodalJointArea;
}
}
}
else
{
BaseType::FinalizeSolutionStep(rModelPart, rA, rDx, rb);
}

KRATOS_CATCH("")
}







protected:








double mDeltaTime;
double mAlpha;
double mBeta;
double mTheta;
double mGCoefficient;







private:








}; 




} 

#endif 
