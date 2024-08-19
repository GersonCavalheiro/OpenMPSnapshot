
#if !defined(KRATOS_MPM_EXPLICIT_SCHEME)
#define KRATOS_MPM_EXPLICIT_SCHEME



#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/variables.h"
#include "solving_strategies/schemes/scheme.h"
#include "custom_utilities/mpm_boundary_rotation_utility.h"
#include "custom_utilities/mpm_explicit_utilities.h"

namespace Kratos {

template <class TSparseSpace,
class TDenseSpace 
>
class MPMExplicitScheme
: public Scheme<TSparseSpace, TDenseSpace> {

public:



KRATOS_CLASS_POINTER_DEFINITION(MPMExplicitScheme);

typedef Scheme<TSparseSpace, TDenseSpace>                      BaseType;

typedef typename BaseType::TDataType                         TDataType;

typedef typename BaseType::DofsArrayType                 DofsArrayType;

typedef typename Element::DofsVectorType                DofsVectorType;

typedef typename BaseType::TSystemMatrixType         TSystemMatrixType;

typedef typename BaseType::TSystemVectorType         TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef ModelPart::ElementsContainerType             ElementsArrayType;

typedef ModelPart::ConditionsContainerType         ConditionsArrayType;

typedef typename BaseType::Pointer                     BaseTypePointer;

typedef ModelPart::NodesContainerType NodesArrayType;

typedef typename ModelPart::NodeIterator NodeIterator;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

static constexpr double numerical_limit = std::numeric_limits<double>::epsilon();


explicit MPMExplicitScheme(
ModelPart& grid_model_part
)
: Scheme<TSparseSpace, TDenseSpace>(),
mr_grid_model_part(grid_model_part)
{
}

virtual ~MPMExplicitScheme() {}


BaseTypePointer Clone() override
{
return BaseTypePointer(new MPMExplicitScheme(*this));
}


void Initialize(ModelPart& rModelPart) override
{
KRATOS_TRY
BaseType::SetSchemeIsInitialized();
KRATOS_CATCH("")
}

/
void Update(
ModelPart& r_model_part,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY
const ProcessInfo& r_current_process_info = r_model_part.GetProcessInfo();

const SizeType dim = r_current_process_info[DOMAIN_SIZE];
const double delta_time = r_current_process_info[DELTA_TIME];

const auto it_node_begin = r_model_part.NodesBegin();

const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);

#pragma omp parallel for schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_model_part.Nodes().size()); ++i) {
auto it_node = it_node_begin + i;
if ((it_node)->Is(ACTIVE))
{
this->UpdateTranslationalDegreesOfFreedom(r_current_process_info, it_node, disppos, delta_time, dim);
}
} 

KRATOS_CATCH("")
}

/
void InitializeSolutionStep(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

const ProcessInfo& rCurrentProcessInfo = r_model_part.GetProcessInfo();
BaseType::InitializeSolutionStep(r_model_part, A, Dx, b);
#pragma omp parallel for
for (int iter = 0; iter < static_cast<int>(mr_grid_model_part.Nodes().size()); ++iter)
{
auto i = mr_grid_model_part.NodesBegin() + iter;

if(i->Is(ACTIVE))
{
double& nodal_mass = (i)->FastGetSolutionStepValue(NODAL_MASS);
array_1d<double, 3 >& nodal_momentum = (i)->FastGetSolutionStepValue(NODAL_MOMENTUM);
array_1d<double, 3 >& nodal_inertia = (i)->FastGetSolutionStepValue(NODAL_INERTIA);
array_1d<double, 3 >& nodal_force = (i)->FastGetSolutionStepValue(FORCE_RESIDUAL);
array_1d<double, 3 >& nodal_displacement = (i)->FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3 >& nodal_velocity = (i)->FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3 >& nodal_acceleration = (i)->FastGetSolutionStepValue(ACCELERATION);

double& nodal_old_pressure = (i)->FastGetSolutionStepValue(PRESSURE, 1);
double& nodal_pressure = (i)->FastGetSolutionStepValue(PRESSURE);
if (i->SolutionStepsDataHas(NODAL_MPRESSURE)) {
double& nodal_mpressure = (i)->FastGetSolutionStepValue(NODAL_MPRESSURE);
nodal_mpressure = 0.0;
}

nodal_mass = 0.0;
nodal_momentum.clear();
nodal_inertia.clear();
nodal_force.clear();

nodal_displacement.clear();
nodal_velocity.clear();
nodal_acceleration.clear();
nodal_old_pressure = 0.0;
nodal_pressure = 0.0;
}
}

Scheme<TSparseSpace, TDenseSpace>::InitializeSolutionStep(r_model_part, A, Dx, b);

if (rCurrentProcessInfo.GetValue(EXPLICIT_STRESS_UPDATE_OPTION) == 0 ||
rCurrentProcessInfo.GetValue(IS_EXPLICIT_CENTRAL_DIFFERENCE))
{
calculateGridVelocityAndApplyDirichletBC(rCurrentProcessInfo,true);

const auto it_elem_begin = r_model_part.ElementsBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(r_model_part.Elements().size()); ++i) {
auto it_elem = it_elem_begin + i;
std::vector<bool> dummy;
it_elem->CalculateOnIntegrationPoints(CALCULATE_EXPLICIT_MP_STRESS, dummy, rCurrentProcessInfo);
}
}
KRATOS_CATCH("")
}

void calculateGridVelocityAndApplyDirichletBC(
const ProcessInfo rCurrentProcessInfo,
bool calculateVelocityFromMomenta = false)
{
KRATOS_TRY

const IndexType DisplacementPosition = mr_grid_model_part.NodesBegin()->GetDofPosition(DISPLACEMENT_X);
const SizeType DomainSize = rCurrentProcessInfo[DOMAIN_SIZE];

#pragma omp parallel for
for (int iter = 0; iter < static_cast<int>(mr_grid_model_part.Nodes().size()); ++iter)
{
NodeIterator i = mr_grid_model_part.NodesBegin() + iter;

if ((i)->Is(ACTIVE))
{
double& nodal_mass = (i)->FastGetSolutionStepValue(NODAL_MASS);
array_1d<double, 3 >& nodal_momentum = (i)->FastGetSolutionStepValue(NODAL_MOMENTUM);
array_1d<double, 3 >& nodal_velocity = (i)->FastGetSolutionStepValue(VELOCITY);

std::array<bool, 3> fix_displacements = { false, false, false };
fix_displacements[0] = (i->GetDof(DISPLACEMENT_X, DisplacementPosition).IsFixed());
fix_displacements[1] = (i->GetDof(DISPLACEMENT_Y, DisplacementPosition + 1).IsFixed());
if (DomainSize == 3)
fix_displacements[2] = (i->GetDof(DISPLACEMENT_Z, DisplacementPosition + 2).IsFixed());

for (IndexType j = 0; j < DomainSize; j++)
{
if (fix_displacements[j])
{
nodal_velocity[j] = 0.0;
}
else if (calculateVelocityFromMomenta && nodal_mass > numerical_limit)
{
nodal_velocity[j] = nodal_momentum[j] / nodal_mass;
}
}
}
}

KRATOS_CATCH("")
}



void FinalizeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

ElementsArrayType& rElements = rModelPart.Elements();
const ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();
const auto it_elem_begin = rModelPart.ElementsBegin();


#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rElements.size()); ++i)
{
auto it_elem = it_elem_begin + i;
std::vector<bool> dummy;
it_elem->CalculateOnIntegrationPoints(EXPLICIT_MAP_GRID_TO_MP, dummy, rCurrentProcessInfo);
}

if (rCurrentProcessInfo.GetValue(EXPLICIT_STRESS_UPDATE_OPTION) > 0)
{
this->CalculateUpdatedGridVelocityField(rCurrentProcessInfo, rModelPart);

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rElements.size()); ++i)
{
auto it_elem = it_elem_begin + i;
std::vector<bool> dummy;
it_elem->CalculateOnIntegrationPoints(CALCULATE_EXPLICIT_MP_STRESS, dummy, rCurrentProcessInfo);
}
}

const auto it_cond_begin = rModelPart.ConditionsBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rModelPart.Conditions().size()); ++i) {
auto it_cond = it_cond_begin + i;
it_cond->FinalizeSolutionStep(rCurrentProcessInfo);
}

KRATOS_CATCH("")
}
/
void GetDofList(
const Element& rCurrentElement,
Element::DofsVectorType& ElementalDofList,
const ProcessInfo& CurrentProcessInfo) override
{
rCurrentElement.GetDofList(ElementalDofList, CurrentProcessInfo);
}

/
void GetDofList(
const Condition& rCurrentCondition,
Element::DofsVectorType& rConditionDofList,
const ProcessInfo& rCurrentProcessInfo) override
{
rCurrentCondition.GetDofList(rConditionDofList, rCurrentProcessInfo);
}

/
int Check(const ModelPart& rModelPart) const override
{
KRATOS_TRY

int err = Scheme<TSparseSpace, TDenseSpace>::Check(rModelPart);
if (err != 0) return err;

for (auto it = rModelPart.NodesBegin();
it != rModelPart.NodesEnd(); ++it)
{
KRATOS_ERROR_IF(it->SolutionStepsDataHas(DISPLACEMENT) == false) << "DISPLACEMENT variable is not allocated for node " << it->Id() << std::endl;
KRATOS_ERROR_IF(it->SolutionStepsDataHas(VELOCITY) == false) << "VELOCITY variable is not allocated for node " << it->Id() << std::endl;
KRATOS_ERROR_IF(it->SolutionStepsDataHas(ACCELERATION) == false) << "ACCELERATION variable is not allocated for node " << it->Id() << std::endl;
}

for (auto it = rModelPart.NodesBegin();
it != rModelPart.NodesEnd(); ++it)
{
KRATOS_ERROR_IF(it->HasDofFor(DISPLACEMENT_X) == false) << "Missing DISPLACEMENT_X dof on node " << it->Id() << std::endl;
KRATOS_ERROR_IF(it->HasDofFor(DISPLACEMENT_Y) == false) << "Missing DISPLACEMENT_Y dof on node " << it->Id() << std::endl;
KRATOS_ERROR_IF(it->HasDofFor(DISPLACEMENT_Z) == false) << "Missing DISPLACEMENT_Z dof on node " << it->Id() << std::endl;
}

KRATOS_ERROR_IF(rModelPart.GetBufferSize() < 2) << "Insufficient buffer size. Buffer size should be greater than 2. Current size is" << rModelPart.GetBufferSize() << std::endl;
KRATOS_CATCH("")
return 0;
}

void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& rCurrentProcessInfo
) override
{
KRATOS_TRY
rCurrentElement.CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);
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

protected:
ModelPart& mr_grid_model_part;

private:
}; 
}  

#endif 
