

#if !defined(KRATOS_MPM_EXPLICIT_STRATEGY )
#define  KRATOS_MPM_EXPLICIT_STRATEGY



#include "includes/define.h"
#include "includes/model_part.h"
#include "utilities/variable_utils.h"
#include "includes/kratos_flags.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"

#include "particle_mechanics_application_variables.h"

namespace Kratos
{

template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
class MPMExplicitStrategy
: public ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>
{
public:
KRATOS_CLASS_POINTER_DEFINITION(MPMExplicitStrategy);

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef TSparseSpace SparseSpaceType;

typedef typename BaseType::TSchemeType TSchemeType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;

typedef typename BaseType::ElementsArrayType ElementsArrayType;

typedef typename BaseType::NodesArrayType NodesArrayType;

typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

explicit MPMExplicitStrategy(
ModelPart& model_part,
typename TSchemeType::Pointer pScheme,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false
)
: ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(model_part, MoveMeshFlag),
mpScheme(pScheme),
mReformDofSetAtEachStep(ReformDofSetAtEachStep),
mCalculateReactionsFlag(CalculateReactions)
{
KRATOS_TRY

mKeepSystemConstantDuringIterations = false;
mSolutionStepIsInitialized = false;
mInitializeWasPerformed = false;
mFinalizeSolutionStep = true;
this->SetEchoLevel(1);
this->SetRebuildLevel(1);

KRATOS_CATCH("")
}

virtual ~MPMExplicitStrategy()
{
}


void SetScheme(typename TSchemeType::Pointer pScheme)
{
mpScheme = pScheme;
};

typename TSchemeType::Pointer GetScheme()
{
return mpScheme;
};

void Initialize() override
{
KRATOS_TRY

typename TSchemeType::Pointer pScheme = GetScheme();

if (mInitializeWasPerformed == false)
{
KRATOS_INFO_IF("MPM_Explicit_Strategy", this->GetEchoLevel() > 1) << "Initializing solving strategy" << std::endl;
KRATOS_ERROR_IF(mInitializeWasPerformed == true) << "Initialization was already performed " << mInitializeWasPerformed << std::endl;

KRATOS_INFO_IF("MPM_Explicit_Strategy", this->GetEchoLevel() > 1) << "Initializing scheme" << std::endl;
if (pScheme->SchemeIsInitialized() == false)
pScheme->Initialize(BaseType::GetModelPart());

KRATOS_INFO_IF("MPM_Explicit_Strategy", this->GetEchoLevel() > 1) << "Initializing elements" << std::endl;
if (pScheme->ElementsAreInitialized() == false)
pScheme->InitializeElements(BaseType::GetModelPart());

KRATOS_INFO_IF("MPM_Explicit_Strategy", this->GetEchoLevel() > 1) << "Initializing conditions" << std::endl;
if (pScheme->ConditionsAreInitialized() == false)
pScheme->InitializeConditions(BaseType::GetModelPart());

mInitializeWasPerformed = true;
}

KRATOS_INFO_IF("MPM_Explicit_Strategy", this->GetEchoLevel() == 2) << "CurrentTime = " << BaseType::GetModelPart().GetProcessInfo()[TIME] << std::endl;

KRATOS_CATCH("")
}

void InitializeSolutionStep() override
{
KRATOS_TRY

if (!mSolutionStepIsInitialized)
{
typename TSchemeType::Pointer pScheme = GetScheme();
TSystemMatrixType mA = TSystemMatrixType();
TSystemVectorType mDx = TSystemVectorType();
TSystemVectorType mb = TSystemVectorType();

pScheme->InitializeSolutionStep(BaseType::GetModelPart(), mA, mDx, mb);
}

mSolutionStepIsInitialized = true;

KRATOS_INFO_IF("MPM_Explicit_Strategy", this->GetEchoLevel() >= 3) << "Initialize Solution Step in strategy finished" << std::endl;

KRATOS_CATCH("")
}


bool SolveSolutionStep() override
{
typename TSchemeType::Pointer pScheme = GetScheme();
DofsArrayType dof_set_dummy;
TSystemMatrixType mA = TSystemMatrixType();
TSystemVectorType mDx = TSystemVectorType();
TSystemVectorType mb = TSystemVectorType();

this->CalculateAndAddRHS(pScheme, BaseType::GetModelPart());
pScheme->Update(BaseType::GetModelPart(), dof_set_dummy, mA, mDx, mb);

if (mCalculateReactionsFlag) CalculateReactions(pScheme, BaseType::GetModelPart(), mA, mDx, mb);

return true;
}

/


typename TSchemeType::Pointer mpScheme;


bool mReformDofSetAtEachStep;

bool mCalculateReactionsFlag;

bool mSolutionStepIsInitialized;

bool mInitializeWasPerformed;

bool mKeepSystemConstantDuringIterations;

bool mFinalizeSolutionStep;

/
if (mFinalizeSolutionStep)
{
KRATOS_INFO_IF("MPM_Explicit_Strategy", this->GetEchoLevel() >= 3) << "Calling FinalizeSolutionStep" << std::endl;

pScheme->FinalizeSolutionStep(BaseType::GetModelPart(), mA, mDx, mb);
if (BaseType::MoveMeshFlag()) BaseType::MoveMesh();
}

pScheme->Clean();

mSolutionStepIsInitialized = false;
KRATOS_CATCH("")
}


int Check() override
{
KRATOS_TRY

BaseType::Check();
GetScheme()->Check(BaseType::GetModelPart());
return 0;

KRATOS_CATCH("")
}


void CalculateAndAddRHS(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart
)
{
KRATOS_TRY

ConditionsArrayType& r_conditions = rModelPart.Conditions();
ElementsArrayType& r_elements = rModelPart.Elements();

LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);
Element::EquationIdVectorType equation_id_vector_dummy; 

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_conditions.size()); ++i) {
auto it_cond = r_conditions.begin() + i;
pScheme->CalculateRHSContribution(*it_cond, RHS_Contribution, equation_id_vector_dummy, rModelPart.GetProcessInfo());
}

#pragma omp parallel for firstprivate(RHS_Contribution, equation_id_vector_dummy), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_elements.size()); ++i) {
auto it_elem = r_elements.begin() + i;
pScheme->CalculateRHSContribution(*it_elem, RHS_Contribution, equation_id_vector_dummy, rModelPart.GetProcessInfo());
}

KRATOS_CATCH("")
}


void CalculateReactions(
typename TSchemeType::Pointer pScheme,
ModelPart& rModelPart,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
)
{
auto& r_nodes = rModelPart.Nodes();

const bool has_dof_for_rot_z = (r_nodes.begin())->HasDofFor(ROTATION_Z);

const array_1d<double, 3> zero_array = ZeroVector(3);
array_1d<double, 3> force_residual = ZeroVector(3);
array_1d<double, 3> moment_residual = ZeroVector(3);

const auto it_node_begin = r_nodes.begin();
const IndexType disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const IndexType rotppos = it_node_begin->GetDofPosition(ROTATION_X);

#pragma omp parallel for firstprivate(force_residual, moment_residual), schedule(guided,512)
for (int i = 0; i < static_cast<int>(r_nodes.size()); ++i) {
auto it_node = it_node_begin + i;

noalias(force_residual) = it_node->FastGetSolutionStepValue(FORCE_RESIDUAL);
if (has_dof_for_rot_z) {
noalias(moment_residual) = it_node->FastGetSolutionStepValue(MOMENT_RESIDUAL);
}
else {
noalias(moment_residual) = zero_array;
}

if (it_node->GetDof(DISPLACEMENT_X, disppos).IsFixed()) {
double& r_reaction = it_node->FastGetSolutionStepValue(REACTION_X);
r_reaction = force_residual[0];
}
if (it_node->GetDof(DISPLACEMENT_Y, disppos + 1).IsFixed()) {
double& r_reaction = it_node->FastGetSolutionStepValue(REACTION_Y);
r_reaction = force_residual[1];
}
if (it_node->GetDof(DISPLACEMENT_Z, disppos + 2).IsFixed()) {
double& r_reaction = it_node->FastGetSolutionStepValue(REACTION_Z);
r_reaction = force_residual[2];
}
if (has_dof_for_rot_z) {
if (it_node->GetDof(ROTATION_X, rotppos).IsFixed()) {
double& r_reaction = it_node->FastGetSolutionStepValue(REACTION_MOMENT_X);
r_reaction = moment_residual[0];
}
if (it_node->GetDof(ROTATION_Y, rotppos + 1).IsFixed()) {
double& r_reaction = it_node->FastGetSolutionStepValue(REACTION_MOMENT_Y);
r_reaction = moment_residual[1];
}
if (it_node->GetDof(ROTATION_Z, rotppos + 2).IsFixed()) {
double& r_reaction = it_node->FastGetSolutionStepValue(REACTION_MOMENT_Z);
r_reaction = moment_residual[2];
}
}
}
}

MPMExplicitStrategy(const MPMExplicitStrategy& Other)
{
};
}; 
}; 

#endif 
