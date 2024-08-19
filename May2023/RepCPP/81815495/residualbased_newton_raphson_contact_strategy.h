
#pragma once



#include "contact_structural_mechanics_application_variables.h"
#include "includes/kratos_parameters.h"
#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/variables.h"


#include "solving_strategies/strategies/residualbased_newton_raphson_strategy.h"


#include "utilities/variable_utils.h"
#include "utilities/color_utilities.h"
#include "utilities/math_utils.h"
#include "custom_python/process_factory_utility.h"
#include "custom_utilities/contact_utilities.h"

namespace Kratos {







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ResidualBasedNewtonRaphsonContactStrategy :
public ResidualBasedNewtonRaphsonStrategy< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:


KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedNewtonRaphsonContactStrategy );

typedef SolvingStrategy<TSparseSpace, TDenseSpace>                        SolvingStrategyType;

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>    StrategyBaseType;

typedef ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef ResidualBasedNewtonRaphsonContactStrategy<TSparseSpace, TDenseSpace, TLinearSolver> ClassType;

typedef ConvergenceCriteria<TSparseSpace, TDenseSpace>               TConvergenceCriteriaType;

typedef typename BaseType::TBuilderAndSolverType                        TBuilderAndSolverType;

typedef typename BaseType::TDataType                                                TDataType;

typedef TSparseSpace                                                          SparseSpaceType;

typedef typename BaseType::TSchemeType                                            TSchemeType;

typedef typename BaseType::DofsArrayType                                        DofsArrayType;

typedef typename BaseType::TSystemMatrixType                                TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                                TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType                        LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType                        LocalSystemMatrixType;

typedef typename BaseType::TSystemMatrixPointerType                  TSystemMatrixPointerType;

typedef typename BaseType::TSystemVectorPointerType                  TSystemVectorPointerType;

typedef ModelPart::NodesContainerType                                          NodesArrayType;

typedef ModelPart::ElementsContainerType                                    ElementsArrayType;

typedef ModelPart::ConditionsContainerType                                ConditionsArrayType;

typedef ProcessFactoryUtility::Pointer                                      ProcessesListType;

typedef std::size_t                                                                 IndexType;


explicit ResidualBasedNewtonRaphsonContactStrategy()
{
}


explicit ResidualBasedNewtonRaphsonContactStrategy(ModelPart& rModelPart, Parameters ThisParameters)
: BaseType(rModelPart),
mpMyProcesses(nullptr),
mpPostProcesses(nullptr)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);

mConvergenceCriteriaEchoLevel = BaseType::mpConvergenceCriteria->GetEchoLevel();
}


ResidualBasedNewtonRaphsonContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})"),
ProcessesListType pMyProcesses = nullptr,
ProcessesListType pPostProcesses = nullptr
)
: BaseType(rModelPart, pScheme, pNewConvergenceCriteria, pNewBuilderAndSolver, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag ),
mThisParameters(ThisParameters),
mpMyProcesses(pMyProcesses),
mpPostProcesses(pPostProcesses)
{
KRATOS_TRY;

mConvergenceCriteriaEchoLevel = pNewConvergenceCriteria->GetEchoLevel();

Parameters default_parameters = GetDefaultParameters();
mThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}


ResidualBasedNewtonRaphsonContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TLinearSolver::Pointer pNewLinearSolver,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})"),
ProcessesListType pMyProcesses = nullptr,
ProcessesListType pPostProcesses = nullptr
)
: BaseType(rModelPart, pScheme, pNewLinearSolver, pNewConvergenceCriteria, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag),
mThisParameters(ThisParameters),
mpMyProcesses(pMyProcesses),
mpPostProcesses(pPostProcesses)
{
KRATOS_TRY;

mConvergenceCriteriaEchoLevel = pNewConvergenceCriteria->GetEchoLevel();

Parameters default_parameters = GetDefaultParameters();
mThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}


ResidualBasedNewtonRaphsonContactStrategy(
ModelPart& rModelPart,
typename TSchemeType::Pointer pScheme,
typename TLinearSolver::Pointer pNewLinearSolver,
typename TConvergenceCriteriaType::Pointer pNewConvergenceCriteria,
typename TBuilderAndSolverType::Pointer pNewBuilderAndSolver,
IndexType MaxIterations = 30,
bool CalculateReactions = false,
bool ReformDofSetAtEachStep = false,
bool MoveMeshFlag = false,
Parameters ThisParameters =  Parameters(R"({})"),
ProcessesListType pMyProcesses = nullptr,
ProcessesListType pPostProcesses = nullptr
)
: ResidualBasedNewtonRaphsonStrategy<TSparseSpace, TDenseSpace, TLinearSolver>(rModelPart, pScheme, pNewLinearSolver, pNewConvergenceCriteria, pNewBuilderAndSolver, MaxIterations, CalculateReactions, ReformDofSetAtEachStep, MoveMeshFlag ),
mThisParameters(ThisParameters),
mpMyProcesses(pMyProcesses),
mpPostProcesses(pPostProcesses)
{
KRATOS_TRY;

mConvergenceCriteriaEchoLevel = pNewConvergenceCriteria->GetEchoLevel();

Parameters default_parameters = GetDefaultParameters();
mThisParameters.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}


~ResidualBasedNewtonRaphsonContactStrategy() override
= default;




typename SolvingStrategyType::Pointer Create(
ModelPart& rModelPart,
Parameters ThisParameters
) const override
{
return Kratos::make_shared<ClassType>(rModelPart, ThisParameters);
}


void Predict() override
{
KRATOS_TRY

const array_1d<double, 3> zero_array = ZeroVector(3);

ModelPart& r_model_part = StrategyBaseType::GetModelPart();
NodesArrayType& r_nodes_array = r_model_part.GetSubModelPart("Contact").Nodes();
const bool frictional = r_model_part.Is(SLIP);

if (r_nodes_array.begin()->SolutionStepsDataHas(WEIGHTED_GAP)) {
VariableUtils().SetVariable(WEIGHTED_GAP, 0.0, r_nodes_array);
if (frictional) {
VariableUtils().SetVariable(WEIGHTED_SLIP, zero_array, r_nodes_array);
}

ContactUtilities::ComputeExplicitContributionConditions(r_model_part.GetSubModelPart("ComputingContact"));

ProcessInfo& r_process_info = r_model_part.GetProcessInfo();
const std::size_t step = r_process_info[STEP];

if (step == 1) {
block_for_each(r_nodes_array, [&](NodeType& rNode) {
noalias(rNode.Coordinates()) += rNode.FastGetSolutionStepValue(DISPLACEMENT);
});
} else {
block_for_each(r_nodes_array, [&](NodeType& rNode) {
noalias(rNode.Coordinates()) += (rNode.FastGetSolutionStepValue(DISPLACEMENT) - rNode.FastGetSolutionStepValue(DISPLACEMENT, 1));
});
}
}


KRATOS_CATCH("")
}



void Initialize() override
{
KRATOS_TRY;

BaseType::Initialize();
mFinalizeWasPerformed = false;

ModelPart& r_model_part = StrategyBaseType::GetModelPart();
ProcessInfo& r_process_info = r_model_part.GetProcessInfo();
r_process_info[NL_ITERATION_NUMBER] = 1;

KRATOS_CATCH("");
}


double Solve() override
{
this->Initialize();
this->InitializeSolutionStep();
this->Predict();
this->SolveSolutionStep();
this->FinalizeSolutionStep();


return 0.0;
}



void InitializeSolutionStep() override
{
BaseType::mpConvergenceCriteria->SetEchoLevel(0);
BaseType::InitializeSolutionStep();
BaseType::mpConvergenceCriteria->SetEchoLevel(mConvergenceCriteriaEchoLevel);

mFinalizeWasPerformed = false;
}



void FinalizeSolutionStep() override
{
KRATOS_TRY;

if (mFinalizeWasPerformed == false) {
BaseType::FinalizeSolutionStep();

mFinalizeWasPerformed = true;
}

KRATOS_CATCH("");
}



bool SolveSolutionStep() override
{
KRATOS_TRY;


bool is_converged = false;

ModelPart& r_model_part = StrategyBaseType::GetModelPart();

if (r_model_part.IsNot(INTERACTION)) {
TSystemMatrixType& A = *BaseType::mpA;
TSystemVectorType& Dx = *BaseType::mpDx;
TSystemVectorType& b = *BaseType::mpb;

ProcessInfo& r_process_info = r_model_part.GetProcessInfo();

int inner_iteration = 0;
while (!is_converged && inner_iteration < mThisParameters["inner_loop_iterations"].GetInt()) {
++inner_iteration;

if (mConvergenceCriteriaEchoLevel > 0 && StrategyBaseType::GetModelPart().GetCommunicator().MyPID() == 0 ) {
std::cout << std::endl << BOLDFONT("Simplified semi-smooth strategy. INNER ITERATION: ") << inner_iteration;;
}

r_process_info[NL_ITERATION_NUMBER] = 1;
r_process_info[INNER_LOOP_ITERATION] = inner_iteration;
is_converged = BaseSolveSolutionStep();

BaseType::mpConvergenceCriteria->SetEchoLevel(0);
is_converged = BaseType::mpConvergenceCriteria->PostCriteria(r_model_part, BaseType::GetBuilderAndSolver()->GetDofSet(), A, Dx, b);
BaseType::mpConvergenceCriteria->SetEchoLevel(mConvergenceCriteriaEchoLevel);

if (mConvergenceCriteriaEchoLevel > 0 && StrategyBaseType::GetModelPart().GetCommunicator().MyPID() == 0 ) {
if (is_converged) std::cout << BOLDFONT("Simplified semi-smooth strategy. INNER ITERATION: ") << BOLDFONT(FGRN("CONVERGED")) << std::endl;
else std::cout << BOLDFONT("Simplified semi-smooth strategy. INNER ITERATION: ") << BOLDFONT(FRED("NOT CONVERGED")) << std::endl;
}
}
} else {
r_model_part.GetProcessInfo()[INNER_LOOP_ITERATION] = 1;
is_converged = BaseSolveSolutionStep();
}

if (mThisParameters["adaptative_strategy"].GetBool()) {
if (!is_converged) {
is_converged = AdaptativeStep();
}
}

return is_converged;

KRATOS_CATCH("");
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                                : "newton_raphson_contact_strategy",
"adaptative_strategy"                 : false,
"split_factor"                        : 10.0,
"max_number_splits"                   : 3,
"inner_loop_iterations"               : 5
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "newton_raphson_contact_strategy";
}





protected:



Parameters mThisParameters;        

bool mFinalizeWasPerformed;        
ProcessesListType mpMyProcesses;   
ProcessesListType mpPostProcesses; 

int mConvergenceCriteriaEchoLevel; 




void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

mThisParameters = ThisParameters;
}


bool BaseSolveSolutionStep()
{
KRATOS_TRY;

ModelPart& r_model_part = StrategyBaseType::GetModelPart();
ProcessInfo& r_process_info = r_model_part.GetProcessInfo();
typename TSchemeType::Pointer p_scheme = BaseType::GetScheme();
typename TBuilderAndSolverType::Pointer p_builder_and_solver = BaseType::GetBuilderAndSolver();
auto& r_dof_set = p_builder_and_solver->GetDofSet();

TSystemMatrixType& rA = *BaseType::mpA;
TSystemVectorType& rDx = *BaseType::mpDx;
TSystemVectorType& rb = *BaseType::mpb;

IndexType iteration_number = 1;
r_process_info[NL_ITERATION_NUMBER] = iteration_number;

bool is_converged = false;
bool residual_is_updated = false;
p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);
is_converged = BaseType::mpConvergenceCriteria->PreCriteria(r_model_part, r_dof_set, rA, rDx, rb);

if (mThisParameters["adaptative_strategy"].GetBool()) {
if (CheckGeometryInverted()) {
KRATOS_WARNING("Element inverted") << "INVERTED ELEMENT BEFORE FIRST SOLVE"  << std::endl;
r_process_info[STEP] -= 1; 
return false;
}
}

if (StrategyBaseType::mRebuildLevel > 1 || StrategyBaseType::mStiffnessMatrixIsBuilt == false) {
TSparseSpace::SetToZero(rA);
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildAndSolve(p_scheme, r_model_part, rA, rDx, rb);
} else {
TSparseSpace::SetToZero(rDx); 
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}

BaseType::EchoInfo(iteration_number);

UpdateDatabase(rA, rDx, rb, StrategyBaseType::MoveMeshFlag());

if (mThisParameters["adaptative_strategy"].GetBool()) {
if (CheckGeometryInverted()) {
KRATOS_WARNING("Element inverted") << "INVERTED ELEMENT DURING DATABASE UPDATE" << std::endl;
r_process_info[STEP] -= 1; 
return false;
}
}

p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

if (is_converged) {
BaseType::mpConvergenceCriteria->InitializeSolutionStep(r_model_part, r_dof_set, rA, rDx, rb);

if (BaseType::mpConvergenceCriteria->GetActualizeRHSflag()) {
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
}

is_converged = BaseType::mpConvergenceCriteria->PostCriteria(r_model_part, r_dof_set, rA, rDx, rb);
}

while (is_converged == false && iteration_number++<BaseType::mMaxIterationNumber) {
r_process_info[NL_ITERATION_NUMBER] = iteration_number;

p_scheme->InitializeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->InitializeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

is_converged = BaseType::mpConvergenceCriteria->PreCriteria(r_model_part, r_dof_set, rA, rDx, rb);

if (SparseSpaceType::Size(rDx) != 0) {
if (StrategyBaseType::mRebuildLevel > 1 || StrategyBaseType::mStiffnessMatrixIsBuilt == false ) {
if( BaseType::GetKeepSystemConstantDuringIterations() == false) {
TSparseSpace::SetToZero(rA);
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}
else {
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}
}
else {
TSparseSpace::SetToZero(rDx);
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHSAndSolve(p_scheme, r_model_part, rA, rDx, rb);
}
} else {
KRATOS_WARNING("No DoFs") << "ATTENTION: no free DOFs!! " << std::endl;
}

BaseType::EchoInfo(iteration_number);

UpdateDatabase(rA, rDx, rb, StrategyBaseType::MoveMeshFlag());

if (mThisParameters["adaptative_strategy"].GetBool()) {
if (CheckGeometryInverted()) {
KRATOS_WARNING("Element inverted") << "INVERTED ELEMENT DURING DATABASE UPDATE" << std::endl;
r_process_info[STEP] -= 1; 
return false;
}
}

p_scheme->FinalizeNonLinIteration(r_model_part, rA, rDx, rb);
BaseType::mpConvergenceCriteria->FinalizeNonLinearIteration(r_model_part, r_dof_set, rA, rDx, rb);

residual_is_updated = false;

if (is_converged) {

if (BaseType::mpConvergenceCriteria->GetActualizeRHSflag()) {
TSparseSpace::SetToZero(rb);

p_builder_and_solver->BuildRHS(p_scheme, r_model_part, rb);
residual_is_updated = true;
}

is_converged = BaseType::mpConvergenceCriteria->PostCriteria(r_model_part, r_dof_set, rA, rDx, rb);
}
}

if (iteration_number >= BaseType::mMaxIterationNumber && r_model_part.GetCommunicator().MyPID() == 0)
MaxIterationsExceeded();

if (residual_is_updated == false) {

}

if (BaseType::mCalculateReactionsFlag)
p_builder_and_solver->CalculateReactions(p_scheme, r_model_part, rA, rDx, rb);

return is_converged;

KRATOS_CATCH("");
}


bool AdaptativeStep()
{
KRATOS_TRY;

bool is_converged = false;
if (mpMyProcesses == nullptr && StrategyBaseType::mEchoLevel > 0)
KRATOS_WARNING("No python processes") << "If you have not implemented any method to recalculate BC or loads in function of time, this strategy will be USELESS" << std::endl;

if (mpPostProcesses == nullptr && StrategyBaseType::mEchoLevel > 0)
KRATOS_WARNING("No python post processes") << "If you don't add the postprocesses and the time step if splitted you won't postprocess that steps" << std::endl;

ModelPart& r_model_part = StrategyBaseType::GetModelPart();
ProcessInfo& r_process_info = r_model_part.GetProcessInfo();

const double original_delta_time = r_process_info[DELTA_TIME]; 

int split_number = 0;

while (is_converged == false && split_number <= mThisParameters["max_number_splits"].GetInt()) {
split_number += 1;
double aux_delta_time, current_time;
const double aux_time = SplitTimeStep(aux_delta_time, current_time);
current_time += aux_delta_time;

bool inside_the_split_is_converged = false;
IndexType inner_iteration = 0;
while (current_time <= aux_time) {
inner_iteration += 1;
r_process_info[STEP] += 1;

if (inner_iteration == 1) {
if (StrategyBaseType::MoveMeshFlag())
UnMoveMesh();

NodesArrayType& r_nodes_array = r_model_part.Nodes();
block_for_each(r_nodes_array, [&](NodeType& rNode) {
rNode.OverwriteSolutionStepData(1, 0);
});

r_process_info.SetCurrentTime(current_time); 

FinalizeSolutionStep();
} else {
NodesArrayType& r_nodes_array = r_model_part.Nodes();
block_for_each(r_nodes_array, [&](NodeType& rNode) {
rNode.CloneSolutionStepData();
});

r_process_info.CloneSolutionStepInfo();
r_process_info.ClearHistory(r_model_part.GetBufferSize());
r_process_info.SetAsTimeStepInfo(current_time); 
}

if (mpMyProcesses != nullptr)
mpMyProcesses->ExecuteInitializeSolutionStep();

if (mpPostProcesses != nullptr)
mpPostProcesses->ExecuteInitializeSolutionStep();

BaseType::mInitializeWasPerformed = false;
mFinalizeWasPerformed = false;

this->Initialize();
this->InitializeSolutionStep();
this->Predict();
inside_the_split_is_converged = BaseType::SolveSolutionStep();
this->FinalizeSolutionStep();

if (mpMyProcesses != nullptr)
mpMyProcesses->ExecuteFinalizeSolutionStep();

if (mpPostProcesses != nullptr)
mpPostProcesses->ExecuteFinalizeSolutionStep();

if (mpMyProcesses != nullptr)
mpMyProcesses->ExecuteBeforeOutputStep();

if (mpPostProcesses != nullptr)
mpPostProcesses->PrintOutput();

if (mpMyProcesses != nullptr)
mpMyProcesses->ExecuteAfterOutputStep();

current_time += aux_delta_time;
}

if (inside_the_split_is_converged)
is_converged = true;
}

if (is_converged == false)
MaxIterationsAndSplitsExceeded();

r_process_info[DELTA_TIME] = original_delta_time;

return is_converged;

KRATOS_CATCH("");
}


void UpdateDatabase(
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b,
const bool MoveMesh
) override
{
BaseType::UpdateDatabase(A,Dx,b,MoveMesh);

}


bool CheckGeometryInverted()
{
ModelPart& r_model_part = StrategyBaseType::GetModelPart();
ProcessInfo& r_process_info = r_model_part.GetProcessInfo();
bool inverted_element = false;

ElementsArrayType& elements_array = r_model_part.Elements();

for(int i = 0; i < static_cast<int>(elements_array.size()); ++i) {
auto it_elem = elements_array.begin() + i;
auto& geom = it_elem->GetGeometry();
if (geom.DeterminantOfJacobian(0) < 0.0) {
if (mConvergenceCriteriaEchoLevel > 0) {
KRATOS_WATCH(it_elem->Id())
KRATOS_WATCH(geom.DeterminantOfJacobian(0))
}
return true;
}

std::vector<Matrix> deformation_gradient_matrices;
it_elem->CalculateOnIntegrationPoints( DEFORMATION_GRADIENT, deformation_gradient_matrices, r_process_info);

for (IndexType i_gp = 0; i_gp  < deformation_gradient_matrices.size(); ++i_gp) {
const double det_f = MathUtils<double>::Det(deformation_gradient_matrices[i_gp]);
if (det_f < 0.0) {
if (mConvergenceCriteriaEchoLevel > 0) {
KRATOS_WATCH(it_elem->Id())
KRATOS_WATCH(det_f)
}
return true;
}
}
}

return inverted_element;
}


double SplitTimeStep(
double& AuxDeltaTime,
double& CurrentTime
)
{
KRATOS_TRY;

const double aux_time = StrategyBaseType::GetModelPart().GetProcessInfo()[TIME];
AuxDeltaTime = StrategyBaseType::GetModelPart().GetProcessInfo()[DELTA_TIME];
CurrentTime = aux_time - AuxDeltaTime;

StrategyBaseType::GetModelPart().GetProcessInfo()[TIME] = CurrentTime; 
AuxDeltaTime /= mThisParameters["split_factor"].GetDouble();
StrategyBaseType::GetModelPart().GetProcessInfo()[DELTA_TIME] = AuxDeltaTime; 

CoutSplittingTime(AuxDeltaTime, aux_time);

return aux_time;

KRATOS_CATCH("");
}


void UnMoveMesh()
{
KRATOS_TRY;

if (StrategyBaseType::GetModelPart().NodesBegin()->SolutionStepsDataHas(DISPLACEMENT_X) == false)
KRATOS_ERROR << "It is impossible to move the mesh since the DISPLACEMENT var is not in the model_part. Either use SetMoveMeshFlag(False) or add DISPLACEMENT to the list of variables" << std::endl;

NodesArrayType& r_nodes_array = StrategyBaseType::GetModelPart().Nodes();
block_for_each(r_nodes_array, [&](NodeType& rNode) {
noalias(rNode.Coordinates()) = rNode.GetInitialPosition().Coordinates();
noalias(rNode.Coordinates()) += rNode.FastGetSolutionStepValue(DISPLACEMENT, 1);
});

KRATOS_CATCH("");
}


void CoutSolvingProblem()
{
if (mConvergenceCriteriaEchoLevel != 0) {
std::cout << "STEP: " << StrategyBaseType::GetModelPart().GetProcessInfo()[STEP] << "\t NON LINEAR ITERATION: " << StrategyBaseType::GetModelPart().GetProcessInfo()[NL_ITERATION_NUMBER] << "\t TIME: " << StrategyBaseType::GetModelPart().GetProcessInfo()[TIME] << "\t DELTA TIME: " << StrategyBaseType::GetModelPart().GetProcessInfo()[DELTA_TIME]  << std::endl;
}
}


void CoutSplittingTime(
const double AuxDeltaTime,
const double AuxTime
)
{
if (mConvergenceCriteriaEchoLevel > 0 && StrategyBaseType::GetModelPart().GetCommunicator().MyPID() == 0 ) {
const double Time = StrategyBaseType::GetModelPart().GetProcessInfo()[TIME];
std::cout.precision(4);
std::cout << "|----------------------------------------------------|" << std::endl;
std::cout << "|     " << BOLDFONT("SPLITTING TIME STEP") << "                            |" << std::endl;
std::cout << "| " << BOLDFONT("COMING BACK TO TIME: ") << std::scientific << Time << "                    |" << std::endl;
std::cout << "| " << BOLDFONT("      NEW TIME STEP: ") << std::scientific << AuxDeltaTime << "                    |" << std::endl;
std::cout << "| " << BOLDFONT("         UNTIL TIME: ") << std::scientific << AuxTime << "                    |" << std::endl;
std::cout << "|----------------------------------------------------|" << std::endl;
}
}


void MaxIterationsExceeded() override
{
if (mConvergenceCriteriaEchoLevel > 0 && StrategyBaseType::GetModelPart().GetCommunicator().MyPID() == 0 ) {
std::cout << "|----------------------------------------------------|" << std::endl;
std::cout << "|        " << BOLDFONT(FRED("ATTENTION: Max iterations exceeded")) << "          |" << std::endl;
std::cout << "|----------------------------------------------------|" << std::endl;
}
}


void MaxIterationsAndSplitsExceeded()
{
if (mConvergenceCriteriaEchoLevel > 0 && StrategyBaseType::GetModelPart().GetCommunicator().MyPID() == 0 ) {
std::cout << "|----------------------------------------------------|" << std::endl;
std::cout << "|        " << BOLDFONT(FRED("ATTENTION: Max iterations exceeded")) << "          |" << std::endl;
std::cout << "|        " << BOLDFONT(FRED("   Max number of splits exceeded  ")) << "          |" << std::endl;
std::cout << "|----------------------------------------------------|" << std::endl;
}
}





ResidualBasedNewtonRaphsonContactStrategy(const ResidualBasedNewtonRaphsonContactStrategy& Other)
{
};


}; 
}  
