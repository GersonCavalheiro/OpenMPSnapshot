
#pragma once



#include "contact_structural_mechanics_application_variables.h"
#include "custom_utilities/contact_utilities.h"
#include "utilities/mortar_utilities.h"
#include "utilities/variable_utils.h"
#include "utilities/normal_calculation_utils.h"
#include "custom_processes/aalm_adapt_penalty_value_process.h"
#include "custom_processes/compute_dynamic_factor_process.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"

#include "includes/gid_io.h"

namespace Kratos
{







template<class TSparseSpace, class TDenseSpace>
class BaseMortarConvergenceCriteria
: public  ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( BaseMortarConvergenceCriteria );

KRATOS_DEFINE_LOCAL_FLAG( COMPUTE_DYNAMIC_FACTOR );
KRATOS_DEFINE_LOCAL_FLAG( IO_DEBUG );
KRATOS_DEFINE_LOCAL_FLAG( PURE_SLIP );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace >            BaseType;

typedef BaseMortarConvergenceCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::DofsArrayType                       DofsArrayType;

typedef typename BaseType::TSystemMatrixType               TSystemMatrixType;

typedef typename BaseType::TSystemVectorType               TSystemVectorType;

typedef GidIO<>                                                GidIOBaseType;


explicit BaseMortarConvergenceCriteria(
const bool ComputeDynamicFactor = false,
const bool IODebug = false,
const bool PureSlip = false
)
: BaseType(),
mpIO(nullptr)
{
mOptions.Set(BaseMortarConvergenceCriteria::COMPUTE_DYNAMIC_FACTOR, ComputeDynamicFactor);
mOptions.Set(BaseMortarConvergenceCriteria::IO_DEBUG, IODebug);
mOptions.Set(BaseMortarConvergenceCriteria::PURE_SLIP, PureSlip);

if (mOptions.Is(BaseMortarConvergenceCriteria::IO_DEBUG)) {
mpIO = Kratos::make_shared<GidIOBaseType>("POST_LINEAR_ITER", GiD_PostBinary, SingleFile, WriteUndeformed,  WriteElementsOnly);
}
}


explicit BaseMortarConvergenceCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}

BaseMortarConvergenceCriteria( BaseMortarConvergenceCriteria const& rOther )
:BaseType(rOther),
mOptions(rOther.mOptions),
mpIO(rOther.mpIO)
{
}

~BaseMortarConvergenceCriteria() override = default;




typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}


bool PreCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

ModelPart& r_contact_model_part = rModelPart.GetSubModelPart("Contact");

const auto normal_variation = r_process_info.Has(CONSIDER_NORMAL_VARIATION) ? static_cast<NormalDerivativesComputation>(r_process_info.GetValue(CONSIDER_NORMAL_VARIATION)) : NO_DERIVATIVES_COMPUTATION;
if (normal_variation != NO_DERIVATIVES_COMPUTATION) {
ComputeNodesMeanNormalModelPartWithPairedNormal(rModelPart); 
}

const bool frictional_problem = rModelPart.IsDefined(SLIP) ? rModelPart.Is(SLIP) : false;
if (frictional_problem) {
const bool has_lm = rModelPart.HasNodalSolutionStepVariable(VECTOR_LAGRANGE_MULTIPLIER);
if (has_lm && mOptions.IsNot(BaseMortarConvergenceCriteria::PURE_SLIP)) {
MortarUtilities::ComputeNodesTangentModelPart(r_contact_model_part);
} else {
MortarUtilities::ComputeNodesTangentModelPart(r_contact_model_part, &WEIGHTED_SLIP, 1.0, true);
}
}

const bool adapt_penalty = r_process_info.Has(ADAPT_PENALTY) ? r_process_info.GetValue(ADAPT_PENALTY) : false;
const bool dynamic_case = rModelPart.HasNodalSolutionStepVariable(VELOCITY);


if (adapt_penalty || dynamic_case) {
ResetWeightedGap(rModelPart);

ContactUtilities::ComputeExplicitContributionConditions(rModelPart.GetSubModelPart("ComputingContact"));
}

if ( dynamic_case && mOptions.Is(BaseMortarConvergenceCriteria::COMPUTE_DYNAMIC_FACTOR)) {
ComputeDynamicFactorProcess compute_dynamic_factor_process( r_contact_model_part );
compute_dynamic_factor_process.Execute();
}

if ( adapt_penalty ) {
AALMAdaptPenaltyValueProcess aalm_adaptation_of_penalty( r_contact_model_part );
aalm_adaptation_of_penalty.Execute();
}

return true;
}


bool PostCriteria(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
auto& r_nodes_array = rModelPart.GetSubModelPart("Contact").Nodes();
block_for_each(r_nodes_array, [&](NodeType& rNode) {
rNode.FastGetSolutionStepValue(WEIGHTED_GAP, 1) = rNode.FastGetSolutionStepValue(WEIGHTED_GAP);
});

ResetWeightedGap(rModelPart);

ContactUtilities::ComputeExplicitContributionConditions(rModelPart.GetSubModelPart("ComputingContact"));

if (mOptions.Is(BaseMortarConvergenceCriteria::IO_DEBUG)) {
const bool frictional_problem = rModelPart.IsDefined(SLIP) ? rModelPart.Is(SLIP) : false;
const int nl_iter = rModelPart.GetProcessInfo()[NL_ITERATION_NUMBER];
const double label = static_cast<double>(nl_iter);

if (nl_iter == 1) {
mpIO->InitializeMesh(label);
mpIO->WriteMesh(rModelPart.GetMesh());
mpIO->FinalizeMesh();
mpIO->InitializeResults(label, rModelPart.GetMesh());
}

mpIO->WriteNodalFlags(INTERFACE, "INTERFACE", rModelPart.Nodes(), label);
mpIO->WriteNodalFlags(ACTIVE, "ACTIVE", rModelPart.Nodes(), label);
mpIO->WriteNodalFlags(SLAVE, "SLAVE", rModelPart.Nodes(), label);
mpIO->WriteNodalFlags(ISOLATED, "ISOLATED", rModelPart.Nodes(), label);
mpIO->WriteNodalResults(NORMAL, rModelPart.Nodes(), label, 0);
mpIO->WriteNodalResultsNonHistorical(DYNAMIC_FACTOR, rModelPart.Nodes(), label);
mpIO->WriteNodalResultsNonHistorical(AUGMENTED_NORMAL_CONTACT_PRESSURE, rModelPart.Nodes(), label);
mpIO->WriteNodalResults(DISPLACEMENT, rModelPart.Nodes(), label, 0);
if (rModelPart.Nodes().begin()->SolutionStepsDataHas(VELOCITY_X)) {
mpIO->WriteNodalResults(VELOCITY, rModelPart.Nodes(), label, 0);
mpIO->WriteNodalResults(ACCELERATION, rModelPart.Nodes(), label, 0);
}
if (r_nodes_array.begin()->SolutionStepsDataHas(LAGRANGE_MULTIPLIER_CONTACT_PRESSURE))
mpIO->WriteNodalResults(LAGRANGE_MULTIPLIER_CONTACT_PRESSURE, rModelPart.Nodes(), label, 0);
else if (r_nodes_array.begin()->SolutionStepsDataHas(VECTOR_LAGRANGE_MULTIPLIER_X))
mpIO->WriteNodalResults(VECTOR_LAGRANGE_MULTIPLIER, rModelPart.Nodes(), label, 0);
mpIO->WriteNodalResults(WEIGHTED_GAP, rModelPart.Nodes(), label, 0);
if (frictional_problem) {
mpIO->WriteNodalFlags(SLIP, "SLIP", rModelPart.Nodes(), label);
mpIO->WriteNodalResults(WEIGHTED_SLIP, rModelPart.Nodes(), label, 0);
mpIO->WriteNodalResultsNonHistorical(AUGMENTED_TANGENT_CONTACT_PRESSURE, rModelPart.Nodes(), label);
}
}

return true;
}


void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
r_process_info.SetValue(ACTIVE_SET_COMPUTED, false);
}


void InitializeSolutionStep(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
ModelPart& r_contact_model_part = rModelPart.GetSubModelPart("Contact");
NormalCalculationUtils().CalculateUnitNormals<ModelPart::ConditionsContainerType>(r_contact_model_part, true);
const bool frictional_problem = rModelPart.IsDefined(SLIP) ? rModelPart.Is(SLIP) : false;
if (frictional_problem) {
const bool has_lm = rModelPart.HasNodalSolutionStepVariable(VECTOR_LAGRANGE_MULTIPLIER);
if (has_lm && mOptions.IsNot(BaseMortarConvergenceCriteria::PURE_SLIP)) {
MortarUtilities::ComputeNodesTangentModelPart(r_contact_model_part);
} else {
MortarUtilities::ComputeNodesTangentModelPart(r_contact_model_part, &WEIGHTED_SLIP, 1.0, true);
}
}

if (mOptions.Is(BaseMortarConvergenceCriteria::IO_DEBUG)) {
mpIO->CloseResultFile();
std::ostringstream new_name ;
new_name << "POST_LINEAR_ITER_STEP=""POST_LINEAR_ITER_STEP=" << rModelPart.GetProcessInfo()[STEP];
mpIO->ChangeOutputName(new_name.str());
}
}


void FinalizeSolutionStep(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
if (mOptions.Is(BaseMortarConvergenceCriteria::IO_DEBUG)) {
mpIO->FinalizeResults();
}
}


void FinalizeNonLinearIteration(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
const TSystemMatrixType& rA,
const TSystemVectorType& rDx,
const TSystemVectorType& rb
) override
{
BaseType::FinalizeNonLinearIteration(rModelPart, rDofSet, rA, rDx, rb);

ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
r_process_info.SetValue(ACTIVE_SET_COMPUTED, false);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                   : "base_mortar_criteria",
"compute_dynamic_factor" : false,
"gidio_debug"            : false,
"pure_slip"              : false
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "base_mortar_criteria";
}




std::string Info() const override
{
return "BaseMortarConvergenceCriteria";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info();
}


protected:



Flags mOptions; 




void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

mOptions.Set(BaseMortarConvergenceCriteria::COMPUTE_DYNAMIC_FACTOR, ThisParameters["compute_dynamic_factor"].GetBool());
mOptions.Set(BaseMortarConvergenceCriteria::IO_DEBUG, ThisParameters["gidio_debug"].GetBool());
mOptions.Set(BaseMortarConvergenceCriteria::PURE_SLIP, ThisParameters["pure_slip"].GetBool());

if (mOptions.Is(BaseMortarConvergenceCriteria::IO_DEBUG)) {
mpIO = Kratos::make_shared<GidIOBaseType>("POST_LINEAR_ITER", GiD_PostBinary, SingleFile, WriteUndeformed,  WriteElementsOnly);
}
}


virtual void ResetWeightedGap(ModelPart& rModelPart)
{
auto& r_nodes_array = rModelPart.GetSubModelPart("Contact").Nodes();
VariableUtils().SetVariable(WEIGHTED_GAP, 0.0, r_nodes_array);
}




private:


GidIOBaseType::Pointer mpIO; 




inline void ComputeNodesMeanNormalModelPartWithPairedNormal(ModelPart& rModelPart)
{
ModelPart& r_contact_model_part = rModelPart.GetSubModelPart("Contact");
NormalCalculationUtils().CalculateUnitNormals<ModelPart::ConditionsContainerType>(r_contact_model_part, true);

ModelPart& r_computing_contact_model_part = rModelPart.GetSubModelPart("ComputingContact");
auto& r_conditions_array = r_computing_contact_model_part.Conditions();
block_for_each(r_conditions_array, [&](Condition& rCond) {
Point::CoordinatesArrayType aux_coords;

GeometryType& r_parent_geometry = rCond.GetGeometry().GetGeometryPart(0);
aux_coords = r_parent_geometry.PointLocalCoordinates(aux_coords, r_parent_geometry.Center());
rCond.SetValue(NORMAL, r_parent_geometry.UnitNormal(aux_coords));
});
}





}; 


template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags BaseMortarConvergenceCriteria<TSparseSpace, TDenseSpace>::COMPUTE_DYNAMIC_FACTOR(Kratos::Flags::Create(0));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags BaseMortarConvergenceCriteria<TSparseSpace, TDenseSpace>::IO_DEBUG(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags BaseMortarConvergenceCriteria<TSparseSpace, TDenseSpace>::PURE_SLIP(Kratos::Flags::Create(2));

}  
