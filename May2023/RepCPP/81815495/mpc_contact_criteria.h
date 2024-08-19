
#pragma once



#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "utilities/color_utilities.h"
#include "utilities/variable_utils.h"
#include "custom_utilities/contact_utilities.h"
#include "processes/simple_mortar_mapper_wrapper_process.h"

namespace Kratos
{






template<class TSparseSpace, class TDenseSpace>
class MPCContactCriteria
: public  ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

KRATOS_CLASS_POINTER_DEFINITION( MPCContactCriteria );

typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;

typedef MPCContactCriteria< TSparseSpace, TDenseSpace > ClassType;

typedef typename BaseType::DofsArrayType            DofsArrayType;

typedef typename BaseType::TSystemMatrixType    TSystemMatrixType;

typedef typename BaseType::TSystemVectorType    TSystemVectorType;

typedef TableStreamUtility::Pointer       TablePrinterPointerType;

typedef std::size_t                                     IndexType;

typedef Node                                          NodeType;
typedef CouplingGeometry<NodeType>           CouplingGeometryType;



explicit MPCContactCriteria()
: BaseType()
{
}


explicit MPCContactCriteria(Kratos::Parameters ThisParameters)
: BaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}

MPCContactCriteria( MPCContactCriteria const& rOther )
: BaseType(rOther)
{
}

~MPCContactCriteria() override = default;




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
BaseType::PreCriteria(rModelPart, rDofSet, rA, rDx, rb);

const array_1d<double, 3> zero_array = ZeroVector(3);

auto& r_nodes_array = rModelPart.GetSubModelPart("Contact").Nodes();

block_for_each(r_nodes_array, [&](NodeType& rNode) {
rNode.SetValue(CONTACT_FORCE, zero_array);
rNode.FastGetSolutionStepValue(WEIGHTED_GAP, 1) = rNode.FastGetSolutionStepValue(WEIGHTED_GAP);
});

ComputeWeightedGap(rModelPart);

VariableUtils().SetNonHistoricalVariableToZero(NODAL_AREA, r_nodes_array);

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
BaseType::PostCriteria(rModelPart, rDofSet, rA, rDx, rb);

const ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
if (r_process_info[NL_ITERATION_NUMBER] > 0) {
const double reaction_check_stiffness_factor = r_process_info.Has(REACTION_CHECK_STIFFNESS_FACTOR) ?  r_process_info.GetValue(REACTION_CHECK_STIFFNESS_FACTOR) : 1.0e-12;

ComputeWeightedGap(rModelPart);

std::size_t sub_contact_counter = 0;
CounterContactModelParts(rModelPart, sub_contact_counter);

Parameters mapping_parameters = Parameters(R"({"distance_threshold" : 1.0e24, "update_interface" : false, "origin_variable" : "REACTION", "mapping_coefficient" : -1.0e0})" );
if (r_process_info.Has(DISTANCE_THRESHOLD)) {
mapping_parameters["distance_threshold"].SetDouble(r_process_info[DISTANCE_THRESHOLD]);
}
auto& r_contact_model_part = rModelPart.GetSubModelPart("Contact");
for (std::size_t i_contact = 0; i_contact < sub_contact_counter; ++i_contact) {
auto& r_sub = r_contact_model_part.GetSubModelPart("ContactSub" + std::to_string(i_contact));
auto& r_sub_master = r_sub.GetSubModelPart("MasterSubModelPart" + std::to_string(i_contact));
auto& r_sub_slave = r_sub.GetSubModelPart("SlaveSubModelPart" + std::to_string(i_contact));
SimpleMortarMapperProcessWrapper(r_sub_master, r_sub_slave, mapping_parameters).Execute();
}


Properties::Pointer p_properties = rModelPart.Elements().begin()->pGetProperties();
for (auto& r_elements : rModelPart.Elements()) {
if (r_elements.pGetProperties()->Has(YOUNG_MODULUS)) {
p_properties = r_elements.pGetProperties();
}
}

IndexType is_active_set_converged = 0, is_slip_converged = 0;

const double young_modulus = p_properties->Has(YOUNG_MODULUS) ? p_properties->GetValue(YOUNG_MODULUS) : 0.0;
const double auxiliary_check = young_modulus > 0.0 ? -(reaction_check_stiffness_factor * young_modulus) : 0.0;

auto& r_nodes_array = r_contact_model_part.Nodes();

if (rModelPart.IsNot(SLIP)) {
is_active_set_converged = block_for_each<SumReduction<IndexType>>(r_nodes_array, [&](NodeType& rNode) {
if (rNode.Is(SLAVE)) {
const array_1d<double, 3>& r_total_force = rNode.FastGetSolutionStepValue(REACTION);

const double nodal_area = rNode.Has(NODAL_AREA) ? rNode.GetValue(NODAL_AREA) : 1.0;
const double gap = rNode.FastGetSolutionStepValue(WEIGHTED_GAP)/nodal_area;
const array_1d<double, 3>& r_normal = rNode.FastGetSolutionStepValue(NORMAL);
const double contact_force = inner_prod(r_total_force, r_normal);
const double contact_pressure = contact_force/rNode.GetValue(NODAL_MAUX);

if (contact_pressure < auxiliary_check || gap < 0.0) { 
rNode.SetValue(CONTACT_FORCE, contact_force/rNode.GetValue(NODAL_PAUX) * r_normal);
rNode.SetValue(NORMAL_CONTACT_STRESS, contact_pressure);
if (rNode.IsNot(ACTIVE)) {
rNode.Set(ACTIVE, true);
return 1;
}
} else {
if (rNode.Is(ACTIVE)) {
rNode.Set(ACTIVE, false);
return 1;
}
}
}
return 0;
});
} else { 
using TwoReduction = CombinedReduction<SumReduction<IndexType>, SumReduction<IndexType>>;
std::tie(is_active_set_converged, is_slip_converged) = block_for_each<TwoReduction>(r_nodes_array, [&](NodeType& rNode) {
if (rNode.Is(SLAVE)) {
const double auxiliary_check = young_modulus > 0.0 ? -(reaction_check_stiffness_factor * young_modulus) : 0.0;
const array_1d<double, 3>& r_total_force = rNode.FastGetSolutionStepValue(REACTION);

const double nodal_area = rNode.Has(NODAL_AREA) ? rNode.GetValue(NODAL_AREA) : 1.0;
const double gap = rNode.FastGetSolutionStepValue(WEIGHTED_GAP)/nodal_area;
const array_1d<double, 3>& r_normal = rNode.FastGetSolutionStepValue(NORMAL);
const double contact_force = inner_prod(r_total_force, r_normal);
const double normal_contact_pressure = contact_force/rNode.GetValue(NODAL_MAUX);

if (normal_contact_pressure < auxiliary_check || gap < 0.0) { 
rNode.SetValue(CONTACT_FORCE, r_total_force/rNode.GetValue(NODAL_PAUX));
rNode.SetValue(NORMAL_CONTACT_STRESS, normal_contact_pressure);
if (rNode.IsNot(ACTIVE)) {
rNode.Set(ACTIVE, true);
return std::make_tuple(1,0);
}

const double tangential_contact_pressure = norm_2(r_total_force - contact_force * r_normal)/rNode.GetValue(NODAL_MAUX);
const bool is_slip = rNode.Is(SLIP);
const double mu = rNode.GetValue(FRICTION_COEFFICIENT);

if (tangential_contact_pressure <= - mu * contact_force) { 
rNode.SetValue(TANGENTIAL_CONTACT_STRESS, tangential_contact_pressure);
if (is_slip) {
rNode.Set(SLIP, false);
return std::make_tuple(0,1);
}
} else { 
rNode.SetValue(TANGENTIAL_CONTACT_STRESS, - mu * contact_force);
if (!is_slip) {
rNode.Set(SLIP, true);
return std::make_tuple(0,1);
}
}
} else {
if (rNode.Is(ACTIVE)) {
rNode.Set(ACTIVE, false);
rNode.Reset(SLIP);
return std::make_tuple(1,0);
}
}
}
return std::make_tuple(0,0);
});
}

auto& r_conditions_array = rModelPart.GetSubModelPart("ComputingContact").Conditions();
block_for_each(r_conditions_array, [&](Condition& rCond) {
const auto& r_slave_geometry = rCond.GetGeometry().GetGeometryPart(CouplingGeometryType::Master);
std::size_t counter = 0;
for (auto& r_node : r_slave_geometry) {
if (r_node.IsNot(ACTIVE)) {
++counter;
}
}

if (counter == r_slave_geometry.size()) {
rCond.Set(ACTIVE, false);
if (rCond.Has(CONSTRAINT_POINTER)) {
auto p_const = rCond.GetValue(CONSTRAINT_POINTER);

p_const->Set(ACTIVE, false);
} else {
KRATOS_ERROR << "Contact conditions must have defined CONSTRAINT_POINTER" << std::endl;
}
}
});

const bool active_set_converged = (is_active_set_converged == 0 ? true : false);
const bool slip_set_converged = (is_slip_converged == 0 ? true : false);

if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
if (active_set_converged) {
KRATOS_INFO("MPCContactCriteria")  << BOLDFONT("\tActive set") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
} else {
KRATOS_INFO("MPCContactCriteria")  << BOLDFONT("\tActive set") << " convergence is " << BOLDFONT(FRED("not achieved")) << std::endl;
}
if (slip_set_converged) {
KRATOS_INFO("MPCContactCriteria")  << BOLDFONT("\tSlip set") << " convergence is " << BOLDFONT(FGRN("achieved")) << std::endl;
} else {
KRATOS_INFO("MPCContactCriteria")  << BOLDFONT("\tSlip set") << " convergence is " << BOLDFONT(FRED("not achieved")) << std::endl;
}
}

return (active_set_converged && slip_set_converged);
}

return true;
}


void Initialize(ModelPart& rModelPart) override
{
BaseType::Initialize(rModelPart);
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name" : "mpc_contact_criteria"
})" );

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "mpc_contact_criteria";
}




std::string Info() const override
{
return "MPCContactCriteria";
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






void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);
}





private:





void ComputeWeightedGap(ModelPart& rModelPart)
{
auto& r_nodes_array = rModelPart.GetSubModelPart("Contact").Nodes();
if (rModelPart.Is(SLIP)) {
VariableUtils().SetHistoricalVariableToZero(WEIGHTED_GAP, r_nodes_array);
VariableUtils().SetHistoricalVariableToZero(WEIGHTED_SLIP, r_nodes_array);
} else {
VariableUtils().SetHistoricalVariableToZero(WEIGHTED_GAP, r_nodes_array);
}

ContactUtilities::ComputeExplicitContributionConditions(rModelPart.GetSubModelPart("ComputingContact"));
}


void CounterContactModelParts(
ModelPart& rModelPart,
std::size_t& rCounter
)
{
for (auto& r_name : rModelPart.GetSubModelPartNames()) {
if (r_name.find("ContactSub") != std::string::npos && r_name.find("ComputingContactSub") == std::string::npos) {
++rCounter;
}
auto& r_sub = rModelPart.GetSubModelPart(r_name);
if (r_sub.IsSubModelPart()) {
CounterContactModelParts(r_sub, rCounter);
}
}
}






}; 


}  

