


#include "containers/model.h"
#include "includes/checks.h"
#include "utilities/openmp_utils.h"
#include "utilities/parallel_utilities.h"
#include "processes/find_nodal_h_process.h"

#include "distance_modification_process.h"
#include "fluid_dynamics_application_variables.h"

namespace Kratos
{

constexpr std::array<std::array<std::size_t,2>, 3> DistanceModificationProcess::NodeIDs2D;
constexpr std::array<std::array<std::size_t,2>, 6> DistanceModificationProcess::NodeIDs3D; 


DistanceModificationProcess::DistanceModificationProcess(
ModelPart& rModelPart,
const double FactorCoeff, 
const double DistanceThreshold,
const bool CheckAtEachStep,
const bool NegElemDeactivation,
const bool RecoverOriginalDistance)
: Process(),
mrModelPart(rModelPart)
{
mDistanceThreshold = DistanceThreshold;
mCheckAtEachStep = CheckAtEachStep;
mNegElemDeactivation = NegElemDeactivation;
mRecoverOriginalDistance = RecoverOriginalDistance;

this->InitializeEmbeddedIsActive();
}

DistanceModificationProcess::DistanceModificationProcess(
ModelPart& rModelPart,
Parameters& rParameters)
: Process(),
mrModelPart(rModelPart)
{
this->CheckDefaultsAndProcessSettings(rParameters);

this->InitializeEmbeddedIsActive();
}

DistanceModificationProcess::DistanceModificationProcess(
Model &rModel,
Parameters &rParameters)
: Process(),
mrModelPart(rModel.GetModelPart(rParameters["model_part_name"].GetString()))
{
this->CheckDefaultsAndProcessSettings(rParameters);

this->InitializeEmbeddedIsActive();
}

void DistanceModificationProcess::InitializeEmbeddedIsActive()
{
for (std::size_t i_node = 0; i_node < static_cast<std::size_t>(mrModelPart.NumberOfNodes()); ++i_node) {
auto it_node = mrModelPart.NodesBegin() + i_node;
it_node->SetValue(EMBEDDED_IS_ACTIVE, 0);
}
}

void DistanceModificationProcess::CheckDefaultsAndProcessSettings(Parameters &rParameters)
{
Parameters default_parameters( R"(
{
"model_part_name"                             : "",
"distance_threshold"                          : 0.001,
"continuous_distance"                         : true,
"check_at_each_time_step"                     : true,
"avoid_almost_empty_elements"                 : true,
"deactivate_full_negative_elements"           : true,
"recover_original_distance_at_each_step"      : false,
"full_negative_elements_fixed_variables_list" : ["PRESSURE","VELOCITY"]
}  )" );

rParameters.ValidateAndAssignDefaults(default_parameters);

mIsModified = false;
mDistanceThreshold = rParameters["distance_threshold"].GetDouble();
mContinuousDistance = rParameters["continuous_distance"].GetBool();
mCheckAtEachStep = rParameters["check_at_each_time_step"].GetBool();
mAvoidAlmostEmptyElements = rParameters["avoid_almost_empty_elements"].GetBool();
mNegElemDeactivation = rParameters["deactivate_full_negative_elements"].GetBool();
mRecoverOriginalDistance = rParameters["recover_original_distance_at_each_step"].GetBool();
if (mNegElemDeactivation) {
this->CheckAndStoreVariablesList(rParameters["full_negative_elements_fixed_variables_list"].GetStringArray());
}
}

void DistanceModificationProcess::Execute()
{
this->ExecuteInitialize();
this->ExecuteInitializeSolutionStep();
}

void DistanceModificationProcess::ExecuteInitialize()
{
KRATOS_TRY;

if (mContinuousDistance){
KRATOS_ERROR_IF_NOT(mrModelPart.HasNodalSolutionStepVariable(DISTANCE)) << "DISTANCE variable not found in solution step variables list in " << mrModelPart.FullName() << ".\n";

FindNodalHProcess<FindNodalHSettings::SaveAsNonHistoricalVariable> nodal_h_calculator(mrModelPart);
nodal_h_calculator.Execute();
}

KRATOS_CATCH("");
}

void DistanceModificationProcess::ExecuteBeforeSolutionLoop()
{
this->ExecuteInitializeSolutionStep();
this->ExecuteFinalizeSolutionStep();
}

void DistanceModificationProcess::ExecuteInitializeSolutionStep()
{
if(!mIsModified){
if (mContinuousDistance) {
this->ModifyDistance();
} else {
this->ModifyDiscontinuousDistance();
}
mIsModified = true;

if (mNegElemDeactivation) {
this->DeactivateFullNegativeElements();
}
}
}

void DistanceModificationProcess::ExecuteFinalizeSolutionStep()
{
if (mCheckAtEachStep){
mIsModified = false;
if (mNegElemDeactivation){
this->RecoverDeactivationPreviousState();
}
}

if(mRecoverOriginalDistance) {
if (mContinuousDistance){
this->RecoverOriginalDistance();
} else {
this->RecoverOriginalDiscontinuousDistance();
}
}
}



void DistanceModificationProcess::ModifyDistance()
{
ModelPart::NodesContainerType& r_nodes = mrModelPart.Nodes();

if (mRecoverOriginalDistance == false) {
#pragma omp parallel for
for (int k = 0; k < static_cast<int>(r_nodes.size()); ++k) {
auto it_node = r_nodes.begin() + k;
const double h = it_node->GetValue(NODAL_H);
const double tol_d = mDistanceThreshold*h;
double& d = it_node->FastGetSolutionStepValue(DISTANCE);

if(std::abs(d) < tol_d){
if (d <= 0.0){
d = -tol_d;
} else {
if (mAvoidAlmostEmptyElements){
d = -tol_d;
} else {
d = tol_d;
}
}
}
}
}
else {

const int num_chunks = 2 * ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector partition_vec;
OpenMPUtils::DivideInPartitions(r_nodes.size(),num_chunks,partition_vec);

#pragma omp parallel for
for (int i_chunk = 0; i_chunk < num_chunks; ++i_chunk)
{
auto nodes_begin = r_nodes.begin() + partition_vec[i_chunk];
auto nodes_end = r_nodes.begin() + partition_vec[i_chunk + 1];

std::vector<std::size_t> aux_modified_distances_ids;
std::vector<double> aux_modified_distance_values;

for (auto it_node = nodes_begin; it_node < nodes_end; ++it_node) {
const double h = it_node->GetValue(NODAL_H);
const double tol_d = mDistanceThreshold*h;
double &d = it_node->FastGetSolutionStepValue(DISTANCE);

if(std::abs(d) < tol_d){

aux_modified_distances_ids.push_back(it_node->Id());
aux_modified_distance_values.push_back(d);

if (d <= 0.0){
d = -tol_d;
} else {
if (mAvoidAlmostEmptyElements){
d = -tol_d;
} else {
d = tol_d;
}
}
}
}

#pragma omp critical
{
mModifiedDistancesIDs.insert(mModifiedDistancesIDs.end(),aux_modified_distances_ids.begin(),aux_modified_distances_ids.end());
mModifiedDistancesValues.insert(mModifiedDistancesValues.end(), aux_modified_distance_values.begin(), aux_modified_distance_values.end());
}
}
}

mrModelPart.GetCommunicator().SynchronizeCurrentDataToMin(DISTANCE);

this->SetContinuousDistanceToSplitFlag();
}

void DistanceModificationProcess::ModifyDiscontinuousDistance()
{
auto r_elems = mrModelPart.Elements();
auto elems_begin = mrModelPart.ElementsBegin();
const auto n_elems = mrModelPart.NumberOfElements();
const std::size_t n_edges_extrapolated = elems_begin->GetValue(ELEMENTAL_EDGE_DISTANCES_EXTRAPOLATED).size();

if (mRecoverOriginalDistance == false) {

if (n_edges_extrapolated > 0) {
block_for_each(r_elems, [&](Element& rElement){
const double tol_d = mDistanceThreshold*(rElement.GetGeometry()).Length();

bool is_modified = false;
Vector &r_elem_dist = rElement.GetValue(ELEMENTAL_DISTANCES);
for (std::size_t i_node = 0; i_node < r_elem_dist.size(); ++i_node){
if (std::abs(r_elem_dist(i_node)) < tol_d){
is_modified = true;
r_elem_dist(i_node) = r_elem_dist(i_node) > 0.0 ? tol_d : -tol_d;
}
}

if (is_modified) {
Vector &r_elem_edge_dist_extra = rElement.GetValue(ELEMENTAL_EDGE_DISTANCES_EXTRAPOLATED);
for (std::size_t i_edge = 0; i_edge < n_edges_extrapolated; ++i_edge) {

const std::array<std::size_t, 2> node_ids = GetNodeIDs(n_edges_extrapolated, i_edge);
const double node_i_distance = r_elem_dist(node_ids[0]);
const double node_j_distance = r_elem_dist(node_ids[1]);

if (r_elem_edge_dist_extra(i_edge)) {
r_elem_edge_dist_extra(i_edge) = std::abs( node_i_distance / (node_j_distance - node_i_distance) );
}
}
}
});
} else {
block_for_each(r_elems, [&](Element& rElement){
const double tol_d = mDistanceThreshold*(rElement.GetGeometry()).Length();

Vector &r_elem_dist = rElement.GetValue(ELEMENTAL_DISTANCES);
for (std::size_t i_node = 0; i_node < r_elem_dist.size(); ++i_node){
if (std::abs(r_elem_dist(i_node)) < tol_d){
r_elem_dist(i_node) = r_elem_dist(i_node) > 0.0 ? tol_d : -tol_d;
}
}
});
}
} else {

const int num_chunks = 2 * ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector partition_vec;
OpenMPUtils::DivideInPartitions(n_elems,num_chunks,partition_vec);

if (n_edges_extrapolated > 0) {
#pragma omp parallel for
for (int i_chunk = 0; i_chunk < num_chunks; ++i_chunk)
{
auto elems_begin = r_elems.begin() + partition_vec[i_chunk];
auto elems_end = r_elems.begin() + partition_vec[i_chunk + 1];

std::vector<std::size_t> aux_modified_distances_ids;
std::vector<Vector> aux_modified_elemental_distances;
std::vector<std::size_t> aux_modified_edge_dist_extra_ids;
std::vector<Vector> aux_modified_edge_dist_extra;

for (auto it_elem = elems_begin; it_elem < elems_end; ++it_elem) {
const double tol_d = mDistanceThreshold * (it_elem->GetGeometry()).Length();

bool is_saved = false;
Vector &r_elem_dist = it_elem->GetValue(ELEMENTAL_DISTANCES);
for (std::size_t i_node = 0; i_node < r_elem_dist.size(); ++i_node){
if (std::abs(r_elem_dist(i_node)) < tol_d){
if (!is_saved){
aux_modified_distances_ids.push_back(it_elem->Id());
aux_modified_elemental_distances.push_back(r_elem_dist);
is_saved = true;
}
r_elem_dist(i_node) = r_elem_dist(i_node) > 0.0 ? tol_d : -tol_d;
}
}

if (is_saved) {
Vector &r_elem_edge_dist_extra = it_elem->GetValue(ELEMENTAL_EDGE_DISTANCES_EXTRAPOLATED);
for (std::size_t i_edge = 0; i_edge < n_edges_extrapolated; ++i_edge) {

const std::array<std::size_t, 2> node_ids = GetNodeIDs(n_edges_extrapolated,i_edge);
const double node_i_distance = r_elem_dist(node_ids[0]);
const double node_j_distance = r_elem_dist(node_ids[1]);

if (r_elem_edge_dist_extra(i_edge) > 0) {
r_elem_edge_dist_extra(i_edge) = std::abs( node_i_distance / (node_j_distance - node_i_distance) );
}
}
}
}

#pragma omp critical
{
mModifiedDistancesIDs.insert(mModifiedDistancesIDs.end(),aux_modified_distances_ids.begin(),aux_modified_distances_ids.end());
mModifiedElementalDistancesValues.insert(mModifiedElementalDistancesValues.end(),aux_modified_elemental_distances.begin(),aux_modified_elemental_distances.end());
}
}
} else {
#pragma omp parallel for
for (int i_chunk = 0; i_chunk < num_chunks; ++i_chunk)
{
auto elems_begin = r_elems.begin() + partition_vec[i_chunk];
auto elems_end = r_elems.begin() + partition_vec[i_chunk + 1];

std::vector<std::size_t> aux_modified_distances_ids;
std::vector<Vector> aux_modified_elemental_distances;

for (auto it_elem = elems_begin; it_elem < elems_end; ++it_elem) {
const double tol_d = mDistanceThreshold * (it_elem->GetGeometry()).Length();

bool is_saved = false;
Vector &r_elem_dist = it_elem->GetValue(ELEMENTAL_DISTANCES);
for (std::size_t i_node = 0; i_node < r_elem_dist.size(); ++i_node){
if (std::abs(r_elem_dist(i_node)) < tol_d){
if (!is_saved){
aux_modified_distances_ids.push_back(it_elem->Id());
aux_modified_elemental_distances.push_back(r_elem_dist);
is_saved = true;
}
r_elem_dist(i_node) = r_elem_dist(i_node) > 0.0 ? tol_d : -tol_d;
}
}
}

#pragma omp critical
{
mModifiedDistancesIDs.insert(mModifiedDistancesIDs.end(),aux_modified_distances_ids.begin(),aux_modified_distances_ids.end());
mModifiedElementalDistancesValues.insert(mModifiedElementalDistancesValues.end(),aux_modified_elemental_distances.begin(),aux_modified_elemental_distances.end());
}
}
}
}

this->SetDiscontinuousDistanceToSplitFlag();
}

void DistanceModificationProcess::RecoverDeactivationPreviousState()
{
#pragma omp parallel for
for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem){
auto it_elem = mrModelPart.ElementsBegin() + i_elem;
it_elem->Set(ACTIVE,true);
}
if ((mDoubleVariablesList.size() > 0.0) || (mComponentVariablesList.size() > 0.0)){
#pragma omp parallel for
for (int i_node = 0; i_node < static_cast<int>(mrModelPart.NumberOfNodes()); ++i_node){
auto it_node = mrModelPart.NodesBegin() + i_node;
if (it_node->GetValue(EMBEDDED_IS_ACTIVE) == 0){
for (std::size_t i_var = 0; i_var < mDoubleVariablesList.size(); i_var++){
const auto& r_double_var = *mDoubleVariablesList[i_var];
it_node->Free(r_double_var);
}
for (std::size_t i_comp = 0; i_comp < mComponentVariablesList.size(); i_comp++){
const auto& r_component_var = *mComponentVariablesList[i_comp];
it_node->Free(r_component_var);
}
}
}
}
}

void DistanceModificationProcess::RecoverOriginalDistance()
{
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mModifiedDistancesIDs.size()); ++i) {
const auto node_id = mModifiedDistancesIDs[i];
mrModelPart.GetNode(node_id).FastGetSolutionStepValue(DISTANCE) = mModifiedDistancesValues[i];
}

mrModelPart.GetCommunicator().SynchronizeCurrentDataToMin(DISTANCE);

mModifiedDistancesIDs.resize(0);
mModifiedDistancesValues.resize(0);
mModifiedDistancesIDs.shrink_to_fit();
mModifiedDistancesValues.shrink_to_fit();

this->SetContinuousDistanceToSplitFlag();
}

void DistanceModificationProcess::RecoverOriginalDiscontinuousDistance()
{    
#pragma omp parallel for
for (int i_elem = 0; i_elem < static_cast<int>(mModifiedDistancesIDs.size()); ++i_elem) {
const std::size_t elem_id = mModifiedDistancesIDs[i_elem];
const auto& r_elem_dist = mModifiedElementalDistancesValues[i_elem];
mrModelPart.GetElement(elem_id).GetValue(ELEMENTAL_DISTANCES) = r_elem_dist;
}

mModifiedDistancesIDs.resize(0);
mModifiedElementalDistancesValues.resize(0);
mModifiedDistancesIDs.shrink_to_fit();
mModifiedElementalDistancesValues.shrink_to_fit();

this->SetDiscontinuousDistanceToSplitFlag();
}

void DistanceModificationProcess::DeactivateFullNegativeElements()
{
ModelPart::NodesContainerType& rNodes = mrModelPart.Nodes();
ModelPart::ElementsContainerType& rElements = mrModelPart.Elements();

#pragma omp parallel for
for (int i_node = 0; i_node < static_cast<int>(rNodes.size()); ++i_node){
ModelPart::NodesContainerType::iterator it_node = rNodes.begin() + i_node;
it_node->SetValue(EMBEDDED_IS_ACTIVE, 0);
}

#pragma omp parallel for
for (int k = 0; k < static_cast<int>(rElements.size()); ++k){
std::size_t n_neg = 0;
ModelPart::ElementsContainerType::iterator itElement = rElements.begin() + k;
auto& rGeometry = itElement->GetGeometry();

for (std::size_t i_node=0; i_node<rGeometry.size(); i_node++){
if (rGeometry[i_node].FastGetSolutionStepValue(DISTANCE) < 0.0){
n_neg++;
}
}

(n_neg == rGeometry.size()) ? itElement->Set(ACTIVE, false) : itElement->Set(ACTIVE, true);

if (itElement->Is(ACTIVE)){
for (std::size_t i_node = 0; i_node < rGeometry.size(); ++i_node){
int& activation_index = rGeometry[i_node].GetValue(EMBEDDED_IS_ACTIVE);
#pragma omp atomic
activation_index += 1;
}
}
}

mrModelPart.GetCommunicator().AssembleNonHistoricalData(EMBEDDED_IS_ACTIVE);

if ((mDoubleVariablesList.size() > 0.0) || (mComponentVariablesList.size() > 0.0)){
#pragma omp parallel for
for (int i_node = 0; i_node < static_cast<int>(rNodes.size()); ++i_node){
auto it_node = rNodes.begin() + i_node;
if (it_node->GetValue(EMBEDDED_IS_ACTIVE) == 0){
for (std::size_t i_var = 0; i_var < mDoubleVariablesList.size(); i_var++){
const auto& r_double_var = *mDoubleVariablesList[i_var];
it_node->Fix(r_double_var);
it_node->FastGetSolutionStepValue(r_double_var) = 0.0;
}
for (std::size_t i_comp = 0; i_comp < mComponentVariablesList.size(); i_comp++){
const auto& r_component_var = *mComponentVariablesList[i_comp];
it_node->Fix(r_component_var);
it_node->FastGetSolutionStepValue(r_component_var) = 0.0;
}
}
}
}
}

void DistanceModificationProcess::SetContinuousDistanceToSplitFlag()
{
if( mrModelPart.NumberOfElements()>0){
std::size_t nodes_per_element = mrModelPart.ElementsBegin()->GetGeometry().PointsNumber();
std::vector<double> elem_dist(nodes_per_element, 0.0);
#pragma omp parallel for firstprivate(elem_dist)
for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem) {
auto it_elem = mrModelPart.ElementsBegin() + i_elem;
auto &r_geom = it_elem->GetGeometry();
if(elem_dist.size() != r_geom.PointsNumber()) elem_dist.resize(r_geom.PointsNumber());
for (std::size_t i_node = 0; i_node < r_geom.PointsNumber(); ++i_node) {
elem_dist[i_node] = r_geom[i_node].FastGetSolutionStepValue(DISTANCE);
}
this->SetElementToSplitFlag(*it_elem, elem_dist);
}
}
}

void DistanceModificationProcess::SetDiscontinuousDistanceToSplitFlag()
{
#pragma omp parallel for
for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem) {
auto it_elem = mrModelPart.ElementsBegin() + i_elem;
const auto &r_elem_dist = it_elem->GetValue(ELEMENTAL_DISTANCES);
this->SetElementToSplitFlag(*it_elem, r_elem_dist);
}
}

void DistanceModificationProcess::CheckAndStoreVariablesList(const std::vector<std::string>& rVariableStringArray)
{
if (mrModelPart.GetCommunicator().LocalMesh().NumberOfNodes() != 0) {
const auto& r_node = *mrModelPart.NodesBegin();
for (std::size_t i_variable=0; i_variable < rVariableStringArray.size(); i_variable++){
if (KratosComponents<Variable<double>>::Has(rVariableStringArray[i_variable])) {
const auto& r_double_var  = KratosComponents<Variable<double>>::Get(rVariableStringArray[i_variable]);
KRATOS_CHECK_DOF_IN_NODE(r_double_var, r_node);
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(r_double_var, r_node)

mDoubleVariablesList.push_back(&r_double_var);
}
else if (KratosComponents<ComponentType>::Has(rVariableStringArray[i_variable])){
const auto& r_component_var  = KratosComponents<ComponentType>::Get(rVariableStringArray[i_variable]);
KRATOS_CHECK_DOF_IN_NODE(r_component_var, r_node);
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(r_component_var, r_node)

mComponentVariablesList.push_back(&r_component_var);
}
else if (KratosComponents<Variable<array_1d<double,3>>>::Has(rVariableStringArray[i_variable])){
const auto& r_vector_var = KratosComponents<Variable<array_1d<double,3>>>::Get(rVariableStringArray[i_variable]);
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(r_vector_var, r_node)

const auto& r_component_var_x = KratosComponents<ComponentType>::Get(rVariableStringArray[i_variable]+"_X");
KRATOS_CHECK_DOF_IN_NODE(r_component_var_x, r_node);
mComponentVariablesList.push_back(&r_component_var_x);

const auto& r_component_var_y = KratosComponents<ComponentType>::Get(rVariableStringArray[i_variable]+"_Y");
KRATOS_CHECK_DOF_IN_NODE(r_component_var_y, r_node);
mComponentVariablesList.push_back(&r_component_var_y);

if (mrModelPart.GetProcessInfo()[DOMAIN_SIZE] == 3) {
const auto& r_component_var_z = KratosComponents<ComponentType>::Get(rVariableStringArray[i_variable]+"_Z");
KRATOS_CHECK_DOF_IN_NODE(r_component_var_z, r_node);
mComponentVariablesList.push_back(&r_component_var_z);
}
}
else {
KRATOS_ERROR << "The variable defined in the list is not a double variable nor a component variable. Given variable: " << rVariableStringArray[i_variable] << std::endl;
}
}
}
}

const std::array<std::size_t,2> DistanceModificationProcess::GetNodeIDs(
const std::size_t NumEdges, 
const std::size_t EdgeID) 
{
switch (NumEdges)
{
case 3:
return NodeIDs2D[EdgeID];
case 6:
return NodeIDs3D[EdgeID];
default:
KRATOS_ERROR << "The number of edges does not correspond to any supported element type (Triangle2D3 and Tetrahedra3D4). The number of edges is: " << NumEdges << std::endl;
}
}



};  
