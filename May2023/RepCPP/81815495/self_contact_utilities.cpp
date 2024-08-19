

#include <unordered_set>
#include <unordered_map>


#include "utilities/parallel_utilities.h"
#include "contact_structural_mechanics_application_variables.h"
#include "custom_utilities/self_contact_utilities.h"
#include "utilities/variable_utils.h"

namespace Kratos
{
namespace SelfContactUtilities
{
void ComputeSelfContactPairing(
ModelPart& rModelPart,
const std::size_t EchoLevel
)
{
KRATOS_TRY

std::unordered_set<std::size_t> ids_to_clear;

auto& r_conditions_array = rModelPart.Conditions();
const int num_conditions = static_cast<int>(r_conditions_array.size());
const auto it_cond_begin = r_conditions_array.begin();

auto& r_nodes_array = rModelPart.Nodes();
VariableUtils().ResetFlag(SLAVE, r_nodes_array);
VariableUtils().ResetFlag(MASTER, r_nodes_array);
VariableUtils().ResetFlag(SLAVE, r_conditions_array);
VariableUtils().ResetFlag(MASTER, r_conditions_array);

VariableUtils().ResetFlag(ACTIVE, r_conditions_array);

std::unordered_set<std::size_t> conditions_index_set;
conditions_index_set.reserve(num_conditions);
conditions_index_set.insert(it_cond_begin->Id());
std::vector<Condition::Pointer> ordered_conditions(num_conditions, nullptr);
ordered_conditions[0] = *(it_cond_begin.base());

std::unordered_map<std::vector<std::size_t>, std::vector<Condition::Pointer>, VectorIndexHasher<std::vector<std::size_t>>, VectorIndexComparor<std::vector<std::size_t>>> boundaries_map;
std::vector<std::size_t> boundary_vector;
for(int i = 1; i < num_conditions; ++i) {
auto it_cond = it_cond_begin + i;

auto& r_geometry = it_cond->GetGeometry();
const auto boundaries = r_geometry.GenerateBoundariesEntities();
for (auto& r_boundary : boundaries) {
boundary_vector.clear();
boundary_vector.reserve(r_boundary.size());
for (auto& r_node : r_boundary) {
boundary_vector.push_back(r_node.Id());
}
std::sort(boundary_vector.begin(), boundary_vector.end());
const auto& it_map = boundaries_map.find(boundary_vector);
if (it_map != boundaries_map.end()) {
it_map->second.push_back(*(it_cond.base()));
} else {
boundaries_map.insert(std::pair<std::vector<std::size_t>, std::vector<Condition::Pointer>>(boundary_vector, std::vector<Condition::Pointer>({*(it_cond.base())})));
}
}
}

std::size_t counter = 1;
bool inserted = false;
for(auto& p_current_condition : ordered_conditions) {
inserted = false;

auto& r_geometry = p_current_condition->GetGeometry();
const auto boundaries = r_geometry.GenerateBoundariesEntities();
for (auto& r_boundary : boundaries) {
boundary_vector.clear();
boundary_vector.reserve(r_boundary.size());
for (auto& r_node : r_boundary) {
boundary_vector.push_back(r_node.Id());
}
std::sort(boundary_vector.begin(), boundary_vector.end());

const auto& it_map = boundaries_map.find(boundary_vector);
if (it_map != boundaries_map.end()) {
for (auto& p_cond : it_map->second) {
if (conditions_index_set.find(p_cond->Id()) == conditions_index_set.end()) {
conditions_index_set.insert(p_cond->Id());
ordered_conditions[counter] = p_cond;
inserted = true;
++counter;
break;
}
}
if (inserted) {
break;
}
}
}

if (!inserted) {
for(int i = 1; i < num_conditions; ++i) {
auto it_cond = it_cond_begin + i;
if (conditions_index_set.find(it_cond->Id()) == conditions_index_set.end()) {
conditions_index_set.insert(it_cond->Id());
ordered_conditions[counter] = *(it_cond.base());
++counter;
break;
}
}
}
}

KRATOS_ERROR_IF_NOT(static_cast<int>(conditions_index_set.size()) == num_conditions) << "Condition set is not fully filled" << std::endl;
KRATOS_ERROR_IF_NOT(static_cast<int>(ordered_conditions.size()) == num_conditions) << "Condition vector is not fully filled" << std::endl;

for(auto& p_cond : ordered_conditions) {
auto& r_slave_geometry = p_cond->GetGeometry();

auto p_indexes_pairs = p_cond->GetValue(INDEX_MAP);

if (p_cond->IsNotDefined(MASTER) || p_cond->IsNot(MASTER)) {
if (p_indexes_pairs->size() > 0) {
ids_to_clear.clear();
for (auto it_pair = p_indexes_pairs->begin(); it_pair != p_indexes_pairs->end(); ++it_pair ) {
const IndexType master_id = p_indexes_pairs->GetId(it_pair); 
auto p_master_cond = rModelPart.pGetCondition(master_id);
if (p_master_cond->IsNotDefined(MASTER) || p_master_cond->IsNot(MASTER)) {
auto& r_master_geometry = p_master_cond->GetGeometry();

bool shared_nodes = false;
for (auto& r_node_slave : r_slave_geometry) {
for (auto& r_node_master : r_master_geometry) {
if (r_node_master.Id() == r_node_slave.Id()) {
shared_nodes = true;
break;
}
}
}
if (shared_nodes) {
continue;
}

std::size_t counter = 0;
for (auto& r_node : r_master_geometry) {
if (r_node.IsNotDefined(MASTER) || r_node.Is(MASTER)) {
++counter;
}
}

if (counter == r_master_geometry.size()) {
p_master_cond->GetValue(INDEX_MAP)->clear();
p_master_cond->Set(MASTER, true);
p_master_cond->Set(SLAVE, false);
for (auto& r_node : r_master_geometry) {
r_node.Set(MASTER, true);
r_node.Set(SLAVE, false);
}
} else {
ids_to_clear.insert(master_id);
}
}
}
for (std::size_t id : ids_to_clear) {
p_indexes_pairs->RemoveId(id);
}
}
if (p_indexes_pairs->size() > 0) {
p_cond->Set(MASTER, false);
p_cond->Set(SLAVE, true);
for (auto& r_node : r_slave_geometry) {
r_node.Set(MASTER, false);
r_node.Set(SLAVE, true);
}
}
} else { 
p_indexes_pairs->clear();
}
}

struct counter_containers {std::size_t master = 0, slave = 0;};
block_for_each(r_conditions_array, counter_containers(), [&EchoLevel](Condition& rCond, counter_containers& counter) {
counter.master = 0;
counter.slave = 0;

auto& r_geometry = rCond.GetGeometry();
const std::size_t number_of_nodes = r_geometry.size();

for (auto& r_node : r_geometry) {
if (r_node.Is(MASTER)) {
++counter.master;
}
if (r_node.Is(SLAVE)) {
++counter.slave;
}
}

KRATOS_ERROR_IF((counter.slave + counter.master) > number_of_nodes) << "The MASTER/SLAVE flags are inconsistent" << std::endl;

if (counter.slave == number_of_nodes || counter.master == number_of_nodes) {
rCond.Set(ACTIVE, true);
} else {
KRATOS_WARNING_IF("SelfContactUtilities", EchoLevel > 0) << "Condition " << rCond.Id() << " must be isolated for sharing MASTER/SLAVE nodes in it" << std::endl;
rCond.Set(ACTIVE, false);
rCond.Set(SLAVE, false);
rCond.Set(MASTER, true);
}
});

KRATOS_CATCH("")
}




void FullAssignmentOfPairs(ModelPart& rModelPart)
{
KRATOS_TRY

for (auto& r_cond : rModelPart.Conditions()) {
r_cond.SetValue(INDEX_MAP, Kratos::make_shared<IndexMap>());
}
for (auto& r_cond_1 : rModelPart.Conditions()) {
auto p_pairs = r_cond_1.GetValue(INDEX_MAP);
for (auto& r_cond_2 : rModelPart.Conditions()) {
if (r_cond_1.Id() != r_cond_2.Id()) {
p_pairs->AddId(r_cond_2.Id());
}
}
}

KRATOS_CATCH("")
}




void NotPredefinedMasterSlave(ModelPart& rModelPart)
{
KRATOS_TRY

auto& r_conditions_array = rModelPart.Conditions();
const auto it_cond_begin = r_conditions_array.begin();
const int num_conditions = static_cast<int>(r_conditions_array.size());

std::vector<IndexType> master_conditions_ids;

#pragma omp parallel
{
std::vector<IndexType> master_conditions_ids_buffer;

#pragma omp for
for(int i = 0; i < num_conditions; ++i) {
auto it_cond = it_cond_begin + i;
IndexMap::Pointer p_indexes_pairs = it_cond->GetValue(INDEX_MAP);
if (p_indexes_pairs->size() > 0) {
it_cond->Set(SLAVE, true);
for (auto& i_pair : *p_indexes_pairs) {
master_conditions_ids_buffer.push_back(i_pair.first);
}
}
}

#pragma omp critical
{
std::move(master_conditions_ids_buffer.begin(),master_conditions_ids_buffer.end(),back_inserter(master_conditions_ids));
}
}

rModelPart.CreateSubModelPart("AuxMasterModelPart");
ModelPart& aux_model_part = rModelPart.GetSubModelPart("AuxMasterModelPart");

std::sort( master_conditions_ids.begin(), master_conditions_ids.end() );
master_conditions_ids.erase( std::unique( master_conditions_ids.begin(), master_conditions_ids.end() ), master_conditions_ids.end() );

aux_model_part.AddConditions(master_conditions_ids);

VariableUtils().SetFlag(MASTER, true, aux_model_part.Conditions());

rModelPart.RemoveSubModelPart("AuxMasterModelPart");

block_for_each(r_conditions_array, [&](Condition& rCond) {
if (rCond.Is(SLAVE)) {
GeometryType& r_geometry = rCond.GetGeometry();
for (NodeType& r_node : r_geometry) {
r_node.SetLock();
r_node.Set(SLAVE, true);
r_node.UnSetLock();
}
}
if (rCond.Is(MASTER)) {
GeometryType& r_geometry = rCond.GetGeometry();
for (NodeType& r_node : r_geometry) {
r_node.SetLock();
r_node.Set(MASTER, true);
r_node.UnSetLock();
}
}
});

KRATOS_CATCH("")
}

} 
} 
