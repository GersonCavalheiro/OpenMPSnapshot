
#include "custom_processes/regenerate_pfem_pressure_conditions_process.h"
#include "processes/generic_find_elements_neighbours_process.h"

namespace Kratos {

template <SizeType TDim>
RegeneratePfemPressureConditionsProcess<TDim>::RegeneratePfemPressureConditionsProcess(
ModelPart& rModelPart)
: mrModelPart(rModelPart)
{
}



template <SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::CreateLineLoads(
const int Id1,
const int Id2,
ElementIterator itElem,
ModelPart& rSubModelPart,
ModelPart::PropertiesType::Pointer pProperties,
int& rMaximumConditionId
)
{
std::vector<IndexType> condition_nodes_id(2);
auto& r_geom = (itElem)->GetGeometry();
condition_nodes_id[0] = r_geom[Id1].Id();
condition_nodes_id[1] = r_geom[Id2].Id();
rMaximumConditionId++;
rSubModelPart.AddNodes(condition_nodes_id);

const auto p_line_cond = rSubModelPart.CreateNewCondition(
"LineLoadCondition2D2N",
rMaximumConditionId,
condition_nodes_id,
pProperties, 0);

mrModelPart.GetSubModelPart("computing_domain").AddCondition(p_line_cond);
}



template <SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::CreatePressureLoads(
const int Id1,
const int Id2,
const int Id3,
ElementIterator itElem,
ModelPart& rSubModelPart,
ModelPart::PropertiesType::Pointer pProperties,
int& rMaximumConditionId
)
{
auto& r_geom = (itElem)->GetGeometry();
std::vector<IndexType> condition_nodes_id(3);
condition_nodes_id[0] = r_geom[Id1].Id();
condition_nodes_id[1] = r_geom[Id2].Id();
condition_nodes_id[2] = r_geom[Id3].Id();
rMaximumConditionId++;

rSubModelPart.AddNodes(condition_nodes_id);

const auto p_pressure_condition = rSubModelPart.CreateNewCondition(
"SurfaceLoadCondition3D3N",
rMaximumConditionId,
condition_nodes_id,
pProperties, 0);

mrModelPart.GetSubModelPart("computing_domain").AddCondition(p_pressure_condition);
}



template<SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::GenerateLineLoads2Nodes(
const int NonWetLocalIdNode,
int& rMaximumConditionId,
ElementIterator itElem
)
{
auto& r_sub_model_part = mrModelPart.GetSubModelPart("PFEMPressureConditions");
const auto it_cond = mrModelPart.ConditionsBegin();
ModelPart::PropertiesType::Pointer p_properties = it_cond->pGetProperties();

auto& r_elem_neigb = (itElem)->GetValue(NEIGHBOUR_ELEMENTS);
if (r_elem_neigb(NonWetLocalIdNode).get()!=nullptr && r_elem_neigb[NonWetLocalIdNode].Id() == (itElem)->Id()) {
const IndexType id_2 = (NonWetLocalIdNode == 0) ? 1 : (NonWetLocalIdNode == 1) ? 2 : 0;
const IndexType id_3 = (NonWetLocalIdNode == 0) ? 2 : (NonWetLocalIdNode == 1) ? 0 : 1;
this->CreateLineLoads(id_2, id_3, itElem, r_sub_model_part, p_properties, rMaximumConditionId);
}
}



template<SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::GenerateLineLoads3Nodes(
int& rMaximumConditionId,
ElementIterator itElem
)
{
auto& r_sub_model_part = mrModelPart.GetSubModelPart("PFEMPressureConditions");
const auto it_cond = mrModelPart.ConditionsBegin();
ModelPart::PropertiesType::Pointer p_properties = it_cond->pGetProperties();

GlobalPointersVector<Element>& r_elem_neigb = (itElem)->GetValue(NEIGHBOUR_ELEMENTS);

IndexType alone_edge_local_id = 10;
int number_of_free_edges = 0, non_free_edge;
for (IndexType i = 0; i < r_elem_neigb.size(); ++i) {
if (r_elem_neigb(i).get()!=nullptr && (itElem)->Id() == r_elem_neigb[i].Id()) {
alone_edge_local_id = i;
number_of_free_edges++;
} else {
non_free_edge = i;
}
}
if (number_of_free_edges == 2) {
const IndexType id_1 = (non_free_edge == 0) ? 0 : (non_free_edge == 1) ? 1 : 2;
const IndexType id_2 = (non_free_edge == 0) ? 1 : (non_free_edge == 1) ? 2 : 0;
const IndexType id_3 = (non_free_edge == 0) ? 2 : (non_free_edge == 1) ? 0 : 1;
this->CreateLineLoads(id_1, id_2, itElem, r_sub_model_part, p_properties, rMaximumConditionId);
this->CreateLineLoads(id_3, id_1, itElem, r_sub_model_part, p_properties, rMaximumConditionId);
} else if (number_of_free_edges == 1) {
const IndexType id_2 = (alone_edge_local_id == 0) ? 1 : (alone_edge_local_id == 1) ? 2 : 0;
const IndexType id_3 = (alone_edge_local_id == 0) ? 2 : (alone_edge_local_id == 1) ? 0 : 1;
this->CreateLineLoads(id_3, id_2, itElem, r_sub_model_part, p_properties, rMaximumConditionId);
}
}



template<SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::GeneratePressureLoads3WetNodes(
const int NonWetLocalIdNode,
int& rMaximumConditionId,
ElementIterator itElem
)
{
auto& r_sub_model_part = mrModelPart.GetSubModelPart("PFEMPressureConditions");
auto it_cond = mrModelPart.ConditionsBegin();
ModelPart::PropertiesType::Pointer p_properties = it_cond->pGetProperties();

const IndexType id_2 = (NonWetLocalIdNode == 0) ? 3 : (NonWetLocalIdNode == 1) ? 0 : (NonWetLocalIdNode == 2) ? 3 : 0;
const IndexType id_3 = (NonWetLocalIdNode == 0) ? 2 : (NonWetLocalIdNode == 1) ? 2 : (NonWetLocalIdNode == 2) ? 1 : 1;
const IndexType id_4 = (NonWetLocalIdNode == 0) ? 1 : (NonWetLocalIdNode == 1) ? 3 : (NonWetLocalIdNode == 2) ? 0 : 2;

auto& r_elem_neigb = (itElem)->GetValue(NEIGHBOUR_ELEMENTS);
if (r_elem_neigb(NonWetLocalIdNode).get()!=nullptr && r_elem_neigb[NonWetLocalIdNode].Id() == (itElem)->Id()) {
this->CreatePressureLoads(id_2, id_3, id_4, itElem, r_sub_model_part, p_properties, rMaximumConditionId);
}
}



template<SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::GeneratePressureLoads4WetNodes(
int& rMaximumConditionId,
ElementIterator itElem
)
{
auto& r_sub_model_part = mrModelPart.GetSubModelPart("PFEMPressureConditions");
auto it_cond = mrModelPart.ConditionsBegin();
ModelPart::PropertiesType::Pointer p_properties = it_cond->pGetProperties();
const std::size_t id = (itElem)->Id();

auto& r_elem_neigb = (itElem)->GetValue(NEIGHBOUR_ELEMENTS);

for (unsigned int i = 0; i < r_elem_neigb.size(); i++) {
if (r_elem_neigb(i).get()!=nullptr && r_elem_neigb[i].Id() == id) { 
const int excluded_local_id_node = i;
const IndexType id_1 = (excluded_local_id_node == 0) ? 3 : (excluded_local_id_node == 1) ? 0 : (excluded_local_id_node == 2) ? 3 : 0;
const IndexType id_2 = (excluded_local_id_node == 0) ? 2 : (excluded_local_id_node == 1) ? 2 : (excluded_local_id_node == 2) ? 1 : 1;
const IndexType id_3 = (excluded_local_id_node == 0) ? 1 : (excluded_local_id_node == 1) ? 3 : (excluded_local_id_node == 2) ? 0 : 2;
this->CreatePressureLoads(id_1, id_2, id_3, itElem, r_sub_model_part, p_properties, rMaximumConditionId);
}
}
}



template <SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::Execute()
{
auto find_neigh = GenericFindElementalNeighboursProcess(mrModelPart);
find_neigh.Execute();

if (!mrModelPart.HasSubModelPart("PFEMPressureConditions")) {
mrModelPart.CreateSubModelPart("PFEMPressureConditions");
}
this->RemovePreviousPressureLoads();

this->CreateNewConditions();
}



template <SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::GetMaximumConditionIdOnSubmodelPart(
int& rMaximumConditionId
)
{
rMaximumConditionId = 0;
for (auto it_cond = mrModelPart.ConditionsBegin(); it_cond != mrModelPart.ConditionsEnd(); it_cond++) {
rMaximumConditionId = (((*it_cond)).Id() > rMaximumConditionId) ? ((*it_cond)).Id() : rMaximumConditionId;
}
}



template <SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::RemovePreviousPressureLoads()
{
auto& r_sub_model = mrModelPart.GetSubModelPart("PFEMPressureConditions");
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(r_sub_model.Conditions().size()); i++) {
auto it_cond = r_sub_model.ConditionsBegin() + i;
it_cond->Set(TO_ERASE, true);
}
mrModelPart.RemoveConditionsFromAllLevels(TO_ERASE);
}



template <SizeType TDim>
void RegeneratePfemPressureConditionsProcess<TDim>::ResetFlagOnElements()
{
const auto it_elem_begin = mrModelPart.ElementsBegin();
#pragma omp parallel for
for(int i = 0; i < static_cast<int>(mrModelPart.Elements().size()); i++) {
auto it_elem = it_elem_begin + i;
it_elem->SetValue(SMOOTHING, false);
}
}



template <>
void RegeneratePfemPressureConditionsProcess<2>::CreateNewConditions()
{
int maximum_condition_id;
this->GetMaximumConditionIdOnSubmodelPart(maximum_condition_id);

for (auto it_elem = mrModelPart.Elements().ptr_begin(); it_elem != mrModelPart.Elements().ptr_end(); ++it_elem) {

const auto& r_geometry = (*it_elem)->GetGeometry();
int wet_nodes_counter = 0, non_wet_local_id_node = 10;

for (IndexType local_id = 0; local_id < r_geometry.PointsNumber(); ++local_id) {
if (std::abs(r_geometry[local_id].FastGetSolutionStepValue(PRESSURE)) > 0.0) {
wet_nodes_counter++;
} else {
non_wet_local_id_node = local_id;
}
}
if (wet_nodes_counter == 2) {
this->GenerateLineLoads2Nodes(non_wet_local_id_node, maximum_condition_id, it_elem);
} else if (wet_nodes_counter == 3) {
this->GenerateLineLoads3Nodes(maximum_condition_id, it_elem);
}

}
}



template <>
void RegeneratePfemPressureConditionsProcess<3>::CreateNewConditions()
{
int maximum_condition_id;
this->GetMaximumConditionIdOnSubmodelPart(maximum_condition_id);
const auto it_elem_begin = mrModelPart.ElementsBegin();

for (int i = 0; i < static_cast<int>(mrModelPart.Elements().size()); i++) {
auto it_elem = it_elem_begin + i;
const  auto& r_geometry = (it_elem)->GetGeometry();
int wet_nodes_counter = 0, non_wet_local_id_node = 10;

for (IndexType local_id = 0; local_id < r_geometry.PointsNumber(); ++local_id) {
if (std::abs(r_geometry[local_id].FastGetSolutionStepValue(PRESSURE)) > 0.0) {
wet_nodes_counter++;
} else {
non_wet_local_id_node = local_id;
}
}
if (wet_nodes_counter == 3) {
this->GeneratePressureLoads3WetNodes(non_wet_local_id_node, maximum_condition_id, it_elem);
} else if (wet_nodes_counter == 4) {
this->GeneratePressureLoads4WetNodes(maximum_condition_id, it_elem);
}
}
}




template class RegeneratePfemPressureConditionsProcess<3>;
template class RegeneratePfemPressureConditionsProcess<2>;

}  
