





#include "custom_processes/multiscale_refining_process.h"
#include "geometries/point.h"
#include "processes/fast_transfer_between_model_parts_process.h"
#include "utilities/assign_unique_model_part_collection_tag_utility.h"
#include "custom_includes/meshing_flags.h"

namespace Kratos
{

MultiscaleRefiningProcess::MultiscaleRefiningProcess(
ModelPart& rThisCoarseModelPart,
ModelPart& rThisRefinedModelPart,
ModelPart& rThisVisualizationModelPart,
Parameters ThisParameters)
: mrCoarseModelPart(rThisCoarseModelPart)
, mrRefinedModelPart(rThisRefinedModelPart)
, mrVisualizationModelPart(rThisVisualizationModelPart)
, mParameters(ThisParameters)
, mUniformRefinement(mrRefinedModelPart)
{
mParameters.ValidateAndAssignDefaults(GetDefaultParameters());

mDivisionsAtSubscale = mParameters["number_of_divisions_at_subscale"].GetInt();
mEchoLevel = mParameters["echo_level"].GetInt();

std::string interface_base_name = mParameters["subscale_interface_base_name"].GetString();
mRefinedInterfaceName = interface_base_name + "_" + std::to_string(mrCoarseModelPart.GetValue(SUBSCALE_INDEX) + 1);
mInterfaceConditionName = mParameters["subscale_boundary_condition"].GetString();

KRATOS_INFO_IF("MultiscaleRefiningProcess", mEchoLevel > 1) << "Parameters\n" << mParameters.Info() << std::endl;

mStepDataSize = mrCoarseModelPart.GetNodalSolutionStepDataSize();

Check();

InitializeCoarseModelPartInterface();

InitializeRefinedModelPartInterface();

}


int MultiscaleRefiningProcess::Check()
{
KRATOS_TRY

KRATOS_CHECK(KratosComponents<Condition>::Has(mInterfaceConditionName));

KRATOS_CHECK_NOT_EQUAL(mDivisionsAtSubscale, 0);

KRATOS_CHECK_EQUAL(mStepDataSize, mrRefinedModelPart.GetNodalSolutionStepDataSize());
KRATOS_CHECK_EQUAL(mStepDataSize, mrVisualizationModelPart.GetNodalSolutionStepDataSize());

return 0;

KRATOS_CATCH("")
}


void MultiscaleRefiningProcess::ExecuteRefinement()
{
IndexIndexMapType node_tag, elem_tag, cond_tag;
AssignUniqueModelPartCollectionTagUtility model_part_collection(mrCoarseModelPart);
model_part_collection.ComputeTags(node_tag, cond_tag, elem_tag, mCollections);

IndexType node_id, elem_id, cond_id;
GetLastId(node_id, elem_id, cond_id);

CloneNodesToRefine(node_id);

MarkElementsFromNodalFlag();
MarkConditionsFromNodalFlag();

CreateElementsToRefine(elem_id, elem_tag);
CreateConditionsToRefine(cond_id, cond_tag);

IdentifyCurrentInterface();

int divisions = mrRefinedModelPart.GetValue(SUBSCALE_INDEX) * mDivisionsAtSubscale;
mUniformRefinement.SetCustomIds(node_id, elem_id, cond_id);
mUniformRefinement.Refine(divisions);
mUniformRefinement.GetLastCreatedIds(node_id, elem_id, cond_id);

UpdateRefinedInterface();

UpdateVisualizationAfterRefinement();

FinalizeRefinement();
}


void MultiscaleRefiningProcess::ExecuteCoarsening()
{
IdentifyParentNodesToCoarsen();
IdentifyElementsToErase();
IdentifyConditionsToErase();
IdentifyRefinedNodesToErase();

mUniformRefinement.RemoveRefinedEntities(TO_ERASE);

IdentifyCurrentInterface();
UpdateVisualizationAfterCoarsening();

FinalizeCoarsening();
}


void MultiscaleRefiningProcess::InitializeVisualizationModelPart(ModelPart& rReferenceModelPart, ModelPart& rNewModelPart)
{
InitializeNewModelPart(rReferenceModelPart, rNewModelPart);

VariablesList& variables_list = rNewModelPart.GetNodalSolutionStepVariablesList();
variables_list = rReferenceModelPart.GetNodalSolutionStepVariablesList();

FastTransferBetweenModelPartsProcess(rNewModelPart, rReferenceModelPart)();

StringVectorType sub_model_parts_names;
sub_model_parts_names = rReferenceModelPart.GetSubModelPartNames();

for (const auto& name : sub_model_parts_names)
{
ModelPart& destination = rNewModelPart.GetSubModelPart(name);
ModelPart& origin = rReferenceModelPart.GetSubModelPart(name);
FastTransferBetweenModelPartsProcess(destination, origin)();
}
}


void MultiscaleRefiningProcess::InitializeRefinedModelPart(ModelPart& rReferenceModelPart, ModelPart& rNewModelPart)
{
InitializeNewModelPart(rReferenceModelPart, rNewModelPart);

int subscale_index = rReferenceModelPart.GetValue(SUBSCALE_INDEX);
rNewModelPart.SetValue(SUBSCALE_INDEX, ++subscale_index);
}


void MultiscaleRefiningProcess::InitializeNewModelPart(ModelPart& rReferenceModelPart, ModelPart& rNewModelPart)
{
AddAllTablesToModelPart(rReferenceModelPart, rNewModelPart);
AddAllPropertiesToModelPart(rReferenceModelPart, rNewModelPart);

ProcessInfo& process_info = rNewModelPart.GetProcessInfo();
process_info = rReferenceModelPart.GetProcessInfo();

StringVectorType sub_model_parts_names;
sub_model_parts_names = rReferenceModelPart.GetSubModelPartNames();

for (const auto& name : sub_model_parts_names)
{
ModelPart& sub_model_part = rNewModelPart.CreateSubModelPart(name);

ModelPart& origin_model_part = rReferenceModelPart.GetSubModelPart(name);

AddAllTablesToModelPart(origin_model_part, sub_model_part);
AddAllPropertiesToModelPart(origin_model_part, sub_model_part);
}
}


void MultiscaleRefiningProcess::InitializeCoarseModelPartInterface()
{
if (mrCoarseModelPart.HasSubModelPart(mRefinedInterfaceName))
{
mrCoarseModelPart.GetSubModelPart(mRefinedInterfaceName).RemoveNodesFromAllLevels();
mrCoarseModelPart.GetSubModelPart(mRefinedInterfaceName).RemoveElementsFromAllLevels();
mrCoarseModelPart.GetSubModelPart(mRefinedInterfaceName).RemoveConditionsFromAllLevels();
}
else
mrCoarseModelPart.CreateSubModelPart(mRefinedInterfaceName);
}


void MultiscaleRefiningProcess::InitializeRefinedModelPartInterface()
{
if (mrRefinedModelPart.HasSubModelPart(mRefinedInterfaceName))
{
mrRefinedModelPart.GetSubModelPart(mRefinedInterfaceName).RemoveNodesFromAllLevels();
mrRefinedModelPart.GetSubModelPart(mRefinedInterfaceName).RemoveElementsFromAllLevels();
mrRefinedModelPart.GetSubModelPart(mRefinedInterfaceName).RemoveConditionsFromAllLevels();
}
else
mrRefinedModelPart.CreateSubModelPart(mRefinedInterfaceName);
}


void MultiscaleRefiningProcess::UpdateVisualizationAfterRefinement()
{
mrVisualizationModelPart.RemoveElementsFromAllLevels(MeshingFlags::REFINED);
mrVisualizationModelPart.RemoveConditionsFromAllLevels(MeshingFlags::REFINED);

ModelPart::NodesContainerType::iterator nodes_begin = mrCoarseModelPart.NodesBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrCoarseModelPart.Nodes().size()); i++)
{
auto coarse_node = nodes_begin + i;
if (coarse_node->Is((MeshingFlags::REFINED)) && (coarse_node->IsNot(INTERFACE)))
coarse_node->Set(INSIDE, true);
else
coarse_node->Set(INSIDE, false);
}

FastTransferBetweenModelPartsProcess(mrVisualizationModelPart, mrRefinedModelPart,
FastTransferBetweenModelPartsProcess::EntityTransfered::ALL, NEW_ENTITY)();
}


void MultiscaleRefiningProcess::UpdateVisualizationAfterCoarsening()
{
mrVisualizationModelPart.RemoveNodesFromAllLevels(TO_ERASE);
mrVisualizationModelPart.RemoveElementsFromAllLevels(TO_ERASE);
mrVisualizationModelPart.RemoveConditionsFromAllLevels(TO_ERASE);

FastTransferBetweenModelPartsProcess(mrVisualizationModelPart, mrCoarseModelPart,
FastTransferBetweenModelPartsProcess::EntityTransfered::ALL, MeshingFlags::TO_COARSEN)();
FastTransferBetweenModelPartsProcess(mrVisualizationModelPart, mrCoarseModelPart,
FastTransferBetweenModelPartsProcess::EntityTransfered::NODES, INTERFACE)();
}


void MultiscaleRefiningProcess::AddAllPropertiesToModelPart(ModelPart& rOriginModelPart, ModelPart& rDestinationModelPart)
{
const IndexType nprop = rOriginModelPart.NumberOfProperties();
ModelPart::PropertiesContainerType::iterator prop_begin = rOriginModelPart.PropertiesBegin();

for (IndexType i = 0; i < nprop; i++)
{
auto prop = prop_begin + i;
rDestinationModelPart.AddProperties(*prop.base());
}
}


void MultiscaleRefiningProcess::AddAllTablesToModelPart(ModelPart& rOriginModelPart, ModelPart& rDestinationModelPart)
{
const IndexType ntables = rOriginModelPart.NumberOfTables();
ModelPart::TablesContainerType::iterator table_begin = rOriginModelPart.TablesBegin();

for (IndexType i = 0; i < ntables; i++)
{
auto table = table_begin + i;
rDestinationModelPart.AddTable(table.base()->first, table.base()->second);
}
}


void MultiscaleRefiningProcess::MarkElementsFromNodalFlag()
{
const int nelems = static_cast<int>(mrCoarseModelPart.Elements().size());
ModelPart::ElementsContainerType::iterator elem_begin = mrCoarseModelPart.ElementsBegin();

const IndexType number_of_nodes = elem_begin->GetGeometry().size();

#pragma omp parallel for
for (int i = 0; i < nelems; i++)
{
auto elem = elem_begin + i;
bool to_refine = true;
bool new_entity = false;
for (IndexType node = 0; node < number_of_nodes; node++)
{
if (elem->GetGeometry()[node].IsNot(TO_REFINE))
to_refine = false;

if (elem->GetGeometry()[node].Is(NEW_ENTITY))
new_entity = true;
}
elem->Set(TO_REFINE, (to_refine && new_entity));
}
}


void MultiscaleRefiningProcess::MarkConditionsFromNodalFlag()
{
const int nconds = static_cast<int>(mrCoarseModelPart.Conditions().size());
ModelPart::ConditionsContainerType::iterator cond_begin = mrCoarseModelPart.ConditionsBegin();

const IndexType number_of_nodes = cond_begin->GetGeometry().size();

#pragma omp parallel for
for (int i = 0; i < nconds; i++)
{
auto cond = cond_begin + i;
bool to_refine = true;
bool new_entity = false;
for (IndexType node = 0; node < number_of_nodes; node++)
{
if (cond->GetGeometry()[node].IsNot(TO_REFINE))
to_refine = false;

if (cond->GetGeometry()[node].Is(NEW_ENTITY))
new_entity = true;
}
cond->Set(TO_REFINE, (to_refine && new_entity));
}
}


void MultiscaleRefiningProcess::CloneNodesToRefine(IndexType& rNodeId)
{
const int nnodes = static_cast<int>(mrCoarseModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator nodes_begin = mrCoarseModelPart.NodesBegin();

for (int i = 0; i < nnodes; i++)
{
auto coarse_node = nodes_begin + i;
if (coarse_node->Is(TO_REFINE))
{
if (coarse_node->IsNot(MeshingFlags::REFINED))
{
coarse_node->Set(NEW_ENTITY, true);
NodeType::Pointer new_node = coarse_node->Clone();
new_node->SetId(++rNodeId);
mrRefinedModelPart.AddNode(new_node);
new_node->Set(TO_REFINE, false);
new_node->GetValue(FATHER_NODES).resize(0);
new_node->GetValue(FATHER_NODES).push_back( NodeType::WeakPointer(*coarse_node.base()) );
new_node->GetValue(FATHER_NODES_WEIGHTS).resize(0);
new_node->GetValue(FATHER_NODES_WEIGHTS).push_back(1.0);
coarse_node->SetValue(SLAVE_NODE, new_node);
coarse_node->Set(MeshingFlags::REFINED, true);
}
}
}

StringVectorType sub_model_part_names = mrCoarseModelPart.GetSubModelPartNames();
for (const auto& name : sub_model_part_names)
{
ModelPart& coarse_sub_model_part = mrCoarseModelPart.GetSubModelPart(name);
ModelPart& refined_sub_model_part = mrRefinedModelPart.GetSubModelPart(name);

const int nnodes = static_cast<int>(coarse_sub_model_part.Nodes().size());
ModelPart::NodesContainerType::iterator nodes_begin = coarse_sub_model_part.NodesBegin();

for (int i = 0; i < nnodes; i++)
{
auto coarse_node = nodes_begin + i;
if (coarse_node->Is(NEW_ENTITY))
refined_sub_model_part.AddNode(coarse_node->GetValue(SLAVE_NODE));
}
}
}


void MultiscaleRefiningProcess::IdentifyParentNodesToCoarsen()
{
const int nnodes = static_cast<int>(mrCoarseModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator nodes_begin = mrCoarseModelPart.NodesBegin();

for (int i = 0; i < nnodes; i++)
{
auto coarse_node = nodes_begin + i;
if (coarse_node->IsNot(TO_REFINE))
{
if (coarse_node->Is(MeshingFlags::REFINED))
{
if (coarse_node->GetValue(SLAVE_NODE)->IsNot(MeshingFlags::REFINED))
{
coarse_node->Set(MeshingFlags::TO_COARSEN, true);
coarse_node->Set(MeshingFlags::REFINED, false);
coarse_node->SetValue(SLAVE_NODE, nullptr);
}
}
}
}
}


void MultiscaleRefiningProcess::IdentifyElementsToErase()
{
const int nelems_coarse = static_cast<int>(mrCoarseModelPart.Elements().size());
ModelPart::ElementsContainerType::iterator coarse_elem_begin = mrCoarseModelPart.ElementsBegin();

const IndexType element_nodes = coarse_elem_begin->GetGeometry().size();

#pragma omp parallel for
for (int i = 0; i < nelems_coarse; i++)
{
auto coarse_elem = coarse_elem_begin + i;
if (coarse_elem->Is(MeshingFlags::REFINED))
{
bool to_coarsen = false;
for (IndexType inode = 0; inode < element_nodes; inode++)
{
if (coarse_elem->GetGeometry()[inode].Is(MeshingFlags::TO_COARSEN))
to_coarsen = true;
}
coarse_elem->Set(MeshingFlags::TO_COARSEN, to_coarsen);
if (to_coarsen)
coarse_elem->Set(MeshingFlags::REFINED, false);
}
}

const int nelems_ref = static_cast<int>(mrRefinedModelPart.Elements().size());
ModelPart::ElementsContainerType::iterator refined_elem_begin = mrRefinedModelPart.ElementsBegin();

#pragma omp parallel for
for (int i = 0; i < nelems_ref; i++)
{
auto refined_elem = refined_elem_begin + i;
if ((refined_elem->GetValue(FATHER_ELEMENT))->Is(MeshingFlags::TO_COARSEN))
refined_elem->Set(TO_ERASE, true);
}
}


void MultiscaleRefiningProcess::IdentifyConditionsToErase()
{
const int nconds_coarse = static_cast<int>(mrCoarseModelPart.Conditions().size());
ModelPart::ConditionsContainerType::iterator coarse_cond_begin = mrCoarseModelPart.ConditionsBegin();

const IndexType condition_nodes = coarse_cond_begin->GetGeometry().size();

#pragma omp parallel for
for (int i = 0; i < nconds_coarse; i++)
{
auto coarse_cond = coarse_cond_begin + i;
if (coarse_cond->Is(MeshingFlags::REFINED))
{
bool to_coarsen = false;
for (IndexType inode = 0; inode < condition_nodes; inode++)
{
if (coarse_cond->GetGeometry()[inode].Is(MeshingFlags::TO_COARSEN))
to_coarsen = true;
}
coarse_cond->Set(MeshingFlags::TO_COARSEN, to_coarsen);
if (to_coarsen)
coarse_cond->Set(MeshingFlags::REFINED, false);
}
}

const int nconds_ref = static_cast<int>(mrRefinedModelPart.Conditions().size());
ModelPart::ConditionsContainerType::iterator refined_cond_begin = mrRefinedModelPart.ConditionsBegin();

#pragma omp parallel for
for (int i = 0; i < nconds_ref; i++)
{
auto refined_cond = refined_cond_begin + i;
if ((refined_cond->GetValue(FATHER_CONDITION))->Is(MeshingFlags::TO_COARSEN))
refined_cond->Set(TO_ERASE, true);
}
}


void MultiscaleRefiningProcess::IdentifyRefinedNodesToErase()
{
const IndexType nelems = mrRefinedModelPart.Elements().size();
if (nelems != 0) 
{
ModelPart::NodeIterator nodes_begin = mrRefinedModelPart.NodesBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrRefinedModelPart.Nodes().size()); i++)
{
auto node = nodes_begin + i;
node->Set(TO_ERASE, true);
}

ModelPart::ElementIterator elements_begin = mrRefinedModelPart.ElementsBegin();
const IndexType element_nodes = elements_begin->GetGeometry().size();

for (IndexType i = 0; i < nelems; i++)
{
auto elem = elements_begin + i;
if (elem->IsNot(TO_ERASE))
{
for (IndexType inode = 0; inode < element_nodes; inode++)
{
(elem->GetGeometry()[inode]).Set(TO_ERASE, false);
}
}
}
}
}


void MultiscaleRefiningProcess::CreateElementsToRefine(IndexType& rElemId, IndexIndexMapType& rElemTag)
{
const int nelems = static_cast<int>(mrCoarseModelPart.Elements().size());
ModelPart::ElementsContainerType::iterator elements_begin = mrCoarseModelPart.ElementsBegin();

const IndexType number_of_nodes = elements_begin->GetGeometry().size();

IndexVectorMapType tag_elems_map;

for (int i = 0; i < nelems; i++)
{
auto coarse_elem = elements_begin + i;
if (coarse_elem->Is(TO_REFINE))
{
Geometry<NodeType>::PointsArrayType p_elem_nodes;
for (IndexType node = 0; node < number_of_nodes; node++)
p_elem_nodes.push_back(coarse_elem->GetGeometry()[node].GetValue(SLAVE_NODE));

Element::Pointer aux_elem = coarse_elem->Clone(++rElemId, p_elem_nodes);
mrRefinedModelPart.AddElement(aux_elem);

aux_elem->SetValue(FATHER_ELEMENT, *coarse_elem.base());
aux_elem->Set(NEW_ENTITY, true);

IndexType tag = rElemTag[coarse_elem->Id()];
tag_elems_map[tag].push_back(rElemId);

coarse_elem->Set(MeshingFlags::REFINED, true);
}
}

for (auto& collection : mCollections)
{
const auto tag = collection.first;
if (tag != 0)
{
for (const auto& name : collection.second)
{
ModelPart& sub_model_part = mrRefinedModelPart.GetSubModelPart(name);
sub_model_part.AddElements(tag_elems_map[tag]);
}
}
}
}


void MultiscaleRefiningProcess::CreateConditionsToRefine(IndexType& rCondId, IndexIndexMapType& rCondTag)
{
const int nconds = static_cast<int>(mrCoarseModelPart.Conditions().size());
ModelPart::ConditionsContainerType::iterator conditions_begin = mrCoarseModelPart.ConditionsBegin();

const IndexType number_of_nodes = conditions_begin->GetGeometry().size();

IndexVectorMapType tag_conds_map;

for (int i = 0; i < nconds; i++)
{
auto coarse_cond = conditions_begin + i;
if (coarse_cond->Is(TO_REFINE))
{
Geometry<NodeType>::PointsArrayType p_cond_nodes;
for (IndexType node = 0; node < number_of_nodes; node++)
p_cond_nodes.push_back(coarse_cond->GetGeometry()[node].GetValue(SLAVE_NODE));

Condition::Pointer aux_cond = coarse_cond->Clone(++rCondId, p_cond_nodes);
mrRefinedModelPart.AddCondition(aux_cond);

aux_cond->SetValue(FATHER_CONDITION, *coarse_cond.base());
aux_cond->Set(NEW_ENTITY, true);

IndexType tag = rCondTag[coarse_cond->Id()];
tag_conds_map[tag].push_back(rCondId);

coarse_cond->Set(MeshingFlags::REFINED, true);
}
}

for (auto& collection : mCollections)
{
const auto tag = collection.first;
if (tag != 0)
{
for (const auto& name : collection.second)
{
ModelPart& sub_model_part = mrRefinedModelPart.GetSubModelPart(name);
sub_model_part.AddConditions(tag_conds_map[tag]);
}
}
}
}


void MultiscaleRefiningProcess::FinalizeRefinement()
{
ModelPart::NodeIterator coarse_nodes_begin = mrCoarseModelPart.NodesBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrCoarseModelPart.Nodes().size()); i++)
{
auto node = coarse_nodes_begin + i;
node->Set(NEW_ENTITY, false);
}

ModelPart::NodeIterator refined_nodes_begin = mrRefinedModelPart.NodesBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrRefinedModelPart.Nodes().size()); i++)
{
auto node = refined_nodes_begin + i;
node->Set(NEW_ENTITY, false);
}

ModelPart::ElementIterator refined_elem_begin = mrRefinedModelPart.ElementsBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrRefinedModelPart.Elements().size()); i++)
{
auto elem = refined_elem_begin + i;
elem->Set(NEW_ENTITY, false);
}

ModelPart::ConditionIterator refined_cond_begin = mrRefinedModelPart.ConditionsBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrRefinedModelPart.Conditions().size()); i++)
{
auto cond = refined_cond_begin + i;
cond->Set(NEW_ENTITY, false);
}
}

void MultiscaleRefiningProcess::FinalizeCoarsening()
{
ModelPart::NodeIterator nodes_begin = mrCoarseModelPart.NodesBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrCoarseModelPart.Nodes().size()); i++)
{
auto node = nodes_begin + i;
node->Set(MeshingFlags::TO_COARSEN, false);
}

ModelPart::ElementIterator elements_begin = mrCoarseModelPart.ElementsBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrCoarseModelPart.Elements().size()); i++)
{
auto elem = elements_begin + i;
elem->Set(MeshingFlags::TO_COARSEN, false);
}

ModelPart::ConditionIterator conditions_begin = mrCoarseModelPart.ConditionsBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrCoarseModelPart.Conditions().size()); i++)
{
auto cond = conditions_begin + i;
cond->Set(MeshingFlags::TO_COARSEN, false);
}
}


void MultiscaleRefiningProcess::IdentifyCurrentInterface()
{
int nnodes = static_cast<int>(mrCoarseModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator nodes_begin = mrCoarseModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
auto node = nodes_begin + i;
node->Set(INTERFACE, false);
}

ModelPart::ElementIterator elem_begin = mrCoarseModelPart.ElementsBegin();
const IndexType element_nodes = elem_begin->GetGeometry().size();

for (int i = 0; i < static_cast<int>(mrCoarseModelPart.Elements().size()); i++)
{
auto elem = elem_begin + i;
if (elem->IsNot(MeshingFlags::REFINED))
{
for (IndexType node = 0; node < element_nodes; node++)
{
if (elem->GetGeometry()[node].Is(MeshingFlags::REFINED))
elem->GetGeometry()[node].Set(INTERFACE, true);
}
}
}
}


void MultiscaleRefiningProcess::UpdateRefinedInterface()
{
mRefinedInterfaceContainer.clear();

ModelPart::NodeIterator refined_begin = mrRefinedModelPart.NodesBegin();
for (int i = 0; i < static_cast<int>(mrRefinedModelPart.Nodes().size()); i++)
{
auto node = refined_begin + i;
bool is_refined_interface = true;
GlobalPointersVector<NodeType>& father_nodes = node->GetValue(FATHER_NODES);
for (auto father_node = father_nodes.begin(); father_node < father_nodes.end(); father_node++)
{
if (father_node->IsNot(INTERFACE))
is_refined_interface = false;
}

if (is_refined_interface)
mRefinedInterfaceContainer.push_back(Kratos::intrusive_ptr<NodeType>(*node.base()));
}
}


void MultiscaleRefiningProcess::GetLastId(
IndexType& rNodesId,
IndexType& rElemsId,
IndexType& rCondsId)
{
rNodesId = 0;
rElemsId = 0;
rCondsId = 0;

ModelPart& root_model_part = mrVisualizationModelPart.GetRootModelPart();

const IndexType nnodes = root_model_part.Nodes().size();
ModelPart::NodesContainerType::iterator nodes_begin = root_model_part.NodesBegin();
for (IndexType i = 0; i < nnodes; i++)
{
auto inode = nodes_begin + i;
if (rNodesId < inode->Id())
rNodesId = inode->Id();
}

const IndexType nelems = root_model_part.Elements().size();
ModelPart::ElementsContainerType::iterator elements_begin = root_model_part.ElementsBegin();
for (IndexType i = 0; i < nelems; i++)
{
auto elem = elements_begin + i;
if (rElemsId < elem->Id())
rElemsId = elem->Id();
}

const IndexType nconds = root_model_part.Conditions().size();
ModelPart::ConditionsContainerType::iterator conditions_begin = root_model_part.ConditionsBegin();
for (IndexType i = 0; i < nconds; i++)
{
auto cond = conditions_begin + i;
if (rCondsId < cond->Id())
rCondsId = cond->Id();
}
}

const Parameters MultiscaleRefiningProcess::GetDefaultParameters() const
{
const Parameters default_parameters = Parameters(R"(
{
"number_of_divisions_at_subscale"     : 2,
"echo_level"                          : 0,
"subscale_interface_base_name"        : "refined_interface",
"subscale_boundary_condition"         : "LineCondition2D2N"
})" );

return default_parameters;
}

} 
