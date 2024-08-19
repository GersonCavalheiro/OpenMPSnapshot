
#pragma once

#include <string>
#include <vector>


#include "includes/key_hash.h"
#include "includes/model_part.h"
#include "utilities/model_part_operator_utilities.h"

namespace Kratos {


class KRATOS_API(KRATOS_CORE) ModelPartOperationUtilities {
public:

using IndexType = std::size_t;

using CNodePointersType = std::vector<ModelPart::NodeType const*>;

template<class KeyType, class ValueType>
using RangedKeyMapType = std::unordered_map<KeyType, ValueType, KeyHasherRange<KeyType>, KeyComparorRange<KeyType>>;



static bool CheckValidityOfModelPartsForOperations(
const ModelPart& rMainModelPart,
const std::vector<ModelPart const*>& rCheckModelParts,
const bool ThrowError = false);


template<class TModelPartOperation>
static ModelPart& CreateModelPartWithOperation(
const std::string& rOutputSubModelPartName,
ModelPart& rMainModelPart,
const std::vector<ModelPart const*>& rModelPartOperationModelParts,
const bool AddNeighbourEntities)
{
std::vector<ModelPart::NodeType*> output_nodes;
std::vector<ModelPart::ConditionType*> output_conditions;
std::vector<ModelPart::ElementType*> output_elements;

ModelPartOperation<TModelPartOperation>(output_nodes, output_conditions, output_elements, rMainModelPart, rModelPartOperationModelParts, AddNeighbourEntities);

return CreateOutputModelPart(rOutputSubModelPartName, rMainModelPart, output_nodes, output_conditions, output_elements);
}


static bool HasIntersection(const std::vector<ModelPart*>& rIntersectionModelParts);

private:

template<class TUnaryFunc>
static void AddNodes(
std::vector<ModelPart::NodeType*>& rNodesOutput,
ModelPart::NodesContainerType& rNodesContainer,
TUnaryFunc&& rIsValidEntity)
{
using p_entity_type = ModelPart::NodeType*;

const auto& result = block_for_each<AccumReduction<p_entity_type>>(rNodesContainer, [&rIsValidEntity](auto& rMainEntity) -> p_entity_type {
ModelPart::NodeType const* p_entity = &rMainEntity;
if (rIsValidEntity(p_entity)) {
return &rMainEntity;
} else {
return nullptr;
}
});

std::for_each(result.begin(), result.end(), [&rNodesOutput](auto pEntity) {
if (pEntity != nullptr) {
rNodesOutput.push_back(pEntity);
}
});
}

template<class TContainerType, class TUnaryFunc>
static void AddEntities(
std::vector<typename TContainerType::value_type*>& rEntitiesOutput,
TContainerType& rMainEntityContainer,
TUnaryFunc&& rIsValidEntity)
{
using p_entity_type = typename TContainerType::value_type*;
const auto& result = block_for_each<AccumReduction<p_entity_type>>(rMainEntityContainer, CNodePointersType(), [&rIsValidEntity](auto& rMainEntity, CNodePointersType& rTLS) -> p_entity_type {
const auto& r_geometry = rMainEntity.GetGeometry();
const IndexType number_of_nodes = r_geometry.size();

if (rTLS.size() != number_of_nodes) {
rTLS.resize(number_of_nodes);
}

for (IndexType i = 0; i < number_of_nodes; ++i) {
rTLS[i] = &r_geometry[i];
}

std::sort(rTLS.begin(), rTLS.end());

if (rIsValidEntity(rTLS)) {
return &rMainEntity;
} else {
return nullptr;
}
});

std::for_each(result.begin(), result.end(), [&rEntitiesOutput](auto pEntity) {
if (pEntity != nullptr) {
rEntitiesOutput.push_back(pEntity);
}
});
}

template<class TModelPartOperation>
static void ModelPartOperation(
std::vector<ModelPart::NodeType*>& rOutputNodes,
std::vector<ModelPart::ConditionType*>& rOutputConditions,
std::vector<ModelPart::ElementType*>& rOutputElements,
ModelPart& rMainModelPart,
const std::vector<ModelPart const*>& rModelPartOperationModelParts,
const bool AddNeighbourEntities)
{
const IndexType number_of_operation_model_parts = rModelPartOperationModelParts.size();

std::vector<std::set<ModelPart::NodeType const*>> set_operation_node_sets(number_of_operation_model_parts);
std::vector<std::set<CNodePointersType>> set_operation_condition_sets(number_of_operation_model_parts);
std::vector<std::set<CNodePointersType>> set_operation_element_sets(number_of_operation_model_parts);

for (IndexType i = 0; i < number_of_operation_model_parts; ++i) {
auto p_model_part = rModelPartOperationModelParts[i];

FillNodesPointerSet(set_operation_node_sets[i], p_model_part->Nodes());

FillNodePointersForEntities(set_operation_condition_sets[i], p_model_part->Conditions());

FillNodePointersForEntities(set_operation_element_sets[i], p_model_part->Elements());
}

AddNodes(rOutputNodes, rMainModelPart.Nodes(), [&set_operation_node_sets](auto& rEntity) {
return TModelPartOperation::IsValid(rEntity, set_operation_node_sets);
});

AddEntities(rOutputConditions, rMainModelPart.Conditions(), [&set_operation_condition_sets](auto& rEntity) {
return TModelPartOperation::IsValid(rEntity, set_operation_condition_sets);
});

AddEntities(rOutputElements, rMainModelPart.Elements(), [&set_operation_element_sets](auto& rEntity) {
return TModelPartOperation::IsValid(rEntity, set_operation_element_sets);
});

if (AddNeighbourEntities) {
FillNodesFromEntities<ModelPart::ConditionType>(rOutputNodes, rOutputConditions.begin(), rOutputConditions.end());
FillNodesFromEntities<ModelPart::ElementType>(rOutputNodes, rOutputElements.begin(), rOutputElements.end());

AddNeighbours(rOutputNodes, rOutputConditions, rOutputElements, rMainModelPart);
}
}

static void FillNodesPointerSet(
std::set<ModelPart::NodeType const*>& rOutput,
const ModelPart::NodesContainerType& rNodes);

template <class TContainerType>
static void FillNodePointersForEntities(
std::set<CNodePointersType>& rOutput,
const TContainerType& rEntityContainer);

static void FillCommonNodes(
ModelPart::NodesContainerType& rOutput,
ModelPart::NodesContainerType& rMainNodes,
const std::set<ModelPart::NodeType const*>& rNodesSet);

template<class TEntityType>
static void FillNodesFromEntities(
std::vector<ModelPart::NodeType*>& rOutput,
typename std::vector<TEntityType*>::iterator pEntityBegin,
typename std::vector<TEntityType*>::iterator pEntityEnd);

template<class TContainerType>
static void FindNeighbourEntities(
std::vector<typename TContainerType::value_type*>& rOutputEntities,
const Flags& rNodeSelectionFlag,
TContainerType& rMainEntities);

static void AddNeighbours(
std::vector<ModelPart::NodeType*>& rOutputNodes,
std::vector<ModelPart::ConditionType*>& rOutputConditions,
std::vector<ModelPart::ElementType*>& rOutputElements,
ModelPart& rMainModelPart);

static void FillWithMainNodesFromSearchNodes(
ModelPart::NodesContainerType& rOutput,
ModelPart::NodesContainerType& rMainNodes,
ModelPart::NodesContainerType& rSearchedNodes);

static void SetCommunicator(
ModelPart& rOutputModelPart,
ModelPart& rMainModelPart);

static ModelPart& CreateOutputModelPart(
const std::string& rOutputSubModelPartName,
ModelPart& rMainModelPart,
std::vector<ModelPart::NodeType*>& rOutputNodes,
std::vector<ModelPart::ConditionType*>& rOutputConditions,
std::vector<ModelPart::ElementType*>& rOutputElements);

static void CheckNodes(
std::vector<IndexType>& rNodeIdsWithIssues,
const ModelPart::NodesContainerType& rCheckNodes,
const std::set<ModelPart::NodeType const*>& rMainNodes);

template<class TContainerType>
static void CheckEntities(
std::vector<IndexType>& rEntityIdsWithIssues,
const TContainerType& rCheckEntities,
const std::set<CNodePointersType>& rMainEntities);

};


} 