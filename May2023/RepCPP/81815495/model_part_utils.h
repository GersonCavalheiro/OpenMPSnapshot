
#pragma once

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "includes/define.h"
#include "includes/model_part.h"


namespace Kratos {


class KRATOS_API(OPTIMIZATION_APPLICATION) ModelPartUtils {
public:

using IndexType = std::size_t;

using NodeIdsType = std::vector<IndexType>;

template <class TEntityType>
using EntityPointerType = typename TEntityType::Pointer;

template <class TContainerType>
using ContainerEntityValueType = typename TContainerType::value_type;

template <class TContainerType>
using ContainerEntityPointerType =
typename ContainerEntityValueType<TContainerType>::Pointer;



static std::vector<ModelPart*> GetModelPartsWithCommonReferenceEntities(
const std::vector<ModelPart*>& rExaminedModelPartsList,
const std::vector<ModelPart*>& rReferenceModelParts,
const bool AreNodesConsidered,
const bool AreConditionsConsidered,
const bool AreElementsConsidered,
const bool AreNeighboursConsidered,
const IndexType EchoLevel = 0);


static void RemoveModelPartsWithCommonReferenceEntitiesBetweenReferenceListAndExaminedList(
const std::vector<ModelPart*> rModelParts);

private:

template<class TDataType>
class SetReduction
{
public:
using return_type = std::set<TDataType>;
using value_type = TDataType;

return_type mValue;

return_type GetValue() const;

void LocalReduce(const value_type& rValue);

void ThreadSafeReduce(SetReduction<TDataType>& rOther);
};

template<class TEntityType, class TMapValueType>
class ContainerEntityMapReduction
{
public:
using return_type = std::map<IndexType, TMapValueType>;
using value_type = std::vector<std::pair<IndexType, EntityPointerType<TEntityType>>>;

return_type mValue;

return_type GetValue() const;

void LocalReduce(const value_type& rValue);

void ThreadSafeReduce(ContainerEntityMapReduction<TEntityType, TMapValueType>& rOther);
};


static void AppendModelPartNames(
std::stringstream& rOutputStream,
const std::vector<ModelPart*>& rModelParts);

template<class TContainerType>
static void UpdateEntityIdsSetFromContainer(
std::set<IndexType>& rOutput,
const TContainerType& rContainer);

template<class TContainerType>
static void UpdateEntityGeometryNodeIdsSetFromContainer(
std::set<NodeIdsType>& rOutput,
const TContainerType& rContainer);

template<class TContainerType>
static void UpdateEntityIdEntityPtrMapWithCommonEntitiesFromContainerAndEntityIdsSet(
std::map<IndexType, ContainerEntityPointerType<TContainerType>>& rOutput,
const std::set<IndexType>& rEntityIdsSet,
TContainerType& rContainer);

template<class TContainerType>
static void UpdateEntityIdEntityPtrMapWithCommonEntitiesFromContainerAndEntityGeometryNodeIdsSet(
std::map<IndexType, ContainerEntityPointerType<TContainerType>>& rOutput,
const std::set<NodeIdsType>& rEntityGeometryNodeIdsSet,
TContainerType& rContainer);

template<class TContainerType>
static void UpdateNeighbourMaps(
std::map<IndexType, std::vector<ContainerEntityPointerType<TContainerType>>>& rOutput,
const std::set<IndexType>& rNodeIdsSet,
TContainerType& rContainer);

template<class TEntityPointerType>
static void UpdateEntityIdEntityPtrMapFromNeighbourMap(
std::map<IndexType, TEntityPointerType>& rOutput,
const std::map<IndexType, std::vector<TEntityPointerType>>& rNodeIdNeighbourEntityPtrsMap);

template<class TEntityPointerType>
static void UpdateNodeIdNodePtrMapFromEntityIdEntityPtrMap(
std::map<IndexType, ModelPart::NodeType::Pointer>& rOutput,
const std::map<IndexType, TEntityPointerType>& rInput);

static std::string GetExaminedModelPartsInfo(
const std::vector<ModelPart*>& rExaminedModelPartsList,
const bool AreNodesConsidered,
const bool AreConditionsConsidered,
const bool AreElementsConsidered,
const bool AreNeighboursConsidered);

static void GetModelParts(
std::set<ModelPart*>& rOutput,
ModelPart& rInput);

static void ExamineModelParts(
std::set<IndexType>& rExaminedNodeIds,
std::set<NodeIdsType>& rExaminedConditionGeometryNodeIdsSet,
std::set<NodeIdsType>& rExaminedElementGeometryNodeIdsSet,
const std::vector<ModelPart*> rExaminedModelPartsList,
const bool AreNodesConsidered,
const bool AreConditionsConsidered,
const bool AreElementsConsidered,
const bool AreNeighboursConsidered);

static void PopulateModelPart(
ModelPart& rOutputModelPart,
ModelPart& rReferenceModelPart,
const bool AreNodesConsidered,
const bool AreConditionsConsidered,
const bool AreElementsConsidered,
const bool AreNeighboursConsidered,
const std::set<IndexType>& rExaminedNodeIds,
const std::set<NodeIdsType>& rExaminedConditionGeometryNodeIdsSet,
const std::set<NodeIdsType>& rExaminedElementGeometryNodeIdsSet);

};

} 