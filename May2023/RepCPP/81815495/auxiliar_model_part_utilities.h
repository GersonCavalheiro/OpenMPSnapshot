
#pragma once



#include "includes/model_part.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{


typedef std::size_t IndexType;


enum class DataLocation {
NodeHistorical,
NodeNonHistorical,
Element,
Condition,
ModelPart,
ProcessInfo
};



class KRATOS_API(KRATOS_CORE) AuxiliarModelPartUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION( AuxiliarModelPartUtilities );



AuxiliarModelPartUtilities(ModelPart& rModelPart):
mrModelPart(rModelPart)
{
}

virtual ~AuxiliarModelPartUtilities()= default;







static void CopySubModelPartStructure(const ModelPart& rModelPartToCopyFromIt, ModelPart& rModelPartToCopyIntoIt);


void RecursiveEnsureModelPartOwnsProperties(const bool RemovePreviousProperties = true);


void EnsureModelPartOwnsProperties(const bool RemovePreviousProperties = true);


void RemoveElementAndBelongings(IndexType ElementId, Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveElementAndBelongings(Element& rThisElement, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveElementAndBelongings(Element::Pointer pThisElement, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveElementAndBelongingsFromAllLevels(IndexType ElementId, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveElementAndBelongingsFromAllLevels(Element& rThisElement, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveElementAndBelongingsFromAllLevels(Element::Pointer pThisElement, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveElementsAndBelongings(Flags IdentifierFlag = TO_ERASE);


void RemoveElementsAndBelongingsFromAllLevels(const Flags IdentifierFlag = TO_ERASE);


void RemoveConditionAndBelongings(IndexType ConditionId, Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveConditionAndBelongings(Condition& ThisCondition, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveConditionAndBelongings(Condition::Pointer pThisCondition, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveConditionAndBelongingsFromAllLevels(IndexType ConditionId, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveConditionAndBelongingsFromAllLevels(Condition& rThisCondition, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveConditionAndBelongingsFromAllLevels(Condition::Pointer pThisCondition, const Flags IdentifierFlag = TO_ERASE, IndexType ThisIndex = 0);


void RemoveConditionsAndBelongings(Flags IdentifierFlag = TO_ERASE);


void RemoveConditionsAndBelongingsFromAllLevels(const Flags IdentifierFlag = TO_ERASE);


void RemoveOrphanNodesFromSubModelParts();

template<class TContainerType>
void GetScalarData(
const Variable<typename TContainerType::value_type>& rVariable,
const DataLocation DataLoc,
TContainerType& data) const
{
KRATOS_TRY

switch (DataLoc)
{
case (DataLocation::NodeHistorical):{
data.resize(mrModelPart.NumberOfNodes());

auto inodebegin = mrModelPart.NodesBegin();

IndexPartition<IndexType>(mrModelPart.NumberOfNodes()).for_each([&](IndexType Index){
auto inode = inodebegin + Index;

data[Index] = inode->FastGetSolutionStepValue(rVariable);
});

break;
}
case (DataLocation::NodeNonHistorical):{
data.resize(mrModelPart.NumberOfNodes());

GetScalarDataFromContainer(mrModelPart.Nodes(), rVariable, data);
break;
}
case (DataLocation::Element):{
data.resize(mrModelPart.NumberOfElements());

GetScalarDataFromContainer(mrModelPart.Elements(), rVariable, data);
break;
}
case (DataLocation::Condition):{
data.resize(mrModelPart.NumberOfConditions());

GetScalarDataFromContainer(mrModelPart.Conditions(), rVariable, data);
break;
}
case (DataLocation::ModelPart):{
data.resize(1);
data[0] = mrModelPart[rVariable];
break;
}
case (DataLocation::ProcessInfo):{
data.resize(1);
data[0] = mrModelPart.GetProcessInfo()[rVariable];
break;
}
default:{
KRATOS_ERROR << "unknown Datalocation" << std::endl;
break;
}
}

KRATOS_CATCH("")
}

template<class TContainerType, class TVarType>
void GetVectorData(
const Variable<TVarType>& rVariable,
const DataLocation DataLoc,
TContainerType& data) const
{
KRATOS_TRY

switch (DataLoc)
{
case (DataLocation::NodeHistorical):{
unsigned int TSize = mrModelPart.NumberOfNodes() > 0 ? mrModelPart.NodesBegin()->FastGetSolutionStepValue(rVariable).size() : 0;

TSize = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(TSize);
data.resize(mrModelPart.NumberOfNodes()*TSize);

auto inodebegin = mrModelPart.NodesBegin();

IndexPartition<IndexType>(mrModelPart.NumberOfNodes()).for_each([&](IndexType Index){
auto inode = inodebegin + Index;

const auto& r_val = inode->FastGetSolutionStepValue(rVariable);
for(std::size_t dim = 0 ; dim < TSize ; dim++){
data[(Index*TSize) + dim] = r_val[dim];
}
});

break;
}
case (DataLocation::NodeNonHistorical):{
unsigned int TSize = mrModelPart.NumberOfNodes() > 0 ? mrModelPart.NodesBegin()->GetValue(rVariable).size() : 0;

TSize = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(TSize);

data.resize(mrModelPart.NumberOfNodes()*TSize);

GetVectorDataFromContainer(mrModelPart.Nodes(), TSize, rVariable, data);
break;
}
case (DataLocation::Element):{
unsigned int TSize = mrModelPart.NumberOfElements() > 0 ? mrModelPart.ElementsBegin()->GetValue(rVariable).size() : 0;

TSize = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(TSize);

data.resize(mrModelPart.NumberOfElements()*TSize);

GetVectorDataFromContainer(mrModelPart.Elements(), TSize, rVariable, data);
break;
}
case (DataLocation::Condition):{
unsigned int TSize = mrModelPart.NumberOfConditions() > 0 ? mrModelPart.ConditionsBegin()->GetValue(rVariable).size() : 0;

TSize = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(TSize);

data.resize(mrModelPart.NumberOfConditions()*TSize);

GetVectorDataFromContainer(mrModelPart.Conditions(), TSize, rVariable, data);
break;
}
case (DataLocation::ModelPart):{
std::size_t TSize = mrModelPart[rVariable].size();
data.resize(TSize);

IndexType counter = 0;
auto& r_val = mrModelPart[rVariable];
for(std::size_t dim = 0 ; dim < TSize ; dim++){
data[counter++] = r_val[dim];
}
break;
}
case (DataLocation::ProcessInfo):{
const std::size_t TSize = mrModelPart.GetProcessInfo()[rVariable].size();
data.resize(TSize);

IndexType counter = 0;
auto& r_val = mrModelPart.GetProcessInfo()[rVariable];
for(std::size_t dim = 0 ; dim < TSize ; dim++){
data[counter++] = r_val[dim];
}
break;
}
default:{
KRATOS_ERROR << "unknown Datalocation" << std::endl;
break;
}
}

KRATOS_CATCH("")
}

template<class TContainerType>
void SetScalarData(
const Variable<typename TContainerType::value_type>& rVariable,
const DataLocation DataLoc,
const TContainerType& rData)
{
KRATOS_TRY

switch (DataLoc)
{
case (DataLocation::NodeHistorical):{
auto inodebegin = mrModelPart.NodesBegin();
IndexPartition<IndexType>(mrModelPart.NumberOfNodes()).for_each([&](IndexType Index){
auto inode = inodebegin + Index;

auto& r_val = inode->FastGetSolutionStepValue(rVariable);
r_val = rData[Index];
});

break;
}
case (DataLocation::NodeNonHistorical):{
SetScalarDataFromContainer(mrModelPart.Nodes(), rVariable, rData);
break;
}
case (DataLocation::Element):{
SetScalarDataFromContainer(mrModelPart.Elements(), rVariable, rData);
break;
}
case (DataLocation::Condition):{
SetScalarDataFromContainer(mrModelPart.Conditions(), rVariable, rData);
break;
}
case (DataLocation::ModelPart):{
mrModelPart[rVariable]= rData[0];
break;
}
case (DataLocation::ProcessInfo):{
mrModelPart.GetProcessInfo()[rVariable] = rData[0] ;
break;
}
default:{
KRATOS_ERROR << "unknown Datalocation" << std::endl;
break;
}
}

KRATOS_CATCH("")
}

template<class TContainerType, class TVarType>
void SetVectorData(
const Variable<TVarType>& rVariable,
const DataLocation DataLoc,
const TContainerType& rData)
{
KRATOS_TRY

switch (DataLoc)
{
case (DataLocation::NodeHistorical):{
unsigned int size = mrModelPart.NumberOfNodes() > 0 ? mrModelPart.NodesBegin()->FastGetSolutionStepValue(rVariable).size() : 0;

size = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(size);

auto inodebegin = mrModelPart.NodesBegin();
IndexPartition<IndexType>(mrModelPart.NumberOfNodes()).for_each([&](IndexType Index){
auto inode = inodebegin + Index;
auto& r_val = inode->FastGetSolutionStepValue(rVariable);

KRATOS_DEBUG_ERROR_IF(r_val.size() != size) << "mismatch in size!" << std::endl;

for(std::size_t dim = 0 ; dim < size ; dim++){
r_val[dim] = rData[(Index*size) + dim];
}
});

break;
}
case (DataLocation::NodeNonHistorical):{
unsigned int size = mrModelPart.NumberOfNodes() > 0 ? mrModelPart.NodesBegin()->GetValue(rVariable).size() : 0;

size = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(size);

SetVectorDataFromContainer(mrModelPart.Nodes(), size, rVariable, rData);
break;
}
case (DataLocation::Element):{
unsigned int size = mrModelPart.NumberOfElements() > 0 ? mrModelPart.ElementsBegin()->GetValue(rVariable).size() : 0;

size = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(size);

SetVectorDataFromContainer(mrModelPart.Elements(), size, rVariable, rData);
break;
}
case (DataLocation::Condition):{
unsigned int size = mrModelPart.NumberOfConditions() > 0 ? mrModelPart.ConditionsBegin()->GetValue(rVariable).size() : 0;

size = mrModelPart.GetCommunicator().GetDataCommunicator().MaxAll(size);

SetVectorDataFromContainer(mrModelPart.Conditions(), size, rVariable, rData);
break;
}
case (DataLocation::ModelPart):{
const std::size_t size = mrModelPart[rVariable].size();

IndexType counter = 0;
auto& r_val = mrModelPart[rVariable];
for(std::size_t dim = 0 ; dim < size ; dim++){
r_val[dim] = rData[counter++];
}
break;
}
case (DataLocation::ProcessInfo):{
const std::size_t size = mrModelPart.GetProcessInfo()[rVariable].size();

IndexType counter = 0;
auto& r_val = mrModelPart.GetProcessInfo()[rVariable];
for(std::size_t dim = 0 ; dim < size ; dim++){
r_val[dim] = rData[counter++];
}
break;
}
default:{
KRATOS_ERROR << "unknown Datalocation" << std::endl;
break;
}

}

KRATOS_CATCH("")
}


ModelPart& DeepCopyModelPart(
const std::string& rNewModelPartName,
Model* pModel = nullptr
);


template<class TClassContainer, class TReferenceClassContainer>
void DeepCopyEntities(
ModelPart& rModelPart,
TClassContainer& rEntities,
TReferenceClassContainer& rReferenceEntities,
std::unordered_map<Geometry<Node>::Pointer,Geometry<Node>::Pointer>& rGeometryPointerDatabase
)
{
KRATOS_TRY

auto& r_properties= rModelPart.rProperties();
rEntities.SetMaxBufferSize(rReferenceEntities.GetMaxBufferSize());
rEntities.SetSortedPartSize(rReferenceEntities.GetSortedPartSize());
const auto& r_reference_entities_container = rReferenceEntities.GetContainer();
auto& r_entities_container = rEntities.GetContainer();
const IndexType number_entities = r_reference_entities_container.size();
r_entities_container.resize(number_entities);
const auto it_ent_begin = r_reference_entities_container.begin();
IndexPartition<std::size_t>(number_entities).for_each([&it_ent_begin,&r_entities_container,&rGeometryPointerDatabase,&r_properties](std::size_t i) {
auto it_ent = it_ent_begin + i;
auto& p_old_ent = (*it_ent);
auto p_new_ent = p_old_ent->Create(p_old_ent->Id(), rGeometryPointerDatabase[p_old_ent->pGetGeometry()], r_properties(p_old_ent->pGetProperties()->Id()));
p_new_ent->SetData(p_old_ent->GetData());
p_new_ent->Set(Flags(*p_old_ent));
r_entities_container[i] = p_new_ent;
});

KRATOS_CATCH("")
}

virtual std::string Info() const
{
return "AuxiliarModelPartUtilities";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << Info() << std::endl;
}

virtual void PrintData(std::ostream& rOStream) const
{
rOStream << Info() << std::endl;
}

private:

ModelPart& mrModelPart;



template<typename TDataType, class TContainerType, class TDataContainerType>
void GetScalarDataFromContainer(
const TContainerType& rContainer,
const Variable<TDataType>& rVariable,
TDataContainerType& data) const
{
KRATOS_TRY

DataSizeCheck(rContainer.size(), data.size());

IndexPartition<std::size_t>(rContainer.size()).for_each([&](std::size_t index){
const auto& r_entity = *(rContainer.begin() + index);
data[index] = r_entity.GetValue(rVariable);
});

KRATOS_CATCH("")
}

template<typename TDataType, class TContainerType, class TDataContainerType>
void GetVectorDataFromContainer(
const TContainerType& rContainer,
const std::size_t VectorSize,
const Variable<TDataType>& rVariable,
TDataContainerType& data) const
{
KRATOS_TRY

DataSizeCheck(rContainer.size()*VectorSize, data.size());

IndexPartition<std::size_t>(rContainer.size()).for_each([&](std::size_t index){
const auto& r_entity = *(rContainer.begin() + index);
const auto& r_val = r_entity.GetValue(rVariable);
for(std::size_t dim = 0 ; dim < VectorSize ; dim++){
data[(VectorSize*index) + dim] = r_val[dim];
}
});

KRATOS_CATCH("")
}

template<typename TDataType, class TContainerType, class TDataContainerType>
void SetScalarDataFromContainer(
TContainerType& rContainer,
const Variable<TDataType>& rVariable,
const TDataContainerType& rData) const
{
KRATOS_TRY

DataSizeCheck(rContainer.size(), rData.size());

IndexPartition<std::size_t>(rContainer.size()).for_each([&](std::size_t index){
auto& r_entity = *(rContainer.begin() + index);
r_entity.SetValue(rVariable, rData[index]);
});

KRATOS_CATCH("")
}

template<typename TDataType, class TContainerType, class TDataContainerType>
void SetVectorDataFromContainer(
TContainerType& rContainer,
const std::size_t VectorSize,
const Variable<TDataType>& rVariable,
const TDataContainerType& rData) const
{
KRATOS_TRY

DataSizeCheck(rContainer.size()*VectorSize, rData.size());

IndexPartition<std::size_t>(rContainer.size()).for_each([&](std::size_t index){
auto& r_entity = *(rContainer.begin() + index);
TDataType aux;
KRATOS_DEBUG_ERROR_IF(aux.size() != VectorSize) << "mismatch in size!" << std::endl;
for(std::size_t dim = 0 ; dim < VectorSize ; dim++){
aux[dim] = rData[(VectorSize*index) + dim];
}
r_entity.SetValue(rVariable, aux);
});

KRATOS_CATCH("")
}

void DataSizeCheck(
const std::size_t ContainerSize,
const std::size_t DataSize) const
{
KRATOS_ERROR_IF(ContainerSize != DataSize) << "Mismatch in size! Container size: " << ContainerSize << " | Data size: " << DataSize << std::endl;
}


void DeepCopySubModelPart(
const ModelPart& rOldModelPart,
ModelPart& rNewModelPart
);




};





}  
