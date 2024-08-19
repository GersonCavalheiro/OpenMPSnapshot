
#pragma once



#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/checks.h"
#include "utilities/parallel_utilities.h"
#include "utilities/atomic_utilities.h"
#include "utilities/reduction_utilities.h"
namespace Kratos
{






class KRATOS_API(KRATOS_CORE) VariableUtils
{
public:

typedef ModelPart::NodeType NodeType;

typedef ModelPart::ConditionType ConditionType;

typedef ModelPart::ElementType ElementType;

KRATOS_CLASS_POINTER_DEFINITION(VariableUtils);

typedef ModelPart::NodesContainerType NodesContainerType;

typedef ModelPart::ConditionsContainerType ConditionsContainerType;

typedef ModelPart::ElementsContainerType ElementsContainerType;

typedef Variable< double > DoubleVarType;

typedef Variable< array_1d<double, 3 > > ArrayVarType;









template <class TVarType>
void CopyModelPartNodalVar(
const TVarType &rVariable,
const TVarType &rDestinationVariable,
const ModelPart &rOriginModelPart,
ModelPart &rDestinationModelPart,
const unsigned int ReadBufferStep,
const unsigned int WriteBufferStep )
{
const int n_orig_nodes = rOriginModelPart.NumberOfNodes();
const int n_dest_nodes = rDestinationModelPart.NumberOfNodes();

KRATOS_ERROR_IF_NOT(n_orig_nodes == n_dest_nodes)
<< "Origin and destination model parts have different number of nodes."
<< "\n\t- Number of origin nodes: " << n_orig_nodes
<< "\n\t- Number of destination nodes: " << n_dest_nodes << std::endl;

IndexPartition<std::size_t>(n_orig_nodes).for_each([&](std::size_t index)
{
auto it_dest_node = rDestinationModelPart.NodesBegin() + index;
const auto it_orig_node = rOriginModelPart.NodesBegin() + index;
const auto &r_value = it_orig_node->GetSolutionStepValue(rVariable, ReadBufferStep);
it_dest_node->FastGetSolutionStepValue(rDestinationVariable, WriteBufferStep) = r_value; });

rDestinationModelPart.GetCommunicator().SynchronizeVariable(rDestinationVariable);
}


template <class TVarType>
void CopyModelPartNodalVar(
const TVarType &rVariable,
const TVarType &rDestinationVariable,
const ModelPart &rOriginModelPart,
ModelPart &rDestinationModelPart,
const unsigned int BuffStep = 0)
{
this->CopyModelPartNodalVar(rVariable, rDestinationVariable, rOriginModelPart, rDestinationModelPart, BuffStep, BuffStep);
}


template< class TVarType >
void CopyModelPartNodalVar(
const TVarType& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const unsigned int BuffStep = 0)
{
this->CopyModelPartNodalVar(rVariable, rVariable, rOriginModelPart, rDestinationModelPart, BuffStep);
}

template< class TVarType >
void CopyModelPartNodalVarToNonHistoricalVar(
const TVarType &rVariable,
const TVarType &rDestinationVariable,
const ModelPart &rOriginModelPart,
ModelPart &rDestinationModelPart,
const unsigned int BuffStep = 0)
{
const int n_orig_nodes = rOriginModelPart.NumberOfNodes();
const int n_dest_nodes = rDestinationModelPart.NumberOfNodes();

KRATOS_ERROR_IF_NOT(n_orig_nodes == n_dest_nodes) <<
"Origin and destination model parts have different number of nodes." <<
"\n\t- Number of origin nodes: " << n_orig_nodes <<
"\n\t- Number of destination nodes: " << n_dest_nodes << std::endl;

IndexPartition<std::size_t>(n_orig_nodes).for_each([&](std::size_t index){
auto it_dest_node = rDestinationModelPart.NodesBegin() + index;
const auto it_orig_node = rOriginModelPart.NodesBegin() + index;
const auto& r_value = it_orig_node->GetSolutionStepValue(rVariable, BuffStep);
it_dest_node->GetValue(rDestinationVariable) = r_value;
});

rDestinationModelPart.GetCommunicator().SynchronizeNonHistoricalVariable(rDestinationVariable);
}

template< class TVarType >
void CopyModelPartNodalVarToNonHistoricalVar(
const TVarType &rVariable,
const ModelPart &rOriginModelPart,
ModelPart &rDestinationModelPart,
const unsigned int BuffStep = 0)
{
this->CopyModelPartNodalVarToNonHistoricalVar(rVariable, rVariable, rOriginModelPart, rDestinationModelPart, BuffStep);
}

template <class TDataType>
void CopyModelPartFlaggedNodalHistoricalVarToHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int ReadBufferStep = 0,
const unsigned int WriteBufferStep = 0)
{
KRATOS_TRY

KRATOS_ERROR_IF(
rOriginModelPart.FullName() == rDestinationModelPart.FullName() &&
rOriginVariable == rDestinationVariable &&
ReadBufferStep == WriteBufferStep)
<< "Trying to copy flagged nodal solution step values with the same origin and destination model parts/variables/buffer steps. This is not permitted ( Origin model part: "
<< rOriginModelPart.Name() << ", destination model part: " << rDestinationModelPart.Name()
<< ", variable: " << rOriginVariable.Name() << ", buffer step: " << ReadBufferStep << " ) !";

KRATOS_ERROR_IF_NOT(rOriginModelPart.HasNodalSolutionStepVariable(rOriginVariable))
<< rOriginVariable.Name() << " is not found in nodal solution step variables list in origin model part ( "
<< rOriginModelPart.Name() << " ).";

KRATOS_ERROR_IF_NOT(rDestinationModelPart.HasNodalSolutionStepVariable(rDestinationVariable))
<< rDestinationVariable.Name() << " is not found in nodal solution step variables list in destination model part ( "
<< rDestinationModelPart.Name() << " ).";

KRATOS_ERROR_IF(ReadBufferStep >= rOriginModelPart.GetBufferSize())
<< "Origin model part ( " << rOriginModelPart.Name()
<< " ) buffer size is smaller or equal than read buffer size [ "
<< rOriginModelPart.GetBufferSize() << " <= " << ReadBufferStep << " ].";

KRATOS_ERROR_IF(WriteBufferStep >= rDestinationModelPart.GetBufferSize())
<< "Destination model part ( " << rDestinationModelPart.Name()
<< " ) buffer size is smaller or equal than read buffer size [ "
<< rDestinationModelPart.GetBufferSize() << " <= " << WriteBufferStep << " ].";

CopyModelPartFlaggedVariable<NodesContainerType>(
rOriginModelPart, rDestinationModelPart, rFlag, CheckValue,
[&](NodeType& rDestNode, const TDataType& rValue) {
rDestNode.FastGetSolutionStepValue(
rDestinationVariable, WriteBufferStep) = rValue;
},
[&](const NodeType& rOriginNode) -> const TDataType& {
return rOriginNode.FastGetSolutionStepValue(rOriginVariable, ReadBufferStep);
});

rDestinationModelPart.GetCommunicator().SynchronizeVariable(rDestinationVariable);

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedNodalHistoricalVarToHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int ReadBufferStep = 0,
const unsigned int WriteBufferStep = 0)
{
KRATOS_TRY

CopyModelPartFlaggedNodalHistoricalVarToHistoricalVar(
rOriginVariable, rDestinationVariable, rModelPart, rModelPart,
rFlag, CheckValue, ReadBufferStep, WriteBufferStep);

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedNodalHistoricalVarToHistoricalVar(
const Variable<TDataType>& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int ReadBufferStep = 0,
const unsigned int WriteBufferStep = 0)
{
KRATOS_TRY

CopyModelPartFlaggedNodalHistoricalVarToHistoricalVar(
rVariable, rVariable, rOriginModelPart, rDestinationModelPart,
rFlag, CheckValue, ReadBufferStep, WriteBufferStep);

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedNodalHistoricalVarToNonHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int ReadBufferStep = 0)
{
KRATOS_TRY

KRATOS_ERROR_IF_NOT(rOriginModelPart.HasNodalSolutionStepVariable(rOriginVariable))
<< rOriginVariable.Name() << " is not found in nodal solution step variables list in origin model part ( "
<< rOriginModelPart.Name() << " ).";

KRATOS_ERROR_IF(ReadBufferStep >= rOriginModelPart.GetBufferSize())
<< "Origin model part ( " << rOriginModelPart.Name()
<< " ) buffer size is smaller or equal than read buffer size [ "
<< rOriginModelPart.GetBufferSize() << " <= " << ReadBufferStep << " ].";


CopyModelPartFlaggedVariable<NodesContainerType>(
rOriginModelPart, rDestinationModelPart, rFlag, CheckValue,
[&](NodeType& rDestNode, const TDataType& rValue) {
rDestNode.SetValue(rDestinationVariable, rValue);
},
[&](const NodeType& rOriginNode) -> const TDataType& {
return rOriginNode.FastGetSolutionStepValue(rOriginVariable, ReadBufferStep);
});

rDestinationModelPart.GetCommunicator().SynchronizeNonHistoricalVariable(rDestinationVariable);

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedNodalHistoricalVarToNonHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int ReadBufferStep = 0)
{
CopyModelPartFlaggedNodalHistoricalVarToNonHistoricalVar(
rOriginVariable, rDestinationVariable, rModelPart, rModelPart,
rFlag, CheckValue, ReadBufferStep);
}

template <class TDataType>
void CopyModelPartFlaggedNodalHistoricalVarToNonHistoricalVar(
const Variable<TDataType>& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int ReadBufferStep = 0)
{
CopyModelPartFlaggedNodalHistoricalVarToNonHistoricalVar(
rVariable, rVariable, rOriginModelPart, rDestinationModelPart,
rFlag, CheckValue, ReadBufferStep);
}

template <class TDataType>
void CopyModelPartFlaggedNodalHistoricalVarToNonHistoricalVar(
const Variable<TDataType>& rVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int ReadBufferStep = 0)
{
CopyModelPartFlaggedNodalHistoricalVarToNonHistoricalVar(
rVariable, rVariable, rModelPart, rModelPart,
rFlag, CheckValue, ReadBufferStep);
}

template <class TDataType>
void CopyModelPartFlaggedNodalNonHistoricalVarToHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int WriteBufferStep = 0)
{
KRATOS_TRY

KRATOS_ERROR_IF_NOT(rDestinationModelPart.HasNodalSolutionStepVariable(rDestinationVariable))
<< rDestinationVariable.Name() << " is not found in nodal solution step variables list in destination model part ( "
<< rDestinationModelPart.Name() << " ).";

KRATOS_ERROR_IF(WriteBufferStep >= rDestinationModelPart.GetBufferSize())
<< "Destination model part ( " << rDestinationModelPart.Name()
<< " ) buffer size is smaller or equal than read buffer size [ "
<< rDestinationModelPart.GetBufferSize() << " <= " << WriteBufferStep << " ].";

CopyModelPartFlaggedVariable<NodesContainerType>(
rOriginModelPart, rDestinationModelPart, rFlag, CheckValue,
[&](NodeType& rDestNode, const TDataType& rValue) {
rDestNode.FastGetSolutionStepValue(
rDestinationVariable, WriteBufferStep) = rValue;
},
[&](const NodeType& rOriginNode) -> const TDataType& {
return rOriginNode.GetValue(rOriginVariable);
});

rDestinationModelPart.GetCommunicator().SynchronizeVariable(rDestinationVariable);

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedNodalNonHistoricalVarToHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int WriteBufferStep = 0)
{
CopyModelPartFlaggedNodalNonHistoricalVarToHistoricalVar(
rOriginVariable, rDestinationVariable, rModelPart, rModelPart,
rFlag, CheckValue, WriteBufferStep);
}

template <class TDataType>
void CopyModelPartFlaggedNodalNonHistoricalVarToHistoricalVar(
const Variable<TDataType>& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int WriteBufferStep = 0)
{
CopyModelPartFlaggedNodalNonHistoricalVarToHistoricalVar(
rVariable, rVariable, rOriginModelPart, rDestinationModelPart,
rFlag, CheckValue, WriteBufferStep);
}

template <class TDataType>
void CopyModelPartFlaggedNodalNonHistoricalVarToHistoricalVar(
const Variable<TDataType>& rVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true,
const unsigned int WriteBufferStep = 0)
{
CopyModelPartFlaggedNodalNonHistoricalVarToHistoricalVar(
rVariable, rVariable, rModelPart, rModelPart,
rFlag, CheckValue, WriteBufferStep);
}

template <class TDataType>
void CopyModelPartFlaggedNodalNonHistoricalVarToNonHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
KRATOS_TRY

KRATOS_ERROR_IF(
rOriginModelPart.FullName() == rDestinationModelPart.FullName() &&
rOriginVariable == rDestinationVariable
) << "Trying to copy flagged nodal non-historical values with the same model parts/variables. This is not permitted ( Origin model part: "
<< rOriginModelPart.Name() << ", destination model part: " << rDestinationModelPart.Name()
<< ", variable: " << rOriginVariable.Name() << " ) !";

CopyModelPartFlaggedVariable<NodesContainerType>(
rOriginModelPart, rDestinationModelPart, rFlag, CheckValue,
[&](NodeType& rDestNode, const TDataType& rValue) {
rDestNode.SetValue(rDestinationVariable, rValue);
},
[&](const NodeType& rOriginNode) -> const TDataType& {
return rOriginNode.GetValue(rOriginVariable);
});

rDestinationModelPart.GetCommunicator().SynchronizeNonHistoricalVariable(rDestinationVariable);

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedNodalNonHistoricalVarToNonHistoricalVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
CopyModelPartFlaggedNodalNonHistoricalVarToNonHistoricalVar(
rOriginVariable, rDestinationVariable, rModelPart, rModelPart, rFlag, CheckValue);
}

template <class TDataType>
void CopyModelPartFlaggedNodalNonHistoricalVarToNonHistoricalVar(
const Variable<TDataType>& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
CopyModelPartFlaggedNodalNonHistoricalVarToNonHistoricalVar(
rVariable, rVariable, rOriginModelPart, rDestinationModelPart, rFlag, CheckValue);
}

template <class TDataType>
void CopyModelPartFlaggedElementVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
KRATOS_TRY

KRATOS_ERROR_IF(rOriginModelPart.FullName() == rDestinationModelPart.FullName() && rOriginVariable == rDestinationVariable)
<< "Trying to copy flagged elemental variable data with the same model "
"parts/variables. This is not permitted ( Origin model part: "
<< rOriginModelPart.Name() << ", destination model part: " << rDestinationModelPart.Name()
<< ", variable: " << rOriginVariable.Name() << " ) !";

CopyModelPartFlaggedVariable<ElementsContainerType>(
rOriginModelPart, rDestinationModelPart, rFlag, CheckValue,
[&](ElementType& rDestElement, const TDataType& rValue) {
rDestElement.SetValue(rDestinationVariable, rValue);
},
[&](const ElementType& rOriginElement) -> const TDataType& {
return rOriginElement.GetValue(rOriginVariable);
});

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedElementVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
CopyModelPartFlaggedElementVar(
rOriginVariable, rDestinationVariable, rModelPart, rModelPart, rFlag, CheckValue);
}

template <class TDataType>
void CopyModelPartFlaggedElementVar(
const Variable<TDataType>& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
CopyModelPartFlaggedElementVar(
rVariable, rVariable, rOriginModelPart, rDestinationModelPart, rFlag, CheckValue);
}

template <class TDataType>
void CopyModelPartFlaggedConditionVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
KRATOS_TRY

KRATOS_ERROR_IF(rOriginModelPart.FullName() == rDestinationModelPart.FullName() && rOriginVariable == rDestinationVariable)
<< "Trying to copy flagged condition variable data with the same model "
"parts/variables. This is not permitted ( Origin model part: "
<< rOriginModelPart.Name() << ", destination model part: " << rDestinationModelPart.Name()
<< ", variable: " << rOriginVariable.Name() << " ) !";

CopyModelPartFlaggedVariable<ConditionsContainerType>(
rOriginModelPart, rDestinationModelPart, rFlag, CheckValue,
[&](ConditionType& rDestCondition, const TDataType& rValue) {
rDestCondition.SetValue(rDestinationVariable, rValue);
},
[&](const ConditionType& rOriginCondition) -> const TDataType& {
return rOriginCondition.GetValue(rOriginVariable);
});

KRATOS_CATCH("");
}

template <class TDataType>
void CopyModelPartFlaggedConditionVar(
const Variable<TDataType>& rOriginVariable,
const Variable<TDataType>& rDestinationVariable,
ModelPart& rModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
CopyModelPartFlaggedConditionVar(
rOriginVariable, rDestinationVariable, rModelPart, rModelPart, rFlag, CheckValue);
}

template <class TDataType>
void CopyModelPartFlaggedConditionVar(
const Variable<TDataType>& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue = true)
{
CopyModelPartFlaggedConditionVar(
rVariable, rVariable, rOriginModelPart, rDestinationModelPart, rFlag, CheckValue);
}


template< class TVarType >
void CopyModelPartElementalVar(
const TVarType& rVariable,
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart){

const int n_orig_elems = rOriginModelPart.NumberOfElements();
const int n_dest_elems = rDestinationModelPart.NumberOfElements();

KRATOS_ERROR_IF_NOT(n_orig_elems == n_dest_elems) << "Origin and destination model parts have different number of elements."
<< "\n\t- Number of origin elements: " << n_orig_elems
<< "\n\t- Number of destination elements: " << n_dest_elems << std::endl;

IndexPartition<std::size_t>(n_orig_elems).for_each([&](std::size_t index){
auto it_dest_elems = rDestinationModelPart.ElementsBegin() + index;
const auto it_orig_elems = rOriginModelPart.ElementsBegin() + index;
const auto& r_value = it_orig_elems->GetValue(rVariable);
it_dest_elems->SetValue(rVariable,r_value);
});
}


template<class TDataType, class TVarType = Variable<TDataType> >
void SetVariable(
const TVarType& rVariable,
const TDataType& rValue,
NodesContainerType& rNodes,
const unsigned int Step = 0)
{
KRATOS_TRY

block_for_each(rNodes, [&](Node& rNode) {
rNode.FastGetSolutionStepValue(rVariable, Step) = rValue;
});

KRATOS_CATCH("")
}


template <class TDataType, class TVarType = Variable<TDataType>>
void SetVariable(
const TVarType &rVariable,
const TDataType &rValue,
NodesContainerType &rNodes,
const Flags Flag,
const bool CheckValue = true)
{
KRATOS_TRY

block_for_each(rNodes, [&](Node& rNode){
if(rNode.Is(Flag) == CheckValue){
rNode.FastGetSolutionStepValue(rVariable) = rValue;}
});

KRATOS_CATCH("")
}


template< class TType , class TContainerType>
void SetNonHistoricalVariableToZero(
const Variable< TType >& rVariable,
TContainerType& rContainer)
{
KRATOS_TRY
this->SetNonHistoricalVariable(rVariable, rVariable.Zero(), rContainer);
KRATOS_CATCH("")
}


template<class TContainerType, class... TVariableArgs>
static void SetNonHistoricalVariablesToZero(
TContainerType& rContainer,
const TVariableArgs&... rVariableArgs)
{
block_for_each(rContainer, [&](auto& rEntity){
(rEntity.SetValue(rVariableArgs, rVariableArgs.Zero()), ...);
});
}


template< class TType >
void SetHistoricalVariableToZero(
const Variable< TType >& rVariable,
NodesContainerType& rNodes)
{
KRATOS_TRY
this->SetVariable(rVariable, rVariable.Zero(), rNodes);
KRATOS_CATCH("")
}


template<class... TVariableArgs>
static void SetHistoricalVariablesToZero(
NodesContainerType& rNodes,
const TVariableArgs&... rVariableArgs)
{
block_for_each(rNodes, [&](NodeType& rNode){(
AuxiliaryHistoricalValueSetter<typename TVariableArgs::Type>(rVariableArgs, rVariableArgs.Zero(), rNode), ...);
});
}


template< class TType, class TContainerType, class TVarType = Variable< TType >>
void SetNonHistoricalVariable(
const TVarType& rVariable,
const TType& Value,
TContainerType& rContainer
)
{
KRATOS_TRY

block_for_each(rContainer, [&](typename TContainerType::value_type& rEntity){
rEntity.SetValue(rVariable, Value);
});

KRATOS_CATCH("")
}


template< class TType, class TContainerType, class TVarType = Variable< TType >>
void SetNonHistoricalVariable(
const TVarType& rVariable,
const TType& rValue,
TContainerType& rContainer,
const Flags Flag,
const bool Check = true
)
{
KRATOS_TRY

block_for_each(rContainer, [&](typename TContainerType::value_type& rEntity){
if(rEntity.Is(Flag) == Check){
rEntity.SetValue(rVariable, rValue);}
});

KRATOS_CATCH("")
}


template< class TContainerType, class TVarType>
void EraseNonHistoricalVariable(
const TVarType& rVariable,
TContainerType& rContainer
)
{
KRATOS_TRY

block_for_each(rContainer, [&rVariable](auto& rEntity){
rEntity.GetData().Erase(rVariable);
});

KRATOS_CATCH("")
}


template< class TContainerType>
void ClearNonHistoricalData(TContainerType& rContainer)
{
KRATOS_TRY

block_for_each(rContainer, [&](typename TContainerType::value_type& rEntity){
rEntity.GetData().Clear();
});

KRATOS_CATCH("")
}


template <class TDataType, class TContainerType, class TWeightDataType>
void WeightedAccumulateVariableOnNodes(
ModelPart& rModelPart,
const Variable<TDataType>& rVariable,
const Variable<TWeightDataType>& rWeightVariable,
const bool IsInverseWeightProvided = false);


template< class TContainerType >
void SetFlag(
const Flags& rFlag,
const bool FlagValue,
TContainerType& rContainer
)
{
KRATOS_TRY

block_for_each(rContainer, [&](typename TContainerType::value_type& rEntity){
rEntity.Set(rFlag, FlagValue);
});

KRATOS_CATCH("")

}


template< class TContainerType >
void ResetFlag(
const Flags& rFlag,
TContainerType& rContainer
)
{
KRATOS_TRY

block_for_each(rContainer, [&](typename TContainerType::value_type& rEntity){
rEntity.Reset(rFlag);
});

KRATOS_CATCH("")
}


template< class TContainerType >
void FlipFlag(
const Flags& rFlag,
TContainerType& rContainer
)
{
KRATOS_TRY

block_for_each(rContainer, [&](typename TContainerType::value_type& rEntity){
rEntity.Flip(rFlag);
});

KRATOS_CATCH("")
}


template< class TDataType, class TVariableType = Variable<TDataType> >
void SaveVariable(
const TVariableType &rOriginVariable,
const TVariableType &rSavedVariable,
NodesContainerType &rNodesContainer)
{
KRATOS_TRY

block_for_each(rNodesContainer, [&](Node& rNode){
rNode.SetValue(rSavedVariable, rNode.FastGetSolutionStepValue(rOriginVariable));
});

KRATOS_CATCH("")
}


template< class TDataType, class TContainerType, class TVariableType = Variable<TDataType> >
void SaveNonHistoricalVariable(
const TVariableType &rOriginVariable,
const TVariableType &rSavedVariable,
TContainerType &rContainer
)
{
KRATOS_TRY

block_for_each(rContainer, [&](typename TContainerType::value_type& rEntity){
rEntity.SetValue(rSavedVariable, rEntity.GetValue(rOriginVariable));
});

KRATOS_CATCH("")
}


template< class TDataType, class TVariableType = Variable<TDataType> >
void CopyVariable(
const TVariableType &rOriginVariable,
const TVariableType &rDestinationVariable,
NodesContainerType &rNodesContainer)
{
KRATOS_TRY

block_for_each(rNodesContainer, [&](Node& rNode){
rNode.FastGetSolutionStepValue(rDestinationVariable) = rNode.FastGetSolutionStepValue(rOriginVariable);
});

KRATOS_CATCH("")
}


[[nodiscard]] NodesContainerType SelectNodeList(
const DoubleVarType& Variable,
const double Value,
const NodesContainerType& rOriginNodes
);


template<class TVarType>
int CheckVariableExists(
const TVarType& rVariable,
const NodesContainerType& rNodes
)
{
KRATOS_TRY

for (auto& i_node : rNodes)
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(rVariable, i_node);

return 0;

KRATOS_CATCH("");
}


template< class TVarType >
void ApplyFixity(
const TVarType& rVar,
const bool IsFixed,
NodesContainerType& rNodes
)
{
KRATOS_TRY

if (rNodes.size() != 0) {
KRATOS_ERROR_IF_NOT(rNodes.begin()->HasDofFor(rVar)) << "Trying to fix/free dof of variable " << rVar.Name() << " but this dof does not exist in node #" << rNodes.begin()->Id() << "!" << std::endl;

#ifdef KRATOS_DEBUG
for (const auto& r_node : rNodes) {
KRATOS_ERROR_IF_NOT(r_node.HasDofFor(rVar)) << "Trying to fix/free dof of variable " << rVar.Name() << " but this dof does not exist in node #" << r_node.Id() << "!" << std::endl;
}
#endif

CheckVariableExists(rVar, rNodes);

if (IsFixed) {
block_for_each(rNodes,[&](Node& rNode){
rNode.pGetDof(rVar)->FixDof();
});
} else {
block_for_each(rNodes,[&](Node& rNode){
rNode.pGetDof(rVar)->FreeDof();
});
}
}

KRATOS_CATCH("")
}


template< class TVarType >
void ApplyFixity(
const TVarType& rVariable,
const bool IsFixed,
NodesContainerType& rNodes,
const Flags& rFlag,
const bool CheckValue = true)
{
KRATOS_TRY

if (rNodes.size() != 0) {
KRATOS_ERROR_IF_NOT(rNodes.begin()->HasDofFor(rVariable))
<< "Trying to fix/free dof of variable " << rVariable.Name()
<< " but this dof does not exist in node #"
<< rNodes.begin()->Id() << "!" << std::endl;

#ifdef KRATOS_DEBUG
for (const auto& r_node : rNodes) {
KRATOS_ERROR_IF_NOT(r_node.HasDofFor(rVariable))
<< "Trying to fix/free dof of variable " << rVariable.Name()
<< " but this dof does not exist in node #" << r_node.Id()
<< "!" << std::endl;
}
#endif

CheckVariableExists(rVariable, rNodes);

if (IsFixed) {
block_for_each(rNodes, [&rVariable, &rFlag, CheckValue](NodeType& rNode) {
if (rNode.Is(rFlag) == CheckValue) {
rNode.pGetDof(rVariable)->FixDof();
}
});
}
else {
block_for_each(rNodes, [&rVariable, &rFlag, CheckValue](NodeType& rNode) {
if (rNode.Is(rFlag) == CheckValue) {
rNode.pGetDof(rVariable)->FreeDof();
}
});
}
}

KRATOS_CATCH("");
}



template< class TVarType >
void ApplyVector(
const TVarType& rVar,
const Vector& rData,
NodesContainerType& rNodes
)
{
KRATOS_TRY

if(rNodes.size() != 0 && rNodes.size() == rData.size()) {
CheckVariableExists(rVar, rNodes);

IndexPartition<std::size_t>(rNodes.size()).for_each([&](std::size_t index){
NodesContainerType::iterator it_node = rNodes.begin() + index;
it_node->FastGetSolutionStepValue(rVar) = rData[index];
});
} else
KRATOS_ERROR  << "There is a mismatch between the size of data array and the number of nodes ";

KRATOS_CATCH("")
}


[[nodiscard]] array_1d<double, 3> SumNonHistoricalNodeVectorVariable(
const ArrayVarType& rVar,
const ModelPart& rModelPart
);


template< class TVarType >
[[nodiscard]] double SumNonHistoricalNodeScalarVariable(
const TVarType& rVar,
const ModelPart& rModelPart
)
{
KRATOS_TRY

double sum_value = 0.0;

const auto& r_communicator = rModelPart.GetCommunicator();
const auto& r_local_mesh = r_communicator.LocalMesh();
const auto& r_nodes_array = r_local_mesh.Nodes();

sum_value = block_for_each<SumReduction<double>>(r_nodes_array, [&](Node& rNode){
return rNode.GetValue(rVar);
});

return r_communicator.GetDataCommunicator().SumAll(sum_value);

KRATOS_CATCH("")
}


template< class TDataType, class TVarType = Variable<TDataType> >
[[nodiscard]] TDataType SumHistoricalVariable(
const TVarType &rVariable,
const ModelPart &rModelPart,
const unsigned int BuffStep = 0
)
{
KRATOS_TRY

const auto &r_communicator = rModelPart.GetCommunicator();

TDataType sum_value = block_for_each<SumReduction<TDataType>>(r_communicator.LocalMesh().Nodes(),[&](Node& rNode){
return rNode.GetSolutionStepValue(rVariable, BuffStep);
});

return r_communicator.GetDataCommunicator().SumAll(sum_value);

KRATOS_CATCH("")

}


[[nodiscard]] array_1d<double, 3> SumConditionVectorVariable(
const ArrayVarType& rVar,
const ModelPart& rModelPart
);


template< class TVarType >
[[nodiscard]] double SumConditionScalarVariable(
const TVarType& rVar,
const ModelPart& rModelPart
)
{
KRATOS_TRY

double sum_value = 0.0;

const auto& r_communicator = rModelPart.GetCommunicator();
const auto& r_local_mesh = r_communicator.LocalMesh();
const auto& r_conditions_array = r_local_mesh.Conditions();

sum_value = block_for_each<SumReduction<double>>(r_conditions_array, [&](ConditionType& rCond){
return rCond.GetValue(rVar);
});

return r_communicator.GetDataCommunicator().SumAll(sum_value);

KRATOS_CATCH("")
}


array_1d<double, 3> SumElementVectorVariable(
const ArrayVarType& rVar,
const ModelPart& rModelPart
);


template< class TVarType >
[[nodiscard]] double SumElementScalarVariable(
const TVarType& rVar,
const ModelPart& rModelPart
)
{
KRATOS_TRY

double sum_value = 0.0;

const auto& r_communicator = rModelPart.GetCommunicator();
const auto& r_local_mesh = r_communicator.LocalMesh();
const auto& r_elements_array = r_local_mesh.Elements();

sum_value = block_for_each<SumReduction<double>>(r_elements_array, [&](ElementType& rElem){
return rElem.GetValue(rVar);
});

return r_communicator.GetDataCommunicator().SumAll(sum_value);

KRATOS_CATCH("")
}


template< class TVarType >
void AddDof(
const TVarType& rVar,
ModelPart& rModelPart
)
{
KRATOS_TRY

if(rModelPart.NumberOfNodes() != 0)
KRATOS_ERROR_IF_NOT(rModelPart.NodesBegin()->SolutionStepsDataHas(rVar)) << "ERROR:: Variable : " << rVar << "not included in the Solution step data ";

rModelPart.GetNodalSolutionStepVariablesList().AddDof(&rVar);

block_for_each(rModelPart.Nodes(),[&](Node& rNode){
rNode.AddDof(rVar);
});

KRATOS_CATCH("")
}


template< class TVarType >
void AddDofWithReaction(
const TVarType& rVar,
const TVarType& rReactionVar,
ModelPart& rModelPart
)
{
KRATOS_TRY

if(rModelPart.NumberOfNodes() != 0) {
KRATOS_ERROR_IF_NOT(rModelPart.NodesBegin()->SolutionStepsDataHas(rVar)) << "ERROR:: DoF Variable : " << rVar << "not included in the Soluttion step data ";
KRATOS_ERROR_IF_NOT(rModelPart.NodesBegin()->SolutionStepsDataHas(rReactionVar)) << "ERROR:: Reaction Variable : " << rReactionVar << "not included in the Soluttion step data ";
}

#ifdef KRATOS_DEBUG
CheckVariableExists(rVar, rModelPart.Nodes());
CheckVariableExists(rReactionVar, rModelPart.Nodes());
#endif

rModelPart.GetNodalSolutionStepVariablesList().AddDof(&rVar, &rReactionVar);

block_for_each(rModelPart.Nodes(),[&](Node& rNode){
rNode.AddDof(rVar,rReactionVar);
});

KRATOS_CATCH("")
}


static void AddDofsList(
const std::vector<std::string>& rDofsVarNamesList,
ModelPart& rModelPart);


static void AddDofsWithReactionsList(
const std::vector<std::array<std::string,2>>& rDofsAndReactionsNamesList,
ModelPart& rModelPart);


bool CheckVariableKeys();


void UpdateCurrentToInitialConfiguration(const ModelPart::NodesContainerType& rNodes);


void UpdateInitialToCurrentConfiguration(const ModelPart::NodesContainerType& rNodes);


void UpdateCurrentPosition(
const ModelPart::NodesContainerType& rNodes,
const ArrayVarType& rUpdateVariable = DISPLACEMENT,
const IndexType BufferPosition = 0
);


template<class TVectorType=Vector>
[[nodiscard]] TVectorType GetCurrentPositionsVector(
const ModelPart::NodesContainerType& rNodes,
const unsigned int Dimension
);


template<class TVectorType=Vector>
[[nodiscard]] TVectorType GetInitialPositionsVector(
const ModelPart::NodesContainerType& rNodes,
const unsigned int Dimension
);


void SetCurrentPositionsVector(
ModelPart::NodesContainerType& rNodes,
const Vector& rPositions
);


void SetInitialPositionsVector(
ModelPart::NodesContainerType& rNodes,
const Vector& rPositions
);


[[nodiscard]] Vector GetSolutionStepValuesVector(
const ModelPart::NodesContainerType& rNodes,
const Variable<array_1d<double,3>>& rVar,
const unsigned int Step,
const unsigned int Dimension=3
);


[[nodiscard]] Vector GetSolutionStepValuesVector(
const ModelPart::NodesContainerType& rNodes,
const Variable<double>& rVar,
const unsigned int Step
);


void SetSolutionStepValuesVector(
ModelPart::NodesContainerType& rNodes,
const Variable<array_1d<double,3>>& rVar,
const Vector& rData,
const unsigned int Step
);


void SetSolutionStepValuesVector(
ModelPart::NodesContainerType& rNodes,
const Variable<double>& rVar,
const Vector& rData,
const unsigned int Step
);


[[nodiscard]] Vector GetValuesVector(
const ModelPart::NodesContainerType& rNodes,
const Variable<array_1d<double,3>>& rVar,
const unsigned int Dimension=3
);


[[nodiscard]] Vector GetValuesVector(
const ModelPart::NodesContainerType& rNodes,
const Variable<double>& rVar
);


void SetValuesVector(
ModelPart::NodesContainerType& rNodes,
const Variable<array_1d<double,3>>& rVar,
const Vector& rData
);


void SetValuesVector(
ModelPart::NodesContainerType& rNodes,
const Variable<double>& rVar,
const Vector& rData
);









private:






template< class TVarType >
bool CheckVariableKeysHelper()
{
KRATOS_TRY

for (const auto& var : KratosComponents< TVarType >::GetComponents()) {
if (var.first == "NONE" || var.first == "")
std::cout << " var first is NONE or empty " << var.first << var.second << std::endl;
if (var.second->Name() == "NONE" || var.second->Name() == "")
std::cout << var.first << var.second << std::endl;
if (var.first != var.second->Name()) 
std::cout << "Registration Name = " << var.first << " Variable Name = " << std::endl;
}

return true;
KRATOS_CATCH("")
}

template <class TContainerType>
[[nodiscard]] TContainerType& GetContainer(ModelPart& rModelPart);

template <class TContainerType>
[[nodiscard]] const TContainerType& GetContainer(const ModelPart& rModelPart);

template<class TDataType>
static void AuxiliaryHistoricalValueSetter(
const Variable<TDataType>& rVariable,
const TDataType& rValue,
NodeType& rNode);

template <class TContainerType, class TSetterFunction, class TGetterFunction>
void CopyModelPartFlaggedVariable(
const ModelPart& rOriginModelPart,
ModelPart& rDestinationModelPart,
const Flags& rFlag,
const bool CheckValue,
TSetterFunction&& rSetterFunction,
TGetterFunction&& rGetterFunction)
{
KRATOS_TRY

const auto& r_origin_container = GetContainer<TContainerType>(rOriginModelPart);
auto& r_destination_container = GetContainer<TContainerType>(rDestinationModelPart);

const int number_of_origin_items = r_origin_container.size();
const int number_of_destination_items = r_destination_container.size();

KRATOS_ERROR_IF_NOT(number_of_origin_items == number_of_destination_items)
<< "Origin ( " << rOriginModelPart.Name() << " ) and destination ( "
<< rDestinationModelPart.Name() << " ) model parts have different number of items."
<< "\n\t- Number of origin items: " << number_of_origin_items
<< "\n\t- Number of destination items: " << number_of_destination_items
<< std::endl;

IndexPartition<int>(number_of_origin_items).for_each([&](int i_node) {
const auto& r_orig_item = *(r_origin_container.begin() + i_node);
auto& r_dest_item = *(r_destination_container.begin() + i_node);
if (r_orig_item.Is(rFlag) == CheckValue) {
rSetterFunction(r_dest_item, rGetterFunction(r_orig_item));
}
});

KRATOS_CATCH("");
}








}; 





} 
