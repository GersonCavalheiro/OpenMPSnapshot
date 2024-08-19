
#pragma once


#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/data_communicator.h"


namespace Kratos
{


class KRATOS_API(OPTIMIZATION_APPLICATION) OptimizationUtils
{
public:

using IndexType = std::size_t;


template<class TContainerType>
static GeometryData::KratosGeometryType GetContainerEntityGeometryType(
const TContainerType& rContainer,
const DataCommunicator& rDataCommunicator);

template<class TContainerType, class TDataType>
static bool IsVariableExistsInAllContainerProperties(
const TContainerType& rContainer,
const Variable<TDataType>& rVariable,
const DataCommunicator& rDataCommunicator);

template<class TContainerType, class TDataType>
static bool IsVariableExistsInAtLeastOneContainerProperties(
const TContainerType& rContainer,
const Variable<TDataType>& rVariable,
const DataCommunicator& rDataCommunicator);

template<class TContainerType>
static void CreateEntitySpecificPropertiesForContainer(
ModelPart& rModelPart,
TContainerType& rContainer);

template<class TDataType>
static IndexType GetVariableDimension(
const Variable<TDataType>& rVariable,
const IndexType DomainSize);

static void CopySolutionStepVariablesList(
ModelPart& rDestinationModelPart,
const ModelPart& rOriginModelPart);

};

}