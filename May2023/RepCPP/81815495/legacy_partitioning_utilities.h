
#pragma once



#include "metis.h"


#include "includes/io.h"


namespace Kratos {




class KRATOS_API(METIS_APPLICATION) LegacyPartitioningUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION(LegacyPartitioningUtilities);

using idxtype = idx_t; 
using PartitionIndicesType = std::vector<idxtype>;
using SizeType = std::size_t;


LegacyPartitioningUtilities() = delete;

LegacyPartitioningUtilities(LegacyPartitioningUtilities const& rOther) = delete;


LegacyPartitioningUtilities& operator=(LegacyPartitioningUtilities const& rOther) = delete;


static void CalculateDomainsGraph(
IO::GraphType& rDomainsGraph,
SizeType NumberOfElements,
IO::ConnectivitiesContainerType& ElementsConnectivities,
PartitionIndicesType const& NPart,
PartitionIndicesType const&  EPart);

static void DividingNodes(
IO::PartitionIndicesContainerType& rNodesAllPartitions,
IO::ConnectivitiesContainerType& ElementsConnectivities,
IO::ConnectivitiesContainerType& ConditionsConnectivities,
PartitionIndicesType const& NodesPartitions,
PartitionIndicesType const& ElementsPartitions,
PartitionIndicesType const& ConditionsPartitions);

static void DividingElements(
IO::PartitionIndicesContainerType& rElementsAllPartitions,
PartitionIndicesType const& ElementsPartitions);

static void DividingConditions(
IO::PartitionIndicesContainerType& rConditionsAllPartitions,
PartitionIndicesType const& ConditionsPartitions);

static void ConvertKratosToCSRFormat(
IO::ConnectivitiesContainerType& KratosFormatNodeConnectivities,
idxtype** NodeIndices,
idxtype** NodeConnectivities);


}; 



} 
