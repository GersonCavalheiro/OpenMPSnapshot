#pragma once

#include "metis.h"

#include "includes/io.h"
#include "processes/process.h"

namespace Kratos
{






class KRATOS_API(METIS_APPLICATION) MetisDivideHeterogeneousInputProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MetisDivideHeterogeneousInputProcess);

using SizeType = IO::SizeType;
using GraphType = IO::GraphType;
using PartitioningInfo = IO::PartitioningInfo;
using PartitionIndicesType = IO::PartitionIndicesType;
using PartitionIndicesContainerType = IO::PartitionIndicesContainerType;
using idxtype = idx_t; 


MetisDivideHeterogeneousInputProcess(
IO& rIO,
SizeType NumberOfPartitions,
int Dimension = 3,
int Verbosity = 0,
bool SynchronizeConditions = false):
mrIO(rIO),
mNumberOfPartitions(NumberOfPartitions),
mSynchronizeConditions(SynchronizeConditions),
mVerbosity(Verbosity)
{
}

MetisDivideHeterogeneousInputProcess(MetisDivideHeterogeneousInputProcess const& rOther) = delete;

virtual ~MetisDivideHeterogeneousInputProcess()
{
}



void operator()()
{
this->Execute();
}




void Execute() override;

virtual void GetNodesPartitions(std::vector<idxtype> &rNodePartition, SizeType &rNumNodes);






std::string Info() const override
{
return "MetisDivideHeterogeneousInputProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MetisDivideHeterogeneousInputProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}





protected:


IO& mrIO;

SizeType mNumberOfPartitions;

bool mSynchronizeConditions;

int mVerbosity;
int mNumNodes;

std::vector<std::unordered_set<std::size_t>> mNodeConnectivities;



void ExecutePartitioning(PartitioningInfo& rPartitioningInfo);


int PartitionNodes(SizeType NumNodes,
idxtype* NodeIndices,
idxtype* NodeConnectivities,
std::vector<idxtype>& rNodePartition);

void PartitionMesh(std::vector<idxtype> const& NodePartition,
const IO::ConnectivitiesContainerType& rElemConnectivities,
std::vector<idxtype>& rElemPartition);

void PartitionElementsSynchronous(std::vector<idxtype> const& NodePartition,
const IO::ConnectivitiesContainerType& rElemConnectivities,
std::vector<idxtype>& rElemPartition);

void PartitionConditionsSynchronous(const std::vector<idxtype>& rNodePartition,
const std::vector<idxtype>& rElemPartition,
const IO::ConnectivitiesContainerType& rCondConnectivities,
const IO::ConnectivitiesContainerType& rElemConnectivities,
std::vector<idxtype>& rCondPartition);

void RedistributeHangingNodes(
std::vector<idxtype>& rNodePartition,
std::vector<idxtype> const& rElementPartition,
const IO::ConnectivitiesContainerType& rElementConnectivities,
std::vector<idxtype> const& rConditionPartition,
const IO::ConnectivitiesContainerType& rConditionConnectivities);

SizeType FindMax(SizeType NumTerms, const std::vector<int>& rVect);

void PrintDebugData(const std::string& rLabel,
const std::vector<idxtype>& rPartitionData);






MetisDivideHeterogeneousInputProcess& operator=(MetisDivideHeterogeneousInputProcess const& rOther);




}; 






inline std::istream& operator >> (std::istream& rIStream,
MetisDivideHeterogeneousInputProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const MetisDivideHeterogeneousInputProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}
