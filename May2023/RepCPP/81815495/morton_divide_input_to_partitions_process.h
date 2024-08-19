
#pragma once


#include "processes/graph_coloring_process.h"
#include "custom_processes/morton_partitioning_process.h"

namespace Kratos
{









class MortonDivideInputToPartitionsProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MortonDivideInputToPartitionsProcess);

typedef std::size_t SizeType;
typedef std::size_t IndexType;
typedef GraphColoringProcess::GraphType GraphType;


MortonDivideInputToPartitionsProcess(IO& rIO, SizeType NumberOfPartitions, int Dimension = 3)
:  mrIO(rIO), mNumberOfPartitions(NumberOfPartitions), mDimension(Dimension)
{
}

MortonDivideInputToPartitionsProcess(MortonDivideInputToPartitionsProcess const& rOther)
: mrIO(rOther.mrIO), mNumberOfPartitions(rOther.mNumberOfPartitions), mDimension(rOther.mDimension)
{
}

virtual ~MortonDivideInputToPartitionsProcess()
{
}



void operator()()
{
Execute();
}



void Execute() override
{
KRATOS_TRY;

if(mNumberOfPartitions < 2) 
{
return;
}

IO::ConnectivitiesContainerType elements_connectivities;
IO::ConnectivitiesContainerType conditions_connectivities;
IO::NodesContainerType nodes_container;

int number_of_elements = mrIO.ReadElementsConnectivities(elements_connectivities);

mrIO.ReadConditionsConnectivities(conditions_connectivities);
mrIO.ReadNodes(nodes_container);

MortonPartitioningProcess::PartitionIndicesType nodes_partitions;
MortonPartitioningProcess::PartitionIndicesType elements_partitions;
MortonPartitioningProcess::PartitionIndicesType conditions_partitions;

MortonPartitioningProcess morton_partitioning_process(elements_connectivities, nodes_container, nodes_partitions, elements_partitions, mNumberOfPartitions, mDimension);
morton_partitioning_process.Execute();

GraphType domains_graph(mNumberOfPartitions,mNumberOfPartitions);
domains_graph = ScalarMatrix(mNumberOfPartitions,mNumberOfPartitions,1);

GraphType domains_colored_graph;

int colors_number;

CalculateDomainsGraph(domains_graph, number_of_elements, elements_connectivities, nodes_partitions, elements_partitions);

GraphColoringProcess(mNumberOfPartitions, domains_graph, domains_colored_graph, colors_number).Execute();


IO::PartitionIndicesContainerType nodes_all_partitions;
IO::PartitionIndicesContainerType elements_all_partitions;
IO::PartitionIndicesContainerType conditions_all_partitions;

ConditionsPartitioning(conditions_connectivities, nodes_partitions, conditions_partitions);
KRATOS_WATCH("ConditionsPartitioning finished")
DividingNodes(nodes_all_partitions, elements_connectivities, conditions_connectivities, nodes_partitions, elements_partitions, conditions_partitions);
KRATOS_WATCH("DividingNodes finished")
DividingElements(elements_all_partitions, elements_partitions);
KRATOS_WATCH("DividingElements finished")
DividingConditions(conditions_all_partitions, conditions_partitions);
KRATOS_WATCH("DividingConditions finished")

IO::PartitionIndicesType io_nodes_partitions(nodes_partitions.begin(), nodes_partitions.end());
IO::PartitionIndicesType io_elements_partitions(elements_partitions.begin(), elements_partitions.end());
IO::PartitionIndicesType io_conditions_partitions(conditions_partitions.begin(), conditions_partitions.end());

mrIO.DivideInputToPartitions(mNumberOfPartitions, domains_colored_graph,
io_nodes_partitions, io_elements_partitions, io_conditions_partitions,
nodes_all_partitions, elements_all_partitions, conditions_all_partitions);
KRATOS_WATCH("DivideInputToPartitions finished")
return;

KRATOS_CATCH("")
}

void CalculateDomainsGraph(GraphType& rDomainsGraph, SizeType NumberOfElements, IO::ConnectivitiesContainerType& ElementsConnectivities, MortonPartitioningProcess::PartitionIndicesType const& NPart, MortonPartitioningProcess::PartitionIndicesType const&  EPart )
{
for(SizeType i_element = 0 ; i_element < NumberOfElements ; i_element++)
{
for(std::vector<std::size_t>::iterator i_node = ElementsConnectivities[i_element].begin() ;
i_node != ElementsConnectivities[i_element].end() ; i_node++)
{
SizeType node_rank = NPart[*i_node-1];
SizeType element_rank = EPart[i_element];
if(node_rank != element_rank)
{
rDomainsGraph(node_rank, element_rank) = 1;
rDomainsGraph(element_rank, node_rank) = 1;
}
}
}
}







std::string Info() const override
{
return "MortonDivideInputToPartitionsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MortonDivideInputToPartitionsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}





protected:

class DomainEntitiesIdContainer
{
public:
DomainEntitiesIdContainer(std::size_t NumberOfNeighbours)
{
mLocalIds.resize(NumberOfNeighbours);
mGhostsIds.resize(NumberOfNeighbours);
mInterfacesIds.resize(NumberOfNeighbours);
}

std::vector<std::size_t>& AllIds()
{
return mAllIds;
}

std::vector<std::vector<std::size_t> >& LocalIds()
{
return mLocalIds;
}

std::vector<std::vector<std::size_t> >& GhostsIds()
{
return mGhostsIds;
}

std::vector<std::vector<std::size_t> >& InterfacesIds()
{
return mInterfacesIds;
}
private:
std::vector<std::size_t> mAllIds;
std::vector<std::vector<std::size_t> > mLocalIds;
std::vector<std::vector<std::size_t> > mGhostsIds;
std::vector<std::vector<std::size_t> > mInterfacesIds;

};





IO& mrIO;

SizeType mNumberOfPartitions;

SizeType mDimension;






void ConditionsPartitioning(IO::ConnectivitiesContainerType& ConditionsConnectivities,
MortonPartitioningProcess::PartitionIndicesType const& NodesPartitions,
MortonPartitioningProcess::PartitionIndicesType& ConditionsPartitions)
{
SizeType number_of_conditions = ConditionsConnectivities.size();

ConditionsPartitions.resize(number_of_conditions);

for(SizeType i_condition = 0 ; i_condition < number_of_conditions ; i_condition++)
{
if(ConditionsConnectivities[i_condition].size() > 0)
{
double average_index = 0.00;
for(IO::ConnectivitiesContainerType::value_type::iterator i_node = ConditionsConnectivities[i_condition].begin() ;
i_node != ConditionsConnectivities[i_condition].end() ; i_node++)
{
const int my_gid = *i_node-1;

average_index += NodesPartitions[my_gid];

}
average_index /= ConditionsConnectivities[i_condition].size();

double difference = mNumberOfPartitions + 10; 

for(IO::ConnectivitiesContainerType::value_type::iterator i_node = ConditionsConnectivities[i_condition].begin() ;
i_node != ConditionsConnectivities[i_condition].end() ; i_node++)
{
const int my_gid = *i_node-1;

const int node_partition = NodesPartitions[my_gid];

if(difference > fabs(average_index - node_partition))
{
difference = fabs(average_index - node_partition);
ConditionsPartitions[i_condition] = node_partition;
}
}

}
}
}

void DividingNodes(IO::PartitionIndicesContainerType& rNodesAllPartitions,
IO::ConnectivitiesContainerType& ElementsConnectivities,
IO::ConnectivitiesContainerType& ConditionsConnectivities,
MortonPartitioningProcess::PartitionIndicesType const& NodesPartitions,
MortonPartitioningProcess::PartitionIndicesType const& ElementsPartitions,
MortonPartitioningProcess::PartitionIndicesType const& ConditionsPartitions)
{
SizeType number_of_nodes = NodesPartitions.size();
SizeType number_of_elements = ElementsPartitions.size();
SizeType number_of_conditions = ConditionsPartitions.size();

rNodesAllPartitions.resize(number_of_nodes);

for(SizeType i_element = 0 ; i_element < number_of_elements ; i_element++)
{
const int element_partition = ElementsPartitions[i_element];

for(IO::ConnectivitiesContainerType::value_type::iterator i_node = ElementsConnectivities[i_element].begin() ;
i_node != ElementsConnectivities[i_element].end() ; i_node++)
{
const int my_gid = *i_node-1;

const int node_partition = NodesPartitions[my_gid];

if(element_partition != node_partition) 
{
std::cout << "Partiton element differnet form partition node" << std::endl;
rNodesAllPartitions[my_gid].push_back(element_partition);
}
}
}

for(SizeType i_condition = 0 ; i_condition < number_of_conditions ; i_condition++)
{
const int condition_partition = ConditionsPartitions[i_condition];

for(IO::ConnectivitiesContainerType::value_type::iterator i_node = ConditionsConnectivities[i_condition].begin() ;
i_node != ConditionsConnectivities[i_condition].end() ; i_node++)
{
const int my_gid = *i_node-1;

const int node_partition = NodesPartitions[my_gid];

if(condition_partition != node_partition) 
rNodesAllPartitions[my_gid].push_back(condition_partition);
}
}

for(SizeType i_node = 0 ; i_node < number_of_nodes ; i_node++)
{
IO::PartitionIndicesContainerType::value_type& node_partitions = rNodesAllPartitions[i_node];
node_partitions.push_back(NodesPartitions[i_node]);

std::sort(node_partitions.begin(), node_partitions.end());
IO::PartitionIndicesContainerType::value_type::iterator new_end=std::unique(node_partitions.begin(), node_partitions.end());
node_partitions.resize(new_end - node_partitions.begin());
}
}

void DividingElements(IO::PartitionIndicesContainerType& rElementsAllPartitions, MortonPartitioningProcess::PartitionIndicesType const& ElementsPartitions)
{
SizeType number_of_elements = ElementsPartitions.size();

rElementsAllPartitions.resize(number_of_elements);

for(SizeType i_element = 0 ; i_element < number_of_elements ; i_element++)
{
rElementsAllPartitions[i_element].push_back(ElementsPartitions[i_element]);
}
}

void DividingConditions(IO::PartitionIndicesContainerType& rConditionsAllPartitions, MortonPartitioningProcess::PartitionIndicesType const& ConditionsPartitions)
{
SizeType number_of_conditions = ConditionsPartitions.size();

rConditionsAllPartitions.resize(number_of_conditions);

for(SizeType i_condition = 0 ; i_condition < number_of_conditions ; i_condition++)
{
rConditionsAllPartitions[i_condition].push_back(ConditionsPartitions[i_condition]);
}
}










private:











MortonDivideInputToPartitionsProcess& operator=(MortonDivideInputToPartitionsProcess const& rOther);




}; 






inline std::istream& operator >> (std::istream& rIStream,
MortonDivideInputToPartitionsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const MortonDivideInputToPartitionsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
