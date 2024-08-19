



#pragma once


#include "includes/io.h"
#include "processes/process.h"

namespace Kratos
{









class MortonPartitioningProcess
: public Process
{
public:

typedef idx_t idxtype;

KRATOS_CLASS_POINTER_DEFINITION(MortonPartitioningProcess);

typedef std::size_t SizeType;
typedef std::size_t IndexType;
typedef std::vector<idxtype> PartitionIndicesType;


MortonPartitioningProcess(IO::ConnectivitiesContainerType& rElementsConnectivities,
IO::NodesContainerType& rNodesContainer,
PartitionIndicesType& rNodesPartitions,
PartitionIndicesType& rElementsPartitions,
SizeType NumberOfPartitions, int Dimension = 3)
: mrElementsConnectivities(rElementsConnectivities),
mrNodesContainer(rNodesContainer),
mrNodesPartitions(rNodesPartitions),
mrElementsPartitions(rElementsPartitions),
mNumberOfPartitions(NumberOfPartitions),
mDimension(Dimension)
{
}

MortonPartitioningProcess(MortonPartitioningProcess const& rOther)
: mrElementsConnectivities(rOther.mrElementsConnectivities),
mrNodesContainer(rOther.mrNodesContainer),
mrNodesPartitions(rOther.mrNodesPartitions),
mrElementsPartitions(rOther.mrElementsPartitions),
mNumberOfPartitions(rOther.mNumberOfPartitions),
mDimension(rOther.mDimension)
{
}

virtual ~MortonPartitioningProcess()
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

int number_of_elements = mrElementsConnectivities.size();

int number_of_nodes = 0;
int real_r_of_nodes = 0;
for(IO::ConnectivitiesContainerType::iterator i_element = mrElementsConnectivities.begin() ; i_element != mrElementsConnectivities.end() ; i_element++)
for(IO::ConnectivitiesContainerType::value_type::iterator i_node_id = i_element->begin() ; i_node_id != i_element->end() ; i_node_id++)
{
if(static_cast<int>(*i_node_id) > number_of_nodes)
number_of_nodes = *i_node_id;

real_r_of_nodes++;
}

std::vector< bool > aux(number_of_nodes,false);
for(IO::ConnectivitiesContainerType::iterator i_element = mrElementsConnectivities.begin() ; i_element != mrElementsConnectivities.end() ; i_element++)
for(IO::ConnectivitiesContainerType::value_type::iterator i_node_id = i_element->begin() ; i_node_id != i_element->end() ; i_node_id++)
{
aux[static_cast<int>(*i_node_id)-1] = true;
}

mrElementsPartitions.resize(number_of_elements);
mrNodesPartitions.resize(number_of_nodes);

idxtype* epart = &(*(mrElementsPartitions.begin()));
idxtype* npart = &(*(mrNodesPartitions.begin()));

AssignPartition(number_of_nodes, real_r_of_nodes, npart, epart);

KRATOS_CATCH("")
}







std::string Info() const override
{
return "MortonPartitioningProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MortonPartitioningProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}





protected:



IO::ConnectivitiesContainerType& mrElementsConnectivities;
IO::NodesContainerType& mrNodesContainer;
PartitionIndicesType& mrNodesPartitions;
PartitionIndicesType& mrElementsPartitions;
SizeType mNumberOfPartitions;
SizeType mDimension;





void AssignPartition(SizeType NumberOfNodes, SizeType NumberOfElements, idxtype* NPart, idxtype* EPart)
{
for(SizeType i_element = 0 ; i_element < NumberOfElements ; i_element++)
{
for(IO::ConnectivitiesContainerType::value_type::iterator i_node = mrElementsConnectivities[i_element].begin() ;
i_node != mrElementsConnectivities[i_element].end() ; i_node++)
{

NPart[(*i_node-1)] = i_element%mNumberOfPartitions;
EPart[i_element] = i_element%mNumberOfPartitions;
}
}
}








private:











MortonPartitioningProcess& operator=(MortonPartitioningProcess const& rOther);




}; 






inline std::istream& operator >> (std::istream& rIStream,
MortonPartitioningProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const MortonPartitioningProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
