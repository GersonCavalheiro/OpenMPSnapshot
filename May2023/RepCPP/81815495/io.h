
#pragma once

#include <string>
#include <iostream>
#include <unordered_set>


#include "includes/model_part.h"

namespace Kratos
{








class KRATOS_API(KRATOS_CORE) IO
{
public:

KRATOS_CLASS_POINTER_DEFINITION(IO);

KRATOS_DEFINE_LOCAL_FLAG( READ );
KRATOS_DEFINE_LOCAL_FLAG( WRITE );
KRATOS_DEFINE_LOCAL_FLAG( APPEND );
KRATOS_DEFINE_LOCAL_FLAG( IGNORE_VARIABLES_ERROR );
KRATOS_DEFINE_LOCAL_FLAG( SKIP_TIMER );
KRATOS_DEFINE_LOCAL_FLAG( MESH_ONLY );
KRATOS_DEFINE_LOCAL_FLAG( SCIENTIFIC_PRECISION );

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

typedef Mesh<NodeType, Properties, Element, Condition> MeshType;

typedef MeshType::NodesContainerType NodesContainerType;

typedef MeshType::PropertiesContainerType PropertiesContainerType;

typedef ModelPart::GeometryContainerType GeometryContainerType;

typedef MeshType::ElementsContainerType ElementsContainerType;

typedef MeshType::ConditionsContainerType ConditionsContainerType;

typedef std::vector<std::vector<std::size_t> > ConnectivitiesContainerType;

typedef std::vector<std::vector<std::size_t> > PartitionIndicesContainerType;

typedef std::vector<std::size_t> PartitionIndicesType;

typedef std::size_t SizeType;

typedef DenseMatrix<int> GraphType;

struct PartitioningInfo
{
GraphType Graph;
PartitionIndicesType NodesPartitions; 
PartitionIndicesType ElementsPartitions; 
PartitionIndicesType ConditionsPartitions; 
PartitionIndicesContainerType NodesAllPartitions; 
PartitionIndicesContainerType ElementsAllPartitions; 
PartitionIndicesContainerType ConditionsAllPartitions; 
};


IO() = default;

virtual ~IO() = default;

IO(IO const& rOther) = delete;


IO& operator=(IO const& rOther) = delete;



virtual bool ReadNode(NodeType& rThisNode)
{
KRATOS_ERROR << "Calling base class method (ReadNode). Please check the definition of derived class." << std::endl;
}


virtual bool ReadNodes(NodesContainerType& rThisNodes)
{
KRATOS_ERROR << "Calling base class method (ReadNodes). Please check the definition of derived class" << std::endl;
}


virtual std::size_t ReadNodesNumber()
{
KRATOS_ERROR << "Calling base class method (ReadNodesNumber). Please check the definition of derived class." << std::endl;;
}


virtual void WriteNodes(NodesContainerType const& rThisNodes)
{
KRATOS_ERROR << "Calling base class method (WriteNodes). Please check the definition of derived class" << std::endl;
}


virtual void ReadProperties(Properties& rThisProperties)
{
KRATOS_ERROR << "Calling base class method (ReadProperties). Please check the definition of derived class" << std::endl;
}


virtual void ReadProperties(PropertiesContainerType& rThisProperties)
{
KRATOS_ERROR << "Calling base class method (ReadProperties). Please check the definition of derived class" << std::endl;
}


virtual void WriteProperties(Properties const& rThisProperties)
{
KRATOS_ERROR << "Calling base class method (WriteProperties). Please check the definition of derived class" << std::endl;
}


virtual void WriteProperties(PropertiesContainerType const& rThisProperties)
{
KRATOS_ERROR << "Calling base class method (WriteProperties). Please check the definition of derived class" << std::endl;
}


virtual void ReadGeometry(
NodesContainerType& rThisNodes,
GeometryType::Pointer& pThisGeometry
)
{
KRATOS_ERROR << "Calling base class method (ReadGeometry). Please check the definition of derived class" << std::endl;
}


virtual void ReadGeometries(
NodesContainerType& rThisNodes,
GeometryContainerType& rThisGeometries
)
{
KRATOS_ERROR << "Calling base class method (ReadGeometries). Please check the definition of derived class" << std::endl;
}


virtual std::size_t ReadGeometriesConnectivities(ConnectivitiesContainerType& rGeometriesConnectivities)
{
KRATOS_ERROR << "Calling base class method (ReadGeometriesConnectivities). Please check the definition of derived class" << std::endl;
}


virtual void WriteGeometries(GeometryContainerType const& rThisGeometries)
{
KRATOS_ERROR << "Calling base class method (WriteGeometries). Please check the definition of derived class" << std::endl;
}


virtual void ReadElement(
NodesContainerType& rThisNodes,
PropertiesContainerType& rThisProperties,
Element::Pointer& pThisElement
)
{
KRATOS_ERROR << "Calling base class method (ReadElement). Please check the definition of derived class" << std::endl;
}


virtual void ReadElements(
NodesContainerType& rThisNodes,
PropertiesContainerType& rThisProperties,
ElementsContainerType& rThisElements
)
{
KRATOS_ERROR << "Calling base class method (ReadElements). Please check the definition of derived class" << std::endl;
}


virtual std::size_t ReadElementsConnectivities(ConnectivitiesContainerType& rElementsConnectivities)
{
KRATOS_ERROR << "Calling base class method (ReadElementsConnectivities). Please check the definition of derived class" << std::endl;
}


virtual void WriteElements(ElementsContainerType const& rThisElements)
{
KRATOS_ERROR << "Calling base class method (WriteElements). Please check the definition of derived class" << std::endl;
}


virtual void ReadCondition(
NodesContainerType& rThisNodes,
PropertiesContainerType& rThisProperties,
Condition::Pointer& pThisCondition
)
{
KRATOS_ERROR << "Calling base class method (ReadCondition). Please check the definition of derived class" << std::endl;
}


virtual void ReadConditions(
NodesContainerType& rThisNodes,
PropertiesContainerType& rThisProperties,
ConditionsContainerType& rThisConditions
)
{
KRATOS_ERROR << "Calling base class method (ReadConditions). Please check the definition of derived class" << std::endl;
}


virtual std::size_t ReadConditionsConnectivities(ConnectivitiesContainerType& rConditionsConnectivities)
{
KRATOS_ERROR << "Calling base class method (ReadConditionsConnectivities). Please check the definition of derived class" << std::endl;
}


virtual void WriteConditions(ConditionsContainerType const& rThisConditions)
{
KRATOS_ERROR << "Calling base class method (WriteConditions). Please check the definition of derived class" << std::endl;
}


virtual void ReadInitialValues(ModelPart& rThisModelPart)
{
KRATOS_ERROR << "Calling base class method (ReadInitialValues). Please check the definition of derived class" << std::endl;
}


virtual void ReadInitialValues(NodesContainerType& rThisNodes, ElementsContainerType& rThisElements, ConditionsContainerType& rThisConditions)
{
KRATOS_ERROR << "Calling base class method (ReadInitialValues). Please check the definition of derived class" << std::endl;
}


virtual void ReadMesh(MeshType & rThisMesh)
{
KRATOS_ERROR << "Calling base class method (ReadMesh). Please check the definition of derived class" << std::endl;
}


KRATOS_DEPRECATED_MESSAGE("'WriteMesh' with a non-const Mesh as input is deprecated. Please use the version of this function that accepts a const Mesh instead.")
virtual void WriteMesh( MeshType& rThisMesh )
{
KRATOS_ERROR << "Calling base class method (WriteMesh). Please check the implementation of derived classes" << std::endl;
}


virtual void WriteMesh(const MeshType& rThisMesh )
{
MeshType& non_const_mesh = const_cast<MeshType&>(rThisMesh);
KRATOS_START_IGNORING_DEPRECATED_FUNCTION_WARNING
this->WriteMesh(non_const_mesh);
KRATOS_STOP_IGNORING_DEPRECATED_FUNCTION_WARNING

}


virtual void ReadModelPart(ModelPart & rThisModelPart)
{
KRATOS_ERROR << "Calling base class method (ReadModelPart). Please check the definition of derived class" << std::endl;
}


KRATOS_DEPRECATED_MESSAGE("'WriteModelPart' with a non-const ModelPart as input is deprecated. Please use the version of this function that accepts a const ModelPart instead.")
virtual void WriteModelPart(ModelPart & rThisModelPart)
{
KRATOS_ERROR << "Calling base class method (WriteModelPart). Please check the definition of derived class" << std::endl;
}


virtual void WriteModelPart(const ModelPart& rThisModelPart)
{
ModelPart& non_const_model_part = const_cast<ModelPart&>(rThisModelPart);
KRATOS_START_IGNORING_DEPRECATED_FUNCTION_WARNING
this->WriteModelPart(non_const_model_part);
KRATOS_STOP_IGNORING_DEPRECATED_FUNCTION_WARNING

}


KRATOS_DEPRECATED_MESSAGE("'WriteNodeMesh' with a non-const Mesh as input is deprecated. Please use the version of this function that accepts a const Mesh instead.")
virtual void WriteNodeMesh( MeshType& rThisMesh )
{
KRATOS_ERROR << "Calling base class method (WriteNodeMesh). Please check the implementation of derived classes" << std::endl;
}


virtual void WriteNodeMesh(const MeshType& rThisMesh )
{
MeshType& non_const_mesh = const_cast<MeshType&>(rThisMesh);
KRATOS_START_IGNORING_DEPRECATED_FUNCTION_WARNING
this->WriteNodeMesh(non_const_mesh);
KRATOS_STOP_IGNORING_DEPRECATED_FUNCTION_WARNING

}


virtual std::size_t ReadNodalGraph(ConnectivitiesContainerType& rAuxConnectivities)
{
KRATOS_ERROR << "Calling base class method (ReadNodalGraph). Please check the definition of derived class" << std::endl;;
}


virtual void DivideInputToPartitions(SizeType NumberOfPartitions,
const PartitioningInfo& rPartitioningInfo)
{
KRATOS_START_IGNORING_DEPRECATED_FUNCTION_WARNING
DivideInputToPartitions(NumberOfPartitions, rPartitioningInfo.Graph, rPartitioningInfo.NodesPartitions, rPartitioningInfo.ElementsPartitions, rPartitioningInfo.ConditionsPartitions, rPartitioningInfo.NodesAllPartitions, rPartitioningInfo.ElementsAllPartitions, rPartitioningInfo.ConditionsAllPartitions); 
KRATOS_STOP_IGNORING_DEPRECATED_FUNCTION_WARNING

}


KRATOS_DEPRECATED_MESSAGE("'This version of \"DivideInputToPartitions\" is deprecated, please use the interface that accepts a \"PartitioningInfo\"")
virtual void DivideInputToPartitions(SizeType NumberOfPartitions,
GraphType const& rDomainsColoredGraph,
PartitionIndicesType const& rNodesPartitions,
PartitionIndicesType const& rElementsPartitions,
PartitionIndicesType const& rConditionsPartitions,
PartitionIndicesContainerType const& rNodesAllPartitions,
PartitionIndicesContainerType const& rElementsAllPartitions,
PartitionIndicesContainerType const& rConditionsAllPartitions)
{
KRATOS_ERROR << "Calling base class method (DivideInputToPartitions). Please check the definition of derived class" << std::endl;
}


virtual void DivideInputToPartitions(Kratos::shared_ptr<std::iostream> * pStreams,
SizeType NumberOfPartitions,
const PartitioningInfo& rPartitioningInfo)
{
KRATOS_START_IGNORING_DEPRECATED_FUNCTION_WARNING
DivideInputToPartitions(pStreams, NumberOfPartitions, rPartitioningInfo.Graph, rPartitioningInfo.NodesPartitions, rPartitioningInfo.ElementsPartitions, rPartitioningInfo.ConditionsPartitions, rPartitioningInfo.NodesAllPartitions, rPartitioningInfo.ElementsAllPartitions, rPartitioningInfo.ConditionsAllPartitions); 
KRATOS_STOP_IGNORING_DEPRECATED_FUNCTION_WARNING

}


KRATOS_DEPRECATED_MESSAGE("'This version of \"DivideInputToPartitions\" is deprecated, please use the interface that accepts a \"PartitioningInfo\"")
virtual void DivideInputToPartitions(Kratos::shared_ptr<std::iostream> * pStreams,
SizeType NumberOfPartitions,
GraphType const& rDomainsColoredGraph,
PartitionIndicesType const& rNodesPartitions,
PartitionIndicesType const& rElementsPartitions,
PartitionIndicesType const& rConditionsPartitions,
PartitionIndicesContainerType const& rNodesAllPartitions,
PartitionIndicesContainerType const& rElementsAllPartitions,
PartitionIndicesContainerType const& rConditionsAllPartitions)
{
KRATOS_ERROR << "Calling base class method (DivideInputToPartitions). Please check the definition of derived class" << std::endl;
}

virtual void ReadSubModelPartElementsAndConditionsIds(
std::string const& rModelPartName,
std::unordered_set<SizeType> &rElementsIds,
std::unordered_set<SizeType> &rConditionsIds)
{
KRATOS_ERROR << "Calling base class method (ReadSubModelPartElementsAndConditionsIds). Please check the definition of derived class" << std::endl;
}

virtual std::size_t ReadNodalGraphFromEntitiesList(
ConnectivitiesContainerType& rAuxConnectivities,
std::unordered_set<SizeType> &rElementsIds,
std::unordered_set<SizeType> &rConditionsIds)
{
KRATOS_ERROR << "Calling base class method (ReadNodalGraphFromEntitiesList). Please check the definition of derived class" << std::endl;
}






virtual std::string Info() const
{
return "IO";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "IO";
}

virtual void PrintData(std::ostream& rOStream) const
{
}



protected:















private:














}; 





inline std::istream& operator >> (std::istream& rIStream,
IO& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const IO& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
