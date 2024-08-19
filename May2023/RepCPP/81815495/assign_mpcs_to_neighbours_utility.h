
#pragma once

#include <string>
#include <iostream>


#include "includes/define.h"

#include "utilities/variable_utils.h"
#include "utilities/rbf_shape_functions_utility.h"

#include "spatial_containers/spatial_search.h"
#include "spatial_containers/bins_dynamic.h"
#include "utilities/builtin_timer.h"

namespace Kratos {




class KRATOS_API(KRATOS_CORE) AssignMPCsToNeighboursUtility
{
public:

using NodeType = ModelPart::NodeType;

using DofType = NodeType::DofType;

using NodesContainerType = ModelPart::NodesContainerType;

using NodeBinsType = BinsDynamic<3, NodeType, NodesContainerType::ContainerType>;

using ConstraintContainerType = ModelPart::MasterSlaveConstraintContainerType;
using ResultNodesContainerType = NodesContainerType::ContainerType;
using VectorResultNodesContainerType = std::vector<ResultNodesContainerType>;
using RadiusArrayType = std::vector<double>;

using DofPointerVectorType = std::vector< Dof<double>::Pointer >;

KRATOS_CLASS_POINTER_DEFINITION(AssignMPCsToNeighboursUtility);


AssignMPCsToNeighboursUtility(NodesContainerType& rStructureNodes);

virtual ~AssignMPCsToNeighboursUtility(){
}




void SearchNodesInRadiusForNodes(
NodesContainerType const& rNodes,
const double Radius,
VectorResultNodesContainerType& rResults,
const double MinNumOfNeighNodes);


void SearchNodesInRadiusForNode(
NodeType::Pointer pNode,
double const Radius,
ResultNodesContainerType& rResults);


const Variable<double>& GetComponentVariable(
const Variable<array_1d<double, 3>>& rVectorVariable, 
const std::size_t ComponentIndex);



void GetDofsAndCoordinatesForNode(
NodeType::Pointer pNode,
const Variable<double>& rVariable,
DofPointerVectorType& rCloud_of_dofs,
array_1d<double,3>& rSlave_coordinates
);

void GetDofsAndCoordinatesForNode(
NodeType::Pointer pNode,
const Variable<array_1d<double, 3>>& rVariable,
DofPointerVectorType& rCloud_of_dofs_x,
DofPointerVectorType& rCloud_of_dofs_y,
DofPointerVectorType& rCloud_of_dofs_z,
array_1d<double,3>& rSlave_coordinates
);

void GetDofsAndCoordinatesForNodes(
ResultNodesContainerType nodes_array,
const Variable<double>& rVariable,
DofPointerVectorType& rCloud_of_dofs,
Matrix& rCloud_of_nodes_coordinates
);

void GetDofsAndCoordinatesForNodes(
ResultNodesContainerType nodes_array,
const Variable<array_1d<double, 3>>& rVariable,
DofPointerVectorType& rCloud_of_dofs_x,
DofPointerVectorType& rCloud_of_dofs_y,
DofPointerVectorType& rCloud_of_dofs_z,
Matrix& rCloud_of_nodes_coordinates
);


void AssignRotationToNodes(
NodesContainerType pNodes,
Matrix RotationMatrix
);



void AssignMPCsToNodes(
NodesContainerType pNodes,
double const Radius,
ModelPart& rComputingModelPart,
const Variable<double>& rVariable,
double const MinNumOfNeighNodes
);

void AssignMPCsToNodes(
NodesContainerType pNodes,
double const Radius,
ModelPart& rComputingModelPart,
const Variable<array_1d<double, 3>>& rVariable,
double const MinNumOfNeighNodes
);


virtual std::string Info() const
{
std::stringstream buffer;
buffer << "AssignMPCsToNeighboursUtility" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const  {rOStream << "AssignMPCsToNeighboursUtility";}

virtual void PrintData(std::ostream& rOStream) const  {}


private:
Kratos::unique_ptr<NodeBinsType> mpBins;
int mMaxNumberOfNodes;

AssignMPCsToNeighboursUtility& operator=(AssignMPCsToNeighboursUtility const& rOther)
{
return *this;
}

}; 
}  