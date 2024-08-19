
#pragma once

#include <string>
#include <iostream>

#include "includes/define.h"

#include "spatial_containers/bins_dynamic_objects.h"

#include "node_configure_for_node_search.h"


namespace Kratos {




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) NodeSearchUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION(NodeSearchUtility);

typedef ModelPart::NodeType                           NodeType;
typedef NodeConfigureForNodeSearch                    NodeConfigureType;
typedef ModelPart::NodesContainerType                 NodesContainerType;
typedef BinsObjectDynamic<NodeConfigureType>          NodeBinsType;

typedef NodesContainerType::ContainerType             ResultNodesContainerType;


NodeSearchUtility(NodesContainerType& rStructureNodes) {
KRATOS_TRY;
NodesContainerType::ContainerType& nodes_model_part = rStructureNodes.GetContainer();
mpBins = Kratos::make_unique<NodeBinsType>(nodes_model_part.begin(), nodes_model_part.end());
mMaxNumberOfNodes = rStructureNodes.size();
KRATOS_CATCH("");
}

~NodeSearchUtility(){
}



void SearchNodesInRadius(
NodeType::Pointer pNode,
double const Radius,
ResultNodesContainerType& rResults );


virtual std::string Info() const
{
std::stringstream buffer;
buffer << "NodeSearchUtility" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const  {rOStream << "NodeSearchUtility";}

virtual void PrintData(std::ostream& rOStream) const  {}


private:
Kratos::unique_ptr<NodeBinsType> mpBins;

int mMaxNumberOfNodes;


NodeSearchUtility& operator=(NodeSearchUtility const& rOther)
{
return *this;
}

NodeSearchUtility(NodeSearchUtility const& rOther)
{
*this = rOther;
}


}; 



}  