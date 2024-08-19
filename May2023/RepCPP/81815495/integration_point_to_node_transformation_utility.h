
#if !defined(KRATOS_INTEGRATION_POINT_TO_NODE_TRANSFORMATION_UTILITY_H_INCLUDED)
#define  KRATOS_INTEGRATION_POINT_TO_NODE_TRANSFORMATION_UTILITY_H_INCLUDED

#include <string>
#include <iostream>


#include "includes/define.h"
#include "includes/element.h"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"

#include "fluid_dynamics_application_variables.h"

namespace Kratos
{





template<unsigned int TDim, unsigned int TNumNodes = TDim + 1>
class IntegrationPointToNodeTransformationUtility {

public:


KRATOS_CLASS_POINTER_DEFINITION(IntegrationPointToNodeTransformationUtility);

template<class TVariableType>
void TransformFromIntegrationPointsToNodes(const Variable<TVariableType>& rVariable, 
ModelPart& rModelPart) const
{
#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
itNode->FastGetSolutionStepValue(rVariable) = rVariable.Zero();
itNode->FastGetSolutionStepValue(NODAL_AREA) = 0.0;
}
}

#pragma omp parallel
{
ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Elements(),ElemBegin,ElemEnd);
std::vector<TVariableType> ValuesOnIntPoint;

for (ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
const auto& r_process_info = rModelPart.GetProcessInfo();
itElem->CalculateOnIntegrationPoints(rVariable,ValuesOnIntPoint,r_process_info);
Element::GeometryType& rGeom = itElem->GetGeometry();
const double Weight = rGeom.Volume() / (double) TNumNodes;
for (unsigned int iNode = 0; iNode < rGeom.size(); iNode++)
{
rGeom[iNode].SetLock();
rGeom[iNode].FastGetSolutionStepValue(rVariable) += Weight * ValuesOnIntPoint[0];
rGeom[iNode].FastGetSolutionStepValue(NODAL_AREA) += Weight;
rGeom[iNode].UnSetLock();
}
}
}

rModelPart.GetCommunicator().AssembleCurrentData(rVariable);
rModelPart.GetCommunicator().AssembleCurrentData(NODAL_AREA);

#pragma omp parallel
{
ModelPart::NodeIterator NodesBegin;
ModelPart::NodeIterator NodesEnd;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),NodesBegin,NodesEnd);

for (ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
const double NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
itNode->FastGetSolutionStepValue(rVariable) /= NodalArea;
}
}
}

}; 



} 

#endif  
