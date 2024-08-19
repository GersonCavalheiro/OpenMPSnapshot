
#ifndef KRATOS_EMBEDDED_NODES_INITIALIZATION_PROCESS_H
#define KRATOS_EMBEDDED_NODES_INITIALIZATION_PROCESS_H

#include <string>
#include <iostream>


#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/cfd_variables.h"



namespace Kratos
{






class EmbeddedNodesInitializationProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(EmbeddedNodesInitializationProcess);

typedef Node                     NodeType;
typedef NodeType::Pointer    NodePointerType;
typedef Geometry<NodeType>      GeometryType;


EmbeddedNodesInitializationProcess(ModelPart& rModelPart, unsigned int MaxIterations = 10) : mrModelPart(rModelPart)
{
mMaxIterations = MaxIterations;
}

~EmbeddedNodesInitializationProcess() override{}




void ExecuteInitializeSolutionStep() override
{
const unsigned int BufferSize = mrModelPart.GetBufferSize();
ModelPart::NodesContainerType& rNodes = mrModelPart.Nodes();
ModelPart::ElementsContainerType& rElements = mrModelPart.Elements();

if( mrModelPart.NodesBegin()->SolutionStepsDataHas( DISTANCE ) == false )
KRATOS_ERROR << "Nodes do not have DISTANCE variable!";
if( mrModelPart.NodesBegin()->SolutionStepsDataHas( PRESSURE ) == false )
KRATOS_ERROR << "Nodes do not have PRESSURE variable!";
if( mrModelPart.NodesBegin()->SolutionStepsDataHas( VELOCITY ) == false )
KRATOS_ERROR << "Nodes do not have VELOCITY variable!";

#pragma omp parallel for
for (int k = 0; k < static_cast<int>(rNodes.size()); ++k)
{
ModelPart::NodesContainerType::iterator itNode = rNodes.begin() + k;
const double& d  = itNode->FastGetSolutionStepValue(DISTANCE);      
const double& dn = itNode->FastGetSolutionStepValue(DISTANCE,1);    

if ((d>0) && (dn<=0))
itNode->Set(SELECTED, true);
}

for (unsigned int it=0; it<mMaxIterations; ++it)
{
for (int k = 0; k < static_cast<int>(rElements.size()); ++k)
{
unsigned int NewNodes = 0;
ModelPart::ElementsContainerType::iterator itElement = rElements.begin() + k;
const GeometryType& rGeometry = itElement->GetGeometry();
const unsigned int ElemNumNodes = rGeometry.PointsNumber();

for (unsigned int j=0; j<ElemNumNodes; ++j)
{
if (rGeometry[j].Is(SELECTED))
NewNodes++;
}

if (NewNodes == 1)
{
NodeType::Pointer pNode;
double p_avg = 0.0;
array_1d<double, 3> v_avg = ZeroVector(3);

for (unsigned int j=0; j<ElemNumNodes; ++j)
{
if (rGeometry[j].IsNot(SELECTED))
{
p_avg += rGeometry[j].FastGetSolutionStepValue(PRESSURE);
v_avg += rGeometry[j].FastGetSolutionStepValue(VELOCITY);
}
else
{
pNode = rGeometry(j);
}
}

p_avg /= (ElemNumNodes-1);
v_avg /= (ElemNumNodes-1);

pNode->Set(SELECTED, false);                                    
for (unsigned int step=0; step<BufferSize; ++step)              
{
pNode->FastGetSolutionStepValue(PRESSURE, step) = p_avg;
pNode->FastGetSolutionStepValue(VELOCITY, step) = v_avg;
}
}
}
}

#pragma omp parallel for
for (int k = 0; k < static_cast<int>(rNodes.size()); ++k)
{
ModelPart::NodesContainerType::iterator itNode = rNodes.begin() + k;

if (itNode->Is(SELECTED))
{
itNode->Set(SELECTED, false);                                   
for (unsigned int step=0; step<BufferSize; ++step)              
{
itNode->FastGetSolutionStepValue(PRESSURE, step) = 0.0;
itNode->FastGetSolutionStepValue(VELOCITY, step) = ZeroVector(3);
}
}
}
}



std::string Info() const override
{
std::stringstream buffer;
buffer << "EmbeddedNodesInitializationProcess" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override {rOStream << "EmbeddedNodesInitializationProcess";}

void PrintData(std::ostream& rOStream) const override {}





protected:


ModelPart&                                 mrModelPart;
unsigned int                            mMaxIterations;










private:













EmbeddedNodesInitializationProcess() = delete;

EmbeddedNodesInitializationProcess& operator=(EmbeddedNodesInitializationProcess const& rOther) = delete;

EmbeddedNodesInitializationProcess(EmbeddedNodesInitializationProcess const& rOther) = delete;



}; 





};  

#endif 
