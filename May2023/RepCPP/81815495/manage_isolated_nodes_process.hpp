
#if !defined(KRATOS_MANAGE_ISOLATED_NODES_PROCESS_H_INCLUDED)
#define  KRATOS_MANAGE_ISOLATED_NODES_PROCESS_H_INCLUDED




#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"
#include "custom_bounding/spatial_bounding_box.hpp"

namespace Kratos
{


class ManageIsolatedNodesProcess : public Process
{
public:
typedef GlobalPointersVector<Node > NodeWeakPtrVectorType;

KRATOS_CLASS_POINTER_DEFINITION(ManageIsolatedNodesProcess);


ManageIsolatedNodesProcess(ModelPart& rModelPart) : Process(Flags()) , mrModelPart(rModelPart)
{
}

virtual ~ManageIsolatedNodesProcess() {}



void operator()()
{
Execute();
}



void Execute() override
{
}

void ExecuteInitialize() override
{
}

void ExecuteBeforeSolutionLoop() override
{
KRATOS_TRY

double Radius = 0.0;
mBoundingBox = SpatialBoundingBox(mrModelPart,Radius,0.1);

KRATOS_CATCH("")
}


void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY

const int nnodes = mrModelPart.Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin =
mrModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; ++i)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if( it->Is(ISOLATED) || (it->Is(RIGID) && (it->IsNot(SOLID) && it->IsNot(FLUID))) ){

if(it->SolutionStepsDataHas(PRESSURE)){
it->FastGetSolutionStepValue(PRESSURE,0) = 0.0;
it->FastGetSolutionStepValue(PRESSURE,1) = 0.0;
}
if(it->SolutionStepsDataHas(PRESSURE_VELOCITY)){
it->FastGetSolutionStepValue(PRESSURE_VELOCITY,0) = 0.0;
it->FastGetSolutionStepValue(PRESSURE_VELOCITY,1) = 0.0;
}
if(it->SolutionStepsDataHas(PRESSURE_ACCELERATION)){
it->FastGetSolutionStepValue(PRESSURE_ACCELERATION,0) = 0.0;
it->FastGetSolutionStepValue(PRESSURE_ACCELERATION,1) = 0.0;
}

}
}
}

KRATOS_CATCH("")
}

void ExecuteFinalizeSolutionStep() override
{
KRATOS_TRY

const int nnodes = mrModelPart.Nodes().size();
const double TimeStep = mrModelPart.GetProcessInfo()[DELTA_TIME];
if (nnodes != 0)
{
this->SetSemiIsolatedNodes(mrModelPart);

ModelPart::NodesContainerType::iterator it_begin = mrModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; ++i)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if( it->Is(ISOLATED) ){
if(it->SolutionStepsDataHas(VOLUME_ACCELERATION)){
const array_1d<double,3>& VolumeAcceleration = it->FastGetSolutionStepValue(VOLUME_ACCELERATION);
noalias(it->FastGetSolutionStepValue(ACCELERATION)) = VolumeAcceleration;
}

noalias(it->FastGetSolutionStepValue(VELOCITY)) = it->FastGetSolutionStepValue(VELOCITY,1) + it->FastGetSolutionStepValue(ACCELERATION)*TimeStep;
noalias(it->Coordinates()) -= it->FastGetSolutionStepValue(DISPLACEMENT);
noalias(it->FastGetSolutionStepValue(DISPLACEMENT)) = it->FastGetSolutionStepValue(DISPLACEMENT,1) + it->FastGetSolutionStepValue(VELOCITY)*TimeStep + 0.5*it->FastGetSolutionStepValue(ACCELERATION)*TimeStep*TimeStep;
noalias(it->Coordinates()) += it->FastGetSolutionStepValue(DISPLACEMENT);



if( !mBoundingBox.IsInside( it->Coordinates() ) ){
it->Set(TO_ERASE);
std::cout<<" ISOLATED to erase "<<std::endl;
}
}
else if( it->Is(VISITED) ){

if(it->SolutionStepsDataHas(VOLUME_ACCELERATION)){
const array_1d<double,3>& VolumeAcceleration = it->FastGetSolutionStepValue(VOLUME_ACCELERATION);
noalias(it->FastGetSolutionStepValue(ACCELERATION)) = VolumeAcceleration;
}

noalias(it->FastGetSolutionStepValue(VELOCITY)) = 0.5 * (it->FastGetSolutionStepValue(VELOCITY) + it->FastGetSolutionStepValue(VELOCITY,1) + it->FastGetSolutionStepValue(ACCELERATION)*TimeStep);
noalias(it->Coordinates()) -= it->FastGetSolutionStepValue(DISPLACEMENT);
noalias(it->FastGetSolutionStepValue(DISPLACEMENT)) = 0.5 * (it->FastGetSolutionStepValue(DISPLACEMENT) + it->FastGetSolutionStepValue(DISPLACEMENT,1) + it->FastGetSolutionStepValue(VELOCITY)*TimeStep + 0.5*it->FastGetSolutionStepValue(ACCELERATION)*TimeStep*TimeStep);
noalias(it->Coordinates()) += it->FastGetSolutionStepValue(DISPLACEMENT);

}
}

this->ResetSemiIsolatedNodes(mrModelPart);

}

KRATOS_CATCH("")
}


void ExecuteBeforeOutputStep() override
{
}


void ExecuteAfterOutputStep() override
{
}


void ExecuteFinalize() override
{
}







std::string Info() const override
{
return "ManageIsolatedNodesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ManageIsolatedNodesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


ManageIsolatedNodesProcess(ManageIsolatedNodesProcess const& rOther);


private:


ModelPart& mrModelPart;

SpatialBoundingBox mBoundingBox;


void SetSemiIsolatedNodes(ModelPart& rModelPart)
{

const int nnodes = mrModelPart.Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; ++i)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if( it->Is(FREE_SURFACE) ){

NodeWeakPtrVectorType& nNodes = it->GetValue(NEIGHBOUR_NODES);
unsigned int rigid = 0;
for(auto& i_nnodes : nNodes)
{
if(i_nnodes.Is(RIGID))
++rigid;
}
if( rigid == nNodes.size() )
it->Set(VISITED,true);
}
}
}


}

void ResetSemiIsolatedNodes(ModelPart& rModelPart)
{

const int nnodes = mrModelPart.Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; ++i)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if(it->Is(VISITED))
it->Set(VISITED,false);
}
}

}



ManageIsolatedNodesProcess& operator=(ManageIsolatedNodesProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
ManageIsolatedNodesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ManageIsolatedNodesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
