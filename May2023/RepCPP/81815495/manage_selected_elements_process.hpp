
#if !defined(KRATOS_MANAGE_SELECTED_ELEMENTS_PROCESS_H_INCLUDED)
#define  KRATOS_MANAGE_SELECTED_ELEMENTS_PROCESS_H_INCLUDED




#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"
#include "custom_bounding/spatial_bounding_box.hpp"

namespace Kratos
{



class ManageSelectedElementsProcess : public Process
{
public:

typedef ModelPart::ElementType::GeometryType       GeometryType;

KRATOS_CLASS_POINTER_DEFINITION(ManageSelectedElementsProcess);


ManageSelectedElementsProcess(ModelPart& rModelPart) : Process(Flags()) , mrModelPart(rModelPart)
{
}

virtual ~ManageSelectedElementsProcess() {}



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

const int nelements = mrModelPart.NumberOfElements();

if (nelements != 0)
{
ModelPart::ElementsContainerType::iterator it_begin = mrModelPart.ElementsBegin();

#pragma omp parallel for
for (int i = 0; i < nelements; ++i)
{
ModelPart::ElementsContainerType::iterator it = it_begin + i;

if( it->Is(FLUID) ){

GeometryType& rGeometry = it->GetGeometry();
const unsigned int number_of_nodes = rGeometry.size();
unsigned int selected_nodes = 0;
for(unsigned int j=0; j<number_of_nodes; ++j)
{
if(rGeometry[j].Is(SELECTED))
++selected_nodes;
}

if(selected_nodes == number_of_nodes){
it->Set(SELECTED,true);
it->Set(ACTIVE,false);
}
}
}
}

KRATOS_CATCH("")
}

void ExecuteFinalizeSolutionStep() override
{
KRATOS_TRY

const int nelements = mrModelPart.NumberOfElements();

if (nelements != 0)
{
ModelPart::ElementsContainerType::iterator it_begin = mrModelPart.ElementsBegin();

for (int i = 0; i < nelements; ++i)
{
ModelPart::ElementsContainerType::iterator it = it_begin + i;

if( it->Is(FLUID) && it->Is(SELECTED) ){

GeometryType& rGeometry = it->GetGeometry();
const unsigned int number_of_nodes = rGeometry.size();

for(unsigned int j=0; j<number_of_nodes; ++j)
{
rGeometry[j].Set(SELECTED,true);
}
it->Set(SELECTED,false);
it->Set(ACTIVE,true);
}
}
}

const int nnodes = mrModelPart.NumberOfNodes();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; ++i)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if( it->Is(SELECTED) ){

if( norm_2(it->FastGetSolutionStepValue(VELOCITY)) > 1.5 * norm_2(it->FastGetSolutionStepValue(VELOCITY,1)) ||
norm_2(it->FastGetSolutionStepValue(ACCELERATION)) > 1.5 * norm_2(it->FastGetSolutionStepValue(ACCELERATION,1)) ){


noalias(it->FastGetSolutionStepValue(VELOCITY)) = it->FastGetSolutionStepValue(VELOCITY,1);
noalias(it->FastGetSolutionStepValue(ACCELERATION)) = it->FastGetSolutionStepValue(ACCELERATION,1);
noalias(it->Coordinates()) -= it->FastGetSolutionStepValue(DISPLACEMENT);
noalias(it->FastGetSolutionStepValue(DISPLACEMENT)) = it->FastGetSolutionStepValue(DISPLACEMENT,1);
noalias(it->Coordinates()) += it->FastGetSolutionStepValue(DISPLACEMENT);



}

if( it->Is(FREE_SURFACE) ){
if( !mBoundingBox.IsInside( it->Coordinates() ) ){
it->Set(TO_ERASE,true);
std::cout<<" SELECTED to erase "<<std::endl;
}
}

it->Set(SELECTED,false);
}


}
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
return "ManageSelectedElementsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ManageSelectedElementsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


ManageSelectedElementsProcess(ManageSelectedElementsProcess const& rOther);


private:


ModelPart& mrModelPart;

SpatialBoundingBox mBoundingBox;


ManageSelectedElementsProcess& operator=(ManageSelectedElementsProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
ManageSelectedElementsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ManageSelectedElementsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
