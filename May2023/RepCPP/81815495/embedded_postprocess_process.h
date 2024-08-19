
#ifndef KRATOS_EMBEDDED_POSTPROCESS_PROCESS_H
#define KRATOS_EMBEDDED_POSTPROCESS_PROCESS_H

#include <string>
#include <iostream>


#include "processes/process.h"
#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/cfd_variables.h"
#include "utilities/openmp_utils.h"



namespace Kratos
{






class EmbeddedPostprocessProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(EmbeddedPostprocessProcess);


EmbeddedPostprocessProcess(ModelPart& rModelPart) : mrModelPart(rModelPart)
{

}

~EmbeddedPostprocessProcess() override{}




void ExecuteFinalizeSolutionStep() override
{
const array_1d<double, 3> aux_zero = ZeroVector(3);
ModelPart::NodesContainerType& rNodes = mrModelPart.Nodes();

if( mrModelPart.NodesBegin()->SolutionStepsDataHas( DISTANCE ) == false )
KRATOS_ERROR << "Nodes do not have DISTANCE variable!";
if( mrModelPart.NodesBegin()->SolutionStepsDataHas( PRESSURE ) == false )
KRATOS_ERROR << "Nodes do not have PRESSURE variable!";
if( mrModelPart.NodesBegin()->SolutionStepsDataHas( VELOCITY ) == false )
KRATOS_ERROR << "Nodes do not have VELOCITY variable!";
if( mrModelPart.NodesBegin()->SolutionStepsDataHas( EMBEDDED_WET_PRESSURE ) == false )
KRATOS_ERROR << "Nodes do not have EMBEDDED_WET_PRESSURE variable!";
if( mrModelPart.NodesBegin()->SolutionStepsDataHas( EMBEDDED_WET_VELOCITY ) == false )
KRATOS_ERROR << "Nodes do not have EMBEDDED_WET_VELOCITY variable!";

#pragma omp parallel for
for (int k = 0; k < static_cast<int>(rNodes.size()); ++k)
{
ModelPart::NodesContainerType::iterator itNode = rNodes.begin() + k;
const double dist = itNode->FastGetSolutionStepValue(DISTANCE);
double& emb_wet_pres = itNode->FastGetSolutionStepValue(EMBEDDED_WET_PRESSURE);
array_1d<double, 3>& emb_wet_vel = itNode->FastGetSolutionStepValue(EMBEDDED_WET_VELOCITY);

if (dist >= 0.0)
{
emb_wet_pres = itNode->FastGetSolutionStepValue(PRESSURE);      
emb_wet_vel = itNode->FastGetSolutionStepValue(VELOCITY);       
}
else
{
emb_wet_pres = 0.0;         
emb_wet_vel = aux_zero;     
}
}
}



std::string Info() const override
{
std::stringstream buffer;
buffer << "EmbeddedPostprocessProcess" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override {rOStream << "EmbeddedPostprocessProcess";}

void PrintData(std::ostream& rOStream) const override {}





protected:


ModelPart&                                  mrModelPart;












private:













EmbeddedPostprocessProcess() = delete;

EmbeddedPostprocessProcess& operator=(EmbeddedPostprocessProcess const& rOther) = delete;

EmbeddedPostprocessProcess(EmbeddedPostprocessProcess const& rOther) = delete;



}; 





};  

#endif 
