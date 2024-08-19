
#if !defined(KRATOS_ASSIGN_ROTATION_ABOUT_AN_AXIS_TO_NODES_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_ROTATION_ABOUT_AN_AXIS_TO_NODES_PROCESS_H_INCLUDED





#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"
#include "utilities/beam_math_utilities.hpp"

namespace Kratos
{



class AssignRotationAboutAnAxisToNodesProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignRotationAboutAnAxisToNodesProcess);



AssignRotationAboutAnAxisToNodesProcess(ModelPart& model_part) : Process(Flags()) , mrModelPart(model_part) {}



AssignRotationAboutAnAxisToNodesProcess(ModelPart& model_part,
Parameters rParameters
) : Process(Flags()) , mrModelPart(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"MODEL_PART_NAME",
"variable_name": "VARIABLE_NAME",
"modulus" : 1.0,
"direction" : [],
"center" : []
}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name = rParameters["variable_name"].GetString();


if( KratosComponents< Variable<array_1d<double, 3> > >::Has( mvariable_name ) ) 
{

mvalue = rParameters["modulus"].GetDouble();

for( unsigned int i=0; i<3; i++)
{
mdirection[i] = rParameters["direction"][i].GetDouble();
mcenter[i] = rParameters["center"][i].GetDouble();
}

double norm = norm_2(mdirection);
if(norm!=0)
mdirection/=norm;

}
else 
{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is " << mvariable_name <<std::endl;

}

KRATOS_CATCH("");
}


~AssignRotationAboutAnAxisToNodesProcess() override {}



void operator()()
{
Execute();
}




void Execute() override
{

KRATOS_TRY;

this->AssignRotationAboutAnAxis();

KRATOS_CATCH("");

}

void ExecuteInitialize() override
{
}

void ExecuteBeforeSolutionLoop() override
{
}


void ExecuteInitializeSolutionStep() override
{
}

void ExecuteFinalizeSolutionStep() override
{
mprevious_value = mvalue;
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
return "AssignRotationAboutAnAxisToNodesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignRotationAboutAnAxisToNodesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


ModelPart& mrModelPart;
std::string mvariable_name;
double mvalue;
double mprevious_value;
array_1d<double,3> mdirection;
array_1d<double,3> mcenter;


AssignRotationAboutAnAxisToNodesProcess(AssignRotationAboutAnAxisToNodesProcess const& rOther);


private:





void AssignRotationAboutAnAxis()
{
KRATOS_TRY

const int nnodes = mrModelPart.GetMesh().Nodes().size();


if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh().NodesBegin();

Matrix rotation_matrix;
array_1d<double,3> radius;
array_1d<double,3> distance;

bool dynamic_angular_velocity = false;
bool dynamic_angular_acceleration = false;

const ProcessInfo& rCurrentProcessInfo = mrModelPart.GetProcessInfo();
const double& rDeltaTime = rCurrentProcessInfo[DELTA_TIME];

array_1d<double,3> angular_velocity;
angular_velocity.clear();
array_1d<double,3> angular_acceleration;
angular_acceleration.clear();

double time_factor = 0.0;
if(mvariable_name == "ROTATION"){

time_factor = 1.0;

}
else if(mvariable_name == "ANGULAR_VELOCITY"){

dynamic_angular_velocity = true;
time_factor = rDeltaTime;

}
else if(mvariable_name == "ANGULAR_ACCELERATION"){

dynamic_angular_velocity = true;
dynamic_angular_acceleration = true;
time_factor = rDeltaTime * rDeltaTime;

}

array_1d<double,3> delta_rotation = time_factor * (mvalue - mprevious_value) * mdirection;

array_1d<double,3> rotation = time_factor * mvalue * mdirection;

if( dynamic_angular_velocity ){
angular_velocity = delta_rotation / rDeltaTime;
if( dynamic_angular_acceleration ){
angular_acceleration = angular_velocity / rDeltaTime;
}
}

Quaternion<double> total_quaternion = Quaternion<double>::FromRotationVector<array_1d<double,3> >(rotation);

#pragma omp parallel for private(distance,radius,rotation_matrix)
for(int i = 0; i<nnodes; i++)
{

ModelPart::NodesContainerType::iterator it = it_begin + i;

distance = it->GetInitialPosition() - mcenter;

total_quaternion.ToRotationMatrix(rotation_matrix);

noalias(radius) = prod(rotation_matrix, distance);

array_1d<double,3>& displacement = it->FastGetSolutionStepValue(DISPLACEMENT);
displacement =  radius - distance; 

if( dynamic_angular_velocity ){

BeamMathUtils<double>::VectorToSkewSymmetricTensor(angular_velocity, rotation_matrix);

array_1d<double,3>& velocity = it->FastGetSolutionStepValue(VELOCITY);
velocity = prod(rotation_matrix,radius);


if( dynamic_angular_acceleration ){
array_1d<double,3>& acceleration = it->FastGetSolutionStepValue(ACCELERATION);
acceleration = prod(rotation_matrix,velocity);

BeamMathUtils<double>::VectorToSkewSymmetricTensor(angular_acceleration, rotation_matrix);
acceleration += prod(rotation_matrix,radius);

}

}
}

}

KRATOS_CATCH( "" )
}



AssignRotationAboutAnAxisToNodesProcess& operator=(AssignRotationAboutAnAxisToNodesProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
AssignRotationAboutAnAxisToNodesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignRotationAboutAnAxisToNodesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
