
#if !defined(KRATOS_ASSIGN_ROTATION_FIELD_ABOUT_AN_AXIS_TO_NODES_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_ROTATION_FIELD_ABOUT_AN_AXIS_TO_NODES_PROCESS_H_INCLUDED





#include "custom_processes/assign_rotation_about_an_axis_to_nodes_process.hpp"

namespace Kratos
{



class AssignRotationFieldAboutAnAxisToNodesProcess : public AssignRotationAboutAnAxisToNodesProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignRotationFieldAboutAnAxisToNodesProcess);

AssignRotationFieldAboutAnAxisToNodesProcess(ModelPart& model_part,
pybind11::object& rPyObject,
const std::string& rPyMethodName,
const bool SpatialFieldFunction,
Parameters rParameters
) : AssignRotationAboutAnAxisToNodesProcess(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"MODEL_PART_NAME",
"variable_name": "VARIABLE_NAME",
"direction" : [],
"center" : []
}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name  = rParameters["variable_name"].GetString();

mPyObject      =  rPyObject;
mPyMethodName  =  rPyMethodName;

mIsSpatialField = SpatialFieldFunction;

for( unsigned int i=0; i<3; i++)
{
mdirection[i] = rParameters["direction"][i].GetDouble();
mcenter[i] = rParameters["center"][i].GetDouble();
}

double norm = norm_2(mdirection);
if(norm!=0)
mdirection/=norm;

KRATOS_CATCH("");
}



~AssignRotationFieldAboutAnAxisToNodesProcess() override {}



void operator()()
{
Execute();
}




void Execute()  override
{

KRATOS_TRY;

if( ! mIsSpatialField ){

const ProcessInfo& rCurrentProcessInfo = mrModelPart.GetProcessInfo();
const double& rCurrentTime  = rCurrentProcessInfo[TIME];
const ProcessInfo& rPreviousProcessInfo = rCurrentProcessInfo.GetPreviousTimeStepInfo();
const double& rPreviousTime = rPreviousProcessInfo[TIME];

this->CallTimeFunction(rPreviousTime, mprevious_value);
this->CallTimeFunction(rCurrentTime, mvalue);

AssignRotationAboutAnAxisToNodesProcess::Execute();

}
else{

AssignRotationAboutAnAxis();
}

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
return "AssignRotationFieldAboutAnAxisToNodesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignRotationFieldAboutAnAxisToNodesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


AssignRotationFieldAboutAnAxisToNodesProcess(AssignRotationFieldAboutAnAxisToNodesProcess const& rOther);


private:


pybind11::object mPyObject;
std::string mPyMethodName;

bool mIsSpatialField;


void CallFunction(const Node::Pointer& pNode, const double& time, double& rValue)
{
KRATOS_TRY

if( mIsSpatialField ){

double x = pNode->X(), y = pNode->Y(), z = pNode->Z();
rValue = mPyObject.attr(mPyMethodName.c_str())(x,y,z,time).cast<double>();

}
else{

rValue = mPyObject.attr(mPyMethodName.c_str())(0.0,0.0,0.0,time).cast<double>();

}

KRATOS_CATCH( "" )

}

void CallTimeFunction(const double& time, double& rValue)
{

KRATOS_TRY

rValue = mPyObject.attr(mPyMethodName.c_str())(0.0,0.0,0.0,time).cast<double>();

KRATOS_CATCH( "" )

}


void AssignRotationAboutAnAxis()
{
KRATOS_TRY

const int nnodes = mrModelPart.GetMesh().Nodes().size();

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh().NodesBegin();

Matrix rotation_matrix;
Quaternion<double> total_quaternion;
array_1d<double,3> radius;
array_1d<double,3> distance;
array_1d<double,3> rotation;
array_1d<double,3> delta_rotation;


double value = 0.0;
bool dynamic_angular_velocity = false;
bool dynamic_angular_acceleration = false;

const ProcessInfo& rCurrentProcessInfo = mrModelPart.GetProcessInfo();
const double& rDeltaTime = rCurrentProcessInfo[DELTA_TIME];
const double& rCurrentTime = rCurrentProcessInfo[TIME];
const ProcessInfo& rPreviousProcessInfo = rCurrentProcessInfo.GetPreviousTimeStepInfo();
const double& rPreviousTime = rPreviousProcessInfo[TIME];

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

for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

this->CallFunction(*(it.base()), rCurrentTime, value);

rotation = value * mdirection;

rotation *= time_factor;

if( dynamic_angular_velocity ){

this->CallFunction(*(it.base()), rPreviousTime, value);
delta_rotation  = rotation - time_factor * value * mdirection;

angular_velocity = delta_rotation / rDeltaTime;
if( dynamic_angular_acceleration ){
angular_acceleration = angular_velocity / rDeltaTime;
}
}

total_quaternion = Quaternion<double>::FromRotationVector<array_1d<double,3> >(rotation);

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



AssignRotationFieldAboutAnAxisToNodesProcess& operator=(AssignRotationFieldAboutAnAxisToNodesProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
AssignRotationFieldAboutAnAxisToNodesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignRotationFieldAboutAnAxisToNodesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
