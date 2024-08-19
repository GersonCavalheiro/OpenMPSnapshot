
#if !defined(KRATOS_ASSIGN_SCALAR_FIELD_TO_ENTITIES_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_SCALAR_FIELD_TO_ENTITIES_PROCESS_H_INCLUDED





#include "custom_processes/assign_scalar_variable_to_entities_process.hpp"

namespace Kratos
{



class AssignScalarFieldToEntitiesProcess : public AssignScalarVariableToEntitiesProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignScalarFieldToEntitiesProcess);

typedef AssignScalarVariableToEntitiesProcess  BaseType;

AssignScalarFieldToEntitiesProcess(ModelPart& rModelPart,
pybind11::object& pPyObject,
const std::string& pPyMethodName,
const bool SpatialFieldFunction,
Parameters rParameters
) : BaseType(rModelPart), mPyObject(pPyObject), mPyMethodName(pPyMethodName), mIsSpatialField(SpatialFieldFunction)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"MODEL_PART_NAME",
"variable_name": "VARIABLE_NAME",
"entity_type": "NODES",
"local_axes" : {},
"compound_assignment": "direct"
}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

this->mvariable_name = rParameters["variable_name"].GetString();


mHasLocalOrigin = false;
if( rParameters["local_axes"].Has("origin") ){
mHasLocalOrigin = true;
mLocalOrigin.resize(3,false);
noalias(mLocalOrigin) = ZeroVector(3);
for( unsigned int i=0; i<3; i++)
mLocalOrigin[i] = rParameters["local_axes"]["origin"][i].GetDouble();
}

mHasLocalAxes = false;
if( rParameters["local_axes"].Has("axes") ){
mHasLocalAxes = true;
mTransformationMatrix.resize(3,3,false);
noalias(mTransformationMatrix) = ZeroMatrix(3,3);
for( unsigned int i=0; i<3; i++)
for( unsigned int j=0; j<3; j++)
mTransformationMatrix(i,j) = rParameters["local_axes"]["axes"][i][j].GetDouble();
}

if( rParameters["entity_type"].GetString() == "NODES" ){
this->mEntity = EntityType::NODES;
}
else if(  rParameters["entity_type"].GetString() == "CONDITIONS" ){
this->mEntity = EntityType::CONDITIONS;
}
else if(  rParameters["entity_type"].GetString() == "ELEMENTS" ){
this->mEntity = EntityType::ELEMENTS;
}
else{
KRATOS_ERROR <<" Entity type "<< rParameters["entity_type"].GetString() <<" is not supported "<<std::endl;
}


if( this->mEntity == EntityType::NODES ){

if( KratosComponents< Variable<double> >::Has( this->mvariable_name ) ) 
{
if( rModelPart.GetNodalSolutionStepVariablesList().Has( KratosComponents< Variable<double> >::Get( this->mvariable_name ) ) == false )
{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is " << this->mvariable_name << std::endl;
}

}
else
{
KRATOS_ERROR << "Not able to set the variable type/name. Attempting to set variable:" << this->mvariable_name << std::endl;
}
}
else if( this->mEntity == EntityType::CONDITIONS || this->mEntity == EntityType::ELEMENTS ){

if( KratosComponents< Variable<Vector> >::Has( this->mvariable_name ) == false ) 
{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is " << this->mvariable_name << std::endl;
}

}
else{
KRATOS_ERROR << " Assignment to " << rParameters["entity_type"].GetString() << " not implemented "<< std::endl;
}

this->SetAssignmentType(rParameters["compound_assignment"].GetString(), mAssignment);

KRATOS_CATCH("")
}

AssignScalarFieldToEntitiesProcess(ModelPart& rModelPart,
pybind11::object& pPyObject,
const std::string& pPyMethodName,
const bool SpatialFieldFunction
) : BaseType(rModelPart), mPyObject(pPyObject), mPyMethodName(pPyMethodName), mIsSpatialField(SpatialFieldFunction)
{
KRATOS_TRY
KRATOS_CATCH("")
}

~AssignScalarFieldToEntitiesProcess() override {}



void operator()()
{
Execute();
}




void Execute()  override
{

KRATOS_TRY

ProcessInfo& rCurrentProcessInfo = this->mrModelPart.GetProcessInfo();

const double& rCurrentTime = rCurrentProcessInfo[TIME];

if( this->mEntity == EntityType::NODES || this->mEntity == EntityType::CONDITIONS ){

if( KratosComponents< Variable<double> >::Has( this->mvariable_name ) ) 
{
AssignValueToNodes<>(KratosComponents< Variable<double> >::Get(this->mvariable_name), rCurrentTime);
}
else if( KratosComponents< Variable<Vector> >::Has( this->mvariable_name ) ) 
{
AssignValueToConditions<>(KratosComponents< Variable<Vector> >::Get(this->mvariable_name), rCurrentTime);
}
else
{
KRATOS_ERROR << "Not able to set the variable. Attempting to set variable:" << this->mvariable_name << std::endl;
}

}
else if( this->mEntity == EntityType::ELEMENTS ){

if( KratosComponents< Variable<Vector> >::Has( this->mvariable_name ) ) 
{
AssignValueToElements<>(KratosComponents< Variable<Vector> >::Get(this->mvariable_name), rCurrentTime);
}
else
{
KRATOS_ERROR << "Not able to set the variable. Attempting to set variable:" << this->mvariable_name << std::endl;
}

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

KRATOS_TRY

if( this->mEntity == EntityType::CONDITIONS ){

if( KratosComponents< Variable<Vector> >::Has( this->mvariable_name ) ) 
{
mAssignment = AssignmentType::DIRECT;
Vector Value(3);
noalias(Value) = ZeroVector(3);
AssignValueToConditions<>(KratosComponents< Variable<Vector> >::Get(this->mvariable_name), Value);
}
else
{
KRATOS_ERROR << "Not able to set the variable. Attempting to set variable:" << this->mvariable_name << std::endl;
}
}

KRATOS_CATCH("")

}







std::string Info() const override
{
return "AssignScalarFieldToEntitiesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignScalarFieldToEntitiesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


pybind11::object mPyObject;
std::string mPyMethodName;

Vector mLocalOrigin;
Matrix mTransformationMatrix;

bool mIsSpatialField;

bool mHasLocalOrigin;
bool mHasLocalAxes;


AssignScalarFieldToEntitiesProcess(AssignScalarFieldToEntitiesProcess const& rOther);


void LocalAxesTransform(const double& rX_global, const double& rY_global, const double& rZ_global,
double& rx_local, double& ry_local, double& rz_local)
{
rx_local = rX_global;
ry_local = rY_global;
rz_local = rZ_global;

if( mHasLocalOrigin  || mHasLocalAxes ){

Vector GlobalPosition(3);
GlobalPosition[0] = rX_global;
GlobalPosition[1] = rY_global;
GlobalPosition[2] = rZ_global;

if( mHasLocalOrigin )
GlobalPosition -= mLocalOrigin;

Vector LocalPosition(3);
noalias(LocalPosition) = ZeroVector(3);

if( mHasLocalAxes )
noalias(LocalPosition) = prod(mTransformationMatrix,GlobalPosition);

rx_local = LocalPosition[0];
ry_local = LocalPosition[1];
rz_local = LocalPosition[2];

}
}

void CallFunction(const Node::Pointer& pNode, const double& time, double& rValue)
{

if( mIsSpatialField ){

double x = 0.0, y = 0.0, z = 0.0;

LocalAxesTransform(pNode->X(), pNode->Y(), pNode->Z(), x, y, z);
rValue = mPyObject.attr(mPyMethodName.c_str())(x,y,z,time).cast<double>();

}
else{

rValue = mPyObject.attr(mPyMethodName.c_str())(0.0,0.0,0.0,time).cast<double>();
}

}


void CallFunction(const Condition::Pointer& pCondition, const double& time, Vector& rValue)
{

Condition::GeometryType& rConditionGeometry = pCondition->GetGeometry();
unsigned int size = rConditionGeometry.size();

rValue.resize(size,false);

if( mIsSpatialField ){

double x = 0, y = 0, z = 0;

for(unsigned int i=0; i<size; i++)
{
LocalAxesTransform(rConditionGeometry[i].X(), rConditionGeometry[i].Y(), rConditionGeometry[i].Z(), x, y, z);
rValue[i] = mPyObject.attr(mPyMethodName.c_str())(x,y,z,time).cast<double>();
}

}
else{

double value = mPyObject.attr(mPyMethodName.c_str())(0.0,0.0,0.0,time).cast<double>();
for(unsigned int i=0; i<size; i++)
{
rValue[i] = value;
}

}

}


void CallFunction(const Element::Pointer& pElement, const double& time, Vector& rValue)
{

Element::GeometryType& rElementGeometry = pElement->GetGeometry();
unsigned int size = rElementGeometry.size();

rValue.resize(size,false);

if( mIsSpatialField ){

double x = 0, y = 0, z = 0;

for(unsigned int i=0; i<size; i++)
{
LocalAxesTransform(rElementGeometry[i].X(), rElementGeometry[i].Y(), rElementGeometry[i].Z(), x, y, z);
rValue[i] = mPyObject.attr(mPyMethodName.c_str())(x,y,z,time).cast<double>();
}

}
else{

double value = mPyObject.attr(mPyMethodName.c_str())(0.0,0.0,0.0,time).cast<double>();
for(unsigned int i=0; i<size; i++)
{
rValue[i] = value;
}

}

}

template< class TVarType >
void AssignValueToNodes(TVarType& rVariable, const double& rTime)
{
if( this->mEntity == EntityType::NODES ){

typedef void (BaseType::*AssignmentMethodPointer) (ModelPart::NodeType&, const TVarType&, const double&);

AssignmentMethodPointer AssignmentMethod = this->GetAssignmentMethod<AssignmentMethodPointer>();

const int nnodes = this->mrModelPart.GetMesh().Nodes().size();

double Value = 0;

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = this->mrModelPart.GetMesh().NodesBegin();

for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

this->CallFunction(*(it.base()), rTime, Value);

(this->*AssignmentMethod)(*it, rVariable, Value);
}
}

}
}

template< class TVarType >
void AssignValueToConditions(TVarType& rVariable, const double& rTime)
{
if( this->mEntity == EntityType::CONDITIONS ){

typedef void (BaseType::*AssignmentMethodPointer) (ModelPart::ConditionType&, const Variable<Vector>&, const Vector&);

AssignmentMethodPointer AssignmentMethod = this->GetAssignmentMethod<AssignmentMethodPointer>();

const int nconditions = this->mrModelPart.GetMesh().Conditions().size();

Vector Value;

if(nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin = this->mrModelPart.GetMesh().ConditionsBegin();

for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

this->CallFunction(*(it.base()), rTime, Value);

(this->*AssignmentMethod)(*it, rVariable, Value);
}
}

}
}

template< class TVarType, class TDataType >
void AssignValueToConditions(TVarType& rVariable, const TDataType Value)
{

if( this->mEntity == EntityType::CONDITIONS ){

typedef void (BaseType::*AssignmentMethodPointer) (ModelPart::ConditionType&, const TVarType&, const TDataType&);
AssignmentMethodPointer AssignmentMethod = this->GetAssignmentMethod<AssignmentMethodPointer>();

const int nconditions = this->mrModelPart.GetMesh().Conditions().size();

if(nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin = this->mrModelPart.GetMesh().ConditionsBegin();

#pragma omp parallel for
for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

(this->*AssignmentMethod)(*it, rVariable, Value);
}
}

}

}


template< class TVarType >
void AssignValueToElements(TVarType& rVariable, const double& rTime)
{
if( this->mEntity == EntityType::ELEMENTS ){

typedef void (BaseType::*AssignmentMethodPointer) (ModelPart::ElementType&, const Variable<Vector>&, const Vector&);

AssignmentMethodPointer AssignmentMethod = this->GetAssignmentMethod<AssignmentMethodPointer>();

const int nelements = this->mrModelPart.GetMesh().Elements().size();

Vector Value;

if(nelements != 0)
{
ModelPart::ElementsContainerType::iterator it_begin = this->mrModelPart.GetMesh().ElementsBegin();

for(int i = 0; i<nelements; i++)
{
ModelPart::ElementsContainerType::iterator it = it_begin + i;

this->CallFunction(*(it.base()), rTime, Value);

(this->*AssignmentMethod)(*it, rVariable, Value);
}
}

}
}










private:


AssignScalarFieldToEntitiesProcess& operator=(AssignScalarFieldToEntitiesProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
AssignScalarFieldToEntitiesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignScalarFieldToEntitiesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
