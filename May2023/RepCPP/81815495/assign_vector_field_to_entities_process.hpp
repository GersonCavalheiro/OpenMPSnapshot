
#if !defined(KRATOS_ASSIGN_VECTOR_FIELD_TO_ENTITIES_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_VECTOR_FIELD_TO_ENTITIES_PROCESS_H_INCLUDED





#include "custom_processes/assign_scalar_field_to_entities_process.hpp"

namespace Kratos
{



class AssignVectorFieldToEntitiesProcess : public AssignScalarFieldToEntitiesProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignVectorFieldToEntitiesProcess);

typedef AssignScalarFieldToEntitiesProcess   BaseType;

AssignVectorFieldToEntitiesProcess(ModelPart& rModelPart,
pybind11::object& rPyObject,
const std::string& rPyMethodName,
const bool SpatialFieldFunction,
Parameters rParameters
) : BaseType(rModelPart, rPyObject, rPyMethodName, SpatialFieldFunction)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"MODEL_PART_NAME",
"variable_name": "VARIABLE_NAME",
"entity_type": "NODES",
"value" : [0.0, 0.0, 0.0],
"local_axes" : {},
"compound_assignment": "direct"
}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

this->mvariable_name = rParameters["variable_name"].GetString();


this->mHasLocalOrigin = false;
if( rParameters["local_axes"].Has("origin") ){
this->mHasLocalOrigin = true;
this->mLocalOrigin.resize(3,false);
for( unsigned int i=0; i<3; i++)
this->mLocalOrigin[i] = rParameters["local_axes"]["origin"][i].GetDouble();
}

this->mHasLocalAxes = false;
if( rParameters["local_axes"].Has("axes") ){
this->mHasLocalAxes = true;
this->mTransformationMatrix.resize(3,3,false);
for( unsigned int i=0; i<3; i++)
for( unsigned int j=0; j<3; j++)
this->mTransformationMatrix(i,j) = rParameters["local_axes"]["axes"][i][j].GetDouble();
}

if( rParameters["entity_type"].GetString() == "NODES" ){
this->mEntity = EntityType::NODES;
}
else if(  rParameters["entity_type"].GetString() == "CONDITIONS" ){
this->mEntity = EntityType::CONDITIONS;
}
else{
KRATOS_ERROR <<" Entity type "<< rParameters["entity_type"].GetString() <<" is not supported "<<std::endl;
}

if( this->mEntity == EntityType::CONDITIONS ){

if(KratosComponents< Variable<Vector> >::Has(this->mvariable_name) == false) 
{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is " << mvariable_name << std::endl;
}
else if( KratosComponents< Variable<array_1d<double,3> > >::Has( this->mvariable_name ) ) 
{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is " << mvariable_name << std::endl;
}

}
else{
KRATOS_ERROR << " Assignment to " << rParameters["entity_type"].GetString() << " not implemented "<< std::endl;
}

mvector_value[0] = rParameters["value"][0].GetDouble();
mvector_value[1] = rParameters["value"][1].GetDouble();
mvector_value[2] = rParameters["value"][2].GetDouble();


this->SetAssignmentType(rParameters["compound_assignment"].GetString(), this->mAssignment);

KRATOS_CATCH("");
}

~AssignVectorFieldToEntitiesProcess() override {}



void operator()()
{
Execute();
}




void Execute() override
{

KRATOS_TRY

ProcessInfo& rCurrentProcessInfo = mrModelPart.GetProcessInfo();

const double& rCurrentTime = rCurrentProcessInfo[TIME];

if( KratosComponents< Variable<Vector> >::Has( this->mvariable_name ) ) 
{

Vector Value;
AssignValueToConditions<>(KratosComponents< Variable<Vector> >::Get(this->mvariable_name), Value, rCurrentTime);

}
else if( KratosComponents< Variable<array_1d<double,3> > >::Has( this->mvariable_name ) ) 
{

array_1d<double,3> Value;
AssignValueToConditions<>(KratosComponents< Variable<array_1d<double,3> > >::Get(this->mvariable_name), Value, rCurrentTime);

}
else
{
KRATOS_ERROR << "Not able to set the variable. Attempting to set variable:" << this->mvariable_name << std::endl;
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

mAssignment = AssignmentType::DIRECT;
if( KratosComponents< Variable<Vector> >::Has( this->mvariable_name ) ) 
{

Vector Value;
noalias(Value) = ZeroVector(3);
BaseType::AssignValueToConditions<>(KratosComponents< Variable<Vector> >::Get(this->mvariable_name), Value);

}
else if( KratosComponents< Variable<array_1d<double,3> > >::Has( this->mvariable_name ) ) 
{

array_1d<double,3> Value;
Value.clear();
BaseType::AssignValueToConditions<>(KratosComponents< Variable<array_1d<double,3> > >::Get(this->mvariable_name), Value);

}
else
{
KRATOS_ERROR << "Not able to set the variable. Attempting to set variable:" << this->mvariable_name << std::endl;
}

}

KRATOS_CATCH("");
}







std::string Info() const override
{
return "AssignVectorFieldToEntitiesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignVectorFieldToEntitiesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


AssignVectorFieldToEntitiesProcess(AssignVectorFieldToEntitiesProcess const& rOther);


private:


array_1d<double,3> mvector_value;


template<class TDataType>
void CallFunction(const Condition::Pointer& pCondition, const double& time, TDataType& rValue)
{

Condition::GeometryType& rConditionGeometry = pCondition->GetGeometry();
unsigned int size = rConditionGeometry.size();
double value = 0;
unsigned int counter = 0;

rValue.resize(size*3,false);

if( mIsSpatialField ){

double x = 0.0, y = 0.0, z = 0.0;

for(unsigned int i=0; i<size; i++)
{
this->LocalAxesTransform(rConditionGeometry[i].X(), rConditionGeometry[i].Y(), rConditionGeometry[i].Z(), x, y, z);

value = mPyObject.attr(this->mPyMethodName.c_str())(x,y,z,time).cast<double>();

for(unsigned int j=0; j<3; j++)
{
rValue[counter] = value * mvector_value[j];
counter++;
}
}

}
else{

value = mPyObject.attr(this->mPyMethodName.c_str())(0.0,0.0,0.0,time).cast<double>();
for(unsigned int i=0; i<size; i++)
{
for(unsigned int j=0; j<3; j++)
{
rValue[counter] = value * mvector_value[j];
counter++;
}
}


}

}


template< class TVarType, class TDataType >
void AssignValueToConditions(TVarType& rVariable, TDataType& Value, const double& rTime )
{

typedef void (BaseType::*AssignmentMethodPointer) (ModelPart::ConditionType&, const TVarType&, const TDataType&);

AssignmentMethodPointer AssignmentMethod = this->GetAssignmentMethod<AssignmentMethodPointer>();

const int nconditions = mrModelPart.GetMesh().Conditions().size();

if(nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin = mrModelPart.GetMesh().ConditionsBegin();

for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

this->CallFunction<TDataType>(*(it.base()), rTime, Value);

(this->*AssignmentMethod)(*it, rVariable, Value);
}
}

}






AssignVectorFieldToEntitiesProcess& operator=(AssignVectorFieldToEntitiesProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
AssignVectorFieldToEntitiesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignVectorFieldToEntitiesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
