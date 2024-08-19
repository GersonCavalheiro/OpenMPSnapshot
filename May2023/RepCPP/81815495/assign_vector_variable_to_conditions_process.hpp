
#if !defined(KRATOS_ASSIGN_VECTOR_VARIABLE_TO_CONDITIONS_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_VECTOR_VARIABLE_TO_CONDITIONS_PROCESS_H_INCLUDED





#include "custom_processes/assign_scalar_variable_to_entities_process.hpp"

namespace Kratos
{



class AssignVectorVariableToConditionsProcess : public AssignScalarVariableToEntitiesProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignVectorVariableToConditionsProcess);

typedef AssignScalarVariableToEntitiesProcess   BaseType;

AssignVectorVariableToConditionsProcess(ModelPart& rModelPart,
Parameters rParameters) : BaseType(rModelPart)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"MODEL_PART_NAME",
"variable_name": "VARIABLE_NAME",
"value" : [0.0, 0.0, 0.0],
"compound_assignment": "direct"
}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name = rParameters["variable_name"].GetString();

if(KratosComponents< Variable<array_1d<double,3> > >::Has(mvariable_name) == false)
{
KRATOS_THROW_ERROR(std::runtime_error,"trying to set a variable that is not in the model_part - variable name is ",mvariable_name);
}

mvector_value[0] = rParameters["value"][0].GetDouble();
mvector_value[1] = rParameters["value"][1].GetDouble();
mvector_value[2] = rParameters["value"][2].GetDouble();

this->SetAssignmentType(rParameters["compound_assignment"].GetString(), mAssignment);

KRATOS_CATCH("");
}


AssignVectorVariableToConditionsProcess(ModelPart& rModelPart,
const Variable<array_1d<double,3> >& rVariable,
const array_1d<double,3>& rvector_value) : BaseType(rModelPart), mvector_value(rvector_value)
{
KRATOS_TRY;

mvariable_name = rVariable.Name();

if( KratosComponents< Variable<array_1d<double,3> > >::Has( mvariable_name ) == false ) 
KRATOS_THROW_ERROR(std::runtime_error,"trying to set a variable that is not in the model_part - variable name is ",mvariable_name);

KRATOS_CATCH("")
}


~AssignVectorVariableToConditionsProcess() override {}



void operator()()
{
Execute();
}




void Execute()  override
{

KRATOS_TRY

InternalAssignValue(KratosComponents< Variable<array_1d<double,3> > >::Get(mvariable_name), mvector_value);

KRATOS_CATCH("")

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

mAssignment = AssignmentType::DIRECT;
array_1d<double,3> vector_value;
vector_value.clear();
InternalAssignValue(KratosComponents< Variable<array_1d<double,3> > >::Get(mvariable_name), vector_value);

KRATOS_CATCH("")
}







std::string Info() const override
{
return "AssignVectorVariableToConditionsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignVectorVariableToConditionsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


AssignVectorVariableToConditionsProcess(AssignVectorVariableToConditionsProcess const& rOther);


private:


array_1d<double,3> mvector_value;


void InternalAssignValue(const Variable<array_1d<double,3> >& rVariable,
const array_1d<double,3>& rvector_value)
{

typedef void (AssignVectorVariableToConditionsProcess::*AssignmentMethodPointer) (ModelPart::ConditionType&, const Variable<array_1d<double,3> >&, const array_1d<double,3>&);

AssignmentMethodPointer AssignmentMethod = this->GetAssignmentMethod<AssignmentMethodPointer>();

const int nconditions = this->mrModelPart.GetMesh().Conditions().size();

if(nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin = this->mrModelPart.GetMesh().ConditionsBegin();

#pragma omp parallel for
for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

(this->*AssignmentMethod)(*it, rVariable, rvector_value);
}
}
}


AssignVectorVariableToConditionsProcess& operator=(AssignVectorVariableToConditionsProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
AssignVectorVariableToConditionsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignVectorVariableToConditionsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
