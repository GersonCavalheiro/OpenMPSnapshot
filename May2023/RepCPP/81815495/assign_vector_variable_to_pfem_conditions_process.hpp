
#if !defined(KRATOS_ASSIGN_VECTOR_VARIABLE_TO_PFEM_CONDITIONS_PROCESS_H_INCLUDED)
#define KRATOS_ASSIGN_VECTOR_VARIABLE_TO_PFEM_CONDITIONS_PROCESS_H_INCLUDED



#include "custom_processes/assign_scalar_variable_to_pfem_entities_process.hpp"

namespace Kratos
{



class AssignVectorVariableToPfemConditionsProcess : public AssignScalarVariableToPfemEntitiesProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignVectorVariableToPfemConditionsProcess);

typedef AssignScalarVariableToPfemEntitiesProcess BaseType;

AssignVectorVariableToPfemConditionsProcess(ModelPart &rModelPart,
Parameters rParameters) : BaseType(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"MODEL_PART_NAME",
"variable_name": "VARIABLE_NAME",
"value" : [0.0, 0.0, 0.0],
"compound_assignment": "direct"
}  )");

rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name = rParameters["variable_name"].GetString();

if (KratosComponents<Variable<array_1d<double, 3>>>::Has(mvariable_name) == false)
{
KRATOS_THROW_ERROR(std::runtime_error, "trying to set a variable that is not in the model_part - variable name is ", mvariable_name);
}

mvector_value[0] = rParameters["value"][0].GetDouble();
mvector_value[1] = rParameters["value"][1].GetDouble();
mvector_value[2] = rParameters["value"][2].GetDouble();

this->SetAssignmentType(rParameters["compound_assignment"].GetString(), mAssignment);

KRATOS_CATCH("");
}

AssignVectorVariableToPfemConditionsProcess(ModelPart &rModelPart,
const Variable<array_1d<double, 3>> &rVariable,
const array_1d<double, 3> &rvector_value) : BaseType(rModelPart), mvector_value(rvector_value)
{
KRATOS_TRY;

mvariable_name = rVariable.Name();

if (KratosComponents<Variable<array_1d<double, 3>>>::Has(mvariable_name) == false) 
KRATOS_THROW_ERROR(std::runtime_error, "trying to set a variable that is not in the model_part - variable name is ", mvariable_name);

KRATOS_CATCH("")
}

~AssignVectorVariableToPfemConditionsProcess() override {}


void operator()()
{
Execute();
}


void Execute() override
{

KRATOS_TRY

InternalAssignValue(KratosComponents<Variable<array_1d<double, 3>>>::Get(mvariable_name), mvector_value);

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
array_1d<double, 3> vector_value;
vector_value.clear();
InternalAssignValue(KratosComponents<Variable<array_1d<double, 3>>>::Get(mvariable_name), vector_value);

KRATOS_CATCH("")
}




std::string Info() const override
{
return "AssignVectorVariableToPfemConditionsProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "AssignVectorVariableToPfemConditionsProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

AssignVectorVariableToPfemConditionsProcess(AssignVectorVariableToPfemConditionsProcess const &rOther);


private:

array_1d<double, 3> mvector_value;


void InternalAssignValue(const Variable<array_1d<double, 3>> &rVariable,
const array_1d<double, 3> &rvector_value)
{

typedef void (AssignVectorVariableToPfemConditionsProcess::*AssignmentMethodPointer)(ModelPart::ConditionType &, const Variable<array_1d<double, 3>> &, const array_1d<double, 3> &);

AssignmentMethodPointer AssignmentMethod = this->GetAssignmentMethod<AssignmentMethodPointer>();

const int nconditions = this->mrModelPart.GetMesh().Conditions().size();

if (nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin = this->mrModelPart.GetMesh().ConditionsBegin();

#pragma omp parallel for
for (int i = 0; i < nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

(this->*AssignmentMethod)(*it, rVariable, rvector_value);
}
}
}


AssignVectorVariableToPfemConditionsProcess &operator=(AssignVectorVariableToPfemConditionsProcess const &rOther);


}; 




inline std::istream &operator>>(std::istream &rIStream,
AssignVectorVariableToPfemConditionsProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const AssignVectorVariableToPfemConditionsProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
