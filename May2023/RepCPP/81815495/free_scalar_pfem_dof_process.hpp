
#if !defined(KRATOS_FREE_SCALAR_PFEM_DOF_PROCESS_H_INCLUDED)
#define KRATOS_FREE_SCALAR_PFEM_DOF_PROCESS_H_INCLUDED



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class FreeScalarPfemDofProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(FreeScalarPfemDofProcess);

FreeScalarPfemDofProcess(ModelPart &model_part,
Parameters rParameters) : Process(), mrModelPart(model_part)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME"
}  )");

rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name = rParameters["variable_name"].GetString();

if (KratosComponents<Variable<double>>::Has(mvariable_name)) 
{
if (model_part.GetNodalSolutionStepVariablesList().Has(KratosComponents<Variable<double>>::Get(mvariable_name)) == false)
{
KRATOS_THROW_ERROR(std::runtime_error, "trying to set a variable that is not in the model_part - variable name is ", mvariable_name);
}
}
else if (KratosComponents<Variable<int>>::Has(mvariable_name)) 
{
if (model_part.GetNodalSolutionStepVariablesList().Has(KratosComponents<Variable<int>>::Get(mvariable_name)) == false)
{
KRATOS_THROW_ERROR(std::runtime_error, "trying to set a variable that is not in the model_part - variable name is ", mvariable_name);
}
}
else if (KratosComponents<Variable<bool>>::Has(mvariable_name)) 
{
if (model_part.GetNodalSolutionStepVariablesList().Has(KratosComponents<Variable<bool>>::Get(mvariable_name)) == false)
{
KRATOS_THROW_ERROR(std::runtime_error, "trying to set a variable that is not in the model_part - variable name is ", mvariable_name);
}
}

KRATOS_CATCH("");
}

FreeScalarPfemDofProcess(ModelPart &model_part,
const Variable<double> &rVariable) : Process(), mrModelPart(model_part)
{
KRATOS_TRY;

if (model_part.GetNodalSolutionStepVariablesList().Has(rVariable) == false)
{
KRATOS_THROW_ERROR(std::runtime_error, "trying to set a variable that is not in the model_part - variable name is ", rVariable);
}

mvariable_name = rVariable.Name();

KRATOS_CATCH("");
}

FreeScalarPfemDofProcess(ModelPart &model_part,
const Variable<int> &rVariable) : Process(), mrModelPart(model_part)
{
KRATOS_TRY;

if (model_part.GetNodalSolutionStepVariablesList().Has(rVariable) == false)
{
KRATOS_THROW_ERROR(std::runtime_error, "Trying to set a variable that is not in the model_part - variable name is ", rVariable);
}

mvariable_name = rVariable.Name();

KRATOS_CATCH("");
}

FreeScalarPfemDofProcess(ModelPart &model_part,
const Variable<bool> &rVariable) : Process(), mrModelPart(model_part)
{
KRATOS_TRY;

if (model_part.GetNodalSolutionStepVariablesList().Has(rVariable) == false)
{
KRATOS_THROW_ERROR(std::runtime_error, "Trying to set a variable that is not in the model_part - variable name is ", rVariable);
}

mvariable_name = rVariable.Name();

KRATOS_CATCH("");
}

~FreeScalarPfemDofProcess() override {}


void operator()()
{
Execute();
}


void Execute() override
{

KRATOS_TRY;

if (KratosComponents<Variable<double>>::Has(mvariable_name)) 
{
InternalFreeDof<>(KratosComponents<Variable<double>>::Get(mvariable_name));
}
else
{
KRATOS_THROW_ERROR(std::logic_error, "Not able to set the variable. Attempting to set variable:", mvariable_name);
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
return "FreeScalarPfemDofProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "FreeScalarPfemDofProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

FreeScalarPfemDofProcess(FreeScalarPfemDofProcess const &rOther);


private:

ModelPart &mrModelPart;
std::string mvariable_name;


template <class TVarType>
void InternalFreeDof(TVarType &rVar)
{
const int nnodes = mrModelPart.GetMesh().Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->Free(rVar);
}
}
}


FreeScalarPfemDofProcess &operator=(FreeScalarPfemDofProcess const &rOther);


}; 




inline std::istream &operator>>(std::istream &rIStream,
FreeScalarPfemDofProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const FreeScalarPfemDofProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
