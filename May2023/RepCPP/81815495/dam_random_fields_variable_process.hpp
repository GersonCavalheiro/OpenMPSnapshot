
#if !defined(KRATOS_DAM_RANDOM_FIELDS_VARIABLE_PROCESS )
#define  KRATOS_DAM_RANDOM_FIELDS_VARIABLE_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamRandomFieldsVariableProcess : public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(DamRandomFieldsVariableProcess);

typedef Table<double,double> TableType;

DamRandomFieldsVariableProcess(ModelPart& rModelPart, TableType& Table,
Parameters& rParameters
) : Process(Flags()) , mrModelPart(rModelPart) , mrTable(Table)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name" : "PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name" : "PLEASE_PRESCRIBE_VARIABLE_NAME",
"mean_value" : 0.0,
"min_value" : 0.0,
"max_value" : 0.0,
"variance"  : 0.0,
"corr_length" : 0,
"interval":[
0.0,
0.0
]
}  )" );

rParameters["variable_name"];
rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();

KRATOS_CATCH("");
}


virtual ~DamRandomFieldsVariableProcess() {}



void Execute() override
{
}


void ExecuteInitialize() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

it->FastGetSolutionStepValue(var) = mrTable.GetValue(it->Id());

}
}

KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{
}


std::string Info() const override
{
return "DamRandomFieldsVariableProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "DamRandomFieldsVariableProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrModelPart;
TableType& mrTable;
std::string mVariableName;


private:

DamRandomFieldsVariableProcess& operator=(DamRandomFieldsVariableProcess const& rOther);

};


inline std::istream& operator >> (std::istream& rIStream,
DamRandomFieldsVariableProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const DamRandomFieldsVariableProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 

