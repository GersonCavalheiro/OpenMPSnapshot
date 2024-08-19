
#if !defined(KRATOS_DAM_INPUT_TABLE_NODAL_YOUNG_MODULUS_PROCESS )
#define  KRATOS_DAM_INPUT_TABLE_NODAL_YOUNG_MODULUS_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamInputTableNodalYoungModulusProcess : public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(DamInputTableNodalYoungModulusProcess);

typedef Table<double,double> TableType;

DamInputTableNodalYoungModulusProcess(ModelPart& rModelPart, TableType& Table,
Parameters& rParameters
) : Process(Flags()) , mrModelPart(rModelPart) , mrTable(Table)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name"      : "PLEASE_PRESCRIBE_VARIABLE_NAME",
"initial_value"      : 0.0,
"input_file_name"    : "",
"interval":[
0.0,
0.0
]
}  )" );

rParameters["variable_name"];
rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mInitialValue = rParameters["initial_value"].GetDouble();
mInputFile = rParameters["input_file_name"].GetString();

KRATOS_CATCH("");
}


virtual ~DamInputTableNodalYoungModulusProcess() {}



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

if ((mInputFile == "") || (mInputFile == "- No file") || (mInputFile == "- Add new file"))
{
#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

it->FastGetSolutionStepValue(var) = mInitialValue;

}
}
else
{
#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

it->FastGetSolutionStepValue(var) = mrTable.GetValue(it->Id());

}
}
}

KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();


if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

if ((mInputFile == "") || (mInputFile == "- No file") || (mInputFile == "- Add new file"))
{
#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

it->FastGetSolutionStepValue(var) = mInitialValue;

}
}
else
{
#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

it->FastGetSolutionStepValue(var) = mrTable.GetValue(it->Id());

}
}
}

KRATOS_CATCH("");
}


std::string Info() const override
{
return "DamInputTableNodalYoungModulusProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "DamInputTableNodalYoungModulusProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrModelPart;
TableType& mrTable;
std::string mVariableName;
double mInitialValue;
std::string mInputFile;


private:

DamInputTableNodalYoungModulusProcess& operator=(DamInputTableNodalYoungModulusProcess const& rOther);

};


inline std::istream& operator >> (std::istream& rIStream,
DamInputTableNodalYoungModulusProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const DamInputTableNodalYoungModulusProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 

