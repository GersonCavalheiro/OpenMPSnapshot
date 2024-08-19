
#if !defined(KRATOS_DAM_FIX_TEMPERATURE_CONDITION_PROCESS)
#define KRATOS_DAM_FIX_TEMPERATURE_CONDITION_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamFixTemperatureConditionProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamFixTemperatureConditionProcess);

typedef Table<double, double> TableType;


DamFixTemperatureConditionProcess(ModelPart &rModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name" : "PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name"   : "PLEASE_PRESCRIBE_VARIABLE_NAME",
"is_fixed"        : false,
"value"           : 0.0,
"table"           : 0,
"interval":[
0.0,
0.0
]
}  )");

rParameters["variable_name"];
rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mIsFixed = rParameters["is_fixed"].GetBool();
mTemperature = rParameters["value"].GetDouble();

mTimeUnitConverter = mrModelPart.GetProcessInfo()[TIME_UNIT_CONVERTER];
mTableId = rParameters["table"].GetInt();

if (mTableId != 0)
mpTable = mrModelPart.pGetTable(mTableId);

KRATOS_CATCH("");
}


virtual ~DamFixTemperatureConditionProcess() {}

void Execute() override
{
}


void ExecuteInitialize() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if (mIsFixed)
{
it->Fix(var);
}

it->FastGetSolutionStepValue(var) = mTemperature;
}
}

KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);

if (mTableId != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mTemperature = mpTable->GetValue(time);
}

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if (mIsFixed)
{
it->Fix(var);
}

it->FastGetSolutionStepValue(var) = mTemperature;
}
}

KRATOS_CATCH("");
}


void ExecuteFinalizeSolutionStep() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if (nnodes != 0)
{

ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->Free(var);
}
}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "FixTemperatureConditionProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "FixTemperatureConditionProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

ModelPart &mrModelPart;
std::string mVariableName;
std::string mGravityDirection;
bool mIsFixed;
double mTemperature;
double mTimeUnitConverter;
TableType::Pointer mpTable;
int mTableId;


private:
DamFixTemperatureConditionProcess &operator=(DamFixTemperatureConditionProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamFixTemperatureConditionProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamFixTemperatureConditionProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
