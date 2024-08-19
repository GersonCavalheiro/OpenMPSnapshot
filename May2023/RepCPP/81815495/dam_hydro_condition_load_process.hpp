
#if !defined(KRATOS_DAM_HYDRO_CONDITION_LOAD_PROCESS)
#define KRATOS_DAM_HYDRO_CONDITION_LOAD_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamHydroConditionLoadProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamHydroConditionLoadProcess);

typedef Table<double, double> TableType;


DamHydroConditionLoadProcess(ModelPart &rModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"Modify"                                                : true,
"Gravity_Direction"                                     : "Y",
"Reservoir_Bottom_Coordinate_in_Gravity_Direction"      : 0.0,
"Spe_weight"                                            : 0.0,
"Water_level"                                           : 0.0,
"Water_Table"                                           : 0,
"interval":[
0.0,
0.0
]
}  )");

rParameters["Reservoir_Bottom_Coordinate_in_Gravity_Direction"];
rParameters["variable_name"];
rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mGravityDirection = rParameters["Gravity_Direction"].GetString();
mReferenceCoordinate = rParameters["Reservoir_Bottom_Coordinate_in_Gravity_Direction"].GetDouble();
mSpecific = rParameters["Spe_weight"].GetDouble();
mWaterLevel = rParameters["Water_level"].GetDouble();

mTimeUnitConverter = mrModelPart.GetProcessInfo()[TIME_UNIT_CONVERTER];
mTableId = rParameters["Water_Table"].GetInt();

if (mTableId != 0)
mpTable = mrModelPart.pGetTable(mTableId);

KRATOS_CATCH("");
}


virtual ~DamHydroConditionLoadProcess() {}


void Execute() override
{
}


void ExecuteInitialize() override
{
KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
int direction;

if (mGravityDirection == "X")
direction = 0;
else if (mGravityDirection == "Y")
direction = 1;
else
direction = 2;

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

double pressure = (mSpecific * (mWaterLevel - it->Coordinates()[direction]));

if (pressure > 0.0)
{
it->FastGetSolutionStepValue(var) = pressure;
}
else
{
it->FastGetSolutionStepValue(var) = 0.0;
}
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
mWaterLevel = mpTable->GetValue(time);
}

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

int direction;

if (mGravityDirection == "X")
direction = 0;
else if (mGravityDirection == "Y")
direction = 1;
else
direction = 2;

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

double pressure = (mSpecific * (mWaterLevel - it->Coordinates()[direction]));

if (pressure > 0.0)
{
it->FastGetSolutionStepValue(var) = pressure;
}
else
{
it->FastGetSolutionStepValue(var) = 0.0;
}
}
}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "DamHydroConditionLoadProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamHydroConditionLoadProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

ModelPart &mrModelPart;
std::string mVariableName;
std::string mGravityDirection;
double mReferenceCoordinate;
double mSpecific;
double mWaterLevel;
double mTimeUnitConverter;
TableType::Pointer mpTable;
int mTableId;


private:
DamHydroConditionLoadProcess &operator=(DamHydroConditionLoadProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamHydroConditionLoadProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamHydroConditionLoadProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
