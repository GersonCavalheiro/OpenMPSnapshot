#if !defined(KRATOS_DAM_WESTERGAARD_CONDITION_LOAD_PROCESS)
#define KRATOS_DAM_WESTERGAARD_CONDITION_LOAD_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamWestergaardConditionLoadProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamWestergaardConditionLoadProcess);

typedef Table<double, double> TableType;


DamWestergaardConditionLoadProcess(ModelPart &rModelPart,
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
"Aceleration"                                           : 0.0,
"Aceleration_Table"                                     : 0,
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
mAcceleration = rParameters["Aceleration"].GetDouble();

mTimeUnitConverter = mrModelPart.GetProcessInfo()[TIME_UNIT_CONVERTER];
mTableIdWater = rParameters["Water_Table"].GetInt();
mTableIdAcceleration = rParameters["Aceleration_Table"].GetInt();

if (mTableIdWater != 0)
mpTableWater = mrModelPart.pGetTable(mTableIdWater);

if (mTableIdAcceleration != 0)
mpTableAcceleration = mrModelPart.pGetTable(mTableIdAcceleration);

KRATOS_CATCH("");
}


virtual ~DamWestergaardConditionLoadProcess() {}


void Execute() override
{
}


void ExecuteInitialize() override
{
KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
int direction;
double pressure;

if (mGravityDirection == "X")
direction = 0;
else if (mGravityDirection == "Y")
direction = 1;
else
direction = 2;

double unit_acceleration = mAcceleration / 9.81;

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

double y_water = mWaterLevel - (it->Coordinates()[direction]);

if (y_water < 0.0)
{
y_water = 0.0;
}

pressure = (mSpecific * (y_water)) + 0.875 * (unit_acceleration)*mSpecific * sqrt(y_water * (mWaterLevel - mReferenceCoordinate));
it->FastGetSolutionStepValue(var) = pressure;
}
}

KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);

if (mTableIdWater != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mWaterLevel = mpTableWater->GetValue(time);
}

if (mTableIdAcceleration != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mAcceleration = mpTableAcceleration->GetValue(time);
}

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
int direction;
double pressure;

if (mGravityDirection == "X")
direction = 0;
else if (mGravityDirection == "Y")
direction = 1;
else
direction = 2;

double unit_acceleration = mAcceleration / 9.81;

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

double y_water = mWaterLevel - (it->Coordinates()[direction]);

if (y_water < 0.0)
{
y_water = 0.0;
}

if (unit_acceleration < 0.0)
{
pressure = (mSpecific * (y_water)) + 0.875 * (-1.0 * unit_acceleration) * mSpecific * sqrt(y_water * (mWaterLevel - mReferenceCoordinate));
}
else
{
pressure = (mSpecific * (y_water));
}

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
return "DamWestergaardConditionLoadProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamWestergaardConditionLoadProcess";
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
double mAcceleration;
double mTimeUnitConverter;
TableType::Pointer mpTableWater;
TableType::Pointer mpTableAcceleration;
int mTableIdWater;
int mTableIdAcceleration;


private:
DamWestergaardConditionLoadProcess &operator=(DamWestergaardConditionLoadProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamWestergaardConditionLoadProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamWestergaardConditionLoadProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
