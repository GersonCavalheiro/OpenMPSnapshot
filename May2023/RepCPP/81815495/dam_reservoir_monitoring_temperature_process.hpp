
#if !defined(KRATOS_DAM_RESERVOIR_MONITORING_TEMPERATURE_PROCESS)
#define KRATOS_DAM_RESERVOIR_MONITORING_TEMPERATURE_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamReservoirMonitoringTemperatureProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamReservoirMonitoringTemperatureProcess);

typedef Table<double, double> TableType;


DamReservoirMonitoringTemperatureProcess(ModelPart &rModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"is_fixed"                                         : false,
"Gravity_Direction"                                : "Y",
"Reservoir_Bottom_Coordinate_in_Gravity_Direction" : 0.0,
"Height_Dam"                                       : 0.0,
"Ambient_temp"                                     : 0.0,
"Ambient_temp_Table"                               : 0,
"Water_level"                                      : 0.0,
"Water_level_Table"                                : 0,
"Z_Coord_1"                                        : 0.0,
"Water_temp_1"                                     : 0.0,
"Water_temp_Table_1"                               : 0,
"Z_Coord_2"                                        : 0.0,
"Water_temp_2"                                     : 0.0,
"Water_temp_Table_2"                               : 0,
"Z_Coord_3"                                        : 0.0,
"Water_temp_3"                                     : 0.0,
"Water_temp_Table_3"                               : 0,
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
mIsFixed = rParameters["is_fixed"].GetBool();
mGravityDirection = rParameters["Gravity_Direction"].GetString();
mReferenceCoordinate = rParameters["Reservoir_Bottom_Coordinate_in_Gravity_Direction"].GetDouble();
mHeight = rParameters["Height_Dam"].GetDouble();
mAmbientTemp = rParameters["Ambient_temp"].GetDouble();
mWaterLevel = rParameters["Water_level"].GetDouble();
mZCoord1 = rParameters["Z_Coord_1"].GetDouble();
mWaterTemp1 = rParameters["Water_temp_1"].GetDouble();
mZCoord2 = rParameters["Z_Coord_2"].GetDouble();
mWaterTemp2 = rParameters["Water_temp_2"].GetDouble();
mZCoord3 = rParameters["Z_Coord_3"].GetDouble();
mWaterTemp3 = rParameters["Water_temp_3"].GetDouble();

mTimeUnitConverter = mrModelPart.GetProcessInfo()[TIME_UNIT_CONVERTER];
mTableIdWater = rParameters["Water_level_Table"].GetInt();
mTableIdAmbientTemp = rParameters["Ambient_temp_Table"].GetInt();
mTableIdWaterTemp1 = rParameters["Water_temp_Table_1"].GetInt();
mTableIdWaterTemp2 = rParameters["Water_temp_Table_2"].GetInt();
mTableIdWaterTemp3 = rParameters["Water_temp_Table_3"].GetInt();


if (mTableIdWater != 0)
mpTableWater = mrModelPart.pGetTable(mTableIdWater);

if (mTableIdAmbientTemp != 0)
mpTableAmbientTemp = mrModelPart.pGetTable(mTableIdAmbientTemp);

if (mTableIdWaterTemp1 != 0)
mpTableWaterTemp1 = mrModelPart.pGetTable(mTableIdWaterTemp1);

if (mTableIdWaterTemp2 != 0)
mpTableWaterTemp2 = mrModelPart.pGetTable(mTableIdWaterTemp2);

if (mTableIdWaterTemp3 != 0)
mpTableWaterTemp3 = mrModelPart.pGetTable(mTableIdWaterTemp3);



KRATOS_CATCH("");
}


virtual ~DamReservoirMonitoringTemperatureProcess() {}


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

double aux = it->Coordinates()[direction];
if (aux < mWaterLevel)
{
if (mIsFixed)
{
it->Fix(var);
}
if (aux > mZCoord1)
{
double Temperature = ((mAmbientTemp - mWaterTemp1)/(mWaterLevel - mZCoord1)) * (aux - mZCoord1) + mWaterTemp1;
it->FastGetSolutionStepValue(var) = Temperature;
}
else if ((aux <= mZCoord1) && (aux > mZCoord2))
{
double Temperature = ((mWaterTemp1 - mWaterTemp2)/(mZCoord1 - mZCoord2)) * (aux - mZCoord2) + mWaterTemp2;
it->FastGetSolutionStepValue(var) = Temperature;
}
else if ((aux <= mZCoord2) && (aux > mZCoord3))
{
double Temperature = ((mWaterTemp2 - mWaterTemp3)/(mZCoord2 - mZCoord3)) * (aux - mZCoord3) + mWaterTemp3;
it->FastGetSolutionStepValue(var) = Temperature;
}
else if (aux <= mZCoord3)
{
double Temperature = ((mWaterTemp3 - mWaterTemp2)/(mZCoord3 - mZCoord2)) * (aux - mZCoord3) + mWaterTemp3;
it->FastGetSolutionStepValue(var) = Temperature;
}
}
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

if (mTableIdAmbientTemp != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mAmbientTemp = mpTableAmbientTemp->GetValue(time);
}

if (mTableIdWaterTemp1 != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mWaterTemp1 = mpTableWaterTemp1->GetValue(time);
}

if (mTableIdWaterTemp2 != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mWaterTemp2 = mpTableWaterTemp2->GetValue(time);
}

if (mTableIdWaterTemp3 != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mWaterTemp3 = mpTableWaterTemp3->GetValue(time);
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

double aux = it->Coordinates()[direction];
if (aux < mWaterLevel)
{
if (mIsFixed)
{
it->Fix(var);
}
if (aux > mZCoord1)
{
double Temperature = ((mAmbientTemp - mWaterTemp1)/(mWaterLevel - mZCoord1)) * (aux - mZCoord1) + mWaterTemp1;
it->FastGetSolutionStepValue(var) = Temperature;
}
else if ((aux <= mZCoord1) && (aux > mZCoord2))
{
double Temperature = ((mWaterTemp1 - mWaterTemp2)/(mZCoord1 - mZCoord2)) * (aux - mZCoord2) + mWaterTemp2;
it->FastGetSolutionStepValue(var) = Temperature;
}
else if ((aux <= mZCoord2) && (aux > mZCoord3))
{
double Temperature = ((mWaterTemp2 - mWaterTemp3)/(mZCoord2 - mZCoord3)) * (aux - mZCoord3) + mWaterTemp3;
it->FastGetSolutionStepValue(var) = Temperature;
}
else if (aux <= mZCoord3)
{
double Temperature = ((mWaterTemp3 - mWaterTemp2)/(mZCoord3 - mZCoord2)) * (aux - mZCoord3) + mWaterTemp3;
it->FastGetSolutionStepValue(var) = Temperature;
}
}
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
return "DamReservoirMonitoringTemperatureProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamReservoirMonitoringTemperatureProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

ModelPart &mrModelPart;
std::string mVariableName;
std::string mGravityDirection;
bool mIsFixed;
double mReferenceCoordinate;
double mHeight;
double mAmbientTemp;
double mWaterLevel;
double mZCoord1;
double mWaterTemp1;
double mZCoord2;
double mWaterTemp2;
double mZCoord3;
double mWaterTemp3;
double mTimeUnitConverter;
TableType::Pointer mpTableWater;
TableType::Pointer mpTableAmbientTemp;
TableType::Pointer mpTableWaterTemp1;
TableType::Pointer mpTableWaterTemp2;
TableType::Pointer mpTableWaterTemp3;
int mTableIdWater;
int mTableIdAmbientTemp;
int mTableIdWaterTemp1;
int mTableIdWaterTemp2;
int mTableIdWaterTemp3;


private:
DamReservoirMonitoringTemperatureProcess &operator=(DamReservoirMonitoringTemperatureProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamReservoirMonitoringTemperatureProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamReservoirMonitoringTemperatureProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
