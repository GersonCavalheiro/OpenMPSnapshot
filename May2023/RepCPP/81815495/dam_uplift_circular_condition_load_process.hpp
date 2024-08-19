
#if !defined(KRATOS_DAM_UPLIFT_CIRCULAR_CONDITION_LOAD_PROCESS)
#define KRATOS_DAM_UPLIFT_CIRCULAR_CONDITION_LOAD_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamUpliftCircularConditionLoadProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamUpliftCircularConditionLoadProcess);

typedef Table<double, double> TableType;


DamUpliftCircularConditionLoadProcess(ModelPart &rModelPart,
ModelPart &rJointModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart), mrJointModelPart(rJointModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"Modify"                                           : true,
"joint_group_name"                                 : "PLEASE_CHOOSE_JOINT_GROUP_NAME",
"Gravity_Direction"                                : "Y",
"Reservoir_Bottom_Coordinate_in_Gravity_Direction" : 0.0,
"Upstream_Coordinate_first_bracket"                : [0.0,0.0,0.0],
"Downstream_Coordinate_first_bracket"              : [0.0,0.0,0.0],
"Focus"                                            : [0.0,0.0,0.0],
"Spe_weight"                                       : 10000,
"Water_level"                                      : 0.0,
"Drains"                                           : false,
"Height_drain"                                     : 0.0,
"Distance"                                         : 0.0,
"Effectiveness"                                    : 0.0,
"table"                                            : 0,
"interval":[
0.0,
0.0
]
}  )");

rParameters["Focus"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mGravityDirection = rParameters["Gravity_Direction"].GetString();
mReferenceCoordinate = rParameters["Reservoir_Bottom_Coordinate_in_Gravity_Direction"].GetDouble();
mSpecific = rParameters["Spe_weight"].GetDouble();

mUpstream.resize(3, false);
mUpstream[0] = rParameters["Upstream_Coordinate_first_bracket"][0].GetDouble();
mUpstream[1] = rParameters["Upstream_Coordinate_first_bracket"][1].GetDouble();
mUpstream[2] = rParameters["Upstream_Coordinate_first_bracket"][2].GetDouble();

mDownstream.resize(3, false);
mDownstream[0] = rParameters["Downstream_Coordinate_first_bracket"][0].GetDouble();
mDownstream[1] = rParameters["Downstream_Coordinate_first_bracket"][1].GetDouble();
mDownstream[2] = rParameters["Downstream_Coordinate_first_bracket"][2].GetDouble();

mFocus.resize(3, false);
mFocus[0] = rParameters["Focus"][0].GetDouble();
mFocus[1] = rParameters["Focus"][1].GetDouble();
mFocus[2] = rParameters["Focus"][2].GetDouble();

mDrain = rParameters["Drains"].GetBool();
mHeightDrain = rParameters["Height_drain"].GetDouble();
mDistanceDrain = rParameters["Distance"].GetDouble();
mEffectivenessDrain = rParameters["Effectiveness"].GetDouble();
mWaterLevel = rParameters["Water_level"].GetInt();

mTimeUnitConverter = mrModelPart.GetProcessInfo()[TIME_UNIT_CONVERTER];
mTableId = rParameters["table"].GetInt();

if (mTableId != 0)
mpTable = mrModelPart.pGetTable(mTableId);

KRATOS_CATCH("");
}


virtual ~DamUpliftCircularConditionLoadProcess() {}

void Execute() override
{
}


void ExecuteInitialize() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
array_1d<double, 3> auxiliar_vector;

int direction;
int radius_comp_1;
int radius_comp_2;

if (mGravityDirection == "X")
{
direction = 0;
radius_comp_1 = 1;
radius_comp_2 = 2;
}
else if (mGravityDirection == "Y")
{
direction = 1;
radius_comp_1 = 0;
radius_comp_2 = 2;
}
else
{
direction = 2;
radius_comp_1 = 0;
radius_comp_2 = 1;
}


double up_radius = norm_2(mFocus - mUpstream);
double down_radius = norm_2(mFocus - mDownstream);
double width_dam = up_radius - down_radius;

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

if (mDrain == true)
{
double coefficient_effectiveness = 1.0 - mEffectivenessDrain;
double aux_drain = coefficient_effectiveness * ((mWaterLevel - mReferenceCoordinate) - mHeightDrain) * ((width_dam - mDistanceDrain) / width_dam) + mHeightDrain;

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

auxiliar_vector.resize(3, false);
const array_1d<double,3>& r_coordinates = it->Coordinates();
noalias(auxiliar_vector) = mFocus - r_coordinates;

double current_radius = sqrt(auxiliar_vector[radius_comp_1] * auxiliar_vector[radius_comp_1] + auxiliar_vector[radius_comp_2] * auxiliar_vector[radius_comp_2]);

mUpliftPressure = mSpecific * ((mWaterLevel - aux_drain) - (r_coordinates[direction])) * (1.0 - ((1.0 / mDistanceDrain) * (fabs(current_radius - up_radius)))) + (mSpecific * aux_drain);

if (mUpliftPressure <= mSpecific * aux_drain)
{
mUpliftPressure = (mSpecific * ((mReferenceCoordinate + aux_drain) - (r_coordinates[direction]))) * (1.0 - ((1.0 / (width_dam - mDistanceDrain)) * (fabs(current_radius - (up_radius - mDistanceDrain)))));
}

if (mUpliftPressure < 0.0)
{
it->FastGetSolutionStepValue(var) = 0.0;
}
else
{
it->FastGetSolutionStepValue(var) = mUpliftPressure;
}
}
}
else
{
#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

auxiliar_vector.resize(3, false);
const array_1d<double,3>& r_coordinates = it->Coordinates();
noalias(auxiliar_vector) = mFocus - r_coordinates;

double current_radius = sqrt(auxiliar_vector[radius_comp_1] * auxiliar_vector[radius_comp_1] + auxiliar_vector[radius_comp_2] * auxiliar_vector[radius_comp_2]);

mUpliftPressure = mSpecific * (mWaterLevel - (r_coordinates[direction])) * (1.0 - (1.0 / width_dam) * (fabs(current_radius - up_radius)));

if (mUpliftPressure < 0.0)
{
it->FastGetSolutionStepValue(var) = 0.0;
}
else
{
it->FastGetSolutionStepValue(var) = mUpliftPressure;
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
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
array_1d<double, 3> auxiliar_vector;

if (mTableId != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mWaterLevel = mpTable->GetValue(time);
}

int direction;
int radius_comp_1;
int radius_comp_2;

if (mGravityDirection == "X")
{
direction = 0;
radius_comp_1 = 1;
radius_comp_2 = 2;
}
else if (mGravityDirection == "Y")
{
direction = 1;
radius_comp_1 = 0;
radius_comp_2 = 2;
}
else
{
direction = 2;
radius_comp_1 = 0;
radius_comp_2 = 1;
}


double up_radius = norm_2(mFocus - mUpstream);
double down_radius = norm_2(mFocus - mDownstream);
double width_dam = up_radius - down_radius;

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

if (mDrain == true)
{
double coefficient_effectiveness = 1.0 - mEffectivenessDrain;
double aux_drain = coefficient_effectiveness * ((mWaterLevel - mReferenceCoordinate) - mHeightDrain) * ((width_dam - mDistanceDrain) / width_dam) + mHeightDrain;

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

auxiliar_vector.resize(3, false);
const array_1d<double,3>& r_coordinates = it->Coordinates();
noalias(auxiliar_vector) = mFocus - r_coordinates;

double current_radius = sqrt(auxiliar_vector[radius_comp_1] * auxiliar_vector[radius_comp_1] + auxiliar_vector[radius_comp_2] * auxiliar_vector[radius_comp_2]);

mUpliftPressure = mSpecific * ((mWaterLevel - aux_drain) - (r_coordinates[direction])) * (1.0 - ((1.0 / mDistanceDrain) * (fabs(current_radius - up_radius)))) + (mSpecific * aux_drain);

if (mUpliftPressure <= mSpecific * aux_drain)
{
mUpliftPressure = (mSpecific * ((mReferenceCoordinate + aux_drain) - (r_coordinates[direction]))) * (1.0 - ((1.0 / (width_dam - mDistanceDrain)) * (fabs(current_radius - (up_radius - mDistanceDrain)))));
}

if (mUpliftPressure < 0.0)
{
it->FastGetSolutionStepValue(var) = 0.0;
}
else
{
it->FastGetSolutionStepValue(var) = mUpliftPressure;
}
}
}
else
{
#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

auxiliar_vector.resize(3, false);
const array_1d<double,3>& r_coordinates = it->Coordinates();
noalias(auxiliar_vector) = mFocus - r_coordinates;

double current_radius = sqrt(auxiliar_vector[radius_comp_1] * auxiliar_vector[radius_comp_1] + auxiliar_vector[radius_comp_2] * auxiliar_vector[radius_comp_2]);

mUpliftPressure = mSpecific * (mWaterLevel - (r_coordinates[direction])) * (1.0 - (1.0 / width_dam) * (fabs(current_radius - up_radius)));

if (mUpliftPressure < 0.0)
{
it->FastGetSolutionStepValue(var) = 0.0;
}
else
{
it->FastGetSolutionStepValue(var) = mUpliftPressure;
}
}
}
}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "DamUpliftCircularConditionLoadProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamUpliftCircularConditionLoadProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

ModelPart &mrModelPart;
ModelPart &mrJointModelPart;
std::string mVariableName;
std::string mGravityDirection;
double mReferenceCoordinate;
double mSpecific;
double mWaterLevel;
bool mDrain;
double mHeightDrain;
double mDistanceDrain;
double mEffectivenessDrain;
double mUpliftPressure;
Vector mUpstream;
Vector mDownstream;
Vector mFocus;
double mTimeUnitConverter;
TableType::Pointer mpTable;
int mTableId;


private:
DamUpliftCircularConditionLoadProcess &operator=(DamUpliftCircularConditionLoadProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamUpliftCircularConditionLoadProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamUpliftCircularConditionLoadProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
