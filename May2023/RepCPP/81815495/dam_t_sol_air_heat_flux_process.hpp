
#if !defined(KRATOS_DAM_T_SOL_AIR_HEAT_FLUX_PROCESS)
#define KRATOS_DAM_T_SOL_AIR_HEAT_FLUX_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamTSolAirHeatFluxProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamTSolAirHeatFluxProcess);

typedef Table<double, double> TableType;


DamTSolAirHeatFluxProcess(ModelPart &rModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"h_0"                             : 0.0,
"ambient_temperature"             : 0.0,
"table_ambient_temperature"       : 0,
"emisivity"                       : 0.0,
"delta_R"                         : 0.0,
"absorption_index"                : 0.0,
"total_insolation"                : 0.0,
"interval":[
0.0,
0.0
]
}  )");

rParameters["h_0"];
rParameters["delta_R"];
rParameters["absorption_index"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mH0 = rParameters["h_0"].GetDouble();
mAmbientTemperature = rParameters["ambient_temperature"].GetDouble();
mEmisivity = rParameters["emisivity"].GetDouble();
mDeltaR = rParameters["delta_R"].GetDouble();
mAbsorption_index = rParameters["absorption_index"].GetDouble();
mTotalInsolation = rParameters["total_insolation"].GetDouble();

mTimeUnitConverter = mrModelPart.GetProcessInfo()[TIME_UNIT_CONVERTER];
mTableId = rParameters["table_ambient_temperature"].GetInt();

if (mTableId != 0)
mpTable = mrModelPart.pGetTable(mTableId);

KRATOS_CATCH("");
}


virtual ~DamTSolAirHeatFluxProcess() {}


void Execute() override
{
}



void ExecuteInitialize() override
{

KRATOS_TRY;

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);

double t_sol_air = mAmbientTemperature + (mAbsorption_index * mTotalInsolation / mH0) - (mEmisivity * mDeltaR / mH0);

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

const double temp_current = it->FastGetSolutionStepValue(TEMPERATURE);
const double heat_flux = mH0 * (t_sol_air - temp_current);
it->FastGetSolutionStepValue(var) = heat_flux;
}
}

KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{

KRATOS_TRY;

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);

if (mTableId != 0)
{
double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;
mAmbientTemperature = mpTable->GetValue(time);
}

double t_sol_air = mAmbientTemperature + (mAbsorption_index * mTotalInsolation / mH0) - (mEmisivity * mDeltaR / mH0);

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

const double temp_current = it->FastGetSolutionStepValue(TEMPERATURE);
const double heat_flux = mH0 * (t_sol_air - temp_current);
it->FastGetSolutionStepValue(var) = heat_flux;
}
}

KRATOS_CATCH("");
}


std::string Info() const override
{
return "DamTSolAirHeatFluxProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamTSolAirHeatFluxProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:
ModelPart &mrModelPart;
std::string mVariableName;
double mH0;
double mAmbientTemperature;
double mEmisivity;
double mDeltaR;
double mAbsorption_index;
double mTotalInsolation;
double mTimeUnitConverter;
TableType::Pointer mpTable;
int mTableId;


private:
DamTSolAirHeatFluxProcess &operator=(DamTSolAirHeatFluxProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamTSolAirHeatFluxProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamTSolAirHeatFluxProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
