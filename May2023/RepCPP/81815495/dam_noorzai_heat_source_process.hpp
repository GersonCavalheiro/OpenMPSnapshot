
#if !defined(KRATOS_DAM_NOORZAI_HEAT_SOURCE_PROCESS)
#define KRATOS_DAM_NOORZAI_HEAT_SOURCE_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamNoorzaiHeatFluxProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamNoorzaiHeatFluxProcess);


DamNoorzaiHeatFluxProcess(ModelPart &rModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"density"                             : 0.0,
"specific_heat"                        : 0.0,
"t_max"                               : 0.0,
"alpha"                               : 0.0,
"interval":[
0.0,
0.0
]
}  )");

rParameters["t_max"];
rParameters["alpha"];
rParameters["specific_heat"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mDensity = rParameters["density"].GetDouble();
mSpecificHeat = rParameters["specific_heat"].GetDouble();
mTMax = rParameters["t_max"].GetDouble();
mAlpha = rParameters["alpha"].GetDouble();

KRATOS_CATCH("");
}


virtual ~DamNoorzaiHeatFluxProcess() {}


void Execute() override
{
}


void ExecuteInitialize() override
{
KRATOS_TRY;

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();
const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);

const double time = mrModelPart.GetProcessInfo()[TIME];
const double delta_time = mrModelPart.GetProcessInfo()[DELTA_TIME];

double value = mDensity * mSpecificHeat * mAlpha * mTMax * (exp(-mAlpha * time + 0.5 * delta_time));

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->FastGetSolutionStepValue(var) = value;
}
}
KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY;

this->ExecuteInitialize();

KRATOS_CATCH("");
}


std::string Info() const override
{
return "DamNoorzaiHeatFluxProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamNoorzaiHeatFluxProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:
ModelPart &mrModelPart;
std::string mVariableName;
double mDensity;
double mSpecificHeat;
double mAlpha;
double mTMax;


private:
DamNoorzaiHeatFluxProcess &operator=(DamNoorzaiHeatFluxProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamNoorzaiHeatFluxProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamNoorzaiHeatFluxProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
