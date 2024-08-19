
#if !defined(KRATOS_DAM_GROUTING_REFERENCE_TEMPERATURE_PROCESS )
#define  KRATOS_DAM_GROUTING_REFERENCE_TEMPERATURE_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamGroutingReferenceTemperatureProcess : public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(DamGroutingReferenceTemperatureProcess);


DamGroutingReferenceTemperatureProcess(ModelPart& rModelPart, Parameters& rParameters
) : Process(Flags()) , mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name"      : "PLEASE_PRESCRIBE_VARIABLE_NAME",
"initial_value"      : 0.0,
"time_grouting"      : 0.0,
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
mTimeGrouting = rParameters["time_grouting"].GetDouble();
mTimeUnitConverter = mrModelPart.GetProcessInfo()[TIME_UNIT_CONVERTER];

KRATOS_CATCH("");
}


virtual ~DamGroutingReferenceTemperatureProcess() {}



void Execute() override
{
}



void ExecuteInitialize() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents< Variable<double> >::Get(mVariableName);

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;

#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->FastGetSolutionStepValue(var) = mInitialValue;
}
}

KRATOS_CATCH("");
}



void ExecuteFinalizeSolutionStep() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents< Variable<double> >::Get(mVariableName);

const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

double time = mrModelPart.GetProcessInfo()[TIME];
time = time / mTimeUnitConverter;

if (time == mTimeGrouting)
{
#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
const double current_temp = it->FastGetSolutionStepValue(TEMPERATURE);
it->FastGetSolutionStepValue(var) = current_temp;

}
}
}

KRATOS_CATCH("");
}


std::string Info() const override
{
return "DamGroutingReferenceTemperatureProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "DamGroutingReferenceTemperatureProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrModelPart;
std::string mVariableName;
double mInitialValue;
double mTimeGrouting;
double mTimeUnitConverter;


private:

DamGroutingReferenceTemperatureProcess& operator=(DamGroutingReferenceTemperatureProcess const& rOther);

};


inline std::istream& operator >> (std::istream& rIStream,
DamGroutingReferenceTemperatureProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const DamGroutingReferenceTemperatureProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 

