
#if !defined(KRATOS_APPLY_CONSTANT_HYDROSTATIC_PRESSURE_PROCESS )
#define  KRATOS_APPLY_CONSTANT_HYDROSTATIC_PRESSURE_PROCESS

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "poromechanics_application_variables.h"

namespace Kratos
{

class ApplyConstantHydrostaticPressureProcess : public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(ApplyConstantHydrostaticPressureProcess);


ApplyConstantHydrostaticPressureProcess(ModelPart& model_part,
Parameters rParameters
) : Process(Flags()) , mr_model_part(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"is_fixed": false,
"gravity_direction" : 2,
"reference_coordinate" : 0.0,
"specific_weight" : 10000.0,
"table" : 1
}  )" );

rParameters["reference_coordinate"];
rParameters["variable_name"];
rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name = rParameters["variable_name"].GetString();
mis_fixed = rParameters["is_fixed"].GetBool();
mgravity_direction = rParameters["gravity_direction"].GetInt();
mreference_coordinate = rParameters["reference_coordinate"].GetDouble();
mspecific_weight = rParameters["specific_weight"].GetDouble();

KRATOS_CATCH("");
}


~ApplyConstantHydrostaticPressureProcess() override {}


void Execute() override
{
}

void ExecuteInitialize() override
{
KRATOS_TRY;

const Variable<double>& var = KratosComponents< Variable<double> >::Get(mvariable_name);

const int nnodes = static_cast<int>(mr_model_part.Nodes().size());

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mr_model_part.NodesBegin();

array_1d<double,3> Coordinates;

#pragma omp parallel for private(Coordinates)
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if(mis_fixed)
{
it->Fix(var);
}

noalias(Coordinates) = it->Coordinates();

const double pressure = mspecific_weight*( mreference_coordinate - Coordinates[mgravity_direction] );

if(pressure > 0.0) 
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
return "ApplyConstantHydrostaticPressureProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ApplyConstantHydrostaticPressureProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mr_model_part;
std::string mvariable_name;
bool mis_fixed;
unsigned int mgravity_direction;
double mreference_coordinate;
double mspecific_weight;


private:

ApplyConstantHydrostaticPressureProcess& operator=(ApplyConstantHydrostaticPressureProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
ApplyConstantHydrostaticPressureProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ApplyConstantHydrostaticPressureProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
