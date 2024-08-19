
#if !defined(KRATOS_APPLY_COMPONENT_TABLE_PROCESS )
#define  KRATOS_APPLY_COMPONENT_TABLE_PROCESS

#include "includes/table.h"
#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "poromechanics_application_variables.h"

namespace Kratos
{

class ApplyComponentTableProcess : public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(ApplyComponentTableProcess);

typedef Table<double,double> TableType;


ApplyComponentTableProcess(ModelPart& model_part,
Parameters rParameters
) : Process(Flags()) , mr_model_part(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"is_fixed": false,
"value" : 1.0,
"table" : 1
}  )" );

rParameters["table"];
rParameters["variable_name"];
rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name = rParameters["variable_name"].GetString();
mis_fixed = rParameters["is_fixed"].GetBool();
minitial_value = rParameters["value"].GetDouble();

unsigned int TableId = rParameters["table"].GetInt();
mpTable = model_part.pGetTable(TableId);
mTimeUnitConverter = model_part.GetProcessInfo()[TIME_UNIT_CONVERTER];

KRATOS_CATCH("");
}


~ApplyComponentTableProcess() override {}


void Execute() override
{
}

void ExecuteInitialize() override
{
KRATOS_TRY;

typedef Variable<double> component_type;
const component_type& var_component = KratosComponents< component_type >::Get(mvariable_name);

const int nnodes = static_cast<int>(mr_model_part.Nodes().size());

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mr_model_part.NodesBegin();

#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if(mis_fixed)
{
it->Fix(var_component);
}

it->FastGetSolutionStepValue(var_component) = minitial_value;
}
}

KRATOS_CATCH("");
}

void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY;

typedef Variable<double> component_type;
const component_type& var_component = KratosComponents< component_type >::Get(mvariable_name);

const double Time = mr_model_part.GetProcessInfo()[TIME]/mTimeUnitConverter;
double value = mpTable->GetValue(Time);

const int nnodes = static_cast<int>(mr_model_part.Nodes().size());

if(nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mr_model_part.NodesBegin();

#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

it->FastGetSolutionStepValue(var_component) = value;
}
}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "ApplyComponentTableProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ApplyComponentTableProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mr_model_part;
std::string mvariable_name;
bool mis_fixed;
double minitial_value;
TableType::Pointer mpTable;
double mTimeUnitConverter;


private:

ApplyComponentTableProcess& operator=(ApplyComponentTableProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
ApplyComponentTableProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ApplyComponentTableProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
