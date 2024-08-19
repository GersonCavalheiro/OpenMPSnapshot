
#if !defined(KRATOS_APPLY_DOUBLE_TABLE_PROCESS )
#define  KRATOS_APPLY_DOUBLE_TABLE_PROCESS


#include "custom_processes/apply_component_table_process.hpp"
#include "poromechanics_application_variables.h"

namespace Kratos
{

class ApplyDoubleTableProcess : public ApplyComponentTableProcess
{

public:

KRATOS_CLASS_POINTER_DEFINITION(ApplyDoubleTableProcess);


ApplyDoubleTableProcess(ModelPart& model_part,
Parameters rParameters
) : ApplyComponentTableProcess(model_part, rParameters) {}


~ApplyDoubleTableProcess() override {}


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

#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if(mis_fixed)
{
it->Fix(var);
}

it->FastGetSolutionStepValue(var) = minitial_value;
}
}

KRATOS_CATCH("");
}

void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY;

const Variable<double>& var = KratosComponents< Variable<double> >::Get(mvariable_name);

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

it->FastGetSolutionStepValue(var) = value;
}
}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "ApplyDoubleTableProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ApplyDoubleTableProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:



private:

ApplyDoubleTableProcess& operator=(ApplyDoubleTableProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
ApplyDoubleTableProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ApplyDoubleTableProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
