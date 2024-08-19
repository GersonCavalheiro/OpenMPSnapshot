
#pragma once



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class KRATOS_API(KRATOS_CORE) ReplaceElementsAndConditionsProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ReplaceElementsAndConditionsProcess);



ReplaceElementsAndConditionsProcess(
ModelPart& rModelPart,
Parameters Settings
) : Process(Flags()) ,
mrModelPart(rModelPart),
mSettings( Settings)
{
KRATOS_TRY

const std::string element_name = Settings["element_name"].GetString();
const std::string condition_name = Settings["condition_name"].GetString();

KRATOS_ERROR_IF(element_name != "" && !KratosComponents<Element>::Has(element_name)) << "Element name not found in KratosComponents< Element > -- name is " << element_name << std::endl;
KRATOS_ERROR_IF(condition_name != "" && !KratosComponents<Condition>::Has(condition_name)) << "Condition name not found in KratosComponents< Condition > -- name is " << condition_name << std::endl;

Settings.ValidateAndAssignDefaults(GetDefaultParameters());

KRATOS_CATCH("")
}

ReplaceElementsAndConditionsProcess(ReplaceElementsAndConditionsProcess const& rOther) = delete;

~ReplaceElementsAndConditionsProcess() override = default;


ReplaceElementsAndConditionsProcess& operator=(ReplaceElementsAndConditionsProcess const& rOther) = delete;

void operator()()
{
Execute();
}


void Execute() override;


const Parameters GetDefaultParameters() const override
{
const Parameters default_parameters( R"({
"element_name"   : "PLEASE_CHOOSE_MODEL_PART_NAME",
"condition_name" : "PLEASE_PRESCRIBE_VARIABLE_NAME"
} )" );
return default_parameters;
}


std::string Info() const override
{
return "ReplaceElementsAndConditionsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ReplaceElementsAndConditionsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}

protected:

ModelPart& mrModelPart; 
Parameters mSettings;   


}; 


inline std::istream& operator >> (std::istream& rIStream,
ReplaceElementsAndConditionsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ReplaceElementsAndConditionsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  