
#pragma once



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ReplaceMultipleElementsAndConditionsProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ReplaceMultipleElementsAndConditionsProcess);



ReplaceMultipleElementsAndConditionsProcess(
ModelPart& rModelPart,
Parameters Settings
) : Process(Flags()) ,
mrModelPart(rModelPart),
mSettings( Settings)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"element_name_table": {},
"condition_name_table": {},
"ignore_elements" : [],
"ignore_conditions" : [],
"ignore_undefined_types" : false
}  )" );

Settings.ValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("")
}

ReplaceMultipleElementsAndConditionsProcess(ReplaceMultipleElementsAndConditionsProcess const& rOther) = delete;

~ReplaceMultipleElementsAndConditionsProcess() override = default;


ReplaceMultipleElementsAndConditionsProcess& operator=(ReplaceMultipleElementsAndConditionsProcess const& rOther) = delete;

void operator()()
{
Execute();
}


void Execute() override;


std::string Info() const override
{
return "ReplaceMultipleElementsAndConditionsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ReplaceMultipleElementsAndConditionsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}

private:

ModelPart& mrModelPart; 
Parameters mSettings;   




void UpdateSubModelPart(ModelPart& rModelPart,
ModelPart& rRootModelPart);


}; 


inline std::istream& operator >> (std::istream& rIStream,
ReplaceMultipleElementsAndConditionsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ReplaceMultipleElementsAndConditionsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
