
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ALMFastInit
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ALMFastInit);

typedef Node                                          NodeType;
typedef Geometry<NodeType>                           GeometryType;
typedef ModelPart::NodesContainerType              NodesArrayType;
typedef ModelPart::ConditionsContainerType    ConditionsArrayType;


ALMFastInit( ModelPart& rThisModelPart):mrThisModelPart(rThisModelPart)
{
KRATOS_TRY;

KRATOS_CATCH("");
}

~ALMFastInit() override
= default;






void operator()()
{
Execute();
}


void Execute() override;






std::string Info() const override
{
return "ALMFastInit";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ALMFastInit";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:















private:



ModelPart& mrThisModelPart; 









ALMFastInit& operator=(ALMFastInit const& rOther) = delete;




}; 






}
