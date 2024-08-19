

#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) AALMAdaptPenaltyValueProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AALMAdaptPenaltyValueProcess);

typedef Node                                          NodeType;
typedef Geometry<NodeType>                           GeometryType;
typedef ModelPart::NodesContainerType              NodesArrayType;
typedef ModelPart::ConditionsContainerType    ConditionsArrayType;


AALMAdaptPenaltyValueProcess( ModelPart& rThisModelPart):mrThisModelPart(rThisModelPart)
{
KRATOS_TRY;

KRATOS_CATCH("");
}

~AALMAdaptPenaltyValueProcess() override
= default;






void operator()()
{
Execute();
}


void Execute() override;






std::string Info() const override
{
return "AALMAdaptPenaltyValueProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AALMAdaptPenaltyValueProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:















private:



ModelPart& mrThisModelPart;









AALMAdaptPenaltyValueProcess& operator=(AALMAdaptPenaltyValueProcess const& rOther) = delete;




}; 






}
