
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SolidShellThickComputeProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SolidShellThickComputeProcess);

typedef Node                                          NodeType;
typedef Geometry<NodeType>                           GeometryType;
typedef ModelPart::NodesContainerType              NodesArrayType;
typedef ModelPart::ConditionsContainerType    ConditionsArrayType;
typedef ModelPart::ElementsContainerType        ElementsArrayType;

typedef std::size_t                                     IndexType;
typedef std::size_t                                      SizeType;


SolidShellThickComputeProcess(
ModelPart& rThisModelPart
):mrThisModelPart(rThisModelPart)
{
KRATOS_TRY

KRATOS_CATCH("")
}

~SolidShellThickComputeProcess() override
= default;






void operator()()
{
Execute();
}


void Execute() override;






std::string Info() const override
{
return "SolidShellThickComputeProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SolidShellThickComputeProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:














private:



ModelPart& mrThisModelPart; 









SolidShellThickComputeProcess& operator=(SolidShellThickComputeProcess const& rOther) = delete;




}; 






}
