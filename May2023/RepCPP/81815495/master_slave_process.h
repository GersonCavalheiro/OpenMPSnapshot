
#pragma once



#include "utilities/parallel_utilities.h"
#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) MasterSlaveProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MasterSlaveProcess);

typedef std::size_t IndexType;

typedef Node                                          NodeType;
typedef Geometry<NodeType>                           GeometryType;
typedef ModelPart::NodesContainerType              NodesArrayType;
typedef ModelPart::ConditionsContainerType    ConditionsArrayType;


MasterSlaveProcess( ModelPart& rThisModelPart)
:mrThisModelPart(rThisModelPart)
{
KRATOS_TRY;

KRATOS_CATCH("");
}

~MasterSlaveProcess() override
= default;






void operator()()
{
Execute();
}


void Execute() override;






std::string Info() const override
{
return "MasterSlaveProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MasterSlaveProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:















private:



ModelPart& mrThisModelPart; 









MasterSlaveProcess& operator=(MasterSlaveProcess const& rOther) = delete;




}; 






}
