
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ImposeRigidMovementProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ImposeRigidMovementProcess);

typedef Node                                                      NodeType;

typedef ModelPart::MasterSlaveConstraintContainerType ConstraintContainerType;

typedef std::size_t                                                 IndexType;
typedef std::size_t                                                  SizeType;



ImposeRigidMovementProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);

~ImposeRigidMovementProcess() override
= default;






void operator()()
{
Execute();
}



void Execute() override;


void ExecuteInitialize() override;


const Parameters GetDefaultParameters() const override;






std::string Info() const override
{
return "ImposeRigidMovementProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ImposeRigidMovementProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:














private:



ModelPart& mrThisModelPart; 
Parameters mThisParameters; 









ImposeRigidMovementProcess& operator=(ImposeRigidMovementProcess const& rOther) = delete;




}; 





inline std::istream& operator >> (std::istream& rIStream,
ImposeRigidMovementProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ImposeRigidMovementProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}
