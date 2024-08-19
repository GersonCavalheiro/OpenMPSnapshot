
#pragma once

#include "processes/process.h"

namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SetCylindricalLocalAxesProcess
: public Process
{

public:


KRATOS_CLASS_POINTER_DEFINITION(SetCylindricalLocalAxesProcess);


SetCylindricalLocalAxesProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);


~SetCylindricalLocalAxesProcess() override = default;


void ExecuteInitialize() override;


void ExecuteInitializeSolutionStep() override;


const Parameters GetDefaultParameters() const override;

std::string Info() const override
{
return "SetCylindricalLocalAxesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SetCylindricalLocalAxesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrThisModelPart;
Parameters mThisParameters;

private:

SetCylindricalLocalAxesProcess& operator=(SetCylindricalLocalAxesProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
SetCylindricalLocalAxesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const SetCylindricalLocalAxesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 
