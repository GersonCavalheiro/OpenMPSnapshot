
#pragma once



#include "processes/process.h"

namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SetSphericalLocalAxesProcess
: public Process
{

public:


KRATOS_CLASS_POINTER_DEFINITION(SetSphericalLocalAxesProcess);


SetSphericalLocalAxesProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);


~SetSphericalLocalAxesProcess() override = default;


void ExecuteInitialize() override;


void ExecuteInitializeSolutionStep() override;


const Parameters GetDefaultParameters() const override;

std::string Info() const override
{
return "SetSphericalLocalAxesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SetSphericalLocalAxesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrThisModelPart;
Parameters mThisParameters;

private:

SetSphericalLocalAxesProcess& operator=(SetSphericalLocalAxesProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
SetSphericalLocalAxesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const SetSphericalLocalAxesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 
