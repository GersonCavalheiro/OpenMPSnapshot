
#pragma once

#include "processes/process.h"

namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SetCartesianLocalAxesProcess
: public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(SetCartesianLocalAxesProcess);


SetCartesianLocalAxesProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);


~SetCartesianLocalAxesProcess() override = default;


void ExecuteInitialize() override;


void ExecuteInitializeSolutionStep() override;


const Parameters GetDefaultParameters() const override;

std::string Info() const override
{
return "SetCartesianLocalAxesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SetCartesianLocalAxesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrThisModelPart;
Parameters mThisParameters;


private:

SetCartesianLocalAxesProcess& operator=(SetCartesianLocalAxesProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
SetCartesianLocalAxesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const SetCartesianLocalAxesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 
