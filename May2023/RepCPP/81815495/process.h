
#pragma once



#include "includes/define.h"
#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "includes/define_registry.h"

namespace Kratos
{


class Model;


class Process : public Flags
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Process);


Process() : Flags() {}
explicit Process(const Flags options) : Flags( options ) {}

~Process() override {}


void operator()()
{
Execute();
}



virtual Process::Pointer Create(
Model& rModel,
Parameters ThisParameters
)
{
KRATOS_ERROR << "Calling base class create. Please override this method in the corresonding Process" << std::endl;
return nullptr;
}


virtual void Execute() {}


virtual void ExecuteInitialize()
{
}


virtual void ExecuteBeforeSolutionLoop()
{
}



virtual void ExecuteInitializeSolutionStep()
{
}


virtual void ExecuteFinalizeSolutionStep()
{
}



virtual void ExecuteBeforeOutputStep()
{
}



virtual void ExecuteAfterOutputStep()
{
}



virtual void ExecuteFinalize()
{
}


virtual int Check()
{
return 0;
}


virtual void Clear()
{
}


virtual const Parameters GetDefaultParameters() const
{
KRATOS_ERROR << "Calling the base Process class GetDefaultParameters. Please implement the GetDefaultParameters in your derived process class." << std::endl;
const Parameters default_parameters = Parameters(R"({})" );

return default_parameters;
}




std::string Info() const override
{
return "Process";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Process";
}

void PrintData(std::ostream& rOStream) const override
{
}


private:

KRATOS_REGISTRY_ADD_PROTOTYPE("Processes.KratosMultiphysics", Process)
KRATOS_REGISTRY_ADD_PROTOTYPE("Processes.All", Process)


Process& operator=(Process const& rOther);



}; 




inline std::istream& operator >> (std::istream& rIStream,
Process& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const Process& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
