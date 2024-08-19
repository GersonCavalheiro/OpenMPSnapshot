
#pragma once



#include "containers/model.h"
#include "includes/define_registry.h"
#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "includes/registry.h"

namespace Kratos
{



class Operation
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Operation);


explicit Operation() = default;

virtual ~Operation() {}

Operation(Operation const& rOther) {}


Operation& operator=(Operation const& rOther) = delete;

void operator()()
{
Execute();
}



virtual Operation::Pointer Create(
Model& rModel,
Parameters ThisParameters) const
{
KRATOS_ERROR << "Calling base class Create. Please override this method in the corresonding Operation" << std::endl;
return nullptr;
}


virtual void Execute()
{
KRATOS_ERROR << "Calling base class Execute. Please override this method in the corresonding Operation" << std::endl;
}


virtual const Parameters GetDefaultParameters() const
{
KRATOS_ERROR << "Calling the base Operation class GetDefaultParameters. Please implement the GetDefaultParameters in your derived process class." << std::endl;
const Parameters default_parameters = Parameters(R"({})" );

return default_parameters;
}






std::string Info() const
{
return "Operation";
}

void PrintInfo(std::ostream& rOStream) const
{
rOStream << "Operation";
}

void PrintData(std::ostream& rOStream) const
{
}

private:

KRATOS_REGISTRY_ADD_PROTOTYPE("Operations.KratosMultiphysics", Operation)
KRATOS_REGISTRY_ADD_PROTOTYPE("Operations.All", Operation)

}; 




inline std::istream& operator >> (std::istream& rIStream, Operation& rThis);

inline std::ostream& operator << (std::ostream& rOStream, const Operation& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
