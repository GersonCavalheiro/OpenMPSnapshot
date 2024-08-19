
#pragma once



#include "containers/model.h"
#include "includes/kratos_parameters.h"

namespace Kratos
{



class Controller
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Controller);


Controller() noexcept = default;

virtual ~Controller() = default;

Controller(Controller const& rOther) = default;

Controller(Controller&& rOther) noexcept = default;


Controller& operator=(Controller const& rOther) = delete;



virtual Controller::Pointer Create(
Model& rModel,
Parameters ThisParameters) const = 0;


virtual int Check() const
{
return 0;
}


virtual bool Evaluate() = 0;


virtual Parameters GetDefaultParameters() const
{
return Parameters(R"({})" );
}


std::string Info() const
{
return "Controller";
}

void PrintInfo(std::ostream& rOStream) const
{
rOStream << "Controller";
}

void PrintData(std::ostream& rOStream) const
{
}

}; 



std::istream& operator >> (std::istream& rIStream, Controller& rThis);

std::ostream& operator << (std::ostream& rOStream, const Controller& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
