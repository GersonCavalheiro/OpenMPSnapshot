
#pragma once



#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"

#include "utilities/function_parser_utility.h"
#include "utilities/mortar_utilities.h"
namespace Kratos {





class KRATOS_API(GEO_MECHANICS_APPLICATION) SetParameterFieldProcess : public Process
{
public:

using SizeType = std::size_t;

KRATOS_CLASS_POINTER_DEFINITION(SetParameterFieldProcess);


SetParameterFieldProcess(ModelPart& rModelPart,
const Parameters& rParameters);



void ExecuteInitialize() override;



std::string Info() const override {
return "SetParameterFieldProcess";
}

void PrintInfo(std::ostream& rOStream) const override {
rOStream << "SetParameterFieldProcess";
}



private:

ModelPart& mrModelPart;
Parameters mParameters;




static void SetValueAtElement(Element& rElement, const Variable<double>& rVar, const double Value);


void SetParameterFieldUsingInputFunction(const Variable<double>& rVar);


void SetParameterFieldUsingParametersClass(const Variable<double>& rVar, Parameters& rParameters);


void SetParameterFieldUsingJsonFile(const Variable<double>& rVar);


void SetParameterFieldUsingJsonString(const  Variable<double>& rVar);


}; 


}  
