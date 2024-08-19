
#pragma once



#include "includes/kratos_parameters.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "geometries/point.h"

namespace Kratos
{





class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ALMVariablesCalculationProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ALMVariablesCalculationProcess);

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

typedef ModelPart::NodesContainerType NodesArrayType;

typedef ModelPart::ConditionsContainerType ConditionsArrayType;


ALMVariablesCalculationProcess(
ModelPart& rThisModelPart,
Variable<double>& rNodalLengthVariable = NODAL_H,
Parameters ThisParameters =  Parameters(R"({})")
):mrThisModelPart(rThisModelPart),
mrNodalLengthVariable(rNodalLengthVariable)
{
KRATOS_TRY

ThisParameters.ValidateAndAssignDefaults(GetDefaultParameters());

mFactorStiffness = ThisParameters["stiffness_factor"].GetDouble();
mPenaltyScale = ThisParameters["penalty_scale_factor"].GetDouble();

KRATOS_ERROR_IF_NOT(rThisModelPart.HasNodalSolutionStepVariable( rNodalLengthVariable )) << "Missing variable " << rNodalLengthVariable;

KRATOS_CATCH("")
}

~ALMVariablesCalculationProcess() override
= default;






void operator()()
{
Execute();
}


void Execute() override;


const Parameters GetDefaultParameters() const override
{
const Parameters default_parameters = Parameters(R"(
{
"stiffness_factor"                     : 10.0,
"penalty_scale_factor"                 : 1.0
})" );

return default_parameters;
}






std::string Info() const override
{
return "ALMVariablesCalculationProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ALMVariablesCalculationProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:














private:



ModelPart& mrThisModelPart;              
Variable<double>& mrNodalLengthVariable; 
double mFactorStiffness;                 
double mPenaltyScale;                    









ALMVariablesCalculationProcess& operator=(ALMVariablesCalculationProcess const& rOther) = delete;




}; 






}
