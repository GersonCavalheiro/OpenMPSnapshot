
#pragma once



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "structural_mechanics_application_variables.h"
#include "response_functions/adjoint_response_function.h"


namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) AdjointStructuralResponseFunction : public AdjointResponseFunction
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AdjointStructuralResponseFunction);

typedef std::size_t IndexType;

typedef std::size_t SizeType;



AdjointStructuralResponseFunction(ModelPart& rModelPart, Parameters ResponseSettings);

virtual ~AdjointStructuralResponseFunction() override
{
}



virtual void Initialize() override;

using AdjointResponseFunction::CalculateGradient;

virtual void CalculateGradient(const Condition& rAdjointCondition,
const Matrix& rResidualGradient,
Vector& rResponseGradient,
const ProcessInfo& rProcessInfo) override;

virtual double CalculateValue(ModelPart& rModelPart) override;


protected:

ModelPart& mrModelPart;




private:

unsigned int mGradientMode;
Parameters mResponseSettings;



};



} 
