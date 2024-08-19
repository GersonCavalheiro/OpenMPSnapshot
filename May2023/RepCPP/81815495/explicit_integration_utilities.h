
#pragma once



#include "includes/model_part.h"

namespace Kratos
{





namespace ExplicitIntegrationUtilities
{
typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef ModelPart::ElementsContainerType ElementsArrayType;
typedef ModelPart::NodesContainerType NodesArrayType;


double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateDeltaTime(
ModelPart& rModelPart,
Parameters ThisParameters = Parameters(R"({})")
);


double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) InnerCalculateDeltaTime(
ModelPart& rModelPart,
const double TimeStepPredictionLevel,
const double MaxDeltaTime,
const double SafetyFactor,
const double MassFactor
);

}; 
}  
