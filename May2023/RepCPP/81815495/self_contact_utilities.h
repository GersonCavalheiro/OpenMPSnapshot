
#pragma once



#include "includes/model_part.h"

namespace Kratos
{





namespace SelfContactUtilities
{

void KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ComputeSelfContactPairing(
ModelPart& rModelPart,
const std::size_t EchoLevel = 0
);


void KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) FullAssignmentOfPairs(ModelPart& rModelPart);


void KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) NotPredefinedMasterSlave(ModelPart& rModelPart);

}; 
}  
