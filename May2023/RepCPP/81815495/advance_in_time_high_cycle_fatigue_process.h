
#pragma once

#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{






class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION)AdvanceInTimeHighCycleFatigueProcess : public Process
{


protected:


public:
static constexpr double tolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION(AdvanceInTimeHighCycleFatigueProcess);


AdvanceInTimeHighCycleFatigueProcess(ModelPart& rModelPart, Parameters ThisParameters);

~AdvanceInTimeHighCycleFatigueProcess() override = default;

void Execute() override;


void CyclePeriodPerIntegrationPoint(bool& rCycleFound);


void StableConditionForAdvancingStrategy(bool& rAdvancingStrategy, bool DamageIndicator);


void TimeIncrement(double& rIncrement);


void TimeAndCyclesUpdate(const double Increment);

protected:
ModelPart& mrModelPart;                     
Parameters mThisParameters;

}; 

} 