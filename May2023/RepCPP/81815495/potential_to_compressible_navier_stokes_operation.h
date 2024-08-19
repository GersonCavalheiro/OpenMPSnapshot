
#pragma once



#include "operations/operation.h"

namespace Kratos
{



class KRATOS_API(COMPRESSIBLE_POTENTIAL_FLOW_APPLICATION) PotentialToCompressibleNavierStokesOperation : public Operation
{
public:

KRATOS_CLASS_POINTER_DEFINITION(PotentialToCompressibleNavierStokesOperation);

KRATOS_REGISTRY_ADD_PROTOTYPE("Operations.KratosMultiphysics.CompressiblePotentialFlowApplication", PotentialToCompressibleNavierStokesOperation)
KRATOS_REGISTRY_ADD_PROTOTYPE("Operations.All", PotentialToCompressibleNavierStokesOperation)


PotentialToCompressibleNavierStokesOperation() : Operation() {}

PotentialToCompressibleNavierStokesOperation(
Model& rModel,
Parameters OperationParameters);

~PotentialToCompressibleNavierStokesOperation() override = default;

PotentialToCompressibleNavierStokesOperation(PotentialToCompressibleNavierStokesOperation const& rOther);


PotentialToCompressibleNavierStokesOperation& operator=(PotentialToCompressibleNavierStokesOperation const& rOther) = delete;




Operation::Pointer Create(
Model& rModel,
Parameters ThisParameters) const override;


const Parameters GetDefaultParameters() const override;


void Execute() override;

private:

Model* mpModel = nullptr;
Parameters mParameters;

}; 


}  
