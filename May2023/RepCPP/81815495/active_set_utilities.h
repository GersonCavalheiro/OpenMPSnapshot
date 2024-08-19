
#pragma once



#include "includes/model_part.h"

namespace Kratos
{





namespace ActiveSetUtilities
{

typedef Node                                              NodeType;
typedef Point::CoordinatesArrayType              CoordinatesArrayType;

typedef Geometry<NodeType>                               GeometryType;

typedef std::size_t                                         IndexType;

typedef std::size_t                                          SizeType;



std::size_t KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ComputePenaltyFrictionlessActiveSet(ModelPart& rModelPart);


array_1d<std::size_t, 2> KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ComputePenaltyFrictionalActiveSet(
ModelPart& rModelPart,
const bool PureSlip = false,
const SizeType EchoLevel = 0
);


std::size_t KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ComputeALMFrictionlessActiveSet(ModelPart& rModelPart);


std::size_t KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ComputeALMFrictionlessComponentsActiveSet(ModelPart& rModelPart);


array_1d<std::size_t, 2> KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ComputeALMFrictionalActiveSet(
ModelPart& rModelPart,
const bool PureSlip = false,
const SizeType EchoLevel = 0
);

};

} 