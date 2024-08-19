
#pragma once



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"


namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ProjectVectorOnSurfaceUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ProjectVectorOnSurfaceUtility);

typedef array_1d<double, 3> Vector3;
typedef Variable< array_1d< double, 3> > ArrayVariableType;


ProjectVectorOnSurfaceUtility() = delete;

virtual ~ProjectVectorOnSurfaceUtility() = default;



static void Execute(ModelPart& rModelPart,Parameters ThisParameters);


private:

static void PlanarProjection(
ModelPart& rModelPart,
const Parameters ThisParameters,
const Vector3& rGlobalDirection,
const ArrayVariableType& rVariable,
const int EchoLevel,
const bool rCheckLocalSpaceDimension);

static void RadialProjection(
ModelPart& rModelPart,
const Parameters ThisParameters,
const Vector3& rGlobalDirection,
const ArrayVariableType& rVariable,
const int EchoLevel,
const bool rCheckLocalSpaceDimension);

static void SphericalProjection(
ModelPart& rModelPart,
const Parameters ThisParameters,
const Vector3& rGlobalDirection,
const ArrayVariableType& rVariable,
const int EchoLevel,
const bool rCheckLocalSpaceDimension);


}; 



}  
