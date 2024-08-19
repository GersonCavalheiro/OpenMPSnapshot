
#pragma once



#include "includes/model_part.h"
#include "contact_structural_mechanics_application_variables.h"

namespace Kratos
{






class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ContactUtilities
{
public:

typedef Node                                              NodeType;
typedef Point::CoordinatesArrayType              CoordinatesArrayType;

typedef Geometry<NodeType>                               GeometryType;

typedef ModelPart::NodesContainerType                  NodesArrayType;
typedef ModelPart::ConditionsContainerType        ConditionsArrayType;

typedef std::size_t                                         IndexType;

typedef std::size_t                                          SizeType;

KRATOS_CLASS_POINTER_DEFINITION(ContactUtilities);




ContactUtilities() = default;

virtual ~ContactUtilities() = default;




static double CalculateRelativeSizeMesh(ModelPart& rModelPart);


static double CalculateMaxNodalH(ModelPart& rModelPart);


static double CalculateMeanNodalH(ModelPart& rModelPart);


static double CalculateMinimalNodalH(ModelPart& rModelPart);


template<class TPointType>
static void ScaleNode(
TPointType& rPointToScale,
const array_1d<double, 3>& rNormal,
const double LengthSearch
)
{
noalias(rPointToScale.Coordinates()) = rPointToScale.Coordinates() + rNormal * LengthSearch;
}


static double DistancePoints(
const GeometryType::CoordinatesArrayType& rPointOrigin,
const GeometryType::CoordinatesArrayType& rPointDestiny
);


static void ComputeStepJump(
ModelPart& rModelPart,
const double DeltaTime,
const bool HalfJump = true
);


static bool CheckActivity(
ModelPart& rModelPart,
const bool ThrowError = true
);


static bool CheckModelPartHasRotationDoF(ModelPart& rModelPart);


static void CleanContactModelParts(ModelPart& rModelPart);


static void ComputeExplicitContributionConditions(ModelPart& rModelPart);


static void ActivateConditionWithActiveNodes(ModelPart& rModelPart);


static array_1d<double, 3> GetHalfJumpCenter(GeometryType& rThisGeometry);


template< std::size_t TDim, std::size_t TNumNodes>
static BoundedMatrix<double, TNumNodes, TDim> ComputeTangentMatrixSlip(
const GeometryType& rGeometry,
const std::size_t StepSlip = 1
)
{

const double zero_tolerance = std::numeric_limits<double>::epsilon();
BoundedMatrix<double, TNumNodes, TDim> tangent_matrix;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
const array_1d<double, 3>& r_gt = rGeometry[i_node].FastGetSolutionStepValue(WEIGHTED_SLIP, StepSlip);
const double norm_slip = norm_2(r_gt);
if (norm_slip > zero_tolerance) { 
const array_1d<double, 3> tangent_slip = r_gt/norm_slip;
for (std::size_t i_dof = 0; i_dof < TDim; ++i_dof)
tangent_matrix(i_node, i_dof) = tangent_slip[i_dof];
} else { 
const array_1d<double, 3>& r_normal = rGeometry[i_node].FastGetSolutionStepValue(NORMAL);
array_1d<double, 3> tangent_xi, tangent_eta;
MathUtils<double>::OrthonormalBasis(r_normal, tangent_xi, tangent_eta);
if constexpr (TDim == 3) {
for (std::size_t i_dof = 0; i_dof < 3; ++i_dof)
tangent_matrix(i_node, i_dof) = tangent_xi[i_dof];
} else  {
if (std::abs(tangent_xi[2]) > std::numeric_limits<double>::epsilon()) {
for (std::size_t i_dof = 0; i_dof < 2; ++i_dof)
tangent_matrix(i_node, i_dof) = tangent_eta[i_dof];
} else {
for (std::size_t i_dof = 0; i_dof < 2; ++i_dof)
tangent_matrix(i_node, i_dof) = tangent_xi[i_dof];
}
}
}
}

return tangent_matrix;
}

protected:







private:




static Matrix GetVariableMatrix(
const GeometryType& rNodes,
const Variable<array_1d<double,3> >& rVarName
);




};

}
