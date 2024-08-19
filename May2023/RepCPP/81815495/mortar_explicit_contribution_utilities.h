
#pragma once



#include "utilities/math_utils.h"
#include "custom_conditions/paired_condition.h"
#include "includes/mortar_classes.h"


#include "utilities/exact_mortar_segmentation_utility.h"
#include "custom_utilities/derivatives_utilities.h"


#include "geometries/line_2d_2.h"
#include "geometries/triangle_3d_3.h"

namespace Kratos
{





template< const SizeType TDim, const SizeType TNumNodes, const FrictionalCase TFrictional, const bool TNormalVariation, const SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) MortarExplicitContributionUtilities
{
public:

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef Point                                                                        PointType;

typedef Node                                                                       NodeType;

typedef Geometry<NodeType>                                                        GeometryType;

typedef GeometryType::IntegrationPointsArrayType                         IntegrationPointsType;

typedef typename std::conditional<TNumNodes == 2, PointBelongsLine2D2N, typename std::conditional<TNumNodes == 3, typename std::conditional<TNumNodesMaster == 3, PointBelongsTriangle3D3N, PointBelongsTriangle3D3NQuadrilateral3D4N>::type, typename std::conditional<TNumNodesMaster == 3, PointBelongsQuadrilateral3D4NTriangle3D3N, PointBelongsQuadrilateral3D4N>::type>::type>::type BelongType;

typedef PointBelong<TNumNodes, TNumNodesMaster>                                PointBelongType;

typedef Geometry<PointBelongType>                                      GeometryPointBelongType;

typedef array_1d<PointBelongType,TDim>                                      ConditionArrayType;

typedef typename std::vector<ConditionArrayType>                        ConditionArrayListType;

typedef Line2D2<PointType>                                                            LineType;

typedef Triangle3D3<PointType>                                                    TriangleType;

typedef typename std::conditional<TDim == 2, LineType, TriangleType >::type  DecompositionType;

typedef typename std::conditional<TFrictional == FrictionalCase::FRICTIONAL || TFrictional == FrictionalCase::FRICTIONAL_PENALTY, DerivativeDataFrictional<TDim, TNumNodes, TNumNodesMaster>, DerivativeData<TDim, TNumNodes, TNumNodesMaster> >::type DerivativeDataType;

static constexpr IndexType MatrixSize = (TFrictional == FrictionalCase::FRICTIONLESS) ? TDim * (TNumNodesMaster + TNumNodes) + TNumNodes : (TFrictional == FrictionalCase::FRICTIONLESS_COMPONENTS || TFrictional == FrictionalCase::FRICTIONAL) ? TDim * (TNumNodesMaster + TNumNodes + TNumNodes) :  TDim * (TNumNodesMaster + TNumNodes);

static constexpr bool IsFrictional  = (TFrictional == FrictionalCase::FRICTIONAL || TFrictional == FrictionalCase::FRICTIONAL_PENALTY) ? true: false;

typedef MortarKinematicVariablesWithDerivatives<TDim, TNumNodes,TNumNodesMaster>                               GeneralVariables;

typedef DualLagrangeMultiplierOperatorsWithDerivatives<TDim, TNumNodes, IsFrictional, TNumNodesMaster>                   AeData;

typedef MortarOperatorWithDerivatives<TDim, TNumNodes, IsFrictional, TNumNodesMaster>                   MortarConditionMatrices;

typedef ExactMortarIntegrationUtility<TDim, TNumNodes, true, TNumNodesMaster>                                IntegrationUtility;

typedef DerivativesUtilities<TDim, TNumNodes, IsFrictional, TNormalVariation, TNumNodesMaster>         DerivativesUtilitiesType;





static MortarConditionMatrices AddExplicitContributionOfMortarCondition(
PairedCondition* pCondition,
const ProcessInfo& rCurrentProcessInfo,
const IndexType IntegrationOrder = 2,
const bool AxisymmetricCase = false,
const bool ComputeNodalArea = false,
const bool ComputeDualLM = true,
const Variable<double>& rAreaVariable = NODAL_AREA
);

static MortarConditionMatrices AddExplicitContributionOfMortarFrictionalCondition(
PairedCondition* pCondition,
const ProcessInfo& rCurrentProcessInfo,
const MortarOperator<TNumNodes, TNumNodesMaster>& rPreviousMortarOperators,
const IndexType IntegrationOrder = 2,
const bool AxisymmetricCase = false,
const bool ComputeNodalArea = false,
const bool ComputeDualLM = true,
const Variable<double>& rAreaVariable = NODAL_AREA,
const bool ConsiderObjetiveFormulation = false
);


static bool ExplicitCalculateAe(
const GeometryType& rSlaveGeometry,
GeneralVariables& rVariables,
const ConditionArrayListType& rConditionsPointsSlave,
BoundedMatrix<double, TNumNodes, TNumNodes>& rAe,
const IntegrationMethod& rIntegrationMethod,
const double AxiSymCoeff = 1.0
);


static void ExplicitCalculateKinematics(
const PairedCondition* pCondition,
GeneralVariables& rVariables,
const BoundedMatrix<double, TNumNodes, TNumNodes>& rAe,
const array_1d<double, 3>& rNormalMaster,
const PointType& rLocalPointDecomp,
const PointType& rLocalPointParent,
const GeometryPointType& rGeometryDecomp,
const bool DualLM = true
);


static void ComputeNodalArea(
PairedCondition* pCondition,
const ProcessInfo& rCurrentProcessInfo,
const Variable<double>& rAreaVariable = NODAL_AREA,
const IndexType IntegrationOrder = 2,
const bool AxisymmetricCase = false
);


static bool ComputePreviousMortarOperators(
PairedCondition* pCondition,
const ProcessInfo& rCurrentProcessInfo,
MortarOperator<TNumNodes, TNumNodesMaster>& rPreviousMortarOperators,
const IndexType IntegrationOrder = 2,
const bool AxisymmetricCase = false,
const bool ComputeNodalArea = false,
const bool ComputeDualLM = true,
const Variable<double>& rAreaVariable = NODAL_AREA
);


static void CalculateKinematics(
const PairedCondition* pCondition,
GeneralVariables& rVariables,
const DerivativeDataType& rDerivativeData,
const array_1d<double, 3>& rNormalMaster,
const PointType& rLocalPointDecomp,
const PointType& rLocalPointParent,
const GeometryPointType& rGeometryDecomp,
const bool DualLM = true
);


static void MasterShapeFunctionValue(
const PairedCondition* pCondition,
GeneralVariables& rVariables,
const array_1d<double, 3>& rNormalMaster,
const PointType& rLocalPoint
);

}; 

namespace AuxiliaryOperationsUtilities
{

double KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) GetAxisymmetricCoefficient(
const PairedCondition* pCondition,
const Vector& rNSlave
);


double KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) CalculateRadius(
const PairedCondition* pCondition,
const Vector& rNSlave
);
}

}  
