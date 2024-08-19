
#pragma once



#include "contact_structural_mechanics_application_variables.h"


#include "includes/model_part.h"
#include "includes/mortar_classes.h"


#include "geometries/line_2d_2.h"
#include "geometries/triangle_3d_3.h"

namespace Kratos
{


typedef std::size_t SizeType;



template< const SizeType TDim, const SizeType TNumNodes, bool TFrictional, const bool TNormalVariation, const SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) DerivativesUtilities
{
public:

typedef std::size_t                                                                                                  IndexType;

typedef Geometry<NodeType>                                                                                        GeometryType;

typedef Geometry<NodeType>::PointsArrayType                                                                     NodesArrayType;

typedef typename std::conditional<TNumNodes == 2, PointBelongsLine2D2N, typename std::conditional<TNumNodes == 3, typename std::conditional<TNumNodesMaster == 3, PointBelongsTriangle3D3N, PointBelongsTriangle3D3NQuadrilateral3D4N>::type, typename std::conditional<TNumNodesMaster == 3, PointBelongsQuadrilateral3D4NTriangle3D3N, PointBelongsQuadrilateral3D4N>::type>::type>::type BelongType;

typedef PointBelong<TNumNodes, TNumNodesMaster>                                                                PointBelongType;

typedef Geometry<PointBelongType>                                                                      GeometryPointBelongType;

typedef array_1d<PointBelongType,TDim>                                                                      ConditionArrayType;

typedef typename std::vector<ConditionArrayType>                                                        ConditionArrayListType;

typedef Line2D2<PointType>                                                                                            LineType;

typedef Triangle3D3<PointType>                                                                                    TriangleType;

typedef typename std::conditional<TDim == 2, LineType, TriangleType >::type                                  DecompositionType;

typedef typename std::conditional<TFrictional, DerivativeDataFrictional<TDim, TNumNodes, TNumNodesMaster>, DerivativeData<TDim, TNumNodes, TNumNodesMaster> >::type DerivativeDataType;

typedef MortarKinematicVariablesWithDerivatives<TDim, TNumNodes, TNumNodesMaster>                             GeneralVariables;

typedef DualLagrangeMultiplierOperatorsWithDerivatives<TDim, TNumNodes, TFrictional, TNumNodesMaster>                   AeData;

typedef MortarOperatorWithDerivatives<TDim, TNumNodes, TFrictional, TNumNodesMaster>                   MortarConditionMatrices;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( DerivativesUtilities );








static void CalculateDeltaDetjSlave(
const DecompositionType& DecompGeom,
const GeneralVariables& rVariables,
DerivativeDataType& rDerivativeData
);


static inline array_1d<array_1d<double, 3>, TDim * TNumNodes> GPDeltaNormalSlave(
const Matrix& rJacobian,
const Matrix& rDNDe
);


static inline array_1d<array_1d<double, 3>, TDim * TNumNodesMaster> GPDeltaNormalMaster(
const Matrix& rJacobian,
const Matrix& rDNDe
);


static array_1d<array_1d<double, 3>, TDim * TNumNodes> DeltaNormalCenter(const GeometryType& rThisGeometry);


static void CalculateDeltaNormalSlave(
array_1d<BoundedMatrix<double, TNumNodes, TDim>, TNumNodes * TDim>& rDeltaNormal,
GeometryType& rThisGeometry
);


static void CalculateDeltaNormalMaster(
array_1d<BoundedMatrix<double, TNumNodesMaster, TDim>, TNumNodesMaster * TDim>& rDeltaNormal,
const GeometryType& rThisGeometry
);


static void CalculateDeltaCellVertex(
const GeneralVariables& rVariables,
DerivativeDataType& rDerivativeData,
const array_1d<BelongType, TDim>& rTheseBelongs,
const NormalDerivativesComputation ConsiderNormalVariation,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const array_1d<double, 3>& rNormal
);


static inline void CalculateDeltaN1(
const GeneralVariables& rVariables,
DerivativeDataType& rDerivativeData,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const array_1d<double, 3>& rSlaveNormal,
const DecompositionType& rDecompGeom,
const PointType& rLocalPointDecomp,
const PointType& rLocalPointParent,
const NormalDerivativesComputation ConsiderNormalVariation = NO_DERIVATIVES_COMPUTATION
);


static void CalculateDeltaN(
const GeneralVariables& rVariables,
DerivativeDataType& rDerivativeData,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const array_1d<double, 3>& rSlaveNormal,
const array_1d<double, 3>& rMasterNormal,
const DecompositionType& rDecompGeom,
const PointType& rLocalPointDecomp,
const PointType& rLocalPointParent,
const NormalDerivativesComputation ConsiderNormalVariation = NO_DERIVATIVES_COMPUTATION,
const bool DualLM = false
);


Matrix& CalculateDeltaPosition(
Matrix& rDeltaPosition,
const GeometryType& rThisGeometry,
const ConditionArrayType& rLocalCoordinates
);


static inline Matrix& CalculateDeltaPosition(
Matrix& rDeltaPosition,
const GeometryType& ThisGeometry
);


static inline void CalculateDeltaPosition(
Vector& rDeltaPosition,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const IndexType IndexNode
);


static inline void CalculateDeltaPosition(
Vector& rDeltaPosition,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const IndexType IndexNode,
const IndexType iDoF
);


static inline void CalculateDeltaPosition(
double& rDeltaPosition,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const IndexType IndexNode,
const IndexType iDoF
);


static bool CalculateAeAndDeltaAe(
const GeometryType& rSlaveGeometry,
const array_1d<double, 3>& rSlaveNormal,
const GeometryType& rMasterGeometry,
DerivativeDataType& rDerivativeData,
GeneralVariables& rVariables,
const NormalDerivativesComputation ConsiderNormalVariation,
ConditionArrayListType& rConditionsPointsSlave,
GeometryData::IntegrationMethod ThisIntegrationMethod,
const double AxiSymCoeff = 1.0
);

private:




static inline array_1d<double, 3> LocalDeltaVertex(
const array_1d<double, 3>& rNormal,
const array_1d<double, 3>& rDeltaNormal,
const IndexType iDoF,
const IndexType iBelong,
const NormalDerivativesComputation ConsiderNormalVariation,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
double Coeff = 1.0
);


static inline BoundedMatrix<double, 3, 3> ComputeRenormalizerMatrix(
const array_1d<double, 3>& rDiffVector,
const array_1d<double, 3>& rDeltaNormal
);


static inline BoundedMatrix<double, 3, 3> ComputeRenormalizerMatrix(
const BoundedMatrix<double, TNumNodes, TDim>& rDiffMatrix,
const BoundedMatrix<double, TNumNodes, TDim>& rDeltaNormal,
const IndexType iGeometry
);


static inline array_1d<double, 3> PreviousNormalGeometry(
const GeometryType& rThisGeometry,
const GeometryType::CoordinatesArrayType& rPointLocal
);


static inline void ConvertAuxHashIndex(
const IndexType AuxIndex,
IndexType& riBelongSlaveStart,
IndexType& riBelongSlaveEnd,
IndexType& riBelongMasterStart,
IndexType& riBelongMasterEnd
);


static inline void DeltaPointLocalCoordinatesSlave(
array_1d<double, 2>& rResult,
const array_1d<double, 3>& rDeltaPoint,
const Matrix& rDNDe,
const GeometryType& rThisGeometry,
const array_1d<double, 3>& rThisNormal
);

static inline void DeltaPointLocalCoordinatesMaster(
array_1d<double, 2>& rResult,
const array_1d<double, 3>& rDeltaPoint,
const Matrix& rDNDe,
const GeometryType& rThisGeometry,
const array_1d<double, 3>& rThisNormal
);


static inline double LocalDeltaSegmentN1(
const array_1d<array_1d<double, 3>, TDim * TNumNodes>& rDeltaNormal,
const array_1d<double, 3>& rSlaveNormal,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const Vector& rN1,
const Matrix& rDNDe1,
const IndexType MortarNode,
const IndexType iNode,
const IndexType iDoF,
const NormalDerivativesComputation ConsiderNormalVariation
);


static inline double LocalDeltaSegmentN2(
const array_1d<array_1d<double, 3>, TDim * TNumNodes>& rDeltaNormal,
const array_1d<double, 3>& rSlaveNormal,
const GeometryType& rSlaveGeometry,
const GeometryType& rMasterGeometry,
const Vector& rN2,
const Matrix& rDNDe2,
const IndexType MortarNode,
const IndexType iNode,
const IndexType iDoF,
const NormalDerivativesComputation ConsiderNormalVariation
);




};

}
