
#pragma once



#include "contact_structural_mechanics_application_variables.h"
#include "custom_conditions/paired_condition.h"
#include "utilities/math_utils.h"
#include "includes/kratos_flags.h"
#include "includes/checks.h"
#include "includes/mortar_classes.h"


#include "utilities/exact_mortar_segmentation_utility.h"
#include "custom_utilities/derivatives_utilities.h"


#include "geometries/line_2d_2.h"
#include "geometries/triangle_3d_3.h"

namespace Kratos
{



typedef std::size_t SizeType;





template< const SizeType TDim, const SizeType TNumNodes, const FrictionalCase TFrictional, const bool TNormalVariation, const SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) MortarContactCondition
: public PairedCondition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( MortarContactCondition );

typedef PairedCondition                                                               BaseType;

typedef typename BaseType::VectorType                                               VectorType;

typedef typename BaseType::MatrixType                                               MatrixType;

typedef typename BaseType::IndexType                                                 IndexType;

typedef typename BaseType::GeometryType::Pointer                           GeometryPointerType;

typedef typename BaseType::NodesArrayType                                       NodesArrayType;

typedef typename BaseType::PropertiesType::Pointer                       PropertiesPointerType;

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


MortarContactCondition()
: PairedCondition()
{}

MortarContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
) :PairedCondition(NewId, pGeometry)
{}

MortarContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
) :PairedCondition( NewId, pGeometry, pProperties )
{}

MortarContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties,
GeometryType::Pointer pMasterGeometry
)
:PairedCondition( NewId, pGeometry, pProperties, pMasterGeometry)
{}

MortarContactCondition( MortarContactCondition const& rOther){}

~MortarContactCondition() override;





void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


Condition::Pointer Create(
IndexType NewId,
NodesArrayType const& rThisNodes,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties,
GeometryType::Pointer pMasterGeom
) const override;


void AddExplicitContribution(const ProcessInfo& rCurrentProcessInfo) override;


void AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3> >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;






void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetDofList(
DofsVectorType& rConditionalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;


void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable,
std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;




std::string Info() const override
{
std::stringstream buffer;
buffer << "MortarContactCondition #" << this->Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MortarContactCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
this->GetParentGeometry().PrintData(rOStream);
this->GetPairedGeometry().PrintData(rOStream);
}



protected:









void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateConditionSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& CurrentProcessInfo,
const bool ComputeLHS = true,
const bool ComputeRHS = true
);






virtual void CalculateLocalLHS(
Matrix& rLocalLHS,
const MortarConditionMatrices& rMortarConditionMatrices,
const DerivativeDataType& rDerivativeData,
const IndexType rActiveInactive,
const ProcessInfo& rCurrentProcessInfo
);


virtual void CalculateLocalRHS(
Vector& rLocalRHS,
const MortarConditionMatrices& rMortarConditionMatrices,
const DerivativeDataType& rDerivativeData,
const IndexType rActiveInactive,
const ProcessInfo& rCurrentProcessInfo
);






virtual IndexType GetActiveInactiveValue(const GeometryType& CurrentGeometry) const
{
KRATOS_ERROR << "You are calling to the base class method GetActiveInactiveValue, you are evil, and your seed must be eradicated from the face of the earth" << std::endl;

return 0;
}


bool CheckIsolatedElement(
const double DeltaTime,
const bool HalfJump = true
);


IntegrationMethod GetIntegrationMethod() const override
{
const IndexType integration_order = GetProperties().Has(INTEGRATION_ORDER_CONTACT) ? GetProperties().GetValue(INTEGRATION_ORDER_CONTACT) : 2;
switch (integration_order) {
case 1:
return GeometryData::IntegrationMethod::GI_GAUSS_1;
case 2:
return GeometryData::IntegrationMethod::GI_GAUSS_2;
case 3:
return GeometryData::IntegrationMethod::GI_GAUSS_3;
case 4:
return GeometryData::IntegrationMethod::GI_GAUSS_4;
case 5:
return GeometryData::IntegrationMethod::GI_GAUSS_5;
default:
return GeometryData::IntegrationMethod::GI_GAUSS_2;
}
}


virtual bool IsAxisymmetric() const;


virtual double GetAxisymmetricCoefficient(const GeneralVariables& rVariables) const;


virtual void ResizeLHS(MatrixType& rLeftHandSideMatrix);


virtual void ResizeRHS(VectorType& rRightHandSideVector);


virtual void ZeroLHS(MatrixType& rLeftHandSideMatrix);


virtual void ZeroRHS(VectorType& rRightHandSideVector);




private:








friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, PairedCondition );
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, PairedCondition );
}


}; 





}