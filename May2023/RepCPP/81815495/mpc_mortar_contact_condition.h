
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





template< const SizeType TDim, const SizeType TNumNodes,const SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) MPCMortarContactCondition
: public PairedCondition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( MPCMortarContactCondition );

typedef PairedCondition                                                                      BaseType;

typedef typename BaseType::VectorType                                                      VectorType;

typedef typename BaseType::MatrixType                                                      MatrixType;

typedef typename BaseType::IndexType                                                        IndexType;

typedef typename BaseType::GeometryType::Pointer                                  GeometryPointerType;

typedef typename BaseType::NodesArrayType                                              NodesArrayType;

typedef typename BaseType::PropertiesType::Pointer                              PropertiesPointerType;

typedef Point                                                                               PointType;

typedef array_1d<Point,TDim>                                                       ConditionArrayType;

typedef typename std::vector<ConditionArrayType>                               ConditionArrayListType;

typedef Node                                                                              NodeType;

typedef Geometry<NodeType>                                                               GeometryType;

typedef GeometryType::IntegrationPointsArrayType                                IntegrationPointsType;

typedef Line2D2<PointType>                                                                   LineType;

typedef Triangle3D3<PointType>                                                           TriangleType;

typedef typename std::conditional<TDim == 2, LineType, TriangleType >::type         DecompositionType;

typedef MortarKinematicVariables<TNumNodes, TNumNodesMaster>                         GeneralVariables;

typedef DualLagrangeMultiplierOperators<TNumNodes, TNumNodesMaster>                            AeData;

typedef MortarOperator<TNumNodes, TNumNodesMaster>                            MortarConditionMatrices;

typedef ExactMortarIntegrationUtility<TDim, TNumNodes, false, TNumNodesMaster>     IntegrationUtility;

typedef DerivativesUtilities<TDim, TNumNodes, false, false, TNumNodesMaster> DerivativesUtilitiesType;

static constexpr IndexType MatrixSize = TDim * (TNumNodes + TNumNodesMaster);


MPCMortarContactCondition()
: PairedCondition()
{}

MPCMortarContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
) :PairedCondition(NewId, pGeometry)
{}

MPCMortarContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
) :PairedCondition( NewId, pGeometry, pProperties )
{}

MPCMortarContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties,
GeometryType::Pointer pMasterGeometry
)
:PairedCondition( NewId, pGeometry, pProperties, pMasterGeometry)
{}

MPCMortarContactCondition( MPCMortarContactCondition const& rOther){}

~MPCMortarContactCondition() override;





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


void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetDofList(
DofsVectorType& rConditionalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;



void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(const ProcessInfo& rCurrentProcessInfo) override;






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
buffer << "MPCMortarContactCondition #" << this->Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MPCMortarContactCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
this->GetParentGeometry().PrintData(rOStream);
this->GetPairedGeometry().PrintData(rOStream);
}



protected:


bool mPreviousMortarOperatorsInitialized = false; 

MortarConditionMatrices mPreviousMortarOperators; 








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


IntegrationMethod GetIntegrationMethod() const override
{
const IndexType integration_order = GetProperties().Has(INTEGRATION_ORDER_CONTACT) ? GetProperties().GetValue(INTEGRATION_ORDER_CONTACT) : 2;
switch (integration_order) {
case 1: return GeometryData::IntegrationMethod::GI_GAUSS_1;
case 2: return GeometryData::IntegrationMethod::GI_GAUSS_2;
case 3: return GeometryData::IntegrationMethod::GI_GAUSS_3;
case 4: return GeometryData::IntegrationMethod::GI_GAUSS_4;
case 5: return GeometryData::IntegrationMethod::GI_GAUSS_5;
default: return GeometryData::IntegrationMethod::GI_GAUSS_2;
}
}




private:





void ConstraintDofDatabaseUpdate(
Matrix& rRelationMatrix,
Vector& rConstantVector,
const ProcessInfo& rCurrentProcessInfo
);


void UpdateConstraintFrictionless(
MortarConditionMatrices& rMortarConditionMatrices,
Matrix& rRelationMatrix,
Vector& rConstantVector,
const ProcessInfo& rCurrentProcessInfo,
const bool DualLM = true
);


void UpdateConstraintFrictional(
MortarConditionMatrices& rMortarConditionMatrices,
Matrix& rRelationMatrix,
Vector& rConstantVector,
const ProcessInfo& rCurrentProcessInfo,
const bool DualLM = true
);


void UpdateConstraintTying(
MortarConditionMatrices& rMortarConditionMatrices,
Matrix& rRelationMatrix,
Vector& rConstantVector,
const ProcessInfo& rCurrentProcessInfo,
const bool DualLM = true
);


void ComputePreviousMortarOperators(const ProcessInfo& rCurrentProcessInfo);





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, PairedCondition );
rSerializer.save("PreviousMortarOperators", mPreviousMortarOperators);
rSerializer.save("PreviousMortarOperatorsInitialized", mPreviousMortarOperatorsInitialized);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, PairedCondition );
rSerializer.load("PreviousMortarOperators", mPreviousMortarOperators);
rSerializer.load("PreviousMortarOperatorsInitialized", mPreviousMortarOperatorsInitialized);
}


}; 





}
