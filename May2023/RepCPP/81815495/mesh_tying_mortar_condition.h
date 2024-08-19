
#pragma once



#include "contact_structural_mechanics_application_variables.h"
#include "custom_conditions/paired_condition.h"
#include "includes/mortar_classes.h"


#include "utilities/math_utils.h"
#include "utilities/exact_mortar_segmentation_utility.h"

namespace Kratos
{



typedef std::size_t SizeType;





template< const SizeType TDim, const SizeType TNumNodesElem, const SizeType TNumNodesElemMaster = TNumNodesElem>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) MeshTyingMortarCondition
: public PairedCondition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( MeshTyingMortarCondition );

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

typedef Variable<double> Array1DComponentsType;

typedef typename std::vector<array_1d<PointType,TDim>>                  ConditionArrayListType;

typedef Line2D2<Point>                                                                LineType;

typedef Triangle3D3<Point>                                                        TriangleType;

typedef typename std::conditional<TDim == 2, LineType, TriangleType >::type  DecompositionType;

static constexpr SizeType NumNodes = (TNumNodesElem == 3 || (TDim == 2 && TNumNodesElem == 4)) ? 2 : TNumNodesElem == 4 ? 3 : 4;

static constexpr SizeType NumNodesMaster = (TNumNodesElemMaster == 3 || (TDim == 2 && TNumNodesElemMaster == 4)) ? 2 : TNumNodesElemMaster == 4 ? 3 : 4;

typedef BoundedMatrix<double, NumNodes, NumNodes>                                  MatrixDualLM;

typedef MortarKinematicVariables<NumNodes, NumNodesMaster>                     GeneralVariables;

typedef DualLagrangeMultiplierOperators<NumNodes, NumNodesMaster>                        AeData;

typedef MortarOperator<NumNodes, NumNodesMaster>                        MortarConditionMatrices;

typedef ExactMortarIntegrationUtility<TDim, NumNodes, false, NumNodesMaster> IntegrationUtility;


enum TensorValue {ScalarValue = 1, Vector2DValue = 2, Vector3DValue = 3};


MeshTyingMortarCondition()
: PairedCondition()
{}

MeshTyingMortarCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
) :PairedCondition(NewId, pGeometry)
{}

MeshTyingMortarCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
) :PairedCondition( NewId, pGeometry, pProperties )
{}

MeshTyingMortarCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties,
GeometryType::Pointer pMasterGeometry
)
:PairedCondition( NewId, pGeometry, pProperties, pMasterGeometry)
{}

MeshTyingMortarCondition( MeshTyingMortarCondition const& rOther){}

~MeshTyingMortarCondition() override;





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
buffer << "MeshTyingMortarCondition #" << this->Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MeshTyingMortarCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
this->GetParentGeometry().PrintData(rOStream);
this->GetPairedGeometry().PrintData(rOStream);
}



protected:


template< const TensorValue TTensor >
struct DofData
{
public:

typedef BoundedMatrix<double, NumNodes, TTensor>  MatrixUnknownSlave;
typedef BoundedMatrix<double, NumNodesMaster, TTensor>  MatrixUnknownMaster;

MatrixUnknownSlave LagrangeMultipliers, u1;
MatrixUnknownMaster u2;

~DofData()= default;


void Initialize(const GeometryType& rGeometryInput)
{
u1 = ZeroMatrix(NumNodes, TTensor);
u2 = ZeroMatrix(NumNodesMaster, TTensor);
LagrangeMultipliers = ZeroMatrix(NumNodes, TTensor);
}


void UpdateMasterPair(
const GeometryType& rGeometryInput,
std::vector<const Variable<double>*>& rpDoubleVariables,
std::vector<const Variable<array_1d<double, 3>>*>& rpArray1DVariables
)
{

if constexpr (TTensor == 1) {
for (IndexType i_node = 0; i_node < NumNodesMaster; ++i_node) {
u2(i_node, 0) = rGeometryInput[i_node].FastGetSolutionStepValue(*rpDoubleVariables[0]);
}
} else {
for (IndexType i_node = 0; i_node < NumNodesMaster; ++i_node) {
const array_1d<double, 3>& value = rGeometryInput[i_node].FastGetSolutionStepValue(*rpArray1DVariables[0]);
for (IndexType i_dof = 0; i_dof < TTensor; ++i_dof) {
u2(i_node, i_dof) = value[i_dof];
}
}
}
}

};


MortarConditionMatrices mrThisMortarConditionMatrices;                

std::vector<const Variable<double>*> mpDoubleVariables;               

std::vector<const Variable<array_1d<double, 3>>*> mpArray1DVariables; 








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
const ProcessInfo& rCurrentProcessInfo,
const bool ComputeLHS = true,
const bool ComputeRHS = true
);


template< const TensorValue TTensor >
void InitializeDofData(DofData<TTensor>& rDofData)
{
rDofData.Initialize(GetParentGeometry());

if constexpr (TTensor == ScalarValue) {
for (IndexType i_node = 0; i_node < NumNodes; i_node++) {
const double value = GetParentGeometry()[i_node].FastGetSolutionStepValue(*mpDoubleVariables[0]);
const double lm = GetParentGeometry()[i_node].FastGetSolutionStepValue(SCALAR_LAGRANGE_MULTIPLIER);
rDofData.u1(i_node, 0) = value;
rDofData.LagrangeMultipliers(i_node, 0) = lm;
}
} else {
for (IndexType i_node = 0; i_node < NumNodes; i_node++) {
const array_1d<double, 3>& value = GetParentGeometry()[i_node].FastGetSolutionStepValue(*mpArray1DVariables[0]);
const array_1d<double, 3>& lm = GetParentGeometry()[i_node].FastGetSolutionStepValue(VECTOR_LAGRANGE_MULTIPLIER);
for (IndexType i_dof = 0; i_dof < TDim; i_dof++) {
rDofData.u1(i_node, i_dof) = value[i_dof];
rDofData.LagrangeMultipliers(i_node, i_dof) = lm[i_dof];
}
}
}
}


bool CalculateAe(
const array_1d<double, 3>& rNormalMaster,
MatrixDualLM& rAe,
GeneralVariables& rVariables,
const ConditionArrayListType& rConditionsPointsSlave,
const IntegrationMethod ThisIntegrationMethod
);


void CalculateKinematics(
GeneralVariables& rVariables,
const MatrixDualLM& rAe,
const array_1d<double, 3>& rNormalMaster,
const PointType& rLocalPointDecomp,
const PointType& rLocalPointParent,
const GeometryPointType& rGeometryDecomp,
const bool DualLM = false
);






template< const TensorValue TTensor >
void CalculateLocalLHS(
Matrix& rLocalLHS,
const MortarConditionMatrices& rMortarConditionMatrices,
const DofData<TTensor>& rDofData
);


template< const TensorValue TTensor >
void CalculateLocalRHS(
Vector& rLocalRHS,
const MortarConditionMatrices& rMortarConditionMatrices,
const DofData<TTensor>& rDofData
);






void MasterShapeFunctionValue(
GeneralVariables& rVariables,
const array_1d<double, 3>& rNormalMaster,
const PointType& rLocalPoint
);







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