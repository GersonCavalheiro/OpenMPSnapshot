
#pragma once



#include "custom_utilities/contact_utilities.h"
#include "custom_conditions/mortar_contact_condition.h"

namespace Kratos
{



typedef Point                                     PointType;
typedef Node                                    NodeType;
typedef Geometry<NodeType>                     GeometryType;
typedef Geometry<PointType>               GeometryPointType;
typedef GeometryData::IntegrationMethod   IntegrationMethod;





template< std::size_t TDim, std::size_t TNumNodes, bool TNormalVariation, std::size_t TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) PenaltyMethodFrictionalMortarContactCondition
: public MortarContactCondition<TDim, TNumNodes, FrictionalCase::FRICTIONAL_PENALTY, TNormalVariation, TNumNodesMaster>
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( PenaltyMethodFrictionalMortarContactCondition );

typedef MortarContactCondition<TDim, TNumNodes, FrictionalCase::FRICTIONAL_PENALTY, TNormalVariation, TNumNodesMaster> BaseType;

typedef Condition                                                                                             ConditionBaseType;

typedef PairedCondition                                                                                 PairedConditionBaseType;

typedef typename BaseType::MortarConditionMatrices                                                      MortarConditionMatrices;

typedef typename BaseType::GeneralVariables                                                                    GeneralVariables;

typedef typename BaseType::IntegrationUtility                                                                IntegrationUtility;

typedef typename BaseType::DerivativesUtilitiesType                                                    DerivativesUtilitiesType;

typedef typename BaseType::BelongType                                                                                BelongType;

typedef typename BaseType::ConditionArrayListType                                                        ConditionArrayListType;

typedef MortarOperator<TNumNodes, TNumNodesMaster>                                                  MortarBaseConditionMatrices;

typedef typename ConditionBaseType::VectorType                                                                       VectorType;

typedef typename ConditionBaseType::MatrixType                                                                       MatrixType;

typedef typename ConditionBaseType::IndexType                                                                         IndexType;

typedef typename ConditionBaseType::GeometryType::Pointer                                                   GeometryPointerType;

typedef typename ConditionBaseType::NodesArrayType                                                               NodesArrayType;

typedef typename ConditionBaseType::PropertiesType                                                               PropertiesType;

typedef typename ConditionBaseType::PropertiesType::Pointer                                               PropertiesPointerType;

typedef typename ConditionBaseType::EquationIdVectorType                                                   EquationIdVectorType;

typedef typename ConditionBaseType::DofsVectorType                                                               DofsVectorType;

typedef Line2D2<Point>                                                                                                 LineType;

typedef Triangle3D3<Point>                                                                                         TriangleType;

typedef typename std::conditional<TDim == 2, LineType, TriangleType >::type                                   DecompositionType;

typedef DerivativeDataFrictional<TDim, TNumNodes, TNumNodesMaster>                                           DerivativeDataType;

static constexpr IndexType MatrixSize = TDim * (TNumNodes + TNumNodesMaster);

static constexpr IndexType StepSlip = TNormalVariation ? 0 : 1;


PenaltyMethodFrictionalMortarContactCondition()
: BaseType()
{
}

PenaltyMethodFrictionalMortarContactCondition(
IndexType NewId,
GeometryPointerType pGeometry
):BaseType(NewId, pGeometry)
{
}

PenaltyMethodFrictionalMortarContactCondition(
IndexType NewId,
GeometryPointerType pGeometry,
PropertiesPointerType pProperties
):BaseType( NewId, pGeometry, pProperties )
{
}

PenaltyMethodFrictionalMortarContactCondition(
IndexType NewId,
GeometryPointerType pGeometry,
PropertiesPointerType pProperties,
GeometryType::Pointer pMasterGeometry
):BaseType( NewId, pGeometry, pProperties, pMasterGeometry )
{
}

PenaltyMethodFrictionalMortarContactCondition( PenaltyMethodFrictionalMortarContactCondition const& rOther)
{
}

~PenaltyMethodFrictionalMortarContactCondition() override;





void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


Condition::Pointer Create(
IndexType NewId,
NodesArrayType const& rThisNodes,
PropertiesPointerType pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryPointerType pGeom,
PropertiesPointerType pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryPointerType pGeom,
PropertiesPointerType pProperties,
GeometryPointerType pMasterGeom
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






static void StaticCalculateLocalRHS(
PairedCondition* pCondition,
const MortarBaseConditionMatrices& rPreviousMortarOperators,
const array_1d<double, TNumNodes>& mu,
Vector& rLocalRHS,
const MortarConditionMatrices& rMortarConditionMatrices,
const DerivativeDataType& rDerivativeData,
const IndexType rActiveInactive,
const ProcessInfo& rCurrentProcessInfo
);






void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetDofList(
DofsVectorType& rConditionalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;




std::string Info() const override
{
std::stringstream buffer;
buffer << "PenaltyMethodFrictionalMortarContactCondition #" << this->Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PenaltyMethodFrictionalMortarContactCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
this->GetParentGeometry().PrintData(rOStream);
this->GetPairedGeometry().PrintData(rOStream);
}



protected:


bool mPreviousMortarOperatorsInitialized = false;     

MortarBaseConditionMatrices mPreviousMortarOperators; 









void CalculateLocalLHS(
Matrix& rLocalLHS,
const MortarConditionMatrices& rMortarConditionMatrices,
const DerivativeDataType& rDerivativeData,
const IndexType rActiveInactive,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateLocalRHS(
Vector& rLocalRHS,
const MortarConditionMatrices& rMortarConditionMatrices,
const DerivativeDataType& rDerivativeData,
const IndexType rActiveInactive,
const ProcessInfo& rCurrentProcessInfo
) override;






IndexType GetActiveInactiveValue(const GeometryType& rCurrentGeometry) const override
{
IndexType value = 0;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
if (rCurrentGeometry[i_node].Is(ACTIVE) == true) {
if (rCurrentGeometry[i_node].Is(SLIP) == true)
value += std::pow(3, i_node);
else
value += 2 * std::pow(3, i_node);
}
}

return value;
}


array_1d<double, TNumNodes> GetFrictionCoefficient()
{
array_1d<double, TNumNodes> friction_coeffient_vector;
auto& r_geometry = this->GetParentGeometry();

for (std::size_t i_node = 0; i_node < TNumNodes; ++i_node) {
friction_coeffient_vector[i_node] = r_geometry[i_node].GetValue(FRICTION_COEFFICIENT);
}


return friction_coeffient_vector;
}




private:





void ComputePreviousMortarOperators(const ProcessInfo& rCurrentProcessInfo);


static inline BoundedMatrix<double, TNumNodes, TDim> ComputeTangentMatrixSlip(const GeometryType& rGeometry)
{
return MortarUtilities::ComputeTangentMatrix<TDim, TNumNodes>(rGeometry);
}





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseType );
rSerializer.save("PreviousMortarOperators", mPreviousMortarOperators);
rSerializer.save("PreviousMortarOperatorsInitialized", mPreviousMortarOperatorsInitialized);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseType );
rSerializer.load("PreviousMortarOperators", mPreviousMortarOperators);
rSerializer.load("PreviousMortarOperatorsInitialized", mPreviousMortarOperatorsInitialized);
}


}; 





}