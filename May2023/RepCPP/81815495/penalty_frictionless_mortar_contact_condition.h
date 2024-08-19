
#pragma once



#include "custom_conditions/mortar_contact_condition.h"

namespace Kratos
{



typedef std::size_t SizeType;





template< SizeType TDim, SizeType TNumNodes, bool TNormalVariation, const SizeType TNumNodesMaster = TNumNodes >
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) PenaltyMethodFrictionlessMortarContactCondition
: public MortarContactCondition<TDim, TNumNodes, FrictionalCase::FRICTIONLESS_PENALTY, TNormalVariation, TNumNodesMaster>
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( PenaltyMethodFrictionlessMortarContactCondition );

typedef MortarContactCondition<TDim, TNumNodes, FrictionalCase::FRICTIONLESS_PENALTY, TNormalVariation, TNumNodesMaster> BaseType;

typedef typename BaseType::MortarConditionMatrices                    MortarConditionMatrices;

typedef typename BaseType::GeneralVariables                                  GeneralVariables;

typedef typename BaseType::AeData                                                      AeData;

typedef typename BaseType::IntegrationUtility                              IntegrationUtility;

typedef typename BaseType::DerivativesUtilitiesType                  DerivativesUtilitiesType;

typedef typename BaseType::BelongType                                              BelongType;

typedef typename BaseType::ConditionArrayType                              ConditionArrayType;

typedef typename BaseType::ConditionArrayListType                      ConditionArrayListType;

typedef typename BaseType::DecompositionType                                DecompositionType;

typedef typename BaseType::DerivativeDataType                              DerivativeDataType;

typedef Condition                                                           ConditionBaseType;

typedef PairedCondition                                               PairedConditionBaseType;

typedef typename ConditionBaseType::VectorType                                     VectorType;

typedef typename ConditionBaseType::MatrixType                                     MatrixType;

typedef typename ConditionBaseType::IndexType                                       IndexType;

typedef typename ConditionBaseType::GeometryType::Pointer                 GeometryPointerType;

typedef typename ConditionBaseType::NodesArrayType                             NodesArrayType;

typedef typename ConditionBaseType::PropertiesType                             PropertiesType;

typedef typename ConditionBaseType::PropertiesType::Pointer             PropertiesPointerType;

typedef typename ConditionBaseType::EquationIdVectorType                 EquationIdVectorType;

typedef typename ConditionBaseType::DofsVectorType                             DofsVectorType;

typedef Point                                                                       PointType;

typedef Node                                                                      NodeType;

typedef Geometry<NodeType>                                                       GeometryType;

typedef GeometryType::IntegrationPointsArrayType                        IntegrationPointsType;

static constexpr IndexType MatrixSize = TDim * (TNumNodes + TNumNodesMaster);


PenaltyMethodFrictionlessMortarContactCondition()
: BaseType()
{
}

PenaltyMethodFrictionlessMortarContactCondition(
IndexType NewId,
GeometryPointerType pGeometry
):BaseType(NewId, pGeometry)
{
}

PenaltyMethodFrictionlessMortarContactCondition(
IndexType NewId,
GeometryPointerType pGeometry,
PropertiesPointerType pProperties
):BaseType( NewId, pGeometry, pProperties )
{
}

PenaltyMethodFrictionlessMortarContactCondition(
IndexType NewId,
GeometryPointerType pGeometry,
PropertiesPointerType pProperties,
GeometryType::Pointer pMasterGeometry
):BaseType( NewId, pGeometry, pProperties, pMasterGeometry )
{
}

PenaltyMethodFrictionlessMortarContactCondition( PenaltyMethodFrictionlessMortarContactCondition const& rOther)
{
}

~PenaltyMethodFrictionlessMortarContactCondition() override;





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
buffer << "PenaltyMethodFrictionlessMortarContactCondition #" << this->Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PenaltyMethodFrictionlessMortarContactCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
this->GetParentGeometry().PrintData(rOStream);
this->GetPairedGeometry().PrintData(rOStream);
}



protected:









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
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node)
if (rCurrentGeometry[i_node].Is(ACTIVE) == true)
value += 1 << i_node;

return value;
}




private:








friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseType );
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseType );
}


}; 





}