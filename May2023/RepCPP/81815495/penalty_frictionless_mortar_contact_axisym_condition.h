
#pragma once



#include "custom_conditions/penalty_frictionless_mortar_contact_condition.h"

namespace Kratos
{



typedef Point                                     PointType;
typedef Node                                    NodeType;
typedef Geometry<NodeType>                     GeometryType;
typedef Geometry<PointType>               GeometryPointType;
typedef GeometryData::IntegrationMethod   IntegrationMethod;





template< std::size_t TNumNodes, bool TNormalVariation >
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) PenaltyMethodFrictionlessMortarContactAxisymCondition
: public PenaltyMethodFrictionlessMortarContactCondition<2, TNumNodes, TNormalVariation>
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( PenaltyMethodFrictionlessMortarContactAxisymCondition );

typedef MortarContactCondition<2, TNumNodes, FrictionalCase::FRICTIONLESS_PENALTY, TNormalVariation> MortarBaseType;

typedef PenaltyMethodFrictionlessMortarContactCondition<2, TNumNodes, TNormalVariation>                         BaseType;

typedef typename MortarBaseType::MortarConditionMatrices                                                     MortarConditionMatrices;

typedef typename MortarBaseType::GeneralVariables                                                                   GeneralVariables;

typedef typename MortarBaseType::AeData                                                                                       AeData;

typedef Condition                                                                                                  ConditionBaseType;

typedef typename ConditionBaseType::VectorType                                                                            VectorType;

typedef typename ConditionBaseType::MatrixType                                                                            MatrixType;

typedef typename ConditionBaseType::IndexType                                                                              IndexType;

typedef typename ConditionBaseType::GeometryType::Pointer                                                        GeometryPointerType;

typedef typename ConditionBaseType::NodesArrayType                                                                    NodesArrayType;

typedef typename ConditionBaseType::PropertiesType::Pointer                                                    PropertiesPointerType;

typedef typename ConditionBaseType::EquationIdVectorType                                                        EquationIdVectorType;

typedef typename ConditionBaseType::DofsVectorType                                                                    DofsVectorType;

typedef typename std::vector<array_1d<PointType,2>>                                                           ConditionArrayListType;

typedef Line2D2<Point>                                                                                             DecompositionType;

typedef DerivativeData<2, TNumNodes>                                                                              DerivativeDataType;

static constexpr IndexType MatrixSize = 2 * (TNumNodes + TNumNodes) + TNumNodes;


PenaltyMethodFrictionlessMortarContactAxisymCondition(): BaseType()
{
}

PenaltyMethodFrictionlessMortarContactAxisymCondition(
IndexType NewId,
GeometryPointerType pGeometry
):BaseType(NewId, pGeometry)
{
}

PenaltyMethodFrictionlessMortarContactAxisymCondition(
IndexType NewId,
GeometryPointerType pGeometry,
PropertiesPointerType pProperties
):BaseType( NewId, pGeometry, pProperties )
{
}

PenaltyMethodFrictionlessMortarContactAxisymCondition(
IndexType NewId,
GeometryPointerType pGeometry,
PropertiesPointerType pProperties,
GeometryType::Pointer pMasterGeometry
):BaseType( NewId, pGeometry, pProperties, pMasterGeometry )
{
}

PenaltyMethodFrictionlessMortarContactAxisymCondition( PenaltyMethodFrictionlessMortarContactAxisymCondition const& rOther)
{
}

~PenaltyMethodFrictionlessMortarContactAxisymCondition() override;





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






bool IsAxisymmetric() const override;


double GetAxisymmetricCoefficient(const GeneralVariables& rVariables) const override;



double CalculateRadius(const GeneralVariables& rVariables) const;




std::string Info() const override
{
std::stringstream buffer;
buffer << "PenaltyMethodFrictionlessMortarContactAxisymCondition #" << this->Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PenaltyMethodFrictionlessMortarContactAxisymCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
this->GetParentGeometry().PrintData(rOStream);
this->GetPairedGeometry().PrintData(rOStream);
}



protected:







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