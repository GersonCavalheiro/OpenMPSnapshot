
#pragma once



#include "adjoint_semi_analytic_base_condition.h"

namespace Kratos
{






template <typename TPrimalCondition>
class AdjointSemiAnalyticPointLoadCondition
: public AdjointSemiAnalyticBaseCondition<TPrimalCondition>
{
public:

typedef AdjointSemiAnalyticBaseCondition<TPrimalCondition> BaseType;
typedef typename BaseType::SizeType SizeType;
typedef typename BaseType::IndexType IndexType;
typedef typename BaseType::GeometryType GeometryType;
typedef typename BaseType::PropertiesType PropertiesType;
typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::VectorType VectorType;
typedef typename BaseType::MatrixType MatrixType;
typedef typename BaseType::EquationIdVectorType EquationIdVectorType;
typedef typename BaseType::DofsVectorType DofsVectorType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::IntegrationMethod IntegrationMethod;
typedef typename BaseType::GeometryDataType GeometryDataType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( AdjointSemiAnalyticPointLoadCondition );


AdjointSemiAnalyticPointLoadCondition(IndexType NewId = 0)
: AdjointSemiAnalyticBaseCondition<TPrimalCondition>(NewId)
{
}

AdjointSemiAnalyticPointLoadCondition(IndexType NewId, typename GeometryType::Pointer pGeometry)
: AdjointSemiAnalyticBaseCondition<TPrimalCondition>(NewId, pGeometry)
{
}

AdjointSemiAnalyticPointLoadCondition(IndexType NewId,
typename GeometryType::Pointer pGeometry,
typename PropertiesType::Pointer pProperties)
: AdjointSemiAnalyticBaseCondition<TPrimalCondition>(NewId, pGeometry, pProperties)
{
}


Condition::Pointer Create(IndexType NewId,
NodesArrayType const& ThisNodes,
typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointSemiAnalyticPointLoadCondition<TPrimalCondition>>(
NewId, this->GetGeometry().Create(ThisNodes), pProperties);
}

Condition::Pointer Create(IndexType NewId,
typename GeometryType::Pointer pGeometry,
typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointSemiAnalyticPointLoadCondition<TPrimalCondition>>(
NewId, pGeometry, pProperties);
}

void CalculateSensitivityMatrix(const Variable<array_1d<double,3> >& rDesignVariable,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;










protected:













private:












friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, AdjointSemiAnalyticBaseCondition<TPrimalCondition>);
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, AdjointSemiAnalyticBaseCondition<TPrimalCondition> );
}



}; 





}  


