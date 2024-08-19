
#pragma once



#include "custom_conditions/line_load_condition.h"

namespace Kratos
{







template<std::size_t TDim>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SmallDisplacementLineLoadCondition
: public LineLoadCondition<TDim>
{
public:

typedef LineLoadCondition<TDim> BaseType;

typedef typename BaseLoadCondition::VectorType VectorType;

typedef typename BaseLoadCondition::MatrixType MatrixType;

typedef typename BaseLoadCondition::IndexType IndexType;

typedef typename BaseLoadCondition::SizeType SizeType;

typedef typename BaseLoadCondition::NodeType NodeType;

typedef typename BaseLoadCondition::PropertiesType PropertiesType;

typedef typename BaseLoadCondition::GeometryType GeometryType;

typedef typename BaseLoadCondition::NodesArrayType NodesArrayType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( SmallDisplacementLineLoadCondition );


SmallDisplacementLineLoadCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
);

SmallDisplacementLineLoadCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

~SmallDisplacementLineLoadCondition() override;





Condition::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Clone (
IndexType NewId,
NodesArrayType const& ThisNodes
) const override;






std::string Info() const override
{
std::stringstream buffer;
buffer << "Small displacement line load condition #" << this->Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SmallDisplacementLineLoadCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
this->pGetGeometry()->PrintData(rOStream);
}




protected:








void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
) override;






SmallDisplacementLineLoadCondition() {};


private:







friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseType );
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseType );
}



}; 




template<std::size_t TDim>
inline std::istream& operator >> (std::istream& rIStream,
SmallDisplacementLineLoadCondition<TDim>& rThis);
template<std::size_t TDim>
inline std::ostream& operator << (std::ostream& rOStream,
const SmallDisplacementLineLoadCondition<TDim>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


