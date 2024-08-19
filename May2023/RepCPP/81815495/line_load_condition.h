
#pragma once



#include "custom_conditions/base_load_condition.h"

namespace Kratos
{







template<std::size_t TDim>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) LineLoadCondition
: public BaseLoadCondition
{
public:

typedef BaseLoadCondition BaseType;

typedef BaseType::IndexType IndexType;

typedef BaseType::SizeType SizeType;

typedef BaseType::NodeType NodeType;

typedef BaseType::PropertiesType PropertiesType;

typedef BaseType::GeometryType GeometryType;

typedef BaseType::NodesArrayType NodesArrayType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( LineLoadCondition );


LineLoadCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
);

LineLoadCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

~LineLoadCondition() override;





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


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3>>& rVariable,
std::vector< array_1d<double, 3>>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;






std::string Info() const override
{
std::stringstream buffer;
buffer << "LineLoadCondition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "LineLoadCondition #" << Id();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}




protected:








void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
) override;


void CalculateAndSubKp(
Matrix& rK,
const array_1d<double, 3>& rTangentXi,
const Matrix& rDN_De,
const Vector& rN,
const double Pressure,
const double IntegrationWeight
) const;


void CalculateAndAddPressureForce(
VectorType& rRightHandSideVector,
const Vector& rN,
const array_1d<double, 3>& rNormal,
const double Pressure,
const double IntegrationWeight
) const;


void GetLocalAxis1(
array_1d<double, 3>& rLocalAxis,
const Matrix& rJacobian
) const;


void GetLocalAxis2(array_1d<double, 3>& rLocalAxis) const;


void GetCrossTangentMatrix(
BoundedMatrix<double, TDim, TDim>& rCrossTangentMatrix,
const array_1d<double, 3>& rTangentXi
) const;






LineLoadCondition() {};


private:







friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseLoadCondition );
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseLoadCondition );
}






}; 




template<std::size_t TDim>
inline std::istream& operator >> (std::istream& rIStream,
LineLoadCondition<TDim>& rThis);
template<std::size_t TDim>
inline std::ostream& operator << (std::ostream& rOStream,
const LineLoadCondition<TDim>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


