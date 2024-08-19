
#pragma once



#include "custom_conditions/surface_load_condition_3d.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SmallDisplacementSurfaceLoadCondition3D
: public SurfaceLoadCondition3D
{
public:


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( SmallDisplacementSurfaceLoadCondition3D );


SmallDisplacementSurfaceLoadCondition3D();

SmallDisplacementSurfaceLoadCondition3D(
IndexType NewId,
GeometryType::Pointer pGeometry
);

SmallDisplacementSurfaceLoadCondition3D(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

~SmallDisplacementSurfaceLoadCondition3D() override;





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
buffer << "Small displacement surface load Condition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SmallDisplacementSurfaceLoadCondition3D #" << Id();
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




private:









friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, SurfaceLoadCondition3D );
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, SurfaceLoadCondition3D );
}


}; 



} 
