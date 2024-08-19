
#pragma once 




#include "custom_conditions/base_load_condition.h"

namespace Kratos
{







template<std::size_t TDim, std::size_t TNumNodes>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION)  MovingLoadCondition
: public BaseLoadCondition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(MovingLoadCondition );


MovingLoadCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
);

MovingLoadCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

~MovingLoadCondition() override;





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



bool HasRotDof() const override;






std::string Info() const override
{
std::stringstream buffer;
buffer << "Point load Condition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Point load Condition #" << Id();
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


void CalculateExactNormalShapeFunctions(VectorType& rShapeFunctionsVector, const double LocalXCoord) const;


void CalculateExactShearShapeFunctions(VectorType& rShapeFunctionsVector, const double LocalXCoord) const;


void CalculateExactRotationalShapeFunctions(VectorType& rShapeFunctionsVector, const double LocalXCoord) const;


void CalculateRotationMatrix(BoundedMatrix<double, TDim, TDim>& rRotationMatrix, const GeometryType& rGeom);


Matrix CalculateGlobalMomentMatrix(const VectorType& RotationalShapeFunctionVector, array_1d<double, TDim> LocalMovingLoad) const;







MovingLoadCondition() {};


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





template<std::size_t TDim, std::size_t TNumNodes>
inline std::istream& operator >> (std::istream& rIStream,
MovingLoadCondition<TDim, TNumNodes>& rThis);

template<std::size_t TDim, std::size_t TNumNodes>
inline std::ostream& operator << (std::ostream& rOStream,
const MovingLoadCondition<TDim, TNumNodes>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
