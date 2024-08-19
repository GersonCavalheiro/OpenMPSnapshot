
#pragma once



#include "includes/element.h"

namespace Kratos {

namespace StructuralMechanicsElementUtilities {

typedef std::size_t SizeType;

typedef std::size_t IndexType;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;


int SolidElementCheck(
const Element& rElement,
const ProcessInfo& rCurrentProcessInfo,
const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLaws
);


template<class TVectorType, class TMatrixType>
void ComputeEquivalentF(
const Element& rElement,
const TVectorType& rStrainTensor,
TMatrixType& rF
)
{
const auto& r_geometry = rElement.GetGeometry();
const SizeType dimension = r_geometry.WorkingSpaceDimension();

if(dimension == 2) {
rF(0,0) = 1.0+rStrainTensor(0);
rF(0,1) = 0.5*rStrainTensor(2);
rF(1,0) = 0.5*rStrainTensor(2);
rF(1,1) = 1.0+rStrainTensor(1);
} else {
rF(0,0) = 1.0+rStrainTensor(0);
rF(0,1) = 0.5*rStrainTensor(3);
rF(0,2) = 0.5*rStrainTensor(5);
rF(1,0) = 0.5*rStrainTensor(3);
rF(1,1) = 1.0+rStrainTensor(1);
rF(1,2) = 0.5*rStrainTensor(4);
rF(2,0) = 0.5*rStrainTensor(5);
rF(2,1) = 0.5*rStrainTensor(4);
rF(2,2) = 1.0+rStrainTensor(2);
}
}


template<class TMatrixType1, class TMatrixType2>
void CalculateB(
const GeometricalObject& rElement,
const TMatrixType1& rDN_DX,
TMatrixType2& rB
)
{
const auto& r_geometry = rElement.GetGeometry();
const SizeType number_of_nodes = r_geometry.PointsNumber();
const SizeType dimension = rDN_DX.size2();

rB.clear();

if(dimension == 2) {
for ( IndexType i = 0; i < number_of_nodes; ++i ) {
const IndexType initial_index = i*2;
rB(0, initial_index    ) = rDN_DX(i, 0);
rB(1, initial_index + 1) = rDN_DX(i, 1);
rB(2, initial_index    ) = rDN_DX(i, 1);
rB(2, initial_index + 1) = rDN_DX(i, 0);
}
} else if(dimension == 3) {
for ( IndexType i = 0; i < number_of_nodes; ++i ) {
const IndexType initial_index = i*3;
rB(0, initial_index    ) = rDN_DX(i, 0);
rB(1, initial_index + 1) = rDN_DX(i, 1);
rB(2, initial_index + 2) = rDN_DX(i, 2);
rB(3, initial_index    ) = rDN_DX(i, 1);
rB(3, initial_index + 1) = rDN_DX(i, 0);
rB(4, initial_index + 1) = rDN_DX(i, 2);
rB(4, initial_index + 2) = rDN_DX(i, 1);
rB(5, initial_index    ) = rDN_DX(i, 2);
rB(5, initial_index + 2) = rDN_DX(i, 0);
}
}
}


array_1d<double, 3> GetBodyForce(
const Element& rElement,
const GeometryType::IntegrationPointsArrayType& rIntegrationPoints,
const IndexType PointNumber
);


bool ComputeLumpedMassMatrix(
const Properties& rProperties,
const ProcessInfo& rCurrentProcessInfo);


bool HasRayleighDamping(
const Properties& rProperties,
const ProcessInfo& rCurrentProcessInfo);


double GetRayleighAlpha(
const Properties& rProperties,
const ProcessInfo& rCurrentProcessInfo);


double GetRayleighBeta(
const Properties& rProperties,
const ProcessInfo& rCurrentProcessInfo);


double GetDensityForMassMatrixComputation(const Element& rElement);


void KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateRayleighDampingMatrix(
Element& rElement,
Element::MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo,
const std::size_t MatrixSize);


double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateReferenceLength2D2N(const Element& rElement);


double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateCurrentLength2D2N(const Element& rElement);


double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateReferenceLength3D2N(const Element& rElement);


double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateCurrentLength3D2N(const Element& rElement);


void InitialCheckLocalAxes(
const array_1d<double, 3>& rv1,
const array_1d<double, 3>& rv2,
const array_1d<double, 3>& rv3,
const double Tolerance = 1.0e4*std::numeric_limits<double>::epsilon());


void BuildRotationMatrix(
BoundedMatrix<double, 3, 3>& rRotationMatrix,
const array_1d<double, 3>& rv1,
const array_1d<double, 3>& rv2,
const array_1d<double, 3>& rv3);

} 
}  
