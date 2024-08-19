
#pragma once



#include "includes/element.h"

namespace Kratos
{

class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) MembraneElement
: public Element
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(MembraneElement);

MembraneElement(IndexType NewId, GeometryType::Pointer pGeometry);

MembraneElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

~MembraneElement() = default;



enum class VoigtType {
Strain,
Stress
};

enum class ConfigurationType {
Current,
Reference
};



Element::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Element::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties
) const override;

void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo) const override;

void GetDofList(
DofsVectorType& ElementalDofList,
const ProcessInfo& rCurrentProcessInfo) const override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void GetValuesVector(
Vector& rValues,
int Step = 0) const override;

void GetFirstDerivativesVector(
Vector& rValues,
int Step = 0) const override;

void GetSecondDerivativesVector(
Vector& rValues,
int Step = 0) const override;

int Check(const ProcessInfo& rCurrentProcessInfo) const override;

void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3>>& rVariable,
std::vector<array_1d<double, 3>>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateMassMatrix(MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateConsistentMassMatrix(MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) const;

void CalculateLumpedMassVector(
VectorType& rMassVector,
const ProcessInfo& rCurrentProcessInfo) const override;

void AddExplicitContribution(
const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3>>& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override;

void AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo) override;

void Calculate(const Variable<Matrix>& rVariable,
Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

void Calculate(const Variable<double>& rVariable,
double& rOutput, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateDampingMatrix(MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

const Parameters GetSpecifications() const override;

private:

void CovariantBaseVectors(array_1d<Vector,2>& rBaseVectors,
const Matrix& rShapeFunctionGradientValues, const ConfigurationType& rConfiguration) const;


void CovariantMetric(Matrix& rMetric,const array_1d<Vector,2>& rBaseVectorCovariant);


void ContraVariantBaseVectors(array_1d<Vector,2>& rBaseVectors,const Matrix& rContraVariantMetric,
const array_1d<Vector,2> rCovariantBaseVectors);


void ContravariantMetric(Matrix& rMetric,const Matrix& rCovariantMetric);



void DeriveCurrentCovariantBaseVectors(array_1d<Vector,2>& rBaseVectors,
const Matrix& rShapeFunctionGradientValues, const SizeType DofR);



void Derivative2CurrentCovariantMetric(Matrix& rMetric,
const Matrix& rShapeFunctionGradientValues, const SizeType DofR, const SizeType DofS);



void JacobiDeterminante(double& rDetJacobi, const array_1d<Vector,2>& rReferenceBaseVectors) const;



void Derivative2StrainGreenLagrange(Vector& rStrain,
const Matrix& rShapeFunctionGradientValues, const SizeType DofR, const SizeType DofS,
const Matrix& rTransformationMatrix);




void DerivativeStrainGreenLagrange(Vector& rStrain, const Matrix& rShapeFunctionGradientValues, const SizeType DofR,
const array_1d<Vector,2> rCurrentCovariantBaseVectors, const Matrix& rTransformationMatrix);



void StrainGreenLagrange(Vector& rStrain, const Matrix& rReferenceCoVariantMetric,const Matrix& rCurrentCoVariantMetric,
const Matrix& rTransformationMatrix);


void MaterialResponse(Vector& rStress,
const Matrix& rReferenceContraVariantMetric,const Matrix& rReferenceCoVariantMetric,const Matrix& rCurrentCoVariantMetric,
const array_1d<Vector,2>& rTransformedBaseVectors, const Matrix& rTransformationMatrix, const SizeType& rIntegrationPointNumber,
Matrix& rTangentModulus,const ProcessInfo& rCurrentProcessInfo);



void AddPreStressPk2(Vector& rStress, const array_1d<Vector,2>& rTransformedBaseVectors);


void DerivativeCurrentCovariantMetric(Matrix& rMetric,
const Matrix& rShapeFunctionGradientValues, const SizeType DofR, const array_1d<Vector,2> rCurrentCovariantBaseVectors);



void InternalForces(Vector& rInternalForces,const IntegrationMethod& ThisMethod,const ProcessInfo& rCurrentProcessInfo);



void TotalStiffnessMatrix(Matrix& rStiffnessMatrix,const IntegrationMethod& ThisMethod,const ProcessInfo& rCurrentProcessInfo);



void InitialStressStiffnessMatrixEntryIJ(double& rEntryIJ,
const Vector& rStressVector,
const SizeType& rPositionI, const SizeType& rPositionJ, const Matrix& rShapeFunctionGradientValues,
const Matrix& rTransformationMatrix);




void MaterialStiffnessMatrixEntryIJ(double& rEntryIJ,
const Matrix& rMaterialTangentModulus,
const SizeType& rPositionI, const SizeType& rPositionJ, const Matrix& rShapeFunctionGradientValues,
const array_1d<Vector,2>& rCurrentCovariantBaseVectors,const Matrix& rTransformationMatrix);




void TransformStrains(Vector& rStrains, Vector& rReferenceStrains, const Matrix& rTransformationMatrix);



void TransformBaseVectors(array_1d<Vector,2>& rBaseVectors,
const array_1d<Vector,2>& rLocalBaseVectors);



template <class T>
void InPlaneTransformationMatrix(Matrix& rTransformationMatrix, const array_1d<Vector,2>& rTransformedBaseVectors,
const T& rLocalReferenceBaseVectors);



void PrincipalVector(Vector& rPrincipalVector, const Vector& rNonPrincipalVector);


void CalculateOnIntegrationPoints(const Variable<Vector >& rVariable,
std::vector< Vector >& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

void DeformationGradient(Matrix& rDeformationGradient, double& rDetDeformationGradient,
const array_1d<Vector,2>& rCurrentCovariantBase, const array_1d<Vector,2>& rReferenceContraVariantBase);


void CalculateAndAddBodyForce(VectorType& rRightHandSideVector,const ProcessInfo& rCurrentProcessInfo) const;

void ReferenceLumpingFactors(Vector& rResult) const;

std::vector<ConstitutiveLaw::Pointer> mConstitutiveLawVector; 
double CalculateReferenceArea() const;


friend class Serializer;

MembraneElement() = default;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


};	

}	
