
#pragma once



#include "includes/element.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) IsotropicShellElement
: public Element
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( IsotropicShellElement );


IsotropicShellElement(IndexType NewId, GeometryType::Pointer pGeometry);
IsotropicShellElement(IndexType NewId, GeometryType::Pointer pGeometry,  PropertiesType::Pointer pProperties);

~IsotropicShellElement() override;






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

void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;

void CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;

void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const override;

void GetDofList(DofsVectorType& ElementalDofList, const ProcessInfo& CurrentProcessInfo) const override;

void InitializeSolutionStep(const ProcessInfo& CurrentProcessInfo) override;

void GetValuesVector(Vector& values, int Step) const override;
void GetFirstDerivativesVector(Vector& values, int Step = 0) const override;
void GetSecondDerivativesVector(Vector& values, int Step = 0) const override;

void CalculateOnIntegrationPoints(const Variable<Matrix >& rVariable, std::vector< Matrix >& Output, const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<double >& rVariable, std::vector<double>& Output, const ProcessInfo& rCurrentProcessInfo) override;

void Calculate(const Variable<Matrix >& rVariable, Matrix& Output, const ProcessInfo& rCurrentProcessInfo) override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeNonLinearIteration(const ProcessInfo& CurrentProcessInfo) override;

void CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) override;

int  Check(const ProcessInfo& rCurrentProcessInfo) const override;













protected:















private:

array_1d< BoundedMatrix<double,3,3> , 3 > mTs;
BoundedMatrix<double,3,3> mTE0;

array_1d< array_1d<double,3>, 3> rot_oldit;

double mOrientationAngle;


void CalculateLocalGlobalTransformation(
double& x12,
double& x23,
double& x31,
double& y12,
double& y23,
double& y31,
array_1d<double,3>& v1,
array_1d<double,3>& v2,
array_1d<double,3>& v3,
double& area
);

void CalculateMembraneB(
BoundedMatrix<double,9,3>& B,
const double&  beta0,
const double& loc1,
const double& loc2,
const double& loc3,
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31
);


void CalculateBendingB(
BoundedMatrix<double,9,3>& Bb,
const double& loc2,
const double& loc3,
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31
);

void CalculateMembraneContribution(
const BoundedMatrix<double,9,3>& Bm,
const BoundedMatrix<double,3,3>& Em,
BoundedMatrix<double,9,9>& Km
);


void AssembleMembraneContribution(
const BoundedMatrix<double,9,9>& Km,
const double& coeff,
BoundedMatrix<double,18,18>& Kloc_system
);

void CalculateBendingContribution(
const BoundedMatrix<double,9,3>& Bb,
const BoundedMatrix<double,3,3>& Eb,
BoundedMatrix<double,9,9>& Kb
);

void AssembleBendingContribution(
const BoundedMatrix<double,9,9>& Kb,
const double& coeff,
BoundedMatrix<double,18,18>& Kloc_system
);

void CalculateGaussPointContribution(
BoundedMatrix<double,18,18>& Kloc_system ,
const BoundedMatrix<double,3,3>& Em,
const BoundedMatrix<double,3,3>& Eb,
const double& weight,
const double& h, 
const double& loc1, 
const double& loc2,
const double& loc3,
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31
);

double CalculateBeta(
const BoundedMatrix<double,3,3>& Em
);

void CalculateMembraneElasticityTensor(
BoundedMatrix<double,3,3>& Em,
const double& h
);

void CalculateBendingElasticityTensor(
BoundedMatrix<double,3,3>& Eb,
const double& h );

void CalculateAllMatrices(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
);

Vector& CalculateVolumeForce(
Vector& rVolumeForce
);

void AddBodyForce(
const double& h,
const double& Area,
const Vector& VolumeForce,
VectorType& rRightHandSideVector
);

void RotateToGlobal(
const array_1d<double,3>& v1,
const array_1d<double,3>& v2,
const array_1d<double,3>& v3,
const BoundedMatrix<double,18,18>& Kloc_system,
Matrix& rLeftHandSideMatrix
);

void RotateToGlobal(
const array_1d<double,3>& v1,
const array_1d<double,3>& v2,
const array_1d<double,3>& v3,
const BoundedMatrix<double,18,18>& Kloc_system,
Matrix& rLeftHandSideMatrix,
VectorType& rRightHandSideVector
);



void NicePrint(const Matrix& A);

void  AddVoigtTensorComponents(
const double local_component,
array_1d<double,6>& v,
const array_1d<double,3>& a,
const array_1d<double,3>& b
);

void CalculateAndAddKg(
MatrixType& LHS,
BoundedMatrix<double,18,18>& rWorkMatrix,
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31,
const array_1d<double,3>& v1,
const array_1d<double,3>& v2,
const array_1d<double,3>& v3,
const double& A
);

void CalculateKg_GaussPointContribution(
BoundedMatrix<double,18,18>& Kloc_system ,
const BoundedMatrix<double,3,3>& Em,
const double& weight,
const double& h, 
const double& loc1, 
const double& loc2,
const double& loc3,
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31,
const array_1d<double,9>& membrane_disp
);

void CalculateLocalShapeDerivatives(
double alpha,
BoundedMatrix<double,2,9>& DNu_loc ,
BoundedMatrix<double,2,9>& DNv_loc ,
BoundedMatrix<double,2,9>& DNw_loc ,
const double& a,  
const double& b, 
const double& c, 
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31
);

void CalculateProjectionOperator(
BoundedMatrix<double,18,18>& rProjOperator,
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31
);

void ApplyProjection(
BoundedMatrix<double,18,18>& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
BoundedMatrix<double,18,18>& rWorkMatrix,
array_1d<double,18>& rWorkArray,
const BoundedMatrix<double,18,18>& rProjOperator
);

void UpdateNodalReferenceSystem(
const double& x12,
const double& x23,
const double& x31,
const double& y12,
const double& y23,
const double& y31
);

void SaveOriginalReference(
const array_1d<double,3>& v1,
const array_1d<double,3>& v2,
const array_1d<double,3>& v3
);

void CalculatePureDisplacement(
Vector& values,
const array_1d<double,3>& v1,
const array_1d<double,3>& v2,
const array_1d<double,3>& v3
);

void CalculatePureMembraneDisplacement(
array_1d<double,9>& values,
const array_1d<double,3>& v1,
const array_1d<double,3>& v2,
const array_1d<double,3>& v3
);

void CalculatePureBendingDisplacement(
array_1d<double,9>& values,
const array_1d<double,3>& v1,
const array_1d<double,3>& v2,
const array_1d<double,3>& v3
);

void InvertMatrix(
const BoundedMatrix<double,3,3>& InputMatrix,
BoundedMatrix<double,3,3>& InvertedMatrix,
double& InputMatrixDet
);

void SetupOrientationAngles();






friend class Serializer;

IsotropicShellElement() {}

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer,  Element )
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer,  Element )
}








}; 




}  


