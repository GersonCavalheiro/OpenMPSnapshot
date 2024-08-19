
#pragma once



#include "custom_elements/base_solid_element.h"
#include "structural_mechanics_application_variables.h"
#include "custom_utilities/structural_mechanics_math_utilities.hpp"

namespace Kratos
{






class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SolidShellElementSprism3D6N
: public BaseSolidElement
{
public:



KRATOS_DEFINE_LOCAL_FLAG( COMPUTE_RHS_VECTOR );
KRATOS_DEFINE_LOCAL_FLAG( COMPUTE_LHS_MATRIX );
KRATOS_DEFINE_LOCAL_FLAG( COMPUTE_RHS_VECTOR_WITH_COMPONENTS );
KRATOS_DEFINE_LOCAL_FLAG( COMPUTE_LHS_MATRIX_WITH_COMPONENTS );
KRATOS_DEFINE_LOCAL_FLAG( EAS_IMPLICIT_EXPLICIT );    
KRATOS_DEFINE_LOCAL_FLAG( TOTAL_UPDATED_LAGRANGIAN ); 
KRATOS_DEFINE_LOCAL_FLAG( QUADRATIC_ELEMENT );        
KRATOS_DEFINE_LOCAL_FLAG( EXPLICIT_RHS_COMPUTATION ); 

typedef ConstitutiveLaw ConstitutiveLawType;

typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

typedef ConstitutiveLawType::StressMeasure StressMeasureType;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef Node NodeType;

typedef BaseSolidElement BaseType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef GlobalPointersVector<NodeType> WeakPointerVectorNodesType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(SolidShellElementSprism3D6N);



enum class Configuration {INITIAL = 0, CURRENT = 1};


enum class GeometricLevel {LOWER = 0, CENTER = 5, UPPER = 9};


enum class OrthogonalBaseApproach {X = 0, Y = 1, Z = 2};



SolidShellElementSprism3D6N();


SolidShellElementSprism3D6N(IndexType NewId, GeometryType::Pointer pGeometry);


SolidShellElementSprism3D6N(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);


SolidShellElementSprism3D6N(SolidShellElementSprism3D6N const& rOther);


~SolidShellElementSprism3D6N() override;


SolidShellElementSprism3D6N& operator=(SolidShellElementSprism3D6N const& rOther);



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


Element::Pointer Clone(
IndexType NewId,
NodesArrayType const& ThisNodes
) const override;


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetValuesVector(
Vector& rValues,
int Step = 0
) const override;


void GetFirstDerivativesVector(
Vector& rValues,
int Step = 0
) const override;


void GetSecondDerivativesVector(
Vector& rValues,
int Step = 0
) const override;


void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const MatrixType& rStiffnessMatrix,
const MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo
);


void CalculateOnIntegrationPoints(
const Variable<bool>& rVariable,
std::vector<bool>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<int>& rVariable,
std::vector<int>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3>>& rVariable,
std::vector<array_1d<double, 3>>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 6>>& rVariable,
std::vector<array_1d<double, 6>>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable,
std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Matrix >& rVariable,
std::vector< Matrix >& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;





void SetValuesOnIntegrationPoints(
const Variable<double>& rVariable,
const std::vector<double>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<Vector>& rVariable,
const std::vector<Vector>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<Matrix>& rVariable,
const std::vector<Matrix>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<ConstitutiveLaw::Pointer>& rVariable,
const std::vector<ConstitutiveLaw::Pointer>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;


std::string Info() const override
{
std::stringstream buffer;
buffer << "SPRISM Element #" << Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SPRISM Element #" << Id();
}

void PrintData(std::ostream& rOStream) const override
{
GetGeometry().PrintData(rOStream);
}



void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

protected:



struct CartesianDerivatives
{


array_1d<BoundedMatrix<double, 2, 4 >, 6> InPlaneCartesianDerivativesGauss;


BoundedMatrix<double, 6, 1 > TransversalCartesianDerivativesCenter;
array_1d<BoundedMatrix<double, 6, 1 >, 6> TransversalCartesianDerivativesGauss;


BoundedMatrix<double, 2, 2 > JInvPlaneLower;
BoundedMatrix<double, 2, 2 > JInvPlaneUpper;


void clear()
{
for (IndexType i = 0; i < 6; i++) {
noalias(InPlaneCartesianDerivativesGauss[i]) = ZeroMatrix(2, 4);
noalias(TransversalCartesianDerivativesGauss[i]) = ZeroMatrix(6, 1);
}

noalias(TransversalCartesianDerivativesCenter) = ZeroMatrix(6, 1);

noalias(JInvPlaneLower) = ZeroMatrix(2, 2);
noalias(JInvPlaneUpper) = ZeroMatrix(2, 2);
}
};


struct CommonComponents
{

BoundedMatrix<double, 3, 18 > BMembraneLower; 
BoundedMatrix<double, 3, 18 > BMembraneUpper; 
BoundedMatrix<double, 2, 18 > BShearLower;    
BoundedMatrix<double, 2, 18 > BShearUpper;    
BoundedMatrix<double, 1, 18 > BNormal;         


BoundedMatrix<double, 3, 1 > CMembraneLower; 
BoundedMatrix<double, 3, 1 > CMembraneUpper; 
BoundedMatrix<double, 2, 1 > CShearLower;    
BoundedMatrix<double, 2, 1 > CShearUpper;    
double CNormal;                                                       


void clear()
{
noalias(BMembraneLower) = ZeroMatrix(3, 18);
noalias(BMembraneUpper) = ZeroMatrix(3, 18);
noalias(BShearLower)    = ZeroMatrix(2, 18);
noalias(BShearUpper)    = ZeroMatrix(2, 18);
noalias(BNormal)        = ZeroMatrix(1, 18);

noalias(CMembraneLower) = ZeroMatrix(3, 1);
noalias(CMembraneUpper) = ZeroMatrix(3, 1);
noalias(CShearLower)    = ZeroMatrix(2, 1);
noalias(CShearUpper)    = ZeroMatrix(2, 1);
CNormal                 = 0.0;
}
};


struct StressIntegratedComponents
{

array_1d<double, 3 > SMembraneLower; 
array_1d<double, 3 > SMembraneUpper; 
array_1d<double, 2 > SShearLower;    
array_1d<double, 2 > SShearUpper;    
double SNormal;                      


void clear()
{
noalias(SMembraneLower) = ZeroVector(3);
noalias(SMembraneUpper) = ZeroVector(3);
noalias(SShearLower)    = ZeroVector(2);
noalias(SShearUpper)    = ZeroVector(2);
SNormal                  = 0.0;
}
};


struct OrthogonalBase
{
array_1d<double, 3 > Vxi, Veta, Vzeta;
};


struct TransverseGradient
{
array_1d<double, 3 > F0, F1, F2;
};


struct TransverseGradientIsoParametric
{
array_1d<double, 3 > Ft, Fxi, Feta;
};


struct EASComponents
{

double mRHSAlpha;
double mStiffAlpha;
BoundedMatrix<double, 1, 36 > mHEAS;


void clear()
{
mRHSAlpha      = 0.0;
mStiffAlpha    = 0.0;
noalias(mHEAS) = ZeroMatrix(1, 36);
}
};



struct GeneralVariables
{
private:
const Matrix* pNcontainer;
const GeometryType::ShapeFunctionsGradientsType* pDN_De;

public:
StressMeasureType StressMeasure;

Matrix ConstitutiveMatrix; 
Vector StrainVector;       
Vector StressVector;       
Matrix B;                  
Matrix F;                  
Matrix F0;                 
Matrix FT;                 
double detF;               
double detF0;              
double detFT;              
Vector C ;                 
double detJ;               

Vector  N;
Matrix  DN_DX;

GeometryType::JacobiansType J;
GeometryType::JacobiansType j;


void SetShapeFunctions(const Matrix& rNcontainer)
{
pNcontainer=&rNcontainer;
}

void SetShapeFunctionsGradients(const GeometryType::ShapeFunctionsGradientsType &rDN_De)
{
pDN_De=&rDN_De;
}


const Matrix& GetShapeFunctions()
{
return *pNcontainer;
}

const GeometryType::ShapeFunctionsGradientsType& GetShapeFunctionsGradients()
{
return *pDN_De;
}
};



struct LocalSystemComponents
{
private:

MatrixType *mpLeftHandSideMatrix;
VectorType *mpRightHandSideVector;


std::vector<MatrixType> *mpLeftHandSideMatrices;
std::vector<VectorType> *mpRightHandSideVectors;


const std::vector< Variable< MatrixType > > *mpLeftHandSideVariables;


const std::vector< Variable< VectorType > > *mpRightHandSideVariables;

public:

Flags  CalculationFlags;


void SetLeftHandSideMatrix( MatrixType& rLeftHandSideMatrix ) { mpLeftHandSideMatrix = &rLeftHandSideMatrix; }
void SetLeftHandSideMatrices( std::vector<MatrixType>& rLeftHandSideMatrices ) { mpLeftHandSideMatrices = &rLeftHandSideMatrices; }
void SetLeftHandSideVariables(const std::vector< Variable< MatrixType > >& rLeftHandSideVariables ) { mpLeftHandSideVariables = &rLeftHandSideVariables; }

void SetRightHandSideVector( VectorType& rRightHandSideVector ) { mpRightHandSideVector = &rRightHandSideVector; }
void SetRightHandSideVectors( std::vector<VectorType>& rRightHandSideVectors ) { mpRightHandSideVectors = &rRightHandSideVectors; }
void SetRightHandSideVariables(const std::vector< Variable< VectorType > >& rRightHandSideVariables ) { mpRightHandSideVariables = &rRightHandSideVariables; }


MatrixType& GetLeftHandSideMatrix() { return *mpLeftHandSideMatrix; }
std::vector<MatrixType>& GetLeftHandSideMatrices() { return *mpLeftHandSideMatrices; }
const std::vector< Variable< MatrixType > >& GetLeftHandSideVariables() { return *mpLeftHandSideVariables; }

VectorType& GetRightHandSideVector() { return *mpRightHandSideVector; }
std::vector<VectorType>& GetRightHandSideVectors() { return *mpRightHandSideVectors; }
const std::vector< Variable< VectorType > >& GetRightHandSideVariables() { return *mpRightHandSideVariables; }
};





bool mFinalizedStep;


std::vector< Matrix > mAuxContainer; 


Flags  mELementalFlags;



void CalculateElementalSystem(
LocalSystemComponents& rLocalSystem,
const ProcessInfo& rCurrentProcessInfo
);


void PrintElementCalculation(
LocalSystemComponents& rLocalSystem,
GeneralVariables& rVariables
);



bool HasNeighbour(
const IndexType Index,
const NodeType& NeighbourNode
) const ;


std::size_t NumberOfActiveNeighbours(const GlobalPointersVector< NodeType >& pNeighbourNodes) const;


void GetNodalCoordinates(
BoundedMatrix<double, 12, 3 >& NodesCoord,
const GlobalPointersVector< NodeType >& pNeighbourNodes,
const Configuration ThisConfiguration
) const;


void CalculateCartesianDerivatives(CartesianDerivatives& rCartesianDerivatives);


void CalculateCommonComponents(
CommonComponents& rCommonComponents,
const CartesianDerivatives& rCartesianDerivatives
);


void CalculateLocalCoordinateSystem(
OrthogonalBase& ThisOrthogonalBase,
const OrthogonalBaseApproach ThisOrthogonalBaseApproach,
const double ThisAngle
);


void CalculateIdVector(array_1d<IndexType, 18 >& rIdVector);


void ComputeLocalDerivatives(
BoundedMatrix<double, 6, 3 >& LocalDerivativePatch,
const array_1d<double, 3>& rLocalCoordinates
);


void ComputeLocalDerivativesQuadratic(
BoundedMatrix<double, 4, 2 >& rLocalDerivativePatch,
const IndexType NodeGauss
);


void CalculateJacobianCenterGauss(
GeometryType::JacobiansType& J,
std::vector< Matrix >& Jinv,
Vector& detJ,
const IndexType rPointNumber,
const double ZetaGauss
);


void CalculateJacobian(
double& detJ,
BoundedMatrix<double, 3, 3 >& J,
BoundedMatrix<double, 6, 3 >& LocalDerivativePatch,
const BoundedMatrix<double, 12, 3 >& NodesCoord,
const array_1d<double, 3>& rLocalCoordinates
);


void CalculateJacobianAndInv(
BoundedMatrix<double, 3, 3 >& J,
BoundedMatrix<double, 3, 3 >& Jinv,
BoundedMatrix<double, 6, 3 >& LocalDerivativePatch,
const BoundedMatrix<double, 3, 6 >& NodesCoord,
const array_1d<double, 3>& rLocalCoordinates
);


void CalculateJacobianAndInv(
BoundedMatrix<double, 3, 3 >& J,
BoundedMatrix<double, 3, 3 >& Jinv,
const BoundedMatrix<double, 3, 6 >& NodesCoord,
const array_1d<double, 3>& rLocalCoordinates
);


void CalculateCartesianDerivativesOnCenterPlane(
BoundedMatrix<double, 2, 4 >& CartesianDerivativesCenter,
const OrthogonalBase& ThisOrthogonalBase,
const GeometricLevel Part
);


void CalculateCartesianDerOnGaussPlane(
BoundedMatrix<double, 2, 4 > & InPlaneCartesianDerivativesGauss,
const BoundedMatrix<double, 12, 3 > & NodesCoord,
const OrthogonalBase& ThisOrthogonalBase,
const IndexType NodeGauss,
const GeometricLevel Part
);


void CalculateCartesianDerOnGaussTrans(
BoundedMatrix<double, 6, 1 > & TransversalCartesianDerivativesGauss,
const BoundedMatrix<double, 12, 3 > & NodesCoord,
const OrthogonalBase& ThisOrthogonalBase,
const array_1d<double, 3>& rLocalCoordinates
);


void CalculateCartesianDerOnCenterTrans(
CartesianDerivatives& rCartesianDerivatives,
const BoundedMatrix<double, 12, 3 >& NodesCoord,
const OrthogonalBase& ThisOrthogonalBase,
const GeometricLevel Part
);


void CalculateInPlaneGradientFGauss(
BoundedMatrix<double, 3, 2 >& InPlaneGradientFGauss,
const BoundedMatrix<double, 2, 4 >& InPlaneCartesianDerivativesGauss,
const BoundedMatrix<double, 12, 3 >& NodesCoord,
const IndexType NodeGauss,
const GeometricLevel Part
);


void CalculateTransverseGradientF(
array_1d<double, 3 >& TransverseGradientF,
const BoundedMatrix<double, 6, 1 >& TransversalCartesianDerivativesGauss,
const BoundedMatrix<double, 12, 3 >& NodesCoord
);


void CalculateTransverseGradientFinP(
TransverseGradientIsoParametric& rTransverseGradientIsoParametric,
const BoundedMatrix<double, 12, 3 >& NodesCoord,
const GeometricLevel Part
);


void CalculateAndAddBMembrane(
BoundedMatrix<double, 3, 18 >& BMembrane,
BoundedMatrix<double, 3, 1 >& CMembrane,
const BoundedMatrix<double, 2, 4 >& InPlaneCartesianDerivativesGauss,
const BoundedMatrix<double, 3, 2 >& InPlaneGradientFGauss,
const IndexType NodeGauss
);


void CalculateAndAddMembraneKgeometric(
BoundedMatrix<double, 36, 36 >& Kgeometricmembrane,
const CartesianDerivatives& rCartesianDerivatives,
const array_1d<double, 3 >& SMembrane,
const GeometricLevel Part
);


void CalculateAndAddBShear(
BoundedMatrix<double, 2, 18 >& BShear,
BoundedMatrix<double, 2, 1 >& CShear,
const CartesianDerivatives& rCartesianDerivatives,
const TransverseGradient& rTransverseGradient,
const TransverseGradientIsoParametric& rTransverseGradientIsoParametric,
const GeometricLevel Part
);


void CalculateAndAddShearKgeometric(
BoundedMatrix<double, 36, 36 >& Kgeometricshear,
const CartesianDerivatives& rCartesianDerivatives,
const array_1d<double, 2 >& SShear,
const GeometricLevel Part
);


void CalculateAndAddBNormal(
BoundedMatrix<double, 1, 18 >& BNormal,
double& CNormal,
const BoundedMatrix<double, 6, 1 >& TransversalCartesianDerivativesGaussCenter,
const array_1d<double, 3 >& TransversalDeformationGradientF
);


void CalculateAndAddNormalKgeometric(
BoundedMatrix<double, 36, 36 >& Kgeometricnormal,
const BoundedMatrix<double, 6, 1 >& TransversalCartesianDerivativesGaussCenter,
const double SNormal
);


BoundedMatrix<double, 36, 1 > GetVectorCurrentPosition();


BoundedMatrix<double, 36, 1 > GetVectorPreviousPosition();


void IntegrateStressesInZeta(
GeneralVariables& rVariables,
StressIntegratedComponents& rIntegratedStress,
const double AlphaEAS,
const double ZetaGauss,
const double IntegrationWeight
);


void IntegrateEASInZeta(
GeneralVariables& rVariables,
EASComponents& rEAS,
const double ZetaGauss,
const double IntegrationWeight
);


void CalculateAndAddLHS(
LocalSystemComponents& rLocalSystem,
GeneralVariables& rVariables,
ConstitutiveLaw::Parameters& rValues,
const StressIntegratedComponents& rIntegratedStress,
const CommonComponents& rCommonComponents,
const CartesianDerivatives& rCartesianDerivatives,
const EASComponents& rEAS,
double& AlphaEAS
);


void CalculateAndAddRHS(
LocalSystemComponents& rLocalSystem,
GeneralVariables& rVariables,
Vector& rVolumeForce,
const StressIntegratedComponents& rIntegratedStress,
const CommonComponents& rCommonComponents,
const EASComponents& rEAS,
double& AlphaEAS
);


void CalculateAndAddKuum(
MatrixType& rLeftHandSideMatrix,
GeneralVariables& rVariables,
const double IntegrationWeight
);


void CalculateAndAddKuug(
MatrixType& rLeftHandSideMatrix,
const StressIntegratedComponents& rIntegratedStress,
const CartesianDerivatives& rCartesianDerivatives
);


void ApplyEASLHS(
MatrixType& rLeftHandSideMatrix,
const EASComponents& rEAS
);


void ApplyEASRHS(
BoundedMatrix<double, 36, 1 >& rRHSFull,
const EASComponents& rEAS,
double& AlphaEAS
);


void CalculateAndAddExternalForces(
VectorType& rRightHandSideVector,
GeneralVariables& rVariables,
Vector& rVolumeForce
);


void CalculateAndAddInternalForces(
VectorType& rRightHandSideVector,
const StressIntegratedComponents& rIntegratedStress,
const CommonComponents& rCommonComponents,
const EASComponents& rEAS,
double& AlphaEAS
);


void SetGeneralVariables(
GeneralVariables& rVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType rPointNumber
);


void InitializeSystemMatrices(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
Flags& rCalculationFlags
);


void CalculateDeltaPosition(Matrix & rDeltaPosition);


void CalculateKinematics(
GeneralVariables& rVariables,
const CommonComponents& rCommonComponents,
const GeometryType::IntegrationPointsArrayType& rIntegrationPoints,
const IndexType rPointNumber,
const double AlphaEAS,
const double ZetaGauss
);


void CbartoFbar(
GeneralVariables& rVariables,
const int rPointNumber
);


void CalculateDeformationMatrix(
Matrix& rB,
const CommonComponents& rCommonComponents,
const double ZetaGauss,
const double AlphaEAS
);


void InitializeGeneralVariables(GeneralVariables & rVariables);


void FinalizeStepVariables(
GeneralVariables & rVariables,
const IndexType rPointNumber
);


void GetHistoricalVariables(
GeneralVariables& rVariables,
const IndexType rPointNumber
);


void CalculateVolumeChange(
double& rVolumeChange,
GeneralVariables& rVariables
);


void CalculateVolumeForce(
Vector& rVolumeForce,
GeneralVariables& rVariables,
const double IntegrationWeight
);


private:



template<class TType>
void GetValueOnConstitutiveLaw(
const Variable<TType>& rVariable,
std::vector<TType>& rOutput
)
{
const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints( this->GetIntegrationMethod() );

for ( IndexType point_number = 0; point_number <integration_points.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->GetValue( rVariable,rOutput[point_number]);
}
}


template<class TType>
void CalculateOnConstitutiveLaw(
const Variable<TType>& rVariable,
std::vector<TType>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{

GeneralVariables general_variables;
this->InitializeGeneralVariables(general_variables);


ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);


Flags &ConstitutiveLawOptions = Values.GetOptions();

ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, false);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS);


const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints( this->GetIntegrationMethod() );

double& alpha_eas = this->GetValue(ALPHA_EAS);


CartesianDerivatives this_cartesian_derivatives;
this->CalculateCartesianDerivatives(this_cartesian_derivatives);


CommonComponents common_components;
common_components.clear();
this->CalculateCommonComponents(common_components, this_cartesian_derivatives);

for ( IndexType point_number = 0; point_number < integration_points.size(); ++point_number ) {
const double zeta_gauss = 2.0 * integration_points[point_number].Z() - 1.0;

this->CalculateKinematics(general_variables, common_components, integration_points, point_number, alpha_eas, zeta_gauss);

if( mFinalizedStep )
this->GetHistoricalVariables(general_variables,point_number);

this->SetGeneralVariables(general_variables,Values,point_number);

rOutput[point_number] = mConstitutiveLawVector[point_number]->CalculateValue( Values, rVariable, rOutput[point_number] );
}
}

friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
