
#pragma once



#include "includes/define.h"
#include "includes/element.h"
#include "utilities/integration_utilities.h"
#include "structural_mechanics_application_variables.h"
#include "utilities/geometrical_sensitivity_utility.h"
#include "custom_utilities/structural_mechanics_element_utilities.h"

namespace Kratos
{



typedef std::size_t IndexType;

typedef std::size_t SizeType;





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) BaseSolidElement
: public Element
{
protected:

struct KinematicVariables
{
Vector  N;
Matrix  B;
double  detF;
Matrix  F;
double  detJ0;
Matrix  J0;
Matrix  InvJ0;
Matrix  DN_DX;
Vector Displacements;


KinematicVariables(
const SizeType StrainSize,
const SizeType Dimension,
const SizeType NumberOfNodes
)
{
detF = 1.0;
detJ0 = 1.0;
N = ZeroVector(NumberOfNodes);
B = ZeroMatrix(StrainSize, Dimension * NumberOfNodes);
F = IdentityMatrix(Dimension);
DN_DX = ZeroMatrix(NumberOfNodes, Dimension);
J0 = ZeroMatrix(Dimension, Dimension);
InvJ0 = ZeroMatrix(Dimension, Dimension);
Displacements = ZeroVector(Dimension * NumberOfNodes);
}
};


struct ConstitutiveVariables
{
ConstitutiveLaw::StrainVectorType StrainVector;
ConstitutiveLaw::StressVectorType StressVector;
ConstitutiveLaw::VoigtSizeMatrixType D;


ConstitutiveVariables(const SizeType StrainSize)
{
if (StrainVector.size() != StrainSize)
StrainVector.resize(StrainSize);

if (StressVector.size() != StrainSize)
StressVector.resize(StrainSize);

if (D.size1() != StrainSize || D.size2() != StrainSize)
D.resize(StrainSize, StrainSize);

noalias(StrainVector) = ZeroVector(StrainSize);
noalias(StressVector) = ZeroVector(StrainSize);
noalias(D)            = ZeroMatrix(StrainSize, StrainSize);
}
};
public:


typedef ConstitutiveLaw ConstitutiveLawType;

typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

typedef ConstitutiveLawType::StressMeasure StressMeasureType;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef Node NodeType;

typedef Element BaseType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( BaseSolidElement );

KRATOS_DEFINE_LOCAL_FLAG(ROTATED);


BaseSolidElement()
{};

BaseSolidElement( IndexType NewId, GeometryType::Pointer pGeometry ):Element(NewId,pGeometry)
{};

BaseSolidElement( IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties ):Element(NewId,pGeometry,pProperties)
{
mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
};

BaseSolidElement(BaseSolidElement const& rOther)
:BaseType(rOther)
,mThisIntegrationMethod(rOther.mThisIntegrationMethod)
{};

~BaseSolidElement() override
{};




void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


void ResetConstitutiveLaw() override;


void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


Element::Pointer Clone (
IndexType NewId,
NodesArrayType const& rThisNodes
) const override;


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;


IntegrationMethod GetIntegrationMethod() const override
{
return mThisIntegrationMethod;
}

const virtual GeometryType::IntegrationPointsArrayType  IntegrationPoints() const
{
return GetGeometry().IntegrationPoints();
}

const virtual GeometryType::IntegrationPointsArrayType  IntegrationPoints(IntegrationMethod ThisMethod) const
{
return GetGeometry().IntegrationPoints(ThisMethod);
}

const virtual Matrix& ShapeFunctionsValues(IntegrationMethod ThisMethod) const
{
return GetGeometry().ShapeFunctionsValues(ThisMethod);
}



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


void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3> >& rDestinationVariable,
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
const Variable<Matrix>& rVariable,
std::vector<Matrix>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<ConstitutiveLaw::Pointer>& rVariable,
std::vector<ConstitutiveLaw::Pointer>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<bool>& rVariable,
const std::vector<bool>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<int>& rVariable,
const std::vector<int>& rValues,
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


void SetValuesOnIntegrationPoints(
const Variable<array_1d<double, 3>>& rVariable,
const std::vector<array_1d<double, 3>>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


void SetValuesOnIntegrationPoints(
const Variable<array_1d<double, 6>>& rVariable,
const std::vector<array_1d<double, 6>>& rValues,
const ProcessInfo& rCurrentProcessInfo
) override;


int Check( const ProcessInfo& rCurrentProcessInfo ) const override;







const Parameters GetSpecifications() const override;

std::string Info() const override
{
std::stringstream buffer;
buffer << "Base Solid Element #" << Id() << "\nConstitutive law: " << mConstitutiveLawVector[0]->Info();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Base Solid Element #" << Id() << "\nConstitutive law: " << mConstitutiveLawVector[0]->Info();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:



IntegrationMethod mThisIntegrationMethod; 

std::vector<ConstitutiveLaw::Pointer> mConstitutiveLawVector; 




void SetIntegrationMethod(const IntegrationMethod& ThisIntegrationMethod)
{
mThisIntegrationMethod = ThisIntegrationMethod;
}


void SetConstitutiveLawVector(const std::vector<ConstitutiveLaw::Pointer>& ThisConstitutiveLawVector)
{
mConstitutiveLawVector = ThisConstitutiveLawVector;
}


virtual void InitializeMaterial();


virtual ConstitutiveLaw::StressMeasure GetStressMeasure() const;


virtual bool UseElementProvidedStrain() const;


virtual void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
);


virtual void CalculateKinematicVariables(
KinematicVariables& rThisKinematicVariables,
const IndexType PointNumber,
const GeometryType::IntegrationMethod& rIntegrationMethod
);


virtual void SetConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints
);


virtual void CalculateConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const ConstitutiveLaw::StressMeasure ThisStressMeasure = ConstitutiveLaw::StressMeasure_PK2,
const bool IsElementRotated = true
);


Matrix& CalculateDeltaDisplacement(Matrix& DeltaDisplacement) const;


virtual double CalculateDerivativesOnReferenceConfiguration(
Matrix& rJ0,
Matrix& rInvJ0,
Matrix& rDN_DX,
const IndexType PointNumber,
IntegrationMethod ThisIntegrationMethod
) const;


double CalculateDerivativesOnCurrentConfiguration(
Matrix& rJ,
Matrix& rInvJ,
Matrix& rDN_DX,
const IndexType PointNumber,
IntegrationMethod ThisIntegrationMethod
) const;


virtual array_1d<double, 3> GetBodyForce(
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const IndexType PointNumber
) const;


virtual void CalculateAndAddKm(
MatrixType& rLeftHandSideMatrix,
const Matrix& B,
const Matrix& D,
const double IntegrationWeight
) const;


void CalculateAndAddKg(
MatrixType& rLeftHandSideMatrix,
const Matrix& DN_DX,
const Vector& rStressVector,
const double IntegrationWeight
) const;


virtual void CalculateAndAddResidualVector(
VectorType& rRightHandSideVector,
const KinematicVariables& rThisKinematicVariables,
const ProcessInfo& rCurrentProcessInfo,
const array_1d<double, 3>& rBodyForce,
const Vector& rStressVector,
const double IntegrationWeight
) const;


void CalculateAndAddExtForceContribution(
const Vector& rN,
const ProcessInfo& rCurrentProcessInfo,
const array_1d<double, 3>& rBodyForce,
VectorType& rRightHandSideVector,
const double IntegrationWeight
) const;


virtual double GetIntegrationWeight(
const GeometryType::IntegrationPointsArrayType& rThisIntegrationPoints,
const IndexType PointNumber,
const double detJ
) const;


void CalculateShapeGradientOfMassMatrix(MatrixType& rMassMatrix, ShapeParameter Deriv) const;


virtual bool IsElementRotated() const;


void RotateToLocalAxes(
ConstitutiveLaw::Parameters &rValues,
KinematicVariables& rThisKinematicVariables);


void RotateToGlobalAxes(
ConstitutiveLaw::Parameters &rValues,
KinematicVariables& rThisKinematicVariables);




private:





void CalculateLumpedMassVector(
VectorType& rLumpedMassVector,
const ProcessInfo& rCurrentProcessInfo
) const override;


void CalculateDampingMatrixWithLumpedMass(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
);


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


void BuildRotationSystem(
BoundedMatrix<double, 3, 3> &rRotationMatrix,
const SizeType StrainSize);


template<class TType>
void CalculateOnConstitutiveLaw(
const Variable<TType>& rVariable,
std::vector<TType>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const bool is_rotated = IsElementRotated();
const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints( this->GetIntegrationMethod() );

const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

Values.SetStrainVector(this_constitutive_variables.StrainVector);

for (IndexType point_number = 0; point_number < integration_points.size(); ++point_number) {
this->CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

this->SetConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points);

if (is_rotated)
RotateToLocalAxes(Values, this_kinematic_variables);

rOutput[point_number] = mConstitutiveLawVector[point_number]->CalculateValue( Values, rVariable, rOutput[point_number] );
}
}




friend class Serializer;

void save( Serializer& rSerializer ) const override;

void load( Serializer& rSerializer ) override;

}; 




} 
