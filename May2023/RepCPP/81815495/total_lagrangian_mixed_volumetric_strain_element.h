
#pragma once



#include "includes/define.h"
#include "includes/element.h"
#include "utilities/integration_utilities.h"

#include "structural_mechanics_application_variables.h"

namespace Kratos
{







template<std::size_t TDim>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TotalLagrangianMixedVolumetricStrainElement
: public Element
{

static constexpr std::size_t NumNodes = TDim + 1;
static constexpr std::size_t StrainSize = TDim == 2 ? 3 : 6;
static constexpr std::size_t BlockSize = TDim +1;
static constexpr std::size_t LocalSize = NumNodes*BlockSize;

protected:


struct KinematicVariables
{
double detF;
Matrix F;
double detJ0;
Matrix J0;
Matrix InvJ0;
Vector N;
Matrix DN_DX;
BoundedMatrix<double,NumNodes, TDim> Displacements;
BoundedVector<double,NumNodes> JacobianDeterminant;
Vector EquivalentStrain;

KinematicVariables()
{
detF = 1.0;
detJ0 = 1.0;
F = IdentityMatrix(TDim);
J0 = ZeroMatrix(TDim, TDim);
InvJ0 = ZeroMatrix(TDim, TDim);
N = ZeroVector(NumNodes);
DN_DX = ZeroMatrix(NumNodes, TDim);
Displacements = ZeroMatrix(NumNodes, TDim);
JacobianDeterminant = ZeroVector(NumNodes);
EquivalentStrain = ZeroVector(StrainSize);
}
};


struct ConstitutiveVariables
{
Vector StrainVector;
Vector StressVector;
Matrix ConstitutiveMatrix;


ConstitutiveVariables()
{
StrainVector = ZeroVector(StrainSize);
StressVector = ZeroVector(StrainSize);
ConstitutiveMatrix = ZeroMatrix(StrainSize, StrainSize);
}
};

public:


typedef ConstitutiveLaw ConstitutiveLawType;

typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

typedef ConstitutiveLawType::StressMeasure StressMeasureType;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef Node NodeType;

typedef Element BaseType;

using NodesArrayType = BaseType::NodesArrayType;

using PropertiesType = BaseType::PropertiesType;

using IndexType = BaseType::IndexType;

using SizeType = BaseType::SizeType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( TotalLagrangianMixedVolumetricStrainElement );


TotalLagrangianMixedVolumetricStrainElement()
{};

TotalLagrangianMixedVolumetricStrainElement(
IndexType NewId,
GeometryType::Pointer pGeometry)
: Element(NewId, pGeometry)
{};

TotalLagrangianMixedVolumetricStrainElement(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties)
: Element(NewId, pGeometry, pProperties)
{};

TotalLagrangianMixedVolumetricStrainElement(TotalLagrangianMixedVolumetricStrainElement const& rOther)
: BaseType(rOther)
, mThisIntegrationMethod(rOther.mThisIntegrationMethod)
, mConstitutiveLawVector(rOther.mConstitutiveLawVector)
{};

~TotalLagrangianMixedVolumetricStrainElement() override
{};




void Initialize(const ProcessInfo &rCurrentProcessInfo) override;


void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


Element::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties) const override;


Element::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties) const override;


Element::Pointer Clone(
IndexType NewId,
NodesArrayType const& rThisNodes) const override;


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo) const override;


void GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo) const override;


IntegrationMethod GetIntegrationMethod() const override
{
return mThisIntegrationMethod;
}


void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;


void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable,
std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;






const Parameters GetSpecifications() const override;

std::string Info() const override
{
std::stringstream buffer;
buffer << "Small Displacement Mixed Strain Element #" << Id() << "\nConstitutive law: " << mConstitutiveLawVector[0]->Info();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Small Displacement Mixed Strain Element #" << Id() << "\nConstitutive law: " << mConstitutiveLawVector[0]->Info();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:








private:


Vector mMinShearModulusVector; 

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


virtual bool UseElementProvidedStrain() const;


virtual void SetConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints) const;


virtual void CalculateConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const ConstitutiveLaw::StressMeasure ThisStressMeasure = ConstitutiveLaw::StressMeasure_PK2) const;


void CalculateKinematicVariables(
KinematicVariables& rThisKinematicVariables,
const IndexType PointNumber,
const GeometryType::IntegrationMethod& rIntegrationMethod) const;


void CalculateEquivalentStrain(KinematicVariables& rThisKinematicVariables) const;


void CalculateEquivalentF(KinematicVariables& rThisKinematicVariables) const;


double CalculateShearModulus(const Matrix &rC) const;


double CalculateBulkModulus(const Matrix &rC) const;


template<class TType>
void GetValueOnConstitutiveLaw(
const Variable<TType>& rVariable,
std::vector<TType>& rOutput)
{
const auto& r_geometry = GetGeometry();
const SizeType n_gauss = r_geometry.IntegrationPointsNumber(GetIntegrationMethod());

for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
mConstitutiveLawVector[i_gauss]->GetValue(rVariable, rOutput[i_gauss]);
}
}


template<class TType>
void CalculateOnConstitutiveLaw(
const Variable<TType>& rVariable,
std::vector<TType>& rOutput,
const ProcessInfo& rCurrentProcessInfo)
{
const auto& r_geometry = GetGeometry();
const SizeType n_nodes = r_geometry.size();
const SizeType dim = r_geometry.WorkingSpaceDimension();
const SizeType n_gauss = r_geometry.IntegrationPointsNumber(GetIntegrationMethod());
const auto& r_integration_points = r_geometry.IntegrationPoints(GetIntegrationMethod());

KinematicVariables kinematic_variables;
for (IndexType i_node = 0; i_node < n_nodes; ++i_node) {
const auto& r_disp = r_geometry[i_node].FastGetSolutionStepValue(DISPLACEMENT);
for (IndexType d = 0; d < dim; ++d) {
kinematic_variables.Displacements(i_node, d) = r_disp[d];
}
kinematic_variables.JacobianDeterminant[i_node] = r_geometry[i_node].FastGetSolutionStepValue(VOLUMETRIC_STRAIN);
}

ConstitutiveVariables constitutive_variables;
ConstitutiveLaw::Parameters cons_law_values(r_geometry, GetProperties(), rCurrentProcessInfo);
auto& r_cons_law_options = cons_law_values.GetOptions();
r_cons_law_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
r_cons_law_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, true);
r_cons_law_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
CalculateKinematicVariables(kinematic_variables, i_gauss, GetIntegrationMethod());

SetConstitutiveVariables(kinematic_variables, constitutive_variables, cons_law_values, i_gauss, r_integration_points);

rOutput[i_gauss] = mConstitutiveLawVector[i_gauss]->CalculateValue(cons_law_values, rVariable, rOutput[i_gauss] );
}
}




friend class Serializer;

void save( Serializer& rSerializer ) const override;

void load( Serializer& rSerializer ) override;

}; 




} 
