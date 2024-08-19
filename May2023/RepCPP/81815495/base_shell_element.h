
#pragma once





#include "includes/element.h"
#include "utilities/quaternion.h"
#include "custom_utilities/shell_cross_section.hpp"
#include "structural_mechanics_application_variables.h"


namespace Kratos
{



enum class ShellKinematics
{
LINEAR,
NONLINEAR_COROTATIONAL
};


template <class TCoordinateTransformation>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) BaseShellElement
: public Element
{
public:

typedef Element BaseType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(BaseShellElement);

typedef std::vector< ShellCrossSection::Pointer > CrossSectionContainerType;

typedef Quaternion<double> QuaternionType;

using CoordinateTransformationPointerType = Kratos::unique_ptr<TCoordinateTransformation>;

using SizeType = std::size_t;

using Vector3Type = array_1d<double, 3>;



BaseShellElement(IndexType NewId,
GeometryType::Pointer pGeometry);


BaseShellElement(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~BaseShellElement() override = default;





void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& CurrentProcessInfo) const override;


void GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const override;


void GetValuesVector(Vector& rValues, int Step = 0) const override;

void GetFirstDerivativesVector(Vector& rValues, int Step = 0) const override;

void GetSecondDerivativesVector(Vector& rValues, int Step = 0) const override;

void ResetConstitutiveLaw() override;

void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

void CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) override;

void CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateRightHandSide(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<array_1d<double,3> >& rVariable,
std::vector<array_1d<double, 3> >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<ConstitutiveLaw::Pointer>& rVariable,
std::vector<ConstitutiveLaw::Pointer>& rValues,
const ProcessInfo& rCurrentProcessInfo) override;

void Calculate(const Variable<Matrix >& rVariable,
Matrix& Output,
const ProcessInfo& rCurrentProcessInfo) override;



int Check(const ProcessInfo& rCurrentProcessInfo) const override;


IntegrationMethod GetIntegrationMethod() const override
{
return mIntegrationMethod;
}

void SetCrossSectionsOnIntegrationPoints(std::vector< ShellCrossSection::Pointer >& crossSections);





const Parameters GetSpecifications() const override;

virtual std::string Info() const override;

virtual void PrintInfo(std::ostream& rOStream) const override;

virtual void PrintData(std::ostream& rOStream) const override;



protected:



IntegrationMethod mIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_2;

CoordinateTransformationPointerType mpCoordinateTransformation = nullptr;

CrossSectionContainerType mSections; 



BaseShellElement() : Element()
{
}

SizeType GetNumberOfDofs() const;

SizeType GetNumberOfGPs() const;


virtual void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
);

void SetupOrientationAngles();

void CheckDofs() const;
void CheckProperties(const ProcessInfo& rCurrentProcessInfo) const;
void CheckSpecificProperties() const;


void ComputeLocalAxis(const Variable<array_1d<double, 3> >& rVariable,
std::vector<array_1d<double, 3> >& rOutput) const;


void ComputeLocalMaterialAxis(const Variable<array_1d<double, 3> >& rVariable,
std::vector<array_1d<double, 3> >& rOutput) const;

void DecimalCorrection(Vector& a);


virtual ShellCrossSection::SectionBehaviorType GetSectionBehavior() const;





private:









friend class Serializer;

void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;





}; 





} 
