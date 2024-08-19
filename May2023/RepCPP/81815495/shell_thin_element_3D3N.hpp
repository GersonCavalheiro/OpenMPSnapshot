
#pragma once


#include <type_traits>


#include "custom_elements/base_shell_element.h"
#include "custom_utilities/shellt3_corotational_coordinate_transformation.hpp"
#include "custom_utilities/shellt3_local_coordinate_system.hpp"

namespace Kratos
{







template <ShellKinematics TKinematics>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ShellThinElement3D3N : public
BaseShellElement<typename std::conditional<TKinematics==ShellKinematics::NONLINEAR_COROTATIONAL,
ShellT3_CorotationalCoordinateTransformation,
ShellT3_CoordinateTransformation>::type>
{
public:


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(ShellThinElement3D3N);

using BaseType = BaseShellElement<typename std::conditional<TKinematics==ShellKinematics::NONLINEAR_COROTATIONAL,
ShellT3_CorotationalCoordinateTransformation,
ShellT3_CoordinateTransformation>::type>;

typedef Quaternion<double> QuaternionType;

using GeometryType = Element::GeometryType;

using PropertiesType = Element::PropertiesType;

using NodesArrayType = Element::NodesArrayType;

using MatrixType = Element::MatrixType;

using VectorType = Element::VectorType;

using SizeType = Element::SizeType;

using Element::GetGeometry;

using Element::GetProperties;

using Vector3Type = typename BaseType::Vector3Type;

using CoordinateTransformationPointerType = typename BaseType::CoordinateTransformationPointerType;






ShellThinElement3D3N(IndexType NewId,
GeometryType::Pointer pGeometry);

ShellThinElement3D3N(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);

~ShellThinElement3D3N() override = default;





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


using BaseType::CalculateOnIntegrationPoints;

void CalculateOnIntegrationPoints(const Variable<double>& rVariable,
std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable,
std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;





protected:



ShellThinElement3D3N() : BaseType()
{
}


private:


class CalculationData
{

public:


ShellT3_LocalCoordinateSystem LCS0; 
ShellT3_LocalCoordinateSystem LCS;  

MatrixType L;

MatrixType Q1;
MatrixType Q2;
MatrixType Q3;

MatrixType Te;
MatrixType TTu;

double dA;
double hMean;
double TotalArea;
double TotalVolume;
std::vector< array_1d<double,3> > gpLocations;

MatrixType dNxy; 

VectorType globalDisplacements; 
VectorType localDisplacements;  

bool CalculateRHS; 
bool CalculateLHS; 


double beta0;
SizeType gpIndex;


MatrixType B;   
MatrixType D;   
MatrixType BTD; 

VectorType generalizedStrains;  
VectorType generalizedStresses; 
std::vector<VectorType> rlaminateStrains;	
std::vector<VectorType> rlaminateStresses;	

VectorType N; 

MatrixType Q; 
MatrixType Qh; 
MatrixType TeQ; 

VectorType H1;
VectorType H2;
VectorType H3;
VectorType H4;
MatrixType Bb;

ShellCrossSection::SectionParameters SectionParameters; 

array_1d< Vector3Type, 3 > Sig;

public:

const ProcessInfo& CurrentProcessInfo;

public:

CalculationData(const CoordinateTransformationPointerType& pCoordinateTransformation,
const ProcessInfo& rCurrentProcessInfo);

};



void CheckGeneralizedStressOrStrainOutput(const Variable<Matrix>& rVariable, int& ijob, bool& bGlobal);

void CalculateStressesFromForceResultants(VectorType& rstresses,
const double& rthickness);

void CalculateLaminaStrains(CalculationData& data);

void CalculateLaminaStresses(CalculationData& data);

double CalculateTsaiWuPlaneStress(const CalculationData& data, const Matrix& rLamina_Strengths, const unsigned int& rCurrent_Ply);

void CalculateVonMisesStress(const CalculationData& data, const Variable<double>& rVariable, double& rVon_Mises_Result);

void InitializeCalculationData(CalculationData& data);

void CalculateBMatrix(CalculationData& data);

void CalculateBeta0(CalculationData& data);

void CalculateSectionResponse(CalculationData& data);

void CalculateGaussPointContribution(CalculationData& data, MatrixType& LHS, VectorType& RHS);

void ApplyCorrectionToRHS(CalculationData& data, VectorType& RHS);

void AddBodyForces(CalculationData& data, VectorType& rRightHandSideVector);

void CalculateAll(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag) override;

bool TryCalculateOnIntegrationPoints_GeneralizedStrainsOrStresses(const Variable<Matrix>& rVariable,
std::vector<Matrix>& rValues,
const ProcessInfo& rCurrentProcessInfo);


ShellCrossSection::SectionBehaviorType GetSectionBehavior() const override;




SizeType mStrainSize = 6;



friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;





};

}
