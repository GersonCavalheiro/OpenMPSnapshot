
#pragma once


#include <type_traits>


#include "custom_elements/base_shell_element.h"
#include "custom_utilities/shellt3_corotational_coordinate_transformation.hpp"
#include "custom_utilities/shellt3_local_coordinate_system.hpp"

namespace Kratos
{









template <ShellKinematics TKinematics>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ShellThickElement3D3N : public
BaseShellElement<typename std::conditional<TKinematics==ShellKinematics::NONLINEAR_COROTATIONAL,
ShellT3_CorotationalCoordinateTransformation,
ShellT3_CoordinateTransformation>::type>
{
public:


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(ShellThickElement3D3N);

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





ShellThickElement3D3N(IndexType NewId,
GeometryType::Pointer pGeometry);

ShellThickElement3D3N(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);

~ShellThickElement3D3N() override = default;





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



ShellThickElement3D3N() : BaseType()
{
}


private:


class CalculationData
{

public:


ShellT3_LocalCoordinateSystem LCS0; 
ShellT3_LocalCoordinateSystem LCS;  

double dA;
double hMean;
double TotalArea;

std::vector< array_1d<double, 3> > gpLocations;

MatrixType dNxy = ZeroMatrix(3, 2);  
VectorType N = ZeroVector(3); 

VectorType globalDisplacements = ZeroVector(18); 
VectorType localDisplacements = ZeroVector(18);  

bool CalculateRHS; 
bool CalculateLHS; 

const bool parabolic_composite_transverse_shear_strains = false;


const bool basicTriCST = false;    

const bool ignore_shear_stabilization = false; 

const bool smoothedDSG = false; 

const bool specialDSGc3 = false; 


SizeType gpIndex;


MatrixType B = ZeroMatrix(8, 18);   

double h_e;        
double alpha = 0.1;    
double shearStabilisation;

Matrix D = ZeroMatrix(8, 8);        

VectorType generalizedStrains = ZeroVector(8);  

VectorType generalizedStresses = ZeroVector(8); 

ShellCrossSection::SectionParameters SectionParameters; 

std::vector<VectorType> rlaminateStrains;

std::vector<VectorType> rlaminateStresses;

public:

const ProcessInfo& CurrentProcessInfo;

public:

CalculationData(const CoordinateTransformationPointerType& pCoordinateTransformation,
const ProcessInfo& rCurrentProcessInfo);

};


void CalculateStressesFromForceResultants
(VectorType& rstresses,
const double& rthickness);

void CalculateLaminaStrains(CalculationData& data);

void CalculateLaminaStresses(CalculationData& data);

double CalculateTsaiWuPlaneStress(const CalculationData& data, const Matrix& rLamina_Strengths, const unsigned int& rCurrent_Ply);

void CalculateVonMisesStress(const CalculationData& data, const Variable<double>& rVariable, double& rVon_Mises_Result);

void CalculateShellElementEnergy(const CalculationData& data, const Variable<double>& rVariable, double& rEnergy_Result);

void CheckGeneralizedStressOrStrainOutput(const Variable<Matrix>& rVariable, int& iJob, bool& bGlobal);

void CalculateSectionResponse(CalculationData& data);

void InitializeCalculationData(CalculationData& data);

void CalculateDSGc3Contribution(CalculationData& data, MatrixType& rLeftHandSideMatrix);

void CalculateSmoothedDSGBMatrix(CalculationData& data);

void CalculateDSGShearBMatrix(Matrix& shearBMatrix, const double& a, const double& b, const double& c, const double& d, const double& A);

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






friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;





};

}
