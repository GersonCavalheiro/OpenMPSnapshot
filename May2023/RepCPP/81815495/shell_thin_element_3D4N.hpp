
#pragma once

#include <type_traits>


#include "custom_elements/base_shell_element.h"
#include "custom_utilities/shell_utilities.h"
#include "custom_utilities/shellq4_corotational_coordinate_transformation.hpp"
#include "custom_utilities/shellq4_local_coordinate_system.hpp"

namespace Kratos
{








template <ShellKinematics TKinematics>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ShellThinElement3D4N : public
BaseShellElement<typename std::conditional<TKinematics==ShellKinematics::NONLINEAR_COROTATIONAL,
ShellQ4_CorotationalCoordinateTransformation,
ShellQ4_CoordinateTransformation>::type>
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(ShellThinElement3D4N);

using BaseType = BaseShellElement<typename std::conditional<TKinematics==ShellKinematics::NONLINEAR_COROTATIONAL,
ShellQ4_CorotationalCoordinateTransformation,
ShellQ4_CoordinateTransformation>::type>;

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


ShellThinElement3D4N(IndexType NewId,
GeometryType::Pointer pGeometry);

ShellThinElement3D4N(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);

~ShellThinElement3D4N() override = default;




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


ShellThinElement3D4N() : BaseType()
{
}


private:

class CalculationData
{
public:


ShellQ4_LocalCoordinateSystem LCS;  
ShellQ4_LocalCoordinateSystem LCS0; 

Vector s_xi = ZeroVector(3);    
Vector s_eta = ZeroVector(3);    

array_1d<Vector, 4> r_cartesian;    
array_1d<double, 4> dA;    

VectorType globalDisplacements = ZeroVector(24); 
VectorType localDisplacements = ZeroVector(24);  

bool CalculateRHS; 
bool CalculateLHS; 


const bool basicQuad = false;    


array_1d<double, 4> N;    
SizeType gpIndex;    


const double alpha = 1.5;
MatrixType L_mem = ZeroMatrix(3, 12); 
MatrixType H_mem_mod = ZeroMatrix(7, 12);    
MatrixType Z = ZeroMatrix(12, 12);    
MatrixType B_h_1 = ZeroMatrix(3, 7);    
MatrixType B_h_2 = ZeroMatrix(3, 7);    
MatrixType B_h_3 = ZeroMatrix(3, 7);    
MatrixType B_h_4 = ZeroMatrix(3, 7);    
MatrixType B_h_bar = ZeroMatrix(3, 7);    
MatrixType T_13 = ZeroMatrix(3, 3);
MatrixType T_24 = ZeroMatrix(3, 3);

array_1d<double, 4> DKQ_a;
array_1d<double, 4> DKQ_b;
array_1d<double, 4> DKQ_c;
array_1d<double, 4> DKQ_d;
array_1d<double, 4> DKQ_e;
MatrixType DKQ_indices = ZeroMatrix(4, 2);
array_1d<Matrix, 4> DKQ_invJac;
array_1d<Matrix, 4> DKQ_jac;
array_1d<double, 4> DKQ_jac_det;


MatrixType B = ZeroMatrix(6, 24);   
MatrixType D = ZeroMatrix(6, 6);    
MatrixType BTD = ZeroMatrix(24, 6);  

VectorType generalizedStrains = ZeroVector(6);  
VectorType generalizedStresses = ZeroVector(6); 
std::vector<VectorType> rlaminateStrains;    
std::vector<VectorType> rlaminateStresses;    

ShellUtilities::JacobianOperator jacOp;
ShellCrossSection::SectionParameters SectionParameters; 

public:

const ProcessInfo& CurrentProcessInfo;

public:

CalculationData(const ShellQ4_LocalCoordinateSystem& localcoordsys,
const ShellQ4_LocalCoordinateSystem& refcoordsys,
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

void InitializeCalculationData(CalculationData& data);

void CalculateBMatrix(CalculationData& data);

void CalculateSectionResponse(CalculationData& data);

void CalculateGaussPointContribution(CalculationData& data,
MatrixType& LHS, VectorType& RHS);

void AddBodyForces(CalculationData& data,
VectorType& rRightHandSideVector); 

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
