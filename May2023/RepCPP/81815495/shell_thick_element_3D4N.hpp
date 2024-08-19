

#pragma once


#include <type_traits>


#include "custom_elements/base_shell_element.h"
#include "custom_utilities/shell_utilities.h"
#include "custom_utilities/shellq4_corotational_coordinate_transformation.hpp"
#include "custom_utilities/shellq4_local_coordinate_system.hpp"

namespace Kratos
{







template <ShellKinematics TKinematics>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ShellThickElement3D4N :
public BaseShellElement<typename std::conditional<TKinematics==ShellKinematics::NONLINEAR_COROTATIONAL,
ShellQ4_CorotationalCoordinateTransformation,
ShellQ4_CoordinateTransformation>::type>
{
public:


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(ShellThickElement3D4N);

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




struct MITC4Params {

double Ax;
double Ay;
double Bx;
double By;
double Cx;
double Cy;
Matrix Transformation;
Matrix ShearStrains;

MITC4Params(const ShellQ4_LocalCoordinateSystem& LCS);

};

class EASOperator; 


class EASOperatorStorage
{

public:

friend class EASOperator;

typedef Element::GeometryType GeometryType;

public:

EASOperatorStorage();

inline void Initialize(const GeometryType& geom);

inline void InitializeSolutionStep();

inline void FinalizeSolutionStep();

inline void FinalizeNonLinearIteration(const Vector& displacementVector);

private:

array_1d<double, 5> alpha;              
array_1d<double, 5> alpha_converged;    

array_1d<double, 24> displ;             
array_1d<double, 24> displ_converged;   

array_1d<double, 5>           residual; 
BoundedMatrix<double, 5, 5>  Hinv;     
BoundedMatrix<double, 5, 24> L;        

bool mInitialized;                      

private:

friend class Serializer;

virtual void save(Serializer& rSerializer) const;

virtual void load(Serializer& rSerializer);

};


class EASOperator
{

public:


EASOperator(const ShellQ4_LocalCoordinateSystem& LCS, EASOperatorStorage& storage);

public:


inline void GaussPointComputation_Step1(double xi, double eta, const ShellUtilities::JacobianOperator& jac,
Vector& generalizedStrains,
EASOperatorStorage& storage);


inline void GaussPointComputation_Step2(const Matrix& D,
const Matrix& B,
const Vector& S,
EASOperatorStorage& storage);


inline void ComputeModfiedTangentAndResidual(Matrix& rLeftHandSideMatrix,
Vector& rRightHandSideVector,
EASOperatorStorage& storage);

private:

Matrix mF0inv;           
double mJ0;              
Vector mEnhancedStrains; 
Matrix mG;               
};



ShellThickElement3D4N(IndexType NewId,
GeometryType::Pointer pGeometry);

ShellThickElement3D4N(IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);

~ShellThickElement3D4N() override = default;





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

void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


using BaseType::CalculateOnIntegrationPoints;

void CalculateOnIntegrationPoints(const Variable<double>& rVariable,
std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable,
std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;





protected:



ShellThickElement3D4N() : BaseType()
{
}


private:


void CalculateStressesFromForceResultants(VectorType& rstresses,
const double& rthickness);

void CalculateLaminaStrains(ShellCrossSection::Pointer& section, const Vector& generalizedStrains, std::vector<VectorType>& rlaminateStrains);

void CalculateLaminaStresses(ShellCrossSection::Pointer& section, ShellCrossSection::SectionParameters parameters, const std::vector<VectorType>& rlaminateStrains, std::vector<VectorType>& rlaminateStresses);

double CalculateTsaiWuPlaneStress(const std::vector<VectorType>& rlaminateStresses, const Matrix& rLamina_Strengths, const unsigned int& rCurrent_Ply);

void CalculateVonMisesStress(const Vector& generalizedStresses,
const Variable<double>& rVariable, double& rVon_Mises_Result);

void CheckGeneralizedStressOrStrainOutput(const Variable<Matrix>& rVariable,
int& iJob, bool& bGlobal);

double CalculateStenbergShearStabilization(const ShellQ4_LocalCoordinateSystem& refCoordinateSystem, const double& meanThickness);

void CalculateBMatrix(double xi, double eta,
const ShellUtilities::JacobianOperator& Jac, const MITC4Params& params,
const Vector& N,
Matrix& B, Vector& Bdrill);

void CalculateAll(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag) override;

void AddBodyForces(const array_1d<double,4>& dA, VectorType& rRightHandSideVector);

bool TryCalculateOnIntegrationPoints_GeneralizedStrainsOrStresses(const Variable<Matrix>& rVariable,
std::vector<Matrix>& rValues,
const ProcessInfo& rCurrentProcessInfo);


ShellCrossSection::SectionBehaviorType GetSectionBehavior() const override;




EASOperatorStorage mEASStorage; 



friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;





};

}
