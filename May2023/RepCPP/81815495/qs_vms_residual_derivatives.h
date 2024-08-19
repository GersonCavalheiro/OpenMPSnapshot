
#pragma once



#include "containers/variable.h"
#include "geometries/geometry.h"
#include "geometries/geometry_data.h"
#include "includes/constitutive_law.h"
#include "includes/node.h"
#include "includes/process_info.h"
#include "includes/ublas_interface.h"
#include "utilities/time_discretization.h"

#include "custom_utilities/fluid_element_utilities.h"
#include "custom_utilities/fluid_calculation_utilities.h"
#include "custom_elements/data_containers/qs_vms/qs_vms_derivative_utilities.h"

namespace Kratos
{


template <unsigned int TDim, unsigned int TNumNodes>
class QSVMSResidualDerivatives
{
public:

using IndexType = std::size_t;

using NodeType = Node;

using GeometryType = Geometry<NodeType>;

using PropertiesType = typename Element::PropertiesType;

static constexpr IndexType TBlockSize = TDim + 1;

constexpr static IndexType TStrainSize = (TDim - 1) * 3; 

constexpr static IndexType TElementLocalSize = TBlockSize * TNumNodes;

constexpr static IndexType TNN = TNumNodes;

using ArrayD = array_1d<double, TDim>;

using VectorN = BoundedVector<double, TNumNodes>;

using VectorF = BoundedVector<double, TElementLocalSize>;

using MatrixDD = BoundedMatrix<double, TDim, TDim>;

using MatrixND = BoundedMatrix<double, TNumNodes, TDim>;


static void Check(
const Element& rElement,
const ProcessInfo& rProcessInfo);

static GeometryData::IntegrationMethod GetIntegrationMethod();



class QSVMSResidualData;



class ResidualsContributions
{
public:

constexpr static IndexType NumNodes = TNumNodes;

constexpr static IndexType BlockSize = TBlockSize;


void AddGaussPointResidualsContributions(
VectorF& rResidual,
QSVMSResidualData& rData,
const double W,
const Vector& rN,
const Matrix& rdNdX) const;


private:

void static AddViscousTerms(
QSVMSResidualData& rData,
VectorF& rResidual,
const double W);

};


template<class TDerivativesType>
class VariableDerivatives
{
public:

constexpr static IndexType NumNodes = TNumNodes;

constexpr static IndexType BlockSize = TBlockSize;

constexpr static IndexType TDerivativeDimension = TDerivativesType::TDerivativeDimension;

constexpr static IndexType ComponentIndex = TDerivativesType::ComponentIndex;



void CalculateGaussPointResidualsDerivativeContributions(
VectorF& rResidualDerivative,
QSVMSResidualData& rData,
const int NodeIndex,
const double W,
const Vector& rN,
const Matrix& rdNdX,
const double WDerivative,
const double DetJDerivative,
const Matrix& rdNdXDerivative,
const double MassTermsDerivativesWeight = 1.0) const
{
rResidualDerivative.clear();

constexpr double u_factor = TDerivativesType::VelocityDerivativeFactor;
constexpr double p_factor = TDerivativesType::PressureDerivativeFactor;

const auto& r_geometry = rData.mpElement->GetGeometry();

const TDerivativesType derivatives_type(
NodeIndex, r_geometry, W, rN, rdNdX,
WDerivative, DetJDerivative, rdNdXDerivative);

const auto& velocity_derivative = derivatives_type.CalculateEffectiveVelocityDerivative(rData.mVelocity);
const auto& element_length_derivative = derivatives_type.CalculateElementLengthDerivative(rData.mElementSize);
derivatives_type.CalculateStrainRateDerivative(rData.mStrainRateDerivative, rData.mNodalVelocity);

double effective_viscosity_derivative;

rData.mpConstitutiveLaw->CalculateDerivative(rData.mConstitutiveLawValues, EFFECTIVE_VISCOSITY, derivatives_type.GetDerivativeVariable(), effective_viscosity_derivative);
effective_viscosity_derivative *= rN[NodeIndex];

double effective_viscosity_derivative_value;
ArrayD derivative_variable_gradient;
const auto& r_effective_viscosity_dependent_variables = derivatives_type.GetEffectiveViscosityDependentVariables();
for (const auto& r_effective_viscosity_dependent_variable : r_effective_viscosity_dependent_variables) {
const auto& r_derivative_variable = std::get<0>(r_effective_viscosity_dependent_variable);
FluidCalculationUtilities::EvaluateGradientInPoint(rData.mpElement->GetGeometry(), rdNdXDerivative, std::tie(derivative_variable_gradient, r_derivative_variable));

const auto& r_effective_viscosity_dependent_variable_gradient_component_list = std::get<1>(r_effective_viscosity_dependent_variable);
for (IndexType i = 0; i < TDim; ++i) {
rData.mpConstitutiveLaw->CalculateDerivative(rData.mConstitutiveLawValues, EFFECTIVE_VISCOSITY, *r_effective_viscosity_dependent_variable_gradient_component_list[i], effective_viscosity_derivative_value);
effective_viscosity_derivative += effective_viscosity_derivative_value * (rdNdX(NodeIndex, i) * (r_derivative_variable ==  derivatives_type.GetDerivativeVariable()));
effective_viscosity_derivative += effective_viscosity_derivative_value * (derivative_variable_gradient[i]);
}
}

const double velocity_norm_derivative = CalculateNormDerivative(
rData.mConvectiveVelocityNorm, rData.mConvectiveVelocity, velocity_derivative);

double tau_one_derivative, tau_two_derivative;
CalculateTauDerivative(
tau_one_derivative, tau_two_derivative, rData.mTauOne,
rData.mDensity, rData.mDynamicTau, rData.mDeltaTime,
rData.mElementSize, element_length_derivative,
rData.mEffectiveViscosity, effective_viscosity_derivative,
rData.mConvectiveVelocityNorm, velocity_norm_derivative);

ArrayD pressure_gradient_derivative;
BoundedMatrix<double, TDim, TDim> velocity_gradient_derivative;
FluidCalculationUtilities::EvaluateGradientInPoint(
r_geometry, rdNdXDerivative,
std::tie(pressure_gradient_derivative, PRESSURE),
std::tie(velocity_gradient_derivative, VELOCITY));
for (IndexType l = 0; l < TDim; ++l) {
pressure_gradient_derivative[l] += rdNdX(NodeIndex, l) * p_factor;
}
row(velocity_gradient_derivative, ComponentIndex) += row(rdNdX, NodeIndex) * u_factor;

double velocity_dot_nabla_derivative = 0.0;
for (IndexType a = 0; a < TNumNodes; ++a) {
for (IndexType i = 0; i < TDim; ++i) {
velocity_dot_nabla_derivative += rData.mNodalVelocity(a, i) * rdNdXDerivative(a, i);
}
}

const ArrayD effective_velocity_derivative_dot_velocity_gradient = prod(rData.mVelocityGradient, velocity_derivative);
const ArrayD effective_velocity_dot_velocity_gradient_derivative = prod(velocity_gradient_derivative, rData.mConvectiveVelocity);
const VectorN convective_velocity_derivative_dot_dn_dx = prod(rdNdX, velocity_derivative);
const VectorN relaxed_acceleration_dot_dn_dx_derivative = prod(rdNdXDerivative, rData.mRelaxedAcceleration);
const VectorN convective_velocity_dot_dn_dx_derivative = prod(rdNdXDerivative, rData.mConvectiveVelocity);
const VectorN effective_velocity_derivative_dot_velocity_gradient_dot_shape_gradient = prod(rdNdX, effective_velocity_derivative_dot_velocity_gradient);
const VectorN effective_velocity_dot_velocity_gradient_derivative_dot_shape_gradient = prod(rdNdX, effective_velocity_dot_velocity_gradient_derivative);
const VectorN effective_velocity_dot_velocity_gradient_dot_shape_gradient_derivative = prod(rdNdXDerivative, rData.mEffectiveVelocityDotVelocityGradient);
const VectorN pressure_gradient_derivative_dot_shape_gradient = prod(rdNdX, pressure_gradient_derivative);
const VectorN pressure_gradient_dot_shape_gradient_derivative = prod(rdNdXDerivative, rData.mPressureGradient);

const ArrayD momentum_projection_derivative = ZeroVector(TDim);
const double mass_projection_derivative = 0.0;

for (IndexType a = 0; a < TNumNodes; ++a) {
const IndexType row = a * TBlockSize;

double forcing_derivative = 0.0;
for (IndexType i = 0; i < TDim; ++i) {

double value = 0.0;

value += rData.mDensity * W * tau_one_derivative * rData.mConvectiveVelocityDotDnDx[a] * rData.mBodyForce[i];
value += rData.mDensity * W * rData.mTauOne * convective_velocity_derivative_dot_dn_dx[a] * rData.mBodyForce[i];
value += rData.mDensity * W * rData.mTauOne * convective_velocity_dot_dn_dx_derivative[a] * rData.mBodyForce[i];

value -= rData.mDensity * W * tau_one_derivative * rData.mConvectiveVelocityDotDnDx[a] * rData.mMomentumProjection[i];
value -= rData.mDensity * W * rData.mTauOne * convective_velocity_derivative_dot_dn_dx[a] * rData.mMomentumProjection[i];
value -= rData.mDensity * W * rData.mTauOne * convective_velocity_dot_dn_dx_derivative[a] * rData.mMomentumProjection[i];
value -= rData.mDensity * W * rData.mTauOne * rData.mConvectiveVelocityDotDnDx[a] * momentum_projection_derivative[i];

value -= W * tau_two_derivative * rdNdX(a, i) * rData.mMassProjection;
value -= W * rData.mTauTwo * rdNdXDerivative(a, i) * rData.mMassProjection;
value -= W * rData.mTauTwo * rdNdX(a, i) * mass_projection_derivative;

forcing_derivative += rdNdXDerivative(a, i) * rData.mBodyForce[i];
forcing_derivative -= rdNdXDerivative(a, i) * rData.mMomentumProjection[i];
forcing_derivative -= rdNdX(a, i) * momentum_projection_derivative[i];

value -= W * rData.mDensity * rN[a] * effective_velocity_derivative_dot_velocity_gradient[i];
value -= W * rData.mDensity * rN[a] * effective_velocity_dot_velocity_gradient_derivative[i];

value -= W * rData.mDensity * convective_velocity_derivative_dot_dn_dx[a] * rData.mTauOne * rData.mDensity * rData.mEffectiveVelocityDotVelocityGradient[i];
value -= W * rData.mDensity * convective_velocity_dot_dn_dx_derivative[a] * rData.mTauOne * rData.mDensity * rData.mEffectiveVelocityDotVelocityGradient[i];
value -= W * rData.mDensity * rData.mConvectiveVelocityDotDnDx[a] * tau_one_derivative * rData.mDensity * rData.mEffectiveVelocityDotVelocityGradient[i];
value -= W * rData.mDensity * rData.mConvectiveVelocityDotDnDx[a] * rData.mTauOne * rData.mDensity * effective_velocity_derivative_dot_velocity_gradient[i];
value -= W * rData.mDensity * rData.mConvectiveVelocityDotDnDx[a] * rData.mTauOne * rData.mDensity * effective_velocity_dot_velocity_gradient_derivative[i];

value -= W * tau_one_derivative * rData.mDensity * rData.mConvectiveVelocityDotDnDx[a] * rData.mPressureGradient[i];
value -= W * rData.mTauOne * rData.mDensity * convective_velocity_derivative_dot_dn_dx[a] * rData.mPressureGradient[i];
value -= W * rData.mTauOne * rData.mDensity * convective_velocity_dot_dn_dx_derivative[a] * rData.mPressureGradient[i];
value -= W * rData.mTauOne * rData.mDensity * rData.mConvectiveVelocityDotDnDx[a] * pressure_gradient_derivative[i];

value += W * rdNdXDerivative(a, i) * rData.mPressure;
value += W * rdNdX(a, i) * rN[NodeIndex] * p_factor;

value -= W * tau_two_derivative * rdNdX(a, i) * rData.mVelocityDotNabla;
value -= W * rData.mTauTwo * rdNdXDerivative(a, i) * rData.mVelocityDotNabla;
value -= W * rData.mTauTwo * rdNdX(a, i) * velocity_dot_nabla_derivative;
value -= W * rData.mTauTwo * rdNdX(a, i) * rdNdX(NodeIndex, ComponentIndex) * u_factor;


value -= W * tau_one_derivative * rData.mDensity * rData.mDensity * rData.mConvectiveVelocityDotDnDx[a] * rData.mRelaxedAcceleration[i] * MassTermsDerivativesWeight;
value -= W * rData.mTauOne * rData.mDensity * rData.mDensity * convective_velocity_derivative_dot_dn_dx[a] * rData.mRelaxedAcceleration[i] * MassTermsDerivativesWeight;
value -= W * rData.mTauOne * rData.mDensity * rData.mDensity * convective_velocity_dot_dn_dx_derivative[a] * rData.mRelaxedAcceleration[i] * MassTermsDerivativesWeight;

rResidualDerivative[row + i] += value;
}

double value = 0.0;

const double forcing = rData.mBodyForceDotDnDx[a] - rData.mMomentumProjectionDotDnDx[a];
value += W * tau_one_derivative * forcing;
value += W * rData.mTauOne * forcing_derivative;

value -= W * tau_one_derivative * rData.mDensity * rData.mEffectiveVelocityDotVelocityGradientDotShapeGradient[a];
value -= W * rData.mTauOne * rData.mDensity * effective_velocity_derivative_dot_velocity_gradient_dot_shape_gradient[a];
value -= W * rData.mTauOne * rData.mDensity * effective_velocity_dot_velocity_gradient_derivative_dot_shape_gradient[a];
value -= W * rData.mTauOne * rData.mDensity * effective_velocity_dot_velocity_gradient_dot_shape_gradient_derivative[a];

value -= W * rN[a] * velocity_dot_nabla_derivative;
value -= W * rN[a] * rdNdX(NodeIndex, ComponentIndex) * u_factor;

value -= W * tau_one_derivative * rData.mPressureGradientDotDnDx[a];
value -= W * rData.mTauOne * pressure_gradient_derivative_dot_shape_gradient[a];
value -= W * rData.mTauOne * pressure_gradient_dot_shape_gradient_derivative[a];

value -=  W * tau_one_derivative * rData.mDensity * rData.mRelaxedAccelerationDotDnDx[a] * MassTermsDerivativesWeight;
value -=  W * rData.mTauOne * rData.mDensity * relaxed_acceleration_dot_dn_dx_derivative[a] * MassTermsDerivativesWeight;

rResidualDerivative[row + TDim] += value;
}

mResidualsContributions.AddGaussPointResidualsContributions(
rResidualDerivative, rData, WDerivative, rN, rdNdX);

const auto& r_strain_rate_variables = QSVMSDerivativeUtilities<TDim>::GetStrainRateVariables();
Vector value;
rData.mShearStressDerivative.clear();

for (IndexType i = 0; i < TStrainSize; ++i) {
rData.mpConstitutiveLaw->CalculateDerivative(rData.mConstitutiveLawValues, CAUCHY_STRESS_VECTOR, *r_strain_rate_variables[i], value);
noalias(rData.mShearStressDerivative) += value * rData.mStrainRateDerivative[i];
}
rData.mpConstitutiveLaw->CalculateDerivative(rData.mConstitutiveLawValues, CAUCHY_STRESS_VECTOR, EFFECTIVE_VISCOSITY, value);
noalias(rData.mShearStressDerivative) += value * effective_viscosity_derivative;

AddViscousDerivative(rData, rResidualDerivative, NodeIndex,
W, rN, rdNdX, WDerivative,
DetJDerivative, rdNdXDerivative);
}


private:

ResidualsContributions mResidualsContributions;



void static AddViscousDerivative(
QSVMSResidualData& rData,
VectorF& rResidualDerivative,
const int NodeIndex,
const double W,
const Vector& rN,
const Matrix& rdNdX,
const double WDerivative,
const double DetJDerivative,
const Matrix& rdNdXDerivative)
{
BoundedMatrix<double, TStrainSize, TElementLocalSize> strain_matrix_derivative;
FluidElementUtilities<TNumNodes>::GetStrainMatrix(rdNdXDerivative, strain_matrix_derivative);

const VectorF& rhs_contribution_derivative =
prod(trans(rData.mStrainMatrix), rData.mShearStressDerivative) +
prod(trans(strain_matrix_derivative), rData.mShearStress);

noalias(rResidualDerivative) -= rhs_contribution_derivative * W;
}

};


template<unsigned int TComponentIndex>
class SecondDerivatives
{
public:

constexpr static IndexType NumNodes = TNumNodes;

constexpr static IndexType BlockSize = TBlockSize;


void CalculateGaussPointResidualsDerivativeContributions(
VectorF& rResidualDerivative,
QSVMSResidualData& rData,
const int NodeIndex,
const double W,
const Vector& rN,
const Matrix& rdNdX) const;

};

class QSVMSResidualData
{
public:

static constexpr IndexType TBlockSize = TDim + 1;

static constexpr IndexType TLNumNodes = TNN;

static constexpr IndexType TResidualSize = TBlockSize * TLNumNodes;


void Initialize(
const Element& rElement,
ConstitutiveLaw& rConstitutiveLaw,
const ProcessInfo& rProcessInfo);

void CalculateGaussPointData(
const double W,
const Vector& rN,
const Matrix& rdNdX);

private:

const Element* mpElement;
ConstitutiveLaw* mpConstitutiveLaw;

int mOSSSwitch;
double mDensity;
double mConvectiveVelocityNorm;
double mEffectiveViscosity;
double mDeltaTime;
double mDynamicTau;
double mTauOne;
double mTauTwo;
double mElementSize;
double mMassProjection;
double mDynamicViscosity;
double mPressure;
double mVelocityDotNabla;

ArrayD mBodyForce;
ArrayD mVelocity;
ArrayD mMeshVelocity;
ArrayD mRelaxedAcceleration;
ArrayD mConvectiveVelocity;
ArrayD mMomentumProjection;
ArrayD mPressureGradient;
ArrayD mEffectiveVelocityDotVelocityGradient;

VectorN mConvectiveVelocityDotDnDx;
VectorN mRelaxedAccelerationDotDnDx;
VectorN mEffectiveVelocityDotVelocityGradientDotShapeGradient;
VectorN mBodyForceDotDnDx;
VectorN mMomentumProjectionDotDnDx;
VectorN mPressureGradientDotDnDx;
VectorN mNodalPressure;
MatrixND mNodalVelocity;
MatrixND mNodalMeshVelocity;
MatrixND mNodalEffectiveVelocity;
MatrixDD mVelocityGradient;
MatrixDD mMeshVelocityGradient;
MatrixDD mEffectiveVelocityGradient;
BoundedVector<double, TElementLocalSize> mViscousTermRHSContribution;

ConstitutiveLaw::Parameters mConstitutiveLawValues;
BoundedMatrix<double, TStrainSize, TElementLocalSize> mStrainMatrix;
Vector mStrainRate;
Vector mShearStress;
Matrix mC;

Vector mStrainRateDerivative;
Vector mShearStressDerivative;


template<class TDerivativesType>
friend class VariableDerivatives;

template<unsigned int TComponentIndex>
friend class SecondDerivatives;

friend class ResidualsContributions;

};


static double CalculateNormDerivative(
const double ValueNorm,
const Vector& Value,
const Vector& ValueDerivative);

static void CalculateTau(
double& TauOne,
double& TauTwo,
const double ElementSize,
const double Density,
const double Viscosity,
const double VelocityNorm,
const double DynamicTau,
const double DeltaTime);

static void CalculateTauDerivative(
double& TauOneDerivative,
double& TauTwoDerivative,
const double TauOne,
const double Density,
const double DynamicTau,
const double DeltaTime,
const double ElementSize,
const double ElementSizeDerivative,
const double Viscosity,
const double ViscosityDerivative,
const double VelocityNorm,
const double VelocityNormDerivative);

static void InitializeConstitutiveLaw(
ConstitutiveLaw::Parameters& rParameters,
Vector& rStrainVector,
Vector& rStressVector,
Matrix& rConstitutiveMatrix,
const GeometryType& rGeometry,
const PropertiesType& rProperties,
const ProcessInfo& rProcessInfo);

};

} 