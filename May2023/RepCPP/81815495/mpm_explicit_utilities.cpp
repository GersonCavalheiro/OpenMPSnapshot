
#include "custom_utilities/mpm_explicit_utilities.h"

namespace Kratos
{
void MPMExplicitUtilities::CalculateAndAddExplicitInternalForce(
const ProcessInfo& rProcessInfo,
Element& rElement,
const Vector& rMPStress,
const double rMPVolume,
const SizeType StrainSize,
Vector& rRightHandSideVector)
{
KRATOS_TRY


GeometryType& rGeom = rElement.GetGeometry();
const SizeType dimension = rGeom.WorkingSpaceDimension();
const SizeType number_of_nodes = rGeom.PointsNumber();
array_1d<double, 3> nodal_force_internal_normal = ZeroVector(3);

const bool is_axisym = (rProcessInfo.Has(IS_AXISYMMETRIC))
? rProcessInfo.GetValue(IS_AXISYMMETRIC) : false;

std::vector<Matrix> DN_DX_vec(rGeom.IntegrationPointsNumber());
GetCartesianDerivatives(DN_DX_vec, rGeom);
IndexType active_node_counter;

for (size_t int_p = 0; int_p < rGeom.IntegrationPointsNumber(); ++int_p)
{
active_node_counter = 0;
double weight = (rGeom.IntegrationPointsNumber() > 1) ? rGeom.IntegrationPoints()[int_p].Weight() : 1.0;
for (IndexType i = 0; i < number_of_nodes; i++)
{
if (rGeom.ShapeFunctionValue(int_p,i) >= 0.0)
{
if (dimension == 2 && StrainSize == 3)
{
nodal_force_internal_normal[0] = rMPVolume * weight *
(rMPStress[0] * DN_DX_vec[int_p](active_node_counter, 0) +
rMPStress[2] * DN_DX_vec[int_p](active_node_counter, 1));
nodal_force_internal_normal[1] = rMPVolume * weight *
(rMPStress[1] * DN_DX_vec[int_p](active_node_counter, 1) +
rMPStress[2] * DN_DX_vec[int_p](active_node_counter, 0));
}
else if (is_axisym)
{
nodal_force_internal_normal[0] = rMPVolume * weight *
(rMPStress[0] * DN_DX_vec[int_p](active_node_counter, 0) +
rMPStress[2] * rGeom.ShapeFunctionValue(int_p, i) /
ParticleMechanicsMathUtilities<double>::CalculateRadius(rGeom.ShapeFunctionsValues(), rGeom, Current, int_p) +
rMPStress[3] * DN_DX_vec[int_p](active_node_counter, 1));

nodal_force_internal_normal[1] = rMPVolume * weight *
(rMPStress[1] * DN_DX_vec[int_p](active_node_counter, 1) +
rMPStress[3] * DN_DX_vec[int_p](active_node_counter, 0));
}
else if (dimension == 3 && StrainSize == 6)
{
nodal_force_internal_normal[0] = rMPVolume * weight *
(rMPStress[0] * DN_DX_vec[int_p](active_node_counter, 0) +
rMPStress[3] * DN_DX_vec[int_p](active_node_counter, 1) +
rMPStress[5] * DN_DX_vec[int_p](active_node_counter, 2));
nodal_force_internal_normal[1] = rMPVolume * weight *
(rMPStress[1] * DN_DX_vec[int_p](active_node_counter, 1) +
rMPStress[3] * DN_DX_vec[int_p](active_node_counter, 0) +
rMPStress[4] * DN_DX_vec[int_p](active_node_counter, 2));
nodal_force_internal_normal[2] = rMPVolume * weight *
(rMPStress[2] * DN_DX_vec[int_p](active_node_counter, 2) +
rMPStress[5] * DN_DX_vec[int_p](active_node_counter, 0) +
rMPStress[4] * DN_DX_vec[int_p](active_node_counter, 1));
rRightHandSideVector[dimension * i + 2] -= nodal_force_internal_normal[2]; 
}
else
{
#pragma omp single
KRATOS_ERROR << "Dimension = " << dimension << " and strain size = " << StrainSize
<< " are invalid for MPM explicit internal force calculation." << std::endl;
}
rRightHandSideVector[dimension * i] -= nodal_force_internal_normal[0]; 
rRightHandSideVector[dimension * i + 1] -= nodal_force_internal_normal[1]; 

active_node_counter += 1;
}
}
}

KRATOS_CATCH("")
}


void MPMExplicitUtilities::UpdateGaussPointExplicit(
const ProcessInfo& rCurrentProcessInfo,
Element& rElement)
{
KRATOS_TRY

const double rDeltaTime = rCurrentProcessInfo[DELTA_TIME];
const bool isCentralDifference = rCurrentProcessInfo.GetValue(IS_EXPLICIT_CENTRAL_DIFFERENCE);
GeometryType& rGeom = rElement.GetGeometry();
const SizeType number_of_nodes = rGeom.PointsNumber();
const SizeType dimension = rGeom.WorkingSpaceDimension();


bool isUpdateMPPositionFromUpdatedMPVelocity = false;

std::vector<array_1d<double, 3 > > MP_PreviousVelocity;
std::vector<array_1d<double, 3 > > MP_PreviousAcceleration;
array_1d<double, 3> MP_Velocity = ZeroVector(3);
rElement.CalculateOnIntegrationPoints(MP_VELOCITY, MP_PreviousVelocity, rCurrentProcessInfo);
rElement.CalculateOnIntegrationPoints(MP_ACCELERATION, MP_PreviousAcceleration, rCurrentProcessInfo);

const double gamma = (isCentralDifference) ? 0.5 : 1.0; 
if (isCentralDifference) isUpdateMPPositionFromUpdatedMPVelocity = false;


for (IndexType i = 0; i < dimension; i++) {
MP_Velocity[i] = MP_PreviousVelocity[0][i] + (1.0 - gamma) * rDeltaTime * MP_PreviousAcceleration[0][i];
}


array_1d<double, 3> delta_xg = ZeroVector(3);
array_1d<double, 3> MP_Acceleration = ZeroVector(3);

for (IndexType int_p = 0; int_p < rGeom.IntegrationPointsNumber(); ++int_p) {
double weight = (rGeom.IntegrationPointsNumber() > 1) ? rGeom.IntegrationPoints()[int_p].Weight() : 1.0;
for (IndexType i = 0; i < number_of_nodes; i++)
{
if (rGeom.ShapeFunctionValue(int_p, i) >= 0.0)
{
const double nodal_mass = rGeom[i].FastGetSolutionStepValue(NODAL_MASS);
if (nodal_mass > std::numeric_limits<double>::epsilon())
{
const array_1d<double, 3>& r_nodal_momenta = rGeom[i].FastGetSolutionStepValue(NODAL_MOMENTUM);
const array_1d<double, 3>& r_current_residual = rGeom[i].FastGetSolutionStepValue(FORCE_RESIDUAL);

const array_1d<double, 3>& r_middle_velocity = rGeom[i].FastGetSolutionStepValue(VELOCITY);

for (IndexType j = 0; j < dimension; j++)
{
MP_Acceleration[j] += rGeom.ShapeFunctionValue(int_p, i) * r_current_residual[j] / nodal_mass * weight;

if (isCentralDifference)
{
delta_xg[j] += rDeltaTime * rGeom.ShapeFunctionValue(int_p, i) * r_middle_velocity[j] * weight;
}
else if (!isUpdateMPPositionFromUpdatedMPVelocity)
{
delta_xg[j] += rDeltaTime * rGeom.ShapeFunctionValue(int_p, i) * r_nodal_momenta[j] / nodal_mass * weight;
}
}
}
}
}
}

rElement.SetValuesOnIntegrationPoints(MP_ACCELERATION, { MP_Acceleration }, rCurrentProcessInfo);

for (IndexType j = 0; j < dimension; j++)
{
MP_Velocity[j] += gamma * rDeltaTime * MP_Acceleration[j];
}
rElement.SetValuesOnIntegrationPoints(MP_VELOCITY, { MP_Velocity }, rCurrentProcessInfo);

if (isUpdateMPPositionFromUpdatedMPVelocity)
{
for (IndexType j = 0; j < dimension; j++)
{
delta_xg[j] = rDeltaTime * MP_Velocity[j];
}
}
std::vector<array_1d<double, 3 > > xg;
rElement.CalculateOnIntegrationPoints(MP_COORD, xg, rCurrentProcessInfo);
const array_1d<double, 3>& new_xg = xg[0] + delta_xg;
rElement.SetValuesOnIntegrationPoints(MP_COORD, { new_xg }, rCurrentProcessInfo);

std::vector<array_1d<double, 3 > > MP_Displacement;
rElement.CalculateOnIntegrationPoints(MP_DISPLACEMENT, MP_Displacement, rCurrentProcessInfo);
MP_Displacement[0] += delta_xg;
rElement.SetValuesOnIntegrationPoints(MP_DISPLACEMENT,MP_Displacement, rCurrentProcessInfo);

KRATOS_CATCH("")
}




void MPMExplicitUtilities::CalculateMUSLGridVelocity(const ProcessInfo& rCurrentProcessInfo,
Element& rElement)
{
KRATOS_TRY

GeometryType& rGeom = rElement.GetGeometry();
const SizeType dimension = rGeom.WorkingSpaceDimension();
const SizeType number_of_nodes = rGeom.PointsNumber();

std::vector<array_1d<double, 3 > > MP_Velocity;
std::vector<double> MP_Mass;
rElement.CalculateOnIntegrationPoints(MP_VELOCITY, MP_Velocity, rCurrentProcessInfo);
rElement.CalculateOnIntegrationPoints(MP_MASS, MP_Mass, rCurrentProcessInfo);

for (IndexType int_p = 0; int_p < rGeom.IntegrationPointsNumber(); ++int_p) {
double weight = (rGeom.IntegrationPointsNumber() > 1) ? rGeom.IntegrationPoints()[int_p].Weight() : 1.0;
for (IndexType i = 0; i < number_of_nodes; i++)
{
if (rGeom.ShapeFunctionValue(int_p, i) >= 0.0)
{
const double& r_nodal_mass = rGeom[i].FastGetSolutionStepValue(NODAL_MASS);

if (r_nodal_mass > std::numeric_limits<double>::epsilon())
{
array_1d<double, 3>& r_current_velocity = rGeom[i].FastGetSolutionStepValue(VELOCITY);
for (IndexType j = 0; j < dimension; j++)
{
r_current_velocity[j] += rGeom.ShapeFunctionValue(int_p, i) * MP_Mass[0] * weight * MP_Velocity[0][j] / r_nodal_mass;
}
}
}
}
}

KRATOS_CATCH("")
}




void MPMExplicitUtilities::CalculateExplicitKinematics(
const ProcessInfo& rCurrentProcessInfo,
Element& rElement,
Vector& rMPStrain,
Matrix& rDeformationGradientIncrement,
const SizeType StrainSize)
{
KRATOS_TRY

GeometryType& rGeom = rElement.GetGeometry();
const double deltaTime = rCurrentProcessInfo[DELTA_TIME];
const SizeType dimension = rGeom.WorkingSpaceDimension();
const SizeType number_of_nodes = rGeom.PointsNumber();

const bool is_axisym = (rCurrentProcessInfo.Has(IS_AXISYMMETRIC))
? rCurrentProcessInfo.GetValue(IS_AXISYMMETRIC) : false;

std::vector<Matrix> DN_DX_vec(rGeom.IntegrationPointsNumber());
GetCartesianDerivatives(DN_DX_vec, rGeom);

Matrix velocityGradient = (StrainSize == 3) ? Matrix(2, 2, 0.0) : Matrix(3, 3, 0.0);
IndexType active_node_counter;

for (IndexType int_p = 0; int_p < rGeom.IntegrationPointsNumber(); ++int_p) {
double weight = (rGeom.IntegrationPointsNumber() > 1) ? rGeom.IntegrationPoints()[int_p].Weight() : 1.0;
active_node_counter = 0;
for (IndexType nodeIndex = 0; nodeIndex < number_of_nodes; nodeIndex++)
{
if (rGeom.ShapeFunctionValue(int_p,nodeIndex) >= 0.0)
{
const array_1d<double, 3 >& nodal_velocity = rGeom[nodeIndex].FastGetSolutionStepValue(VELOCITY);

for (IndexType i = 0; i < dimension; i++)
{
for (IndexType j = 0; j < dimension; j++)
{
velocityGradient(i, j) += nodal_velocity[i] * DN_DX_vec[int_p](active_node_counter, j) * weight;
}
}
if (is_axisym) 
{
velocityGradient(2, 2) += nodal_velocity[0] * rGeom.ShapeFunctionValue(int_p, nodeIndex) /
ParticleMechanicsMathUtilities<double>::CalculateRadius(rGeom.ShapeFunctionsValues(), rGeom, Current) * weight;
}
active_node_counter += 1;
}
}
}

const Matrix rateOfDeformation = 0.5 * (velocityGradient + trans(velocityGradient));
const Matrix spinTensor = velocityGradient - rateOfDeformation;

const Matrix jaumannRate = rateOfDeformation -
(prod(spinTensor, rateOfDeformation)) * deltaTime +
prod((rateOfDeformation * deltaTime), spinTensor);
const Matrix strainIncrement = deltaTime * jaumannRate;

rMPStrain(0) += strainIncrement(0, 0); 
rMPStrain(1) += strainIncrement(1, 1); 
if (dimension == 2 && StrainSize == 3)
{
rMPStrain(2) += 2.0 * strainIncrement(0, 1); 
}
else if((dimension == 3 && StrainSize == 6))
{
rMPStrain(2) += strainIncrement(2, 2); 

rMPStrain(3) += 2.0 * strainIncrement(0, 1); 
rMPStrain(4) += 2.0 * strainIncrement(1, 2); 
rMPStrain(5) += 2.0 * strainIncrement(0, 2); 
}
else if (is_axisym)
{
rMPStrain(2) += strainIncrement(2, 2); 
rMPStrain(3) += 2.0 * strainIncrement(0, 1); 
}

rDeformationGradientIncrement = IdentityMatrix(strainIncrement.size1());
if (rCurrentProcessInfo.GetValue(IS_COMPRESSIBLE)) rDeformationGradientIncrement += strainIncrement;

KRATOS_CATCH("")
}

void MPMExplicitUtilities::GetCartesianDerivatives(std::vector<Matrix>& rDN_DXVec, GeometryType& rGeom)
{
KRATOS_TRY
if (rDN_DXVec.size() != rGeom.IntegrationPointsNumber()) rDN_DXVec.resize(rGeom.IntegrationPointsNumber());

for (IndexType i = 0; i < rGeom.IntegrationPointsNumber(); i++) {
Matrix Jacobian;
Matrix InvJ;
double detJ;
rGeom.Jacobian(Jacobian, i);
MathUtils<double>::InvertMatrix(Jacobian, InvJ, detJ);
rDN_DXVec[i] = prod(rGeom.ShapeFunctionLocalGradient(i), InvJ); 
}
KRATOS_CATCH("")
}
} 