

#ifndef _GEOMETRYFUNCTIONS_H
#define	_GEOMETRYFUNCTIONS_H

#include <cmath>
#include "utilities/openmp_utils.h"
#include "utilities/quaternion.h"
#include "includes/model_part.h"
#include "DEM_application_variables.h"

namespace Kratos {

namespace GeometryFunctions {

typedef Geometry<Node > GeometryType;

static inline void RotateAVectorAGivenAngleAroundAUnitaryVector(const array_1d<double, 3>& old_vec, const array_1d<double, 3>& axis,
const double ang, array_1d<double, 3>& new_vec) {
double cang = std::cos(ang);
double sang = std::sin(ang);

new_vec[0] = axis[0] * (axis[0] * old_vec[0] + axis[1] * old_vec[1] + axis[2] * old_vec[2]) * (1 - cang) + old_vec[0] * cang + (-axis[2] * old_vec[1] + axis[1] * old_vec[2]) * sang;
new_vec[1] = axis[1] * (axis[0] * old_vec[0] + axis[1] * old_vec[1] + axis[2] * old_vec[2]) * (1 - cang) + old_vec[1] * cang + ( axis[2] * old_vec[0] - axis[0] * old_vec[2]) * sang;
new_vec[2] = axis[2] * (axis[0] * old_vec[0] + axis[1] * old_vec[1] + axis[2] * old_vec[2]) * (1 - cang) + old_vec[2] * cang + (-axis[1] * old_vec[0] + axis[0] * old_vec[1]) * sang;
}

static inline void TranslateGridOfNodes(const double time, const double velocity_start_time, const double velocity_stop_time, array_1d<double, 3>& center_position,
const array_1d<double, 3>& initial_center, array_1d<double, 3>& previous_displ, array_1d<double, 3>& linear_velocity_changed,
const double linear_period, const double dt, const array_1d<double, 3>& linear_velocity) {

if (time < velocity_start_time || time > velocity_stop_time) {
center_position[0] = initial_center[0] + previous_displ[0];
center_position[1] = initial_center[1] + previous_displ[1];
center_position[2] = initial_center[2] + previous_displ[2];
linear_velocity_changed = ZeroVector(3);
} else {
if (linear_period > 0.0) {
double linear_omega = 2.0 * Globals::Pi / linear_period;
double inv_linear_omega = 1.0 / linear_omega;
noalias(center_position) = initial_center + linear_velocity * std::sin(linear_omega * (time - velocity_start_time)) * inv_linear_omega;
noalias(linear_velocity_changed) = linear_velocity * std::cos(linear_omega * (time - velocity_start_time));
noalias(previous_displ) = center_position - initial_center;
} else {
center_position[0] = initial_center[0] + previous_displ[0] + dt * linear_velocity[0];
center_position[1] = initial_center[1] + previous_displ[1] + dt * linear_velocity[1];
center_position[2] = initial_center[2] + previous_displ[2] + dt * linear_velocity[2];
previous_displ[0] += dt * linear_velocity[0];
previous_displ[1] += dt * linear_velocity[1];
previous_displ[2] += dt * linear_velocity[2];
linear_velocity_changed = linear_velocity;
}
}
}

static inline int sign(const double a)
{
return (0.0 < a) - (a < 0.0);

}

static inline double min(double a, double b)
{
double output;
if (a<=b) output = a;
else output = b;
return output;
}

static inline double max(double a, double b)
{
double output;
if (a>=b) output = a;
else output = b;
return output;
}

static inline void normalize(double Vector[3])
{
double distance = DEM_INNER_PRODUCT_3(Vector, Vector);
double inv_distance = (distance > 0.0) ?  1.0 / sqrt(distance) : 0.00;

Vector[0] = Vector[0] * inv_distance;
Vector[1] = Vector[1] * inv_distance;
Vector[2] = Vector[2] * inv_distance;
}

static inline void normalize(array_1d<double,3>& Vector, double& distance)
{
distance = DEM_MODULUS_3(Vector);
double inv_distance = (distance != 0.0) ?  1.0 / distance : 0.00;

Vector[0] = Vector[0] * inv_distance;
Vector[1] = Vector[1] * inv_distance;
Vector[2] = Vector[2] * inv_distance;
}

static inline void normalize(double Vector[3], double& distance)
{
distance = DEM_MODULUS_3(Vector);
double inv_distance = (distance != 0.0) ?  1.0 / distance : 0.00;

Vector[0] = Vector[0] * inv_distance;
Vector[1] = Vector[1] * inv_distance;
Vector[2] = Vector[2] * inv_distance;
}

static inline void normalize(array_1d<double,3>& Vector)
{
double distance = DEM_MODULUS_3(Vector);
double inv_distance = (distance != 0.0) ?  1.0 / distance : 0.00;

Vector[0] = Vector[0] * inv_distance;
Vector[1] = Vector[1] * inv_distance;
Vector[2] = Vector[2] * inv_distance;
}

static inline void module(const array_1d<double,3>& Vector, double& distance)
{
distance = DEM_MODULUS_3(Vector);
}

static inline double module(const double Vector[3])
{
return DEM_MODULUS_3(Vector);
}

static inline void module(const double Vector[3], double& distance)
{
distance = DEM_MODULUS_3(Vector);
}

static inline double module(const array_1d<double,3>& Vector)
{
double distance = DEM_MODULUS_3(Vector);
return distance;
}

static inline void VectorGlobal2Local(const double LocalCoordSystem[3][3], const double GlobalVector[3], double LocalVector[3])
{
for (int i=0; i<3; i++) {
LocalVector[i] = 0.0;
for (int j=0; j<3; j++) {
LocalVector[i]+=LocalCoordSystem[i][j]*GlobalVector[j];
}
}
}

static inline void VectorGlobal2Local(const double LocalCoordSystem[3][3], const array_1d<double, 3>& GlobalVector, array_1d<double, 3>& LocalVector)
{
for (int i=0; i<3; i++) {
LocalVector[i] = 0.0;
for (int j=0; j<3; j++) {
LocalVector[i]+=LocalCoordSystem[i][j]*GlobalVector[j];
}
}
}

static inline void VectorGlobal2Local(const double LocalCoordSystem[3][3], const array_1d<double, 3>& GlobalVector, double LocalVector[3])
{
for (int i=0; i<3; i++) {
LocalVector[i] = 0.0;
for (int j=0; j<3; j++) {
LocalVector[i]+=LocalCoordSystem[i][j]*GlobalVector[j];
}
}
}

static inline void VectorLocal2Global(const double LocalCoordSystem[3][3], const double LocalVector[3], double GlobalVector[3])
{
for (int i=0; i<3; i++) {
GlobalVector[i] = 0.0;
for (int j=0; j<3; j++) {
GlobalVector[i]+=LocalCoordSystem[j][i]*LocalVector[j];
}
}
}

static inline void VectorLocal2Global(const double LocalCoordSystem[3][3], const array_1d<double, 3>& LocalVector, array_1d<double, 3>& GlobalVector)
{
for (int i=0; i<3; i++) {
GlobalVector[i] = 0.0;
for (int j=0; j<3; j++) {
GlobalVector[i]+=LocalCoordSystem[j][i]*LocalVector[j];
}
}
}

static inline void VectorLocal2Global(const double LocalCoordSystem[3][3], const array_1d<double, 3>& LocalVector, double GlobalVector[3])
{
for (int i=0; i<3; i++) {
GlobalVector[i] = 0.0;
for (int j=0; j<3; j++) {
GlobalVector[i]+=LocalCoordSystem[j][i]*LocalVector[j];
}
}
}

static inline void VectorLocal2Global(const double LocalCoordSystem[3][3], const double LocalVector[3], array_1d<double, 3>& GlobalVector)
{
for (int i=0; i<3; i++) {
GlobalVector[i] = 0.0;
for (int j=0; j<3; j++) {
GlobalVector[i]+=LocalCoordSystem[j][i]*LocalVector[j];
}
}
}

static inline void ProductMatrices3X3(const double Matrix1[3][3], const double Matrix2[3][3], double Matrix3[3][3])
{
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 3; j++) {
Matrix3[i][j] = 0.0;
for (int k = 0; k < 3; k++) {
Matrix3[i][j] += Matrix1[i][k] * Matrix2[k][j];
}
}
}
}

static inline void ProductMatrix3X3Vector3X1(const double Matrix[3][3], const array_1d<double,3>& Vector1, array_1d<double,3>& Vector2)
{
for (int i=0; i<3; i++) {
Vector2[i] = 0.0;
for (int j=0; j<3; j++) {
Vector2[i]+=Matrix[j][i]*Vector1[j];
}
}
}

static inline void TensorGlobal2Local(const double LocalCoordSystem[3][3], const double GlobalTensor[3][3], double LocalTensor[3][3])
{

double TransposedLocalCoordSystem[3][3];
double TemporalResult[3][3];

TransposedLocalCoordSystem[0][0] = LocalCoordSystem[0][0]; TransposedLocalCoordSystem[0][1] = LocalCoordSystem[1][0]; TransposedLocalCoordSystem[0][2] = LocalCoordSystem[2][0];
TransposedLocalCoordSystem[1][0] = LocalCoordSystem[0][1]; TransposedLocalCoordSystem[1][1] = LocalCoordSystem[1][1]; TransposedLocalCoordSystem[1][2] = LocalCoordSystem[2][1];
TransposedLocalCoordSystem[2][0] = LocalCoordSystem[0][2]; TransposedLocalCoordSystem[2][1] = LocalCoordSystem[1][2]; TransposedLocalCoordSystem[2][2] = LocalCoordSystem[2][2];

ProductMatrices3X3(LocalCoordSystem, GlobalTensor, TemporalResult);
ProductMatrices3X3(TemporalResult, TransposedLocalCoordSystem, LocalTensor);
}

static inline void TensorLocal2Global(const double LocalCoordSystem[3][3], const double LocalTensor[3][3], double GlobalTensor[3][3])
{

double TransposedLocalCoordSystem[3][3];
double TemporalResult[3][3];

TransposedLocalCoordSystem[0][0] = LocalCoordSystem[0][0]; TransposedLocalCoordSystem[0][1] = LocalCoordSystem[1][0]; TransposedLocalCoordSystem[0][2] = LocalCoordSystem[2][0];
TransposedLocalCoordSystem[1][0] = LocalCoordSystem[0][1]; TransposedLocalCoordSystem[1][1] = LocalCoordSystem[1][1]; TransposedLocalCoordSystem[1][2] = LocalCoordSystem[2][1];
TransposedLocalCoordSystem[2][0] = LocalCoordSystem[0][2]; TransposedLocalCoordSystem[2][1] = LocalCoordSystem[1][2]; TransposedLocalCoordSystem[2][2] = LocalCoordSystem[2][2];

ProductMatrices3X3(TransposedLocalCoordSystem, LocalTensor, TemporalResult);
ProductMatrices3X3(TemporalResult, LocalCoordSystem, GlobalTensor);
}

static inline void RotaMatrixTensorLocal2Global(const double R[3][3], const double LocalTensor[3][3], double GlobalTensor[3][3])
{
double RT[3][3]; double Temp[3][3];

RT[0][0] = R[0][0]; RT[0][1] = R[1][0]; RT[0][2] = R[2][0];
RT[1][0] = R[0][1]; RT[1][1] = R[1][1]; RT[1][2] = R[2][1];
RT[2][0] = R[0][2]; RT[2][1] = R[1][2]; RT[2][2] = R[2][2];

ProductMatrices3X3(R, LocalTensor, Temp);
ProductMatrices3X3(Temp, RT, GlobalTensor);
}

static inline void ConstructLocalTensor(const double moment_of_inertia, double LocalTensor[3][3])
{
LocalTensor[0][0] = moment_of_inertia; LocalTensor[0][1] = 0.0; LocalTensor[0][2] = 0.0;
LocalTensor[1][0] = 0.0; LocalTensor[1][1] = moment_of_inertia; LocalTensor[1][2] = 0.0;
LocalTensor[2][0] = 0.0; LocalTensor[2][1] = 0.0; LocalTensor[2][2] = moment_of_inertia;
}

static inline void ConstructInvLocalTensor(const double moment_of_inertia, double LocalTensorInv[3][3])
{
double moment_of_inertia_inv = 1/moment_of_inertia;
LocalTensorInv[0][0] = moment_of_inertia_inv; LocalTensorInv[0][1] = 0.0; LocalTensorInv[0][2] = 0.0;
LocalTensorInv[1][0] = 0.0; LocalTensorInv[1][1] = moment_of_inertia_inv; LocalTensorInv[1][2] = 0.0;
LocalTensorInv[2][0] = 0.0; LocalTensorInv[2][1] = 0.0; LocalTensorInv[2][2] = moment_of_inertia_inv;
}

static inline void ConstructLocalTensor(const array_1d<double, 3 >& moments_of_inertia, double LocalTensor[3][3])
{
LocalTensor[0][0] = moments_of_inertia[0]; LocalTensor[0][1] = 0.0; LocalTensor[0][2] = 0.0;
LocalTensor[1][0] = 0.0; LocalTensor[1][1] = moments_of_inertia[1]; LocalTensor[1][2] = 0.0;
LocalTensor[2][0] = 0.0; LocalTensor[2][1] = 0.0; LocalTensor[2][2] = moments_of_inertia[2];
}

static inline void ConstructInvLocalTensor(const array_1d<double, 3 >& moments_of_inertia, double LocalTensorInv[3][3])
{
LocalTensorInv[0][0] = 1/moments_of_inertia[0]; LocalTensorInv[0][1] = 0.0; LocalTensorInv[0][2] = 0.0;
LocalTensorInv[1][0] = 0.0; LocalTensorInv[1][1] = 1/moments_of_inertia[1]; LocalTensorInv[1][2] = 0.0;
LocalTensorInv[2][0] = 0.0; LocalTensorInv[2][1] = 0.0; LocalTensorInv[2][2] = 1/moments_of_inertia[2];
}

static inline double DotProduct(double Vector1[3], double Vector2[3])
{
return Vector1[0] * Vector2[0] + Vector1[1] * Vector2[1] + Vector1[2] * Vector2[2];
}

static inline double DotProduct(const array_1d<double,3>& Vector1, const array_1d<double,3>& Vector2)
{
return Vector1[0] * Vector2[0] + Vector1[1] * Vector2[1] + Vector1[2] * Vector2[2];
}

static inline void CrossProduct(const double u[3], const double v[3], double ReturnVector[3])
{
ReturnVector[0] = u[1]*v[2] - u[2]*v[1];
ReturnVector[1] = v[0]*u[2] - u[0]*v[2];
ReturnVector[2] = u[0]*v[1] - u[1]*v[0];
}

static inline void CrossProduct(const array_1d<double,3>& u, const array_1d<double,3>& v, array_1d<double,3>& ReturnVector)
{
ReturnVector[0] = u[1]*v[2] - u[2]*v[1];
ReturnVector[1] = v[0]*u[2] - u[0]*v[2];
ReturnVector[2] = u[0]*v[1] - u[1]*v[0];
}

static inline void CrossProduct(const double u[3], const array_1d<double,3>& v, double ReturnVector[3])
{
ReturnVector[0] = u[1]*v[2] - u[2]*v[1];
ReturnVector[1] = v[0]*u[2] - u[0]*v[2];
ReturnVector[2] = u[0]*v[1] - u[1]*v[0];
}

static inline void CrossProduct(const array_1d<double,3>& u, const double v[3], double ReturnVector[3])
{
ReturnVector[0] = u[1]*v[2] - u[2]*v[1];
ReturnVector[1] = v[0]*u[2] - u[0]*v[2];
ReturnVector[2] = u[0]*v[1] - u[1]*v[0];
}

static inline void CrossProduct(const array_1d<double,3>& u, const double v[3], array_1d<double,3>& ReturnVector)
{
ReturnVector[0] = u[1]*v[2] - u[2]*v[1];
ReturnVector[1] = v[0]*u[2] - u[0]*v[2];
ReturnVector[2] = u[0]*v[1] - u[1]*v[0];
}

static inline void CrossProduct(const array_1d<double,3>& u, const array_1d<double,3>& v, double ReturnVector[3])
{
ReturnVector[0] = u[1]*v[2] - u[2]*v[1];
ReturnVector[1] = v[0]*u[2] - u[0]*v[2];
ReturnVector[2] = u[0]*v[1] - u[1]*v[0];
}

static inline void RotateRightHandedBasisAroundAxis(const array_1d<double, 3>& e1,  const array_1d<double, 3>& e2,  const array_1d<double, 3>& axis,
const double ang, array_1d<double, 3>& new_axes1, array_1d<double, 3>& new_axes2,
array_1d<double, 3>& new_axes3) {

RotateAVectorAGivenAngleAroundAUnitaryVector(e1, axis, ang, new_axes1);

RotateAVectorAGivenAngleAroundAUnitaryVector(e2, axis, ang, new_axes2);

CrossProduct(new_axes1, new_axes2, new_axes3);
}

static inline void RotateGridOfNodes(const double time, const double angular_velocity_start_time, const double angular_velocity_stop_time,
array_1d<double, 3>& angular_velocity_changed, const double angular_period, const double mod_angular_velocity,
const array_1d<double, 3>& angular_velocity, array_1d<double, 3>& new_axes1, array_1d<double, 3>& new_axes2,
array_1d<double, 3>& new_axes3) {

array_1d<double, 3> angle;
noalias(angle) = ZeroVector(3);
double sign_angle = 1.0;
array_1d<double, 3> final_angle = ZeroVector(3);

if (time < angular_velocity_start_time) angular_velocity_changed = ZeroVector(3);

else if (((time - angular_velocity_start_time) > 0.0) && ((time - angular_velocity_stop_time) < 0.0)) {

if (angular_period > 0.0) {
double angular_omega = 2.0 * Globals::Pi / angular_period;
double inv_angular_omega = 1.0 / angular_omega;
noalias(angle) = angular_velocity * std::sin(angular_omega * (time - angular_velocity_start_time)) * inv_angular_omega;
sign_angle = std::sin(angular_omega * (time - angular_velocity_start_time)) / fabs(sin(angular_omega * (time - angular_velocity_start_time)));
noalias(angular_velocity_changed) = angular_velocity * std::cos(angular_omega * (time - angular_velocity_start_time));
noalias(final_angle) = angle;
} else {
noalias(angle) = angular_velocity * (time - angular_velocity_start_time);
noalias(angular_velocity_changed) = angular_velocity;
}
} else { 
noalias(angular_velocity_changed) = ZeroVector(3);

if (angular_period > 0.0) {
double angular_omega = 2.0 * Globals::Pi / angular_period;
double inv_angular_omega = 1.0 / angular_omega;
noalias(angle) = angular_velocity * std::sin(angular_omega * (angular_velocity_stop_time - angular_velocity_start_time)) * inv_angular_omega;
} else {
noalias(angle) = angular_velocity * (angular_velocity_stop_time - angular_velocity_start_time);
}
}


new_axes1[0] = 1.0;
new_axes1[1] = 0.0;
new_axes1[2] = 0.0;

new_axes2[0] = 0.0;
new_axes2[1] = 1.0;
new_axes2[2] = 0.0;

new_axes3[0] = 0.0;
new_axes3[1] = 0.0;
new_axes3[2] = 1.0;

if (mod_angular_velocity > 0.0) {

double ang = sign_angle * MathUtils<double>::Norm3(angle);
array_1d<double, 3> rotation_axis;
noalias(rotation_axis) = angular_velocity / mod_angular_velocity;
array_1d<double, 3> e1;
e1[0] = 1.0;
e1[1] = 0.0;
e1[2] = 0.0;

array_1d<double, 3> e2;
e2[0] = 0.0;
e2[1] = 1.0;
e2[2] = 0.0;

RotateRightHandedBasisAroundAxis(e1, e2, rotation_axis, ang, new_axes1, new_axes2, new_axes3);
}
}

static inline void UpdateKinematicVariablesOfAGridOfNodes(double mod_angular_velocity, const array_1d<double, 3>& linear_velocity,
const array_1d<double, 3>& initial_center, array_1d<double, 3>& new_axes1, array_1d<double, 3>& new_axes2,
array_1d<double, 3>& new_axes3, array_1d<double, 3>& angular_velocity_changed,
array_1d<double, 3>& linear_velocity_changed, array_1d<double, 3>& center_position,
const bool fixed_mesh, const double dt, ModelPart::NodesContainerType& pNodes)
{
if (mod_angular_velocity > std::numeric_limits<double>::epsilon() || MathUtils<double>::Norm3(linear_velocity) > std::numeric_limits<double>::epsilon()) {

#pragma omp parallel for
for (int k = 0; k < (int)pNodes.size(); k++) {

array_1d<double, 3> local_coordinates = ZeroVector(3);
array_1d<double, 3> relative_position = ZeroVector(3);

ModelPart::NodeIterator node = pNodes.begin() + k;

noalias(local_coordinates) = node->GetInitialPosition().Coordinates() - initial_center;
noalias(relative_position) = new_axes1 * local_coordinates[0] + new_axes2 * local_coordinates[1] + new_axes3 * local_coordinates[2];
array_1d<double, 3> old_coordinates;
noalias(old_coordinates) = node->Coordinates();
array_1d<double, 3> velocity_due_to_rotation;
array_1d<double, 3>& velocity = node->FastGetSolutionStepValue(VELOCITY);

CrossProduct(angular_velocity_changed, relative_position, velocity_due_to_rotation);
noalias(velocity) = linear_velocity_changed + velocity_due_to_rotation;

if (!fixed_mesh) {
noalias(node->Coordinates()) = center_position + relative_position;
noalias(node->FastGetSolutionStepValue(DISPLACEMENT)) = node->Coordinates() - node->GetInitialPosition().Coordinates();
noalias(node->FastGetSolutionStepValue(DELTA_DISPLACEMENT)) = node->Coordinates() - old_coordinates;
} else {
(node->FastGetSolutionStepValue(DISPLACEMENT)).clear(); 
noalias(node->FastGetSolutionStepValue(DELTA_DISPLACEMENT)) = velocity * dt; 
}
}
}
}


static inline void ComputeContactLocalCoordSystem(array_1d<double, 3> NormalDirection, const double& distance, double LocalCoordSystem[3][3])  
{
double inv_distance = (distance != 0.0) ? 1.0 / distance : 0.0;
NormalDirection[0] *= inv_distance;
NormalDirection[1] *= inv_distance;
NormalDirection[2] *= inv_distance;
double N_fast[3];
N_fast[0] = NormalDirection[0];
N_fast[1] = NormalDirection[1];
N_fast[2] = NormalDirection[2];

if (fabs(N_fast[0]) >= 0.577) 
{
LocalCoordSystem[0][0] = - N_fast[1];
LocalCoordSystem[0][1] = N_fast[0];
LocalCoordSystem[0][2] = 0.0;
}
else if (fabs(N_fast[1]) >= 0.577)
{
LocalCoordSystem[0][0] = 0.0;
LocalCoordSystem[0][1] = - N_fast[2];
LocalCoordSystem[0][2] = N_fast[1];
}
else
{
LocalCoordSystem[0][0] = N_fast[2];
LocalCoordSystem[0][1] = 0.0;
LocalCoordSystem[0][2] = - N_fast[0];
}

double distance0 = DEM_MODULUS_3(LocalCoordSystem[0]);
double inv_distance0 = (distance0 != 0.0) ? 1.0 / distance0 : 0.0;
LocalCoordSystem[0][0] = LocalCoordSystem[0][0] * inv_distance0;
LocalCoordSystem[0][1] = LocalCoordSystem[0][1] * inv_distance0;
LocalCoordSystem[0][2] = LocalCoordSystem[0][2] * inv_distance0;

LocalCoordSystem[1][0] = N_fast[1] * LocalCoordSystem[0][2] - N_fast[2] * LocalCoordSystem[0][1];
LocalCoordSystem[1][1] = N_fast[2] * LocalCoordSystem[0][0] - N_fast[0] * LocalCoordSystem[0][2];
LocalCoordSystem[1][2] = N_fast[0] * LocalCoordSystem[0][1] - N_fast[1] * LocalCoordSystem[0][0];


LocalCoordSystem[2][0] = N_fast[0];
LocalCoordSystem[2][1] = N_fast[1];
LocalCoordSystem[2][2] = N_fast[2];
}

static inline double DistanceOfTwoPoint(const double coord1[3], const double coord2[3])
{
double dx = coord1[0] - coord2[0];
double dy = coord1[1] - coord2[1];
double dz = coord1[2] - coord2[2];

return sqrt(dx * dx + dy * dy + dz * dz);
}

static inline double DistanceOfTwoPoint(const array_1d<double,3>& coord1, const double coord2[3])
{
double dx = coord1[0] - coord2[0];
double dy = coord1[1] - coord2[1];
double dz = coord1[2] - coord2[2];

return sqrt(dx * dx + dy * dy + dz * dz);
}

static inline double DistanceOfTwoPointSquared(const array_1d<double,3>& coord1, const array_1d<double,3>& coord2)
{
double dx = coord1[0] - coord2[0];
double dy = coord1[1] - coord2[1];
double dz = coord1[2] - coord2[2];

return (dx * dx + dy * dy + dz * dz);
}

static inline double DistanceOfTwoPointSquared(double coord1[3], double coord2[3])
{
double dx = coord1[0] - coord2[0];
double dy = coord1[1] - coord2[1];
double dz = coord1[2] - coord2[2];

return (dx * dx + dy * dy + dz * dz);
}
static inline double DistancePointToPlane(const array_1d<double,3>& CoordInPlane, double PlaneUnitNormalVector[3], double TestCoord[3])
{
double Vector1[3] = {0.0};

for (unsigned int i = 0; i<3; i++)
{
Vector1[i] = TestCoord[i]- CoordInPlane[i];
}

double dist = fabs (DotProduct(Vector1, PlaneUnitNormalVector));

return dist;
}

static inline void CoordProjectionOnPlane(double CoordOut[3], double CoordIn[3], double LocalCoordSystem[3][3], double IntersectionCoord[3])
{
double out_coord_local[3] = {0.0};
double in_coord_local[3]  = {0.0};

VectorGlobal2Local(LocalCoordSystem, CoordOut, out_coord_local);
VectorGlobal2Local(LocalCoordSystem, CoordIn,  in_coord_local);

double vector1[3] = {0.0};
vector1[0] = out_coord_local[0];
vector1[1] = out_coord_local[1];
vector1[2] = in_coord_local [2];

VectorLocal2Global(LocalCoordSystem, vector1, IntersectionCoord);

}

static inline void CoordProjectionOnPlaneNew(double CoordOut[3], const array_1d<double, 3>& CoordIn, double LocalCoordSystem[3][3], double IntersectionCoord[3])
{
double out_coord_local[3] = {0.0};
double in_coord_local[3]  = {0.0};

VectorGlobal2Local(LocalCoordSystem, CoordOut, out_coord_local);
VectorGlobal2Local(LocalCoordSystem, CoordIn,  in_coord_local);

double vector1[3] = {0.0};
vector1[0] = out_coord_local[0];
vector1[1] = out_coord_local[1];
vector1[2] = in_coord_local [2];

VectorLocal2Global(LocalCoordSystem, vector1, IntersectionCoord);

}

static inline void Compute3DimElementFaceLocalSystem(const array_1d <double,3>& FaceCoord1, const array_1d <double,3>& FaceCoord2, const array_1d <double,3>& FaceCoord3, double ParticleCoord[3],
double LocalCoordSystem[3][3], double& normal_flag)
{

double Vector1[3] = {0.0};
double Vector2[3] = {0.0};
double Vector3[3] = {0.0};
double Normal[3]  = {0.0};

Vector1[0] = FaceCoord2[0] - FaceCoord1[0];
Vector1[1] = FaceCoord2[1] - FaceCoord1[1];
Vector1[2] = FaceCoord2[2] - FaceCoord1[2];

Vector2[0] = FaceCoord3[0] - FaceCoord2[0];
Vector2[1] = FaceCoord3[1] - FaceCoord2[1];
Vector2[2] = FaceCoord3[2] - FaceCoord2[2];

normalize(Vector1);
CrossProduct(Vector1, Vector2, Normal);
normalize(Normal);

CrossProduct(Normal, Vector1, Vector2);
normalize(Vector2);

Vector3[0] = ParticleCoord[0] - FaceCoord1[0];
Vector3[1] = ParticleCoord[1] - FaceCoord1[1];
Vector3[2] = ParticleCoord[2] - FaceCoord1[2];

normalize(Vector3);

if (DotProduct(Vector3, Normal) > 0.0)
{
for (int ia = 0; ia < 3; ia++)
{
normal_flag             = 1.0;
LocalCoordSystem[0][ia] = Vector1[ia];
LocalCoordSystem[1][ia] = Vector2[ia];
LocalCoordSystem[2][ia] = Normal [ia];
}
}
else
{
for (int ia = 0; ia < 3; ia++)
{
normal_flag             = -1.0;
LocalCoordSystem[0][ia] = -Vector1[ia];
LocalCoordSystem[1][ia] = -Vector2[ia];
LocalCoordSystem[2][ia] = -Normal [ia];
}
}
}

static inline void Compute3DimElementFaceLocalSystem(double FaceCoord1[3], double FaceCoord2[3], double FaceCoord3[3], double ParticleCoord[3],
double LocalCoordSystem[3][3], double& normal_flag){


double Vector1[3] = {0.0};
double Vector2[3] = {0.0};
double Vector3[3] = {0.0};
double Normal[3]  = {0.0};

Vector1[0] = FaceCoord2[0] - FaceCoord1[0];
Vector1[1] = FaceCoord2[1] - FaceCoord1[1];
Vector1[2] = FaceCoord2[2] - FaceCoord1[2];

Vector2[0] = FaceCoord3[0] - FaceCoord2[0];
Vector2[1] = FaceCoord3[1] - FaceCoord2[1];
Vector2[2] = FaceCoord3[2] - FaceCoord2[2];

normalize(Vector1);
CrossProduct(Vector1, Vector2, Normal);
normalize(Normal);

CrossProduct(Normal, Vector1, Vector2);
normalize(Vector2);

Vector3[0] = ParticleCoord[0] - FaceCoord1[0];
Vector3[1] = ParticleCoord[1] - FaceCoord1[1];
Vector3[2] = ParticleCoord[2] - FaceCoord1[2];
normalize(Vector3);

if (DotProduct(Vector3, Normal) > 0.0)
{
for (int ia = 0; ia < 3; ia++)
{
normal_flag             = 1.0;
LocalCoordSystem[0][ia] = Vector1[ia];
LocalCoordSystem[1][ia] = Vector2[ia];
LocalCoordSystem[2][ia] = Normal [ia];
}
}
else
{
for (int ia = 0; ia < 3; ia++)
{
normal_flag             = -1.0;
LocalCoordSystem[0][ia] = -Vector1[ia];
LocalCoordSystem[1][ia] = -Vector2[ia];
LocalCoordSystem[2][ia] = -Normal [ia];
}
}
}



static inline void RotatePointAboutArbitraryLine(array_1d<double,3>& TargetPoint, const array_1d<double,3>& CentrePoint, const array_1d<double,3>& LineVector, const double RotationAngle)
{
const double O = RotationAngle;
double x = TargetPoint[0], a = CentrePoint[0], u = LineVector[0];
double y = TargetPoint[1], b = CentrePoint[1], v = LineVector[1];
double z = TargetPoint[2], c = CentrePoint[2], w = LineVector[2];
double L = u*u+v*v+w*w;

if (L==0)
{
}
else
{
const double inv_L = 1.0 / L;
TargetPoint[0] = ((a*(v*v+w*w)-u*(b*v+c*w-u*x-v*y-w*z))*(1-cos(O))+L*x*cos(O)+sqrt(L)*(-c*w+b*w-w*y+v*z)*sin(O))* inv_L;
TargetPoint[1] = ((b*(u*u+w*w)-v*(a*u+c*w-u*x-v*y-w*z))*(1-cos(O))+L*y*cos(O)+sqrt(L)*(c*u-a*w+w*x-u*z)*sin(O))* inv_L;
TargetPoint[2] = ((c*(u*u+v*v)-w*(a*u+b*v-u*x-v*y-w*z))*(1-cos(O))+L*z*cos(O)+sqrt(L)*(-b*u+a*v-v*x+u*y)*sin(O))* inv_L;
}
}


static inline void QuaternionVectorLocal2Global(const Quaternion<double>& Q, const array_1d<double, 3>& LocalVector, array_1d<double, 3>& GlobalVector)
{
Q.RotateVector3(LocalVector, GlobalVector);
}

static inline void QuaternionVectorGlobal2Local(const Quaternion<double>& Q, const array_1d<double, 3>& GlobalVector, array_1d<double, 3>& LocalVector)
{
Quaternion<double> Q_conj = Q.conjugate();
Q_conj.RotateVector3(GlobalVector, LocalVector);
}

static inline void QuaternionTensorLocal2Global(const Quaternion<double>& Q, const double LocalTensor[3][3], double GlobalTensor[3][3])
{
array_1d<double, 3> LocalTensorC1; array_1d<double, 3> LocalTensorC2; array_1d<double, 3> LocalTensorC3;

LocalTensorC1[0] = LocalTensor[0][0]; LocalTensorC2[0] = LocalTensor[0][1]; LocalTensorC3[0] = LocalTensor[0][2];
LocalTensorC1[1] = LocalTensor[1][0]; LocalTensorC2[1] = LocalTensor[1][1]; LocalTensorC3[1] = LocalTensor[1][2];
LocalTensorC1[2] = LocalTensor[2][0]; LocalTensorC2[2] = LocalTensor[2][1]; LocalTensorC3[2] = LocalTensor[2][2];

array_1d<double, 3> TempTensorC1; array_1d<double, 3> TempTensorC2; array_1d<double, 3> TempTensorC3;
array_1d<double, 3> TempTensorTraspC1; array_1d<double, 3> TempTensorTraspC2; array_1d<double, 3> TempTensorTraspC3;

Q.RotateVector3(LocalTensorC1, TempTensorC1);
Q.RotateVector3(LocalTensorC2, TempTensorC2);
Q.RotateVector3(LocalTensorC3, TempTensorC3);

TempTensorTraspC1[0] = TempTensorC1[0]; TempTensorTraspC2[0] = TempTensorC1[1]; TempTensorTraspC3[0] = TempTensorC1[2];
TempTensorTraspC1[1] = TempTensorC2[0]; TempTensorTraspC2[1] = TempTensorC2[1]; TempTensorTraspC3[1] = TempTensorC2[2];
TempTensorTraspC1[2] = TempTensorC3[0]; TempTensorTraspC2[2] = TempTensorC3[1]; TempTensorTraspC3[2] = TempTensorC3[2];

array_1d<double, 3> GlobalTensorTraspC1; array_1d<double, 3> GlobalTensorTraspC2; array_1d<double, 3> GlobalTensorTraspC3;

Q.RotateVector3(TempTensorTraspC1, GlobalTensorTraspC1);
Q.RotateVector3(TempTensorTraspC2, GlobalTensorTraspC2);
Q.RotateVector3(TempTensorTraspC3, GlobalTensorTraspC3);

GlobalTensor[0][0] = GlobalTensorTraspC1[0]; GlobalTensor[0][1] = GlobalTensorTraspC1[1]; GlobalTensor[0][2] = GlobalTensorTraspC1[2];
GlobalTensor[1][0] = GlobalTensorTraspC2[0]; GlobalTensor[1][1] = GlobalTensorTraspC2[1]; GlobalTensor[1][2] = GlobalTensorTraspC2[2];
GlobalTensor[2][0] = GlobalTensorTraspC3[0]; GlobalTensor[2][1] = GlobalTensorTraspC3[1]; GlobalTensor[2][2] = GlobalTensorTraspC3[2];
}

static inline void UpdateOrientation(array_1d<double, 3>& EulerAngles, Quaternion<double>& Orientation, const array_1d<double, 3>& DeltaRotation)
{
Quaternion<double> DeltaOrientation = Quaternion<double>::Identity();

array_1d<double, 3 > theta = DeltaRotation;
DEM_MULTIPLY_BY_SCALAR_3(theta, 0.5);

double thetaMag = DEM_MODULUS_3(theta);
const double epsilon = std::numeric_limits<double>::epsilon();

if (thetaMag * thetaMag * thetaMag * thetaMag / 24.0 < epsilon) { 
double aux = (1 - thetaMag * thetaMag / 6);
DeltaOrientation = Quaternion<double>((1 + thetaMag * thetaMag / 2), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
else {
double aux = std::sin(thetaMag)/thetaMag;
DeltaOrientation = Quaternion<double>(cos(thetaMag), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
Orientation = DeltaOrientation * Orientation;
Orientation.ToEulerAngles(EulerAngles);
}

static inline void UpdateOrientation(Quaternion<double>& Orientation, const array_1d<double, 3>& DeltaRotation)
{
Quaternion<double> DeltaOrientation = Quaternion<double>::Identity();

array_1d<double, 3 > theta = DeltaRotation;
DEM_MULTIPLY_BY_SCALAR_3(theta, 0.5);

double thetaMag = DEM_MODULUS_3(theta);
const double epsilon = std::numeric_limits<double>::epsilon();

if (thetaMag * thetaMag * thetaMag * thetaMag / 24.0 < epsilon) { 
double aux = (1 - thetaMag * thetaMag / 6);
DeltaOrientation = Quaternion<double>((1 + thetaMag * thetaMag * 0.5), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
else {
double aux = std::sin(thetaMag)/thetaMag;
DeltaOrientation = Quaternion<double>(cos(thetaMag), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
Orientation = DeltaOrientation * Orientation;
}

static inline void UpdateOrientation(const Quaternion<double>& Orientation, Quaternion<double>& NewOrientation, const array_1d<double, 3>& DeltaRotation)
{
Quaternion<double> DeltaOrientation = Quaternion<double>::Identity();

array_1d<double, 3 > theta = DeltaRotation;
DEM_MULTIPLY_BY_SCALAR_3(theta, 0.5);

double thetaMag = DEM_MODULUS_3(theta);
const double epsilon = std::numeric_limits<double>::epsilon();

if (thetaMag * thetaMag * thetaMag * thetaMag / 24.0 < epsilon) { 
double aux = (1 - thetaMag * thetaMag / 6);
DeltaOrientation = Quaternion<double>((1 + thetaMag * thetaMag * 0.5), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
else {
double aux = std::sin(thetaMag)/thetaMag;
DeltaOrientation = Quaternion<double>(cos(thetaMag), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
NewOrientation = DeltaOrientation * Orientation;
}

static inline void EulerAnglesFromRotationAngle(array_1d<double, 3>& EulerAngles, const array_1d<double, 3>& RotatedAngle)
{
Quaternion<double> Orientation = Quaternion<double>::Identity();

array_1d<double, 3 > theta = RotatedAngle;
DEM_MULTIPLY_BY_SCALAR_3(theta, 0.5);

double thetaMag = DEM_MODULUS_3(theta);

const double epsilon = std::numeric_limits<double>::epsilon();

if (thetaMag * thetaMag * thetaMag * thetaMag / 24.0 < epsilon) { 
double aux = (1 - thetaMag * thetaMag / 6);
Orientation = Quaternion<double>((1 + thetaMag * thetaMag / 2), theta[0]*aux, theta[1]*aux, theta[2]*aux);
Orientation.normalize();
}
else {
double aux = std::sin(thetaMag)/thetaMag;
Orientation = Quaternion<double>(cos(thetaMag), theta[0]*aux, theta[1]*aux, theta[2]*aux);
Orientation.normalize();
}
Orientation.ToEulerAngles(EulerAngles);
}

static inline void OrientationFromRotationAngle(Quaternion<double>& DeltaOrientation, const array_1d<double, 3>& DeltaRotation)
{
array_1d<double, 3 > theta = DeltaRotation;
DEM_MULTIPLY_BY_SCALAR_3(theta, 0.5);

double thetaMag = DEM_MODULUS_3(theta);
const double epsilon = std::numeric_limits<double>::epsilon();

if (thetaMag * thetaMag * thetaMag * thetaMag / 24.0 < epsilon) { 
double aux = (1 - thetaMag * thetaMag / 6);
DeltaOrientation = Quaternion<double>((1 + thetaMag * thetaMag / 2), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
else {
double aux = std::sin(thetaMag)/thetaMag;
DeltaOrientation = Quaternion<double>(cos(thetaMag), theta[0]*aux, theta[1]*aux, theta[2]*aux);
DeltaOrientation.normalize();
}
}





static inline  bool InsideOutside(const array_1d<double, 3>& Coord1,
const array_1d<double, 3>& Coord2,
const array_1d<double, 3>& JudgeCoord,
const array_1d<double, 3>& normal_element,
double& area){

double b[3];
double p1[3];
double coor[3];
DEM_COPY_SECOND_TO_FIRST_3(coor, Coord1)
b[0] = Coord2[0] - coor[0];
b[1] = Coord2[1] - coor[1];
b[2] = Coord2[2] - coor[2];
p1[0] = JudgeCoord[0] - coor[0];
p1[1] = JudgeCoord[1] - coor[1];
p1[2] = JudgeCoord[2] - coor[2];
DEM_SET_TO_CROSS_OF_FIRST_TWO_3(b, p1, coor)

if (DEM_INNER_PRODUCT_3(coor, normal_element) >= 0){
area = 0.5 * DEM_MODULUS_3(coor);
return true;
}
else return false;

}

static inline bool InsideOutside(const array_1d<double, 3> &Coord1,
const array_1d<double, 3>& Coord2,
const array_1d<double, 3>& JudgeCoord,
const array_1d<double, 3>& normal_element) {

array_1d<double, 3> cp1;
array_1d<double, 3> b_a;
array_1d<double, 3> p1_a;

noalias(b_a)  = Coord2 - Coord1;
noalias(p1_a) = JudgeCoord - Coord1;

GeometryFunctions::CrossProduct(b_a, p1_a, cp1);

if (GeometryFunctions::DotProduct(cp1, normal_element) >= 0)
{
return true;
}
else return false;

}

static inline void WeightsCalculation(std::vector<double> Area, std::vector<double>& Weight)
{
unsigned int facet_size = Area.size();
if (facet_size == 3)
{
const double total_area = Area[0]+Area[1]+Area[2];
const double inv_total_area = 1.0 / total_area;
for (unsigned int i = 0; i< 3; i++)
{
Weight[i] = Area[(i+1)%facet_size] * inv_total_area;
}
}
else if (facet_size == 4)
{
const double total_discriminant = Area[0]*Area[1]+Area[1]*Area[2]+Area[2]*Area[3]+Area[3]*Area[0]; 
const double inv_total_discriminant = 1.0 / total_discriminant;
for (unsigned int i = 0; i< 4; i++)
{
Weight[i] = (Area[(i+1)%facet_size]*Area[(i+2)%facet_size]) * inv_total_discriminant;
}
}
else {
KRATOS_WATCH("WEIGHTS FOR N-SIZE POLYGONAL FE TO BE IMPLEMENTED")
}
}

static inline bool FastFacetCheck(const std::vector< array_1d <double,3> >& Coord, const array_1d <double,3>& Particle_Coord, double rad, double &DistPToB, unsigned int &current_edge_index)
{
double A[3];
double B[3];
double PC[3];

for (unsigned int i = 0; i < 3; i++){
B[i]  = Coord[0][i];
PC[i] = Coord[1][i];
A[i]  = Coord[2][i];
}

for (unsigned int i = 0; i < 3; i++){
A[i] = A[i] - PC[i];
B[i] = B[i] - PC[i];
PC[i] = Particle_Coord[i] - PC[i];
}


double N_fast[3];
DEM_SET_TO_CROSS_OF_FIRST_TWO_3(A, B, N_fast)

double normal_flag = 1.0;

if (DEM_INNER_PRODUCT_3(PC, N_fast) < 0){ 
normal_flag = -1.0;
}

normalize(N_fast);


DistPToB = 0.0;

for (unsigned int i = 0; i < 3; i++){
DistPToB += normal_flag * N_fast[i] * PC[i];
}

if (DistPToB < rad){
array_1d <double, 3> IntersectionCoord;
array_1d <double, 3> N;

for (unsigned int i = 0; i < 3; i++){
IntersectionCoord[i] = Particle_Coord[i] - DistPToB * normal_flag * N_fast[i];
N[i] = N_fast[i];
}

int facet_size = Coord.size();

for (int i = 0; i < facet_size; i++) {
double this_area = 0.0;

if (InsideOutside(Coord[i], Coord[(i+1)%facet_size], IntersectionCoord, N, this_area) == false){
current_edge_index = i;
return false;
}
}
return true;
}

return false;
}

static inline bool FacetCheck(const GeometryType&  Coord, const array_1d <double,3>& Particle_Coord, double rad,
double LocalCoordSystem[3][3], double& DistPToB, std::vector<double>& Weight, unsigned int& current_edge_index, bool& inside)
{
int facet_size = Coord.size();

array_1d <double,3> A;
array_1d <double,3> B;
array_1d <double,3> N;
array_1d <double,3> PC;

for (unsigned int i = 0; i<3; i++)
{
A[i] = Coord[2].Coordinates()[i]-Coord[1].Coordinates()[i];
B[i] = Coord[0].Coordinates()[i]-Coord[1].Coordinates()[i];
PC[i] = Particle_Coord[i]-Coord[1].Coordinates()[i];
}

N[0] = A[1]*B[2] - A[2]*B[1];
N[1] = A[2]*B[0] - A[0]*B[2];
N[2] = A[0]*B[1] - A[1]*B[0];

double normal_flag = 1.0;

if (DotProduct(PC,N) < 0) 
{
normal_flag = - 1.0;
N[0]=-N[0];
N[1]=-N[1];
N[2]=-N[2];
}
normalize(N);


DistPToB = 0.0;

for (unsigned int i = 0; i<3; i++)
{
DistPToB += N[i]*PC[i];
}

array_1d <double,3> IntersectionCoord;

for (unsigned int i = 0; i<3; i++)
{
IntersectionCoord[i] = Particle_Coord[i] - DistPToB*N[i];
}

std::vector<double> Area;
Area.resize(facet_size);

for (int i = 0; i<facet_size; i++)
{
double this_area = 0.0;
if (InsideOutside(Coord[i],
Coord[(i+1)%facet_size],
IntersectionCoord,
normal_flag*N,
this_area) == false)
{
current_edge_index = i;
inside = false;
return false;
}
else
{
inside = true;
Area[i] = this_area; 
}

}

if (DistPToB < rad)
{
double auxiliar_unit_vector[3];
CrossProduct( N,A,auxiliar_unit_vector );
normalize( auxiliar_unit_vector );
normalize( A );
for (unsigned int j = 0; j<3; j++)
{
LocalCoordSystem[0][j] = A[j];
LocalCoordSystem[1][j] = auxiliar_unit_vector[j];
LocalCoordSystem[2][j] = N[j];
}

WeightsCalculation(Area,Weight);
return true;

}

return false;

} 

static inline bool FastEdgeVertexCheck(const array_1d<double,3>& Coord1, const array_1d<double,3>& Coord2, const array_1d<double,3>& Particle_Coord, double Radius)
{
double IntersectionCoordEdge[3];
double normal_unit_vector[3];
double edge_unit_vector[3];
double module_edge_vector = 0.0;
double particle_vector1[3];
double particle_vector2[3];

for (unsigned int j = 0; j<3; j++)
{
edge_unit_vector[j] = Coord2[j] - Coord1[j];
particle_vector1[j]  = Particle_Coord[j] - Coord1[j];
particle_vector2[j]  = Particle_Coord[j] - Coord2[j];
}

normalize( edge_unit_vector, module_edge_vector);
double projection_on_edge = DotProduct(particle_vector1,edge_unit_vector);

double eta = projection_on_edge/module_edge_vector;

if ((eta>=0.0) && (eta<=1.0)) 
{
for (unsigned int j = 0; j<3; j++)
{
IntersectionCoordEdge[j] = Coord1[j] + projection_on_edge*edge_unit_vector[j];
normal_unit_vector[j]   = Particle_Coord[j] - IntersectionCoordEdge[j];
}

double DistParticleToEdge;
normalize( normal_unit_vector, DistParticleToEdge);

if (DistParticleToEdge < Radius)
{
return true;
}
}

if (eta < 0.0)  
{
double dist_to_vertex_sq = 0.0;
double Rad_sq = Radius*Radius;

for (unsigned int j = 0; j<3; j++)
{
dist_to_vertex_sq +=particle_vector1[j]*particle_vector1[j];
}

if (dist_to_vertex_sq < Rad_sq)
{
return true;
}
}

if (eta > 1.0)  
{
double dist_to_vertex_sq = 0.0;
double Rad_sq = Radius*Radius;
for (unsigned int j = 0; j<3; j++)
{
dist_to_vertex_sq +=particle_vector2[j]*particle_vector2[j];
}

if (dist_to_vertex_sq < Rad_sq)
{
return true;
}
}

return false;

}

static inline bool EdgeCheck(const array_1d<double,3>& Coord1, const array_1d<double,3>& Coord2, const array_1d<double,3>& Particle_Coord, double Radius,
double LocalCoordSystem[3][3], double& DistParticleToEdge, double& eta)
{
double IntersectionCoordEdge[3];
double normal_unit_vector[3];
double edge_unit_vector[3];
double module_edge_vector = 0.0;
double particle_vector[3];

for (unsigned int j = 0; j<3; j++)
{
edge_unit_vector[j] = Coord2[j] - Coord1[j];
particle_vector[j]  = Particle_Coord[j] - Coord1[j];
}

normalize(edge_unit_vector, module_edge_vector);
double projection_on_edge = DotProduct(particle_vector,edge_unit_vector);

for (unsigned int j = 0; j<3; j++)
{
IntersectionCoordEdge[j] = Coord1[j] + projection_on_edge*edge_unit_vector[j];
normal_unit_vector[j]   = Particle_Coord[j] - IntersectionCoordEdge[j];
}

normalize( normal_unit_vector, DistParticleToEdge);

eta = projection_on_edge / module_edge_vector;

if (DistParticleToEdge < Radius)
{
if ((eta>=0.0) && (eta<=1.0))
{
double dummy_length = 0.0;
double auxiliar_unit_vector[3];
CrossProduct(normal_unit_vector,edge_unit_vector,auxiliar_unit_vector);
normalize(auxiliar_unit_vector, dummy_length);

for (unsigned int j = 0; j<3; j++)
{
LocalCoordSystem[0][j] = edge_unit_vector[j];
LocalCoordSystem[1][j] = auxiliar_unit_vector[j];
LocalCoordSystem[2][j] = normal_unit_vector[j];
}

return true;
}
} 

return false;

}

static inline bool VertexCheck(const array_1d<double,3>& Coord, const array_1d<double,3>& Particle_Coord, double Radius, double LocalCoordSystem[3][3], double& DistParticleToVertex)
{
double dist_sq = 0.0;
array_1d<double, 3> normal_v;
for (unsigned int j = 0; j < 3; j++)
{
normal_v[j] = Particle_Coord[j] - Coord[j];
dist_sq += normal_v[j] * normal_v[j];
}
if (dist_sq <= Radius * Radius)
{
DistParticleToVertex = sqrt(dist_sq);
ComputeContactLocalCoordSystem(normal_v, DistParticleToVertex, LocalCoordSystem);
return true;
}
return false;
}


static inline bool FastVertexCheck(const array_1d<double,3>& Coord, const array_1d<double,3>& Particle_Coord, double Radius)
{
double dist_sq = 0.0;
array_1d<double, 3> normal_v;
for (unsigned int j = 0; j < 3; j++)
{
normal_v[j] = Particle_Coord[j] - Coord[j];
dist_sq += normal_v[j] * normal_v[j];
}
if (dist_sq <= Radius * Radius) return true;
return false;
}










static inline void GetRotationMatrix(const array_1d<double, 3>& EulerAngles, double rotation_matrix[3][3]) {

double cosA=cos(EulerAngles[0]);
double sinA=sin(EulerAngles[0]);
double cosB=cos(EulerAngles[1]);
double sinB=sin(EulerAngles[1]);
double cosC=cos(EulerAngles[2]);
double sinC=sin(EulerAngles[2]);

rotation_matrix[0][0] = cosC*cosA - cosB*sinA*sinC;
rotation_matrix[0][1] = -sinC*cosA - cosB*sinA*cosC;
rotation_matrix[0][2] = sinB*sinA;
rotation_matrix[1][0] = cosC*sinA + cosB*cosA*sinC;
rotation_matrix[1][1] = -sinC*sinA + cosB*cosA*cosC;
rotation_matrix[1][2] = -sinB*cosA;
rotation_matrix[2][0] = sinC*sinB;
rotation_matrix[2][1] = cosC*sinB;
rotation_matrix[2][2] = cosB;

return;
}

static inline void GetEulerAngles(const double rotation_matrix[3][3], array_1d<double, 3 > & EulerAngles)
{
if (rotation_matrix[2][2] < 1.0)
{
if (rotation_matrix[2][2] > -1.0) {
EulerAngles[0] = atan2(rotation_matrix[0][2], -rotation_matrix[1][2]);
EulerAngles[1] = acos(rotation_matrix[2][2]);
EulerAngles[2] = atan2(rotation_matrix[2][0], rotation_matrix[2][1]);
}
else 
{
EulerAngles[0] = -atan2(-rotation_matrix[0][1], rotation_matrix[0][0]);
EulerAngles[1] = Globals::Pi;
EulerAngles[2] = 0;
}
}
else 
{
EulerAngles[0] = atan2(-rotation_matrix[0][1], rotation_matrix[0][0]);
EulerAngles[1] = 0;
EulerAngles[2] = 0;
}

return;
}

static inline void GetGiDEulerAngles(const BoundedMatrix<double, 3, 3>& rotation_matrix, array_1d<double, 3>& EulerAngles) {
const double numerical_limit = std::numeric_limits<double>::epsilon();
const double two_pi = 3.1415926535897932384626433 * 2.0;
if(rotation_matrix(2, 2)<1.0-numerical_limit && rotation_matrix(2, 2)>-1.0+numerical_limit){
const double senb=sqrt(1.0-rotation_matrix(2, 2)*rotation_matrix(2, 2));
EulerAngles[1]=acos(rotation_matrix(2, 2));
EulerAngles[2]=acos(rotation_matrix(1, 2)/senb);
if(rotation_matrix(0, 2)/senb<0.0) EulerAngles[2]=two_pi - EulerAngles[2];
EulerAngles[0]=acos(-rotation_matrix(2, 1)/senb);
if(rotation_matrix(2, 0)/senb<0.0) EulerAngles[0]=two_pi - EulerAngles[0];
} else {
EulerAngles[1]=acos(rotation_matrix(2, 2));
EulerAngles[0]=0.0;
EulerAngles[2]=acos(rotation_matrix(0, 0));
if(-rotation_matrix(1, 0)<0.0) EulerAngles[2]=two_pi - EulerAngles[2];
}
}

inline void QuaternionToGiDEulerAngles(const Quaternion<double>& quaternion, array_1d<double, 3>& EulerAngles) {
BoundedMatrix<double, 3, 3> rotation_matrix = ZeroMatrix(3,3);
quaternion.ToRotationMatrix(rotation_matrix);
GetGiDEulerAngles(rotation_matrix, EulerAngles);
}


static inline void TriAngleArea(double Coord1[3], double Coord2[3], double Coord3[3], double& area)
{
int k;
double Vector1[3],Vector2[3],Vector0[3];
for (k = 0;k < 3; k++)
{
Vector1[k] = Coord3[k] - Coord1[k];
Vector2[k] = Coord2[k] - Coord1[k];
}

CrossProduct(Vector1, Vector2, Vector0);
area = 0.5 * DEM_MODULUS_3(Vector0);
}

static inline void TriAngleWeight(double Coord1[3], double Coord2[3], double Coord3[3], double JudgeCoord[3], double Weight[3])
{
double area[3], s;
TriAngleArea(Coord1, Coord2, JudgeCoord, area[0]);
TriAngleArea(Coord2, Coord3, JudgeCoord, area[1]);
TriAngleArea(Coord3, Coord1, JudgeCoord, area[2]);

TriAngleArea(Coord1, Coord2, Coord3, s);
const double s_inv = 1.0 / s;
Weight[0] = area[1] * s_inv;
Weight[1] = area[2] * s_inv;
Weight[2] = area[0] * s_inv;
}

static inline void QuadAngleWeight(double Coord1[3], double Coord2[3], double Coord3[3], double Coord4[3], double JudgeCoord[3], double Weight[4])
{
double area[4], s1, s2, s;
TriAngleArea(Coord1, Coord2, JudgeCoord, area[0]);
TriAngleArea(Coord2, Coord3, JudgeCoord, area[1]);
TriAngleArea(Coord3, Coord4, JudgeCoord, area[2]);
TriAngleArea(Coord4, Coord1, JudgeCoord, area[3]);

TriAngleArea(Coord1, Coord2, Coord3, s1);
TriAngleArea(Coord1, Coord3, Coord4, s2);

s = s1 + s2;

if (fabs(area[0] + area[1] + area[2] + area[3] - s) < 1.0e-15) 
{
double QuadNormArea = 1 / ((area[0] + area[2]) * (area[1] + area[3]));

Weight[0] = (area[1] * area[2]) * QuadNormArea;
Weight[1] = (area[2] * area[3]) * QuadNormArea;
Weight[2] = (area[3] * area[0]) * QuadNormArea;
Weight[3] = (area[0] * area[1]) * QuadNormArea;
}
}

static inline void AreaAndCentroidCircularSector(double C[3], double Radius, double P1[3], double P2[3], double Normal[3], double& Area, double CoMSC[3])
{
double a[3]           = {0.0};
double c[3]           = {0.0};
double bisection[3]   = {0.0};
double norm_a         = 0.0;

for (unsigned int index = 0;index<3;index++) {

a[index] = P1[index]-C[index];
c[index] = P2[index]-P1[index];

}

CrossProduct(Normal,c,bisection);
normalize(bisection);
double dot_product = DotProduct(bisection,a);

if (dot_product<0.0) {

for (unsigned int index = 0;index<3;index++) {

bisection[index] = -bisection[index];
}

dot_product = -dot_product;
}

module(a, norm_a);

double cos_alpha = dot_product/norm_a;
double alpha = acos(cos_alpha);
double sin_alpha = std::sin(alpha);

Area = Radius*Radius*alpha;
double dist = 0.66666666666666*(Radius*sin_alpha/alpha);
for (unsigned int index = 0;index<3;index++) {
CoMSC[index] = C[index]+dist*bisection[index];
}

}

static inline void AlternativeAreaCircularSegment(double Radius, double tol_Radius, double V0V1[3], double V0CC[3], double Normal[3], double& AreaSC, bool& flag)
{

double normal_outwards[3] = {0.0};
flag = false;
AreaSC = 0.0;

CrossProduct(V0V1, Normal, normal_outwards);
normalize(normal_outwards);

double dist = DotProduct(normal_outwards,V0CC);
double delta_circle = Radius + dist; 

if ((delta_circle > tol_Radius) && (delta_circle - 2*Radius < -tol_Radius)) {

flag = true;
double b = sqrt(delta_circle*(2*Radius-delta_circle));
AreaSC   = 2.0*Radius*Radius*atan(delta_circle/b)-b*(Radius-delta_circle);
}
}

static inline void AreaAndCentroidCircularSegment(double Centre[3], double Radius, double tol_Radius, double V0[3], double V1[3], double Normal[3], double& AreaSegC, double CoMSegC[3], bool& flag)
{
double V0V1[3]            = {0.0};
double V0CC[3]            = {0.0};
double a[3]               = {0.0};
double normal_outwards[3] = {0.0};
double Radius_SQ          = 0.0;
double distance_V0V1      = 0.0;
double dist_CoM           = 0.0;
AreaSegC                  = 0.0;
flag = false;

for (unsigned int index = 0; index<3; index++) {

V0V1[index]     = V1[index] - V0[index];
V0CC[index]     = Centre[index] - V0[index];
}

GeometryFunctions::CrossProduct(V0V1,Normal,normal_outwards);
GeometryFunctions::normalize(V0V1,distance_V0V1);

double distV0 =  GeometryFunctions::DotProduct(V0CC,V0V1);

if ((distV0 > 0.0) && (distV0 < distance_V0V1)) {

GeometryFunctions::normalize(normal_outwards);
double dist_normal   = GeometryFunctions::DotProduct(normal_outwards,V0CC);
double delta_circle  = Radius + dist_normal; 

if ((delta_circle > tol_Radius) && ( delta_circle - 2.0*Radius < -tol_Radius)) {

Radius_SQ = Radius*Radius;
double semi_dist = sqrt(Radius_SQ - dist_normal*dist_normal);
flag = true;

for (unsigned int index = 0;index<3;index++) {

a[index] = V0[index] + (distV0 - semi_dist)*V0V1[index] - Centre[index]; 
}

double cos_alpha = GeometryFunctions::DotProduct(a,normal_outwards)/(GeometryFunctions::module(a)*GeometryFunctions::module(normal_outwards));
double alpha = acos(cos_alpha);
double sin_alpha = std::sin(alpha);

AreaSegC = Radius_SQ*(alpha-sin_alpha*cos_alpha);

if (fabs(sin_alpha) < tol_Radius) {dist_CoM=0.0;}

else {dist_CoM = 0.6666666666666 * (Radius*sin_alpha*sin_alpha*sin_alpha/(alpha-sin_alpha*cos_alpha));}

for (unsigned int index = 0; index<3; index++) {
CoMSegC[index] = Centre[index] + dist_CoM*normal_outwards[index];
}
} 

} 

}

static inline void AreaAndCentroidTriangle(double Coord1[3], double Coord2[3], double Coord3[3], double& area, double CoMTri[3]) {

TriAngleArea(Coord1,Coord2,Coord3,area);

for (unsigned int index =0; index<3; index++) {

CoMTri[index] = 0.33333333333333 * (Coord1[index]+Coord2[index]+Coord3[index]);
}

} 

} 

} 

#endif	
