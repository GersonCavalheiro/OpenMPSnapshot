
#pragma once





#include "includes/cfd_variables.h"
#include "includes/condition.h"
#include "includes/define.h"
#include "includes/process_info.h"

#include "fluid_dynamics_application_variables.h"


namespace Kratos
{



template<std::size_t TDim, std::size_t TNumNodes>
class LinearLogWallLaw
{
public:

KRATOS_CLASS_POINTER_DEFINITION(LinearLogWallLaw);

static constexpr std::size_t BlockSize = TDim+1;

static constexpr std::size_t LocalSize = TNumNodes*BlockSize;

using SizeType = Condition::SizeType;

using IndexType = Condition::IndexType;

using VectorType = Condition::VectorType;

using MatrixType = Condition::MatrixType;


LinearLogWallLaw() = delete;

LinearLogWallLaw(LinearLogWallLaw const& rOther) = delete;

~LinearLogWallLaw() = default;



static void AddWallModelLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const Condition* pCondition,
const ProcessInfo& rCurrentProcessInfo)
{
WallLawDataContainer wall_law_data;
wall_law_data.Initialize(*pCondition);

const auto& r_geom = pCondition->GetGeometry();
const double w_gauss_lobatto = r_geom.DomainSize() / TNumNodes;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
const auto& r_aux_v = wall_law_data.NodalWallVelocities[i_node];
const double wall_vel_norm = norm_2(r_aux_v);

if (wall_vel_norm > ZeroTol) {
const double u_tau = wall_law_data.CalculateFrictionVelocity(wall_vel_norm, wall_law_data.NodalYWallValues[i_node]);

const double tmp = w_gauss_lobatto * std::pow(u_tau,2) * wall_law_data.Density / wall_vel_norm;
for (IndexType d = 0; d < TDim; ++d) {
rRightHandSideVector(i_node*BlockSize + d) -= tmp * r_aux_v[d];
rLeftHandSideMatrix(i_node*BlockSize + d, i_node*BlockSize + d) += tmp;
}
}
}
}


static void AddWallModelLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const Condition* pCondition,
const ProcessInfo& rCurrentProcessInfo)
{
WallLawDataContainer wall_law_data;
wall_law_data.Initialize(*pCondition);

const auto& r_geom = pCondition->GetGeometry();
const double w_gauss_lobatto = r_geom.DomainSize() / TNumNodes;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
const auto& r_aux_v = wall_law_data.NodalWallVelocities[i_node];
const double wall_vel_norm = norm_2(r_aux_v);

if (wall_vel_norm > ZeroTol) {
const double u_tau = wall_law_data.CalculateFrictionVelocity(wall_vel_norm, wall_law_data.NodalYWallValues[i_node]);

const double tmp = w_gauss_lobatto * std::pow(u_tau,2) * wall_law_data.Density / wall_vel_norm;
for (IndexType d = 0; d < TDim; ++d) {
rLeftHandSideMatrix(i_node*BlockSize + d, i_node*BlockSize + d) += tmp;
}
}
}
}


static void AddWallModelRightHandSide(
VectorType& rRightHandSideVector,
const Condition* pCondition,
const ProcessInfo& rCurrentProcessInfo)
{
WallLawDataContainer wall_law_data;
wall_law_data.Initialize(*pCondition);

const auto& r_geom = pCondition->GetGeometry();
const double w_gauss_lobatto = r_geom.DomainSize() / TNumNodes;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
const auto& r_aux_v = wall_law_data.NodalWallVelocities[i_node];
const double wall_vel_norm = norm_2(r_aux_v);

if (wall_vel_norm > ZeroTol) {
const double u_tau = wall_law_data.CalculateFrictionVelocity(wall_vel_norm, wall_law_data.NodalYWallValues[i_node]);

const double tmp = w_gauss_lobatto * std::pow(u_tau,2) * wall_law_data.Density / wall_vel_norm;
for (IndexType d = 0; d < TDim; ++d) {
rRightHandSideVector(i_node*BlockSize + d) -= tmp * r_aux_v[d];
}
}
}
}


static int Check(
const Condition* pCondition,
const ProcessInfo& rCurrentProcessInfo)
{
return 0;
}






std::string Info() const
{
std::stringstream buffer;
buffer << "LinearLogWallLaw";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const
{
rOStream << "LinearLogWallLaw";
}

void PrintData(std::ostream& rOStream) const {}



private:

static constexpr double BPlus = 5.2; 
static constexpr double InvKappa = 1.0/0.41; 
static constexpr double YPlusLimit = 10.9931899; 
static constexpr double ZeroTol = 1.0e-12; 



struct WallLawDataContainer
{
double Density;
double KinematicViscosity;
array_1d<double,TNumNodes> NodalYWallValues;
array_1d<array_1d<double,3>,TNumNodes> NodalWallVelocities;

void Initialize(const Condition& rCondition)
{
const auto& r_parent_element = rCondition.GetValue(NEIGHBOUR_ELEMENTS)[0];
Density = r_parent_element.GetProperties().GetValue(DENSITY);
KinematicViscosity = r_parent_element.GetProperties().GetValue(DYNAMIC_VISCOSITY) / Density;

const auto& r_geom = rCondition.GetGeometry();
for (std::size_t i_node = 0; i_node < TNumNodes; ++i_node) {
const auto& r_node = r_geom[i_node];
const double aux_y_wall = r_node.GetValue(Y_WALL);
KRATOS_ERROR_IF(aux_y_wall < ZeroTol) << "Negative or zero 'Y_WALL' in condition " << rCondition.Id() << "." << std::endl;
NodalYWallValues[i_node] = aux_y_wall;
noalias(NodalWallVelocities[i_node]) = r_node.FastGetSolutionStepValue(VELOCITY) - r_node.FastGetSolutionStepValue(MESH_VELOCITY);
}
}

double CalculateFrictionVelocity(
const double WallVelocityNorm,
const double YWall)
{
double u_tau = 0.0;
if (WallVelocityNorm > ZeroTol) {
u_tau = sqrt(WallVelocityNorm * KinematicViscosity / YWall); 
double y_plus = YWall * u_tau / KinematicViscosity; 

if (y_plus > YPlusLimit) {
double dx = 1e10;
const double rel_tol = 1e-6;
const std::size_t max_it = 100;
double u_plus = InvKappa * log(y_plus) + BPlus; 

std::size_t it = 0;
while (it < 100 && std::abs(dx) > rel_tol * u_tau) {
double f = u_tau * u_plus - WallVelocityNorm;
double df = u_plus + InvKappa;
dx = f/df;

u_tau -= dx;
y_plus = YWall * u_tau / KinematicViscosity;
u_plus = InvKappa * log(y_plus) + BPlus;
++it;
}
KRATOS_WARNING_IF("LinearLogWallLaw", it == max_it) << "Wall condition Newton-Raphson did not converge. Residual is " << dx << "." << std::endl;
}
}

return u_tau;
}
};

}; 






}  
