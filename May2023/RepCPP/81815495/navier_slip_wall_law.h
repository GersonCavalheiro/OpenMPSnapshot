
#pragma once





#include "includes/cfd_variables.h"
#include "includes/condition.h"
#include "includes/define.h"
#include "includes/process_info.h"

#include "fluid_dynamics_application_variables.h"


namespace Kratos
{







template<std::size_t TDim, std::size_t TNumNodes>
class NavierSlipWallLaw
{
public:

KRATOS_CLASS_POINTER_DEFINITION(NavierSlipWallLaw);

static constexpr std::size_t BlockSize = TDim+1;

static constexpr std::size_t LocalSize = TNumNodes*BlockSize;

using SizeType = Condition::SizeType;

using IndexType = Condition::IndexType;

using VectorType = Condition::VectorType;

using MatrixType = Condition::MatrixType;


NavierSlipWallLaw() = delete;

NavierSlipWallLaw(NavierSlipWallLaw const& rOther) = delete;

~NavierSlipWallLaw() = default;





static void AddWallModelLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const Condition* pCondition,
const ProcessInfo& rCurrentProcessInfo)
{
WallLawDataContainer wall_law_data;
wall_law_data.Initialize(*pCondition);

BoundedMatrix<double,TDim,TDim> tang_proj_mat;
SetTangentialProjectionMatrix(wall_law_data.Normal, tang_proj_mat);

const SizeType n_gauss = wall_law_data.GaussPtsWeights.size();
for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
const double w_gauss = wall_law_data.GaussPtsWeights[i_gauss];
const auto& N_gauss = row(wall_law_data.ShapeFunctionsContainer, i_gauss);

double gp_slip_length = 0.0;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
gp_slip_length += N_gauss[i_node] * wall_law_data.NodalSlipLength[i_node];
}

const double aux_val = w_gauss * wall_law_data.DynamicViscosity / gp_slip_length;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
for (IndexType j_node = 0; j_node < TNumNodes; ++j_node) {
const auto& r_v_j = wall_law_data.NodalWallVelocities[j_node];
for (IndexType d1 = 0; d1 < TDim; ++d1) {
for (IndexType d2 = 0; d2 < TDim; ++d2) {
rRightHandSideVector[i_node*BlockSize + d2] += aux_val * N_gauss[i_node] * N_gauss[j_node] * tang_proj_mat(d1,d2) * r_v_j[d1];
rLeftHandSideMatrix(i_node*BlockSize + d1, j_node*BlockSize + d2) -= aux_val * N_gauss[i_node] * N_gauss[j_node] * tang_proj_mat(d1,d2);
}
}
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

BoundedMatrix<double,TDim,TDim> tang_proj_mat;
SetTangentialProjectionMatrix(wall_law_data.Normal, tang_proj_mat);

const SizeType n_gauss = wall_law_data.GaussPtsWeights.size();
for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
const double w_gauss = wall_law_data.GaussPtsWeights[i_gauss];
const auto& N_gauss = row(wall_law_data.ShapeFunctionsContainer, i_gauss);

double gp_slip_length = 0.0;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
gp_slip_length += N_gauss[i_node] * wall_law_data.NodalSlipLength[i_node];
}

const double aux_val = w_gauss * wall_law_data.DynamicViscosity / gp_slip_length;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
for (IndexType j_node = 0; j_node < TNumNodes; ++j_node) {
for (IndexType d1 = 0; d1 < TDim; ++d1) {
for (IndexType d2 = 0; d2 < TDim; ++d2) {
rLeftHandSideMatrix(i_node*BlockSize + d1, j_node*BlockSize + d2) -= aux_val * N_gauss[i_node] * N_gauss[j_node] * tang_proj_mat(d1,d2);
}
}
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

BoundedMatrix<double,TDim,TDim> tang_proj_mat;
SetTangentialProjectionMatrix(wall_law_data.Normal, tang_proj_mat);

const SizeType n_gauss = wall_law_data.GaussPtsWeights.size();
for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
const double w_gauss = wall_law_data.GaussPtsWeights[i_gauss];
const auto& N_gauss = row(wall_law_data.ShapeFunctionsContainer, i_gauss);

double gp_slip_length = 0.0;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
gp_slip_length += N_gauss[i_node] * wall_law_data.NodalSlipLength[i_node];
}

const double aux_val = w_gauss * wall_law_data.DynamicViscosity / gp_slip_length;
for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
for (IndexType j_node = 0; j_node < TNumNodes; ++j_node) {
const auto& r_v_j = wall_law_data.NodalWallVelocities[j_node];
for (IndexType d1 = 0; d1 < TDim; ++d1) {
for (IndexType d2 = 0; d2 < TDim; ++d2) {
rRightHandSideVector[i_node*BlockSize + d2] += aux_val * N_gauss[i_node] * N_gauss[j_node] * tang_proj_mat(d1,d2) * r_v_j[d1];
}
}
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
buffer << "NavierSlipWallLaw";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const
{
rOStream << "NavierSlipWallLaw";
}

void PrintData(std::ostream& rOStream) const {}



private:


struct WallLawDataContainer
{
double DynamicViscosity;
array_1d<double,3> Normal;
Vector GaussPtsWeights;
Matrix ShapeFunctionsContainer;
array_1d<double, TNumNodes> NodalSlipLength;
array_1d<array_1d<double,3>,TNumNodes> NodalWallVelocities;

void Initialize(const Condition& rCondition)
{
const auto& r_parent_element = rCondition.GetValue(NEIGHBOUR_ELEMENTS)[0];
DynamicViscosity = r_parent_element.GetProperties().GetValue(DYNAMIC_VISCOSITY);

const auto& r_geom = rCondition.GetGeometry();
Normal = r_geom.UnitNormal(0, GeometryData::IntegrationMethod::GI_GAUSS_1);

const auto& r_integration_pts = r_geom.IntegrationPoints(GeometryData::IntegrationMethod::GI_GAUSS_2);
const SizeType n_gauss = r_integration_pts.size();
r_geom.DeterminantOfJacobian(GaussPtsWeights, GeometryData::IntegrationMethod::GI_GAUSS_2);
for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
GaussPtsWeights[i_gauss] = GaussPtsWeights[i_gauss] * r_integration_pts[i_gauss].Weight();
}
ShapeFunctionsContainer = r_geom.ShapeFunctionsValues(GeometryData::IntegrationMethod::GI_GAUSS_2);

for (std::size_t i_node = 0; i_node < TNumNodes; ++i_node) {
const auto& r_node = r_geom[i_node];
const double aux_slip_length = r_node.GetValue(SLIP_LENGTH);
KRATOS_ERROR_IF(aux_slip_length < 1.0e-12) << "Negative or zero 'SLIP_LENGTH' at node " << r_node.Id() << "." << std::endl;
NodalSlipLength[i_node] = aux_slip_length;
noalias(NodalWallVelocities[i_node]) = r_node.FastGetSolutionStepValue(VELOCITY) - r_node.FastGetSolutionStepValue(MESH_VELOCITY);
}
}
};

static void SetTangentialProjectionMatrix(
const array_1d<double,3>& rUnitNormal,
BoundedMatrix<double,TDim,TDim>& rTangProjMat)
{
noalias(rTangProjMat) = IdentityMatrix(TDim,TDim);
for (std::size_t d1 = 0; d1 < TDim; ++d1) {
for (std::size_t d2 = 0; d2 < TDim; ++d2) {
rTangProjMat(d1,d2) -= rUnitNormal[d1]*rUnitNormal[d2];
}
}
}

}; 






}  
