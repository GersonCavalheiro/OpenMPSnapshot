
#if !defined( KRATOS_PARTITIONED_FSI_UTILITIES )
#define  KRATOS_PARTITIONED_FSI_UTILITIES



#include <set>
#include <typeinfo>




#include "includes/define.h"
#include "includes/variables.h"
#include "includes/mesh_moving_variables.h"
#include "includes/fsi_variables.h"
#include "containers/array_1d.h"
#include "includes/model_part.h"
#include "includes/communicator.h"
#include "includes/ublas_interface.h"
#include "utilities/binbased_fast_point_locator.h"
#include "utilities/math_utils.h"
#include "utilities/normal_calculation_utils.h"
#include "utilities/openmp_utils.h"
#include "utilities/variable_utils.h"


namespace Kratos
{






















template<class TSpace, class TValueType, unsigned int TDim>
class PartitionedFSIUtilities
{

public:




typedef typename TSpace::VectorType                     VectorType;
typedef typename TSpace::MatrixType                     MatrixType;

typedef typename TSpace::VectorPointerType              VectorPointerType;
typedef typename TSpace::MatrixPointerType              MatrixPointerType;

KRATOS_CLASS_POINTER_DEFINITION( PartitionedFSIUtilities );





PartitionedFSIUtilities(){}


PartitionedFSIUtilities(const PartitionedFSIUtilities& Other) = delete;


virtual ~PartitionedFSIUtilities() = default;






void CreateCouplingSkin(
const ModelPart &rOriginInterfaceModelPart,
ModelPart &rDestinationInterfaceModelPart)
{
const auto& r_communicator = rOriginInterfaceModelPart.GetCommunicator();
KRATOS_ERROR_IF(r_communicator.GlobalNumberOfNodes() == 0) << "Origin model part has no nodes." << std::endl;
KRATOS_ERROR_IF(r_communicator.GlobalNumberOfConditions() == 0) << "Origin model part has no conditions." << std::endl;

KRATOS_ERROR_IF(rDestinationInterfaceModelPart.IsSubModelPart()) << "Destination model part must be a root model part." << std::endl;
KRATOS_ERROR_IF(rDestinationInterfaceModelPart.NumberOfNodes() != 0) << "Destination interface model part should be empty. Current number of nodes: " << rDestinationInterfaceModelPart.NumberOfNodes() << std::endl;
KRATOS_ERROR_IF(rDestinationInterfaceModelPart.NumberOfElements() != 0) << "Destination interface model part should be empty. Current number of elements: " << rDestinationInterfaceModelPart.NumberOfElements() << std::endl;
KRATOS_ERROR_IF(rDestinationInterfaceModelPart.NumberOfConditions() != 0) << "Destination interface model part should be empty. Current number of conditions: " << rDestinationInterfaceModelPart.NumberOfConditions() << std::endl;

if (rOriginInterfaceModelPart.IsDistributed()) {
for (const auto &r_node : rOriginInterfaceModelPart.Nodes()) {
auto p_new_node = rDestinationInterfaceModelPart.CreateNewNode(r_node.Id(), r_node);
p_new_node->FastGetSolutionStepValue(PARTITION_INDEX) = r_node.FastGetSolutionStepValue(PARTITION_INDEX);
}
} else {
for (const auto &r_node : rOriginInterfaceModelPart.Nodes()) {
rDestinationInterfaceModelPart.CreateNewNode(r_node.Id(), r_node);
}
}

for (const auto &r_cond: rOriginInterfaceModelPart.Conditions()) {
std::vector<ModelPart::IndexType> nodes_vect;
for (const auto &r_node : r_cond.GetGeometry()) {
nodes_vect.push_back(r_node.Id());
}

rDestinationInterfaceModelPart.CreateNewCondition(this->GetSkinConditionName(), r_cond.Id(), nodes_vect, r_cond.pGetProperties());
}
}


int GetInterfaceResidualSize(ModelPart& rInterfaceModelPart)
{
double A; 
unsigned int block_size = typeid(TValueType).hash_code() == typeid(A).hash_code() ? 1 : TDim;
int local_number_of_nodes = (rInterfaceModelPart.GetCommunicator().LocalMesh().NumberOfNodes()) * block_size;
return rInterfaceModelPart.GetCommunicator().GetDataCommunicator().SumAll(local_number_of_nodes);
}


double GetInterfaceArea(ModelPart& rInterfaceModelPart)
{
double interface_area = 0.0;

auto& rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::ConditionIterator local_mesh_conditions_begin = rLocalMesh.ConditionsBegin();
#pragma omp parallel for firstprivate(local_mesh_conditions_begin) reduction(+:interface_area)
for(int k=0; k < static_cast<int>(rLocalMesh.NumberOfConditions()); ++k) {
const ModelPart::ConditionIterator it_cond = local_mesh_conditions_begin+k;
const Condition::GeometryType& rGeom = it_cond->GetGeometry();
interface_area += rGeom.Length();
}

return rInterfaceModelPart.GetCommunicator().GetDataCommunicator().SumAll(interface_area);
}


virtual VectorPointerType SetUpInterfaceVector(ModelPart& rInterfaceModelPart)
{
VectorPointerType p_int_vector = TSpace::CreateEmptyVectorPointer();
const unsigned int residual_size = this->GetInterfaceResidualSize(rInterfaceModelPart);
if (TSpace::Size(*p_int_vector) != residual_size){
TSpace::Resize(p_int_vector, residual_size);
}
TSpace::SetToZero(*p_int_vector);
return p_int_vector;
}


void InitializeInterfaceVector(
const ModelPart& rInterfaceModelPart,
const Variable<TValueType> &rOriginVariable,
VectorType &rInterfaceVector)
{
auto &r_local_mesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
auto nodes_begin = r_local_mesh.NodesBegin();
#pragma omp parallel for firstprivate(nodes_begin)
for (int i_node = 0; i_node < static_cast<int>(r_local_mesh.NumberOfNodes()); ++i_node) {
auto it_node = nodes_begin + i_node;
const auto &r_value = it_node->FastGetSolutionStepValue(rOriginVariable);
this->AuxSetLocalValue(rInterfaceVector, r_value, i_node);
}
}


virtual void ComputeInterfaceResidualVector(
ModelPart &rInterfaceModelPart,
const Variable<TValueType> &rOriginalVariable,
const Variable<TValueType> &rModifiedVariable,
const Variable<TValueType> &rResidualVariable,
VectorType &rInterfaceResidual,
const std::string ResidualType = "nodal",
const Variable<double> &rResidualNormVariable = FSI_INTERFACE_RESIDUAL_NORM)
{
TSpace::SetToZero(rInterfaceResidual);

if (ResidualType == "nodal") {
this->ComputeNodeByNodeResidual(rInterfaceModelPart, rOriginalVariable, rModifiedVariable, rResidualVariable);
} else if (ResidualType == "consistent") {
this->ComputeConsistentResidual(rInterfaceModelPart, rOriginalVariable, rModifiedVariable, rResidualVariable);
} else {
KRATOS_ERROR << "Provided interface residual type " << ResidualType << " is not available. Available options are \"nodal\" and \"consistent\"" << std::endl;
}

auto& rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k = 0; k < static_cast<int>(rLocalMesh.NumberOfNodes()); ++k) {
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;
const auto &r_res_value = it_node->FastGetSolutionStepValue(rResidualVariable);
this->AuxSetLocalValue(rInterfaceResidual, r_res_value, k);
}

rInterfaceModelPart.GetProcessInfo().GetValue(rResidualNormVariable) = TSpace::TwoNorm(rInterfaceResidual);
}


double ComputeInterfaceResidualNorm(
ModelPart &rInterfaceModelPart,
const Variable<TValueType> &rOriginalVariable,
const Variable<TValueType> &rModifiedVariable,
const Variable<TValueType> &rResidualVariable,
const std::string ResidualType = "nodal")
{
VectorPointerType p_interface_residual = this->SetUpInterfaceVector(rInterfaceModelPart);

if (ResidualType == "nodal") {
this->ComputeNodeByNodeResidual(rInterfaceModelPart, rOriginalVariable, rModifiedVariable, rResidualVariable);
} else if (ResidualType == "consistent") {
this->ComputeConsistentResidual(rInterfaceModelPart, rOriginalVariable, rModifiedVariable, rResidualVariable);
} else {
KRATOS_ERROR << "Provided interface residual type " << ResidualType << " is not available. Available options are \"nodal\" and \"consistent\"" << std::endl;
}

auto &rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k=0; k<static_cast<int>(rLocalMesh.NumberOfNodes()); ++k) {
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin + k;
const auto &r_res_value = it_node->FastGetSolutionStepValue(rResidualVariable);
this->AuxSetLocalValue(*p_interface_residual, r_res_value, k);
}

return TSpace::TwoNorm(*p_interface_residual);
}


virtual void AuxSetLocalValue(
VectorType &rInterfaceResidual,
const double &rResidualValue,
const int AuxPosition)
{
this->SetLocalValue(rInterfaceResidual, AuxPosition, rResidualValue);
}


virtual void AuxSetLocalValue(
VectorType &rInterfaceResidual,
const array_1d<double,3> &rResidualValue,
const int AuxPosition)
{
const unsigned int base_i = AuxPosition * TDim;
for (unsigned int jj = 0; jj < TDim; ++jj) {
this->SetLocalValue(rInterfaceResidual, base_i + jj, rResidualValue[jj]);
}
}


virtual void UpdateInterfaceValues(
ModelPart &rInterfaceModelPart,
const Variable<TValueType> &rSolutionVariable,
const VectorType &rCorrectedGuess)
{
auto& rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k = 0; k < static_cast<int>(rLocalMesh.NumberOfNodes()); ++k){
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin + k;
TValueType &r_updated_value = it_node->FastGetSolutionStepValue(rSolutionVariable);
this->UpdateInterfaceLocalValue(rCorrectedGuess, r_updated_value, k);
}

rInterfaceModelPart.GetCommunicator().SynchronizeVariable(rSolutionVariable);
}


void UpdateInterfaceLocalValue(
const VectorType &rCorrectedGuess,
double &rValueToUpdate,
const int AuxPosition)
{
rValueToUpdate = this->GetLocalValue(rCorrectedGuess, AuxPosition);
}


void UpdateInterfaceLocalValue(
const VectorType &rCorrectedGuess,
array_1d<double, 3> &rValueToUpdate,
const int AuxPosition)
{
const int base_i = AuxPosition * TDim;
for (unsigned int jj = 0; jj < TDim; ++jj){
rValueToUpdate[jj] = this->GetLocalValue(rCorrectedGuess, base_i + jj);
}
}


virtual void ComputeAndPrintFluidInterfaceNorms(ModelPart& rInterfaceModelPart)
{
double p_norm = 0.0;
double vx_norm = 0.0;
double vy_norm = 0.0;
double vz_norm = 0.0;
double rx_norm = 0.0;
double ry_norm = 0.0;
double rz_norm = 0.0;
double ux_mesh_norm = 0.0;
double uy_mesh_norm = 0.0;
double uz_mesh_norm = 0.0;

auto& rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin) reduction(+ : p_norm, vx_norm, vy_norm, vz_norm, rx_norm, ry_norm, rz_norm, ux_mesh_norm, uy_mesh_norm, uz_mesh_norm)
for(int k=0; k<static_cast<int>(rLocalMesh.NumberOfNodes()); ++k)
{
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;

p_norm += std::pow(it_node->FastGetSolutionStepValue(PRESSURE), 2);
vx_norm += std::pow(it_node->FastGetSolutionStepValue(VELOCITY_X), 2);
vy_norm += std::pow(it_node->FastGetSolutionStepValue(VELOCITY_Y), 2);
vz_norm += std::pow(it_node->FastGetSolutionStepValue(VELOCITY_Z), 2);
rx_norm += std::pow(it_node->FastGetSolutionStepValue(REACTION_X), 2);
ry_norm += std::pow(it_node->FastGetSolutionStepValue(REACTION_Y), 2);
rz_norm += std::pow(it_node->FastGetSolutionStepValue(REACTION_Z), 2);
ux_mesh_norm += std::pow(it_node->FastGetSolutionStepValue(MESH_DISPLACEMENT_X), 2);
uy_mesh_norm += std::pow(it_node->FastGetSolutionStepValue(MESH_DISPLACEMENT_Y), 2);
uz_mesh_norm += std::pow(it_node->FastGetSolutionStepValue(MESH_DISPLACEMENT_Z), 2);
}

std::vector<double> local_data{p_norm, vx_norm, vy_norm, vz_norm, rx_norm, ry_norm, rz_norm, ux_mesh_norm, uy_mesh_norm, uz_mesh_norm};
std::vector<double> global_sum{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
rInterfaceModelPart.GetCommunicator().GetDataCommunicator().SumAll(local_data, global_sum);

p_norm       = global_sum[0];
vx_norm      = global_sum[1];
vy_norm      = global_sum[2];
vz_norm      = global_sum[3];
rx_norm      = global_sum[4];
ry_norm      = global_sum[5];
rz_norm      = global_sum[6];
ux_mesh_norm = global_sum[7];
uy_mesh_norm = global_sum[8];
uz_mesh_norm = global_sum[9];

if (rInterfaceModelPart.GetCommunicator().MyPID() == 0)
{
std::cout << " " << std::endl;
std::cout << "|p_norm| = " << std::sqrt(p_norm) << std::endl;
std::cout << "|vx_norm| = " << std::sqrt(vx_norm) << std::endl;
std::cout << "|vy_norm| = " << std::sqrt(vy_norm) << std::endl;
std::cout << "|vz_norm| = " << std::sqrt(vz_norm) << std::endl;
std::cout << "|rx_norm| = " << std::sqrt(rx_norm) << std::endl;
std::cout << "|ry_norm| = " << std::sqrt(ry_norm) << std::endl;
std::cout << "|rz_norm| = " << std::sqrt(rz_norm) << std::endl;
std::cout << "|ux_mesh_norm| = " << std::sqrt(ux_mesh_norm) << std::endl;
std::cout << "|uy_mesh_norm| = " << std::sqrt(uy_mesh_norm) << std::endl;
std::cout << "|uz_mesh_norm| = " << std::sqrt(uz_mesh_norm) << std::endl;
std::cout << " " << std::endl;
}
}


virtual void ComputeAndPrintStructureInterfaceNorms(ModelPart& rInterfaceModelPart)
{
double ux_norm = 0.0;
double uy_norm = 0.0;
double uz_norm = 0.0;

auto& rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin) reduction(+ : ux_norm, uy_norm, uz_norm)
for(int k=0; k<static_cast<int>(rLocalMesh.NumberOfNodes()); ++k)
{
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;
const array_1d<double, 3>& disp = it_node->FastGetSolutionStepValue(DISPLACEMENT);

ux_norm += std::pow(disp[0], 2);
uy_norm += std::pow(disp[1], 2);
uz_norm += std::pow(disp[2], 2);
}

std::vector<double> local_data{ux_norm, uy_norm, uz_norm};
std::vector<double> global_sum{0, 0, 0};
rInterfaceModelPart.GetCommunicator().GetDataCommunicator().SumAll(local_data, global_sum);

ux_norm = global_sum[0];
uy_norm = global_sum[1];
uz_norm = global_sum[2];

if (rInterfaceModelPart.GetCommunicator().MyPID() == 0)
{
std::cout << " " << std::endl;
std::cout << "|ux_norm| = " << std::sqrt(ux_norm) << std::endl;
std::cout << "|uy_norm| = " << std::sqrt(uy_norm) << std::endl;
std::cout << "|uz_norm| = " << std::sqrt(uz_norm) << std::endl;
std::cout << " " << std::endl;
}
}


virtual void CheckCurrentCoordinatesFluid(ModelPart& rModelPart, const double tolerance)
{
auto& rLocalMesh = rModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k=0; k<static_cast<int>(rLocalMesh.NumberOfNodes()); ++k)
{
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;
const array_1d<double, 3>& disp = it_node->FastGetSolutionStepValue(MESH_DISPLACEMENT);

if (std::fabs(it_node->X() - (it_node->X0() + disp[0])) > tolerance)
{
KRATOS_ERROR << "Node " << it_node->Id() << " X != X0 + deltaX";
}
if (std::fabs(it_node->Y() - (it_node->Y0() + disp[1])) > tolerance)
{
KRATOS_ERROR << "Node " << it_node->Id() << " Y != Y0 + deltaY";
}
if (std::fabs(it_node->Z() - (it_node->Z0() + disp[2])) > tolerance)
{
KRATOS_ERROR << "Node " << it_node->Id() << " Z != Z0 + deltaZ";
}
}
}


virtual void CheckCurrentCoordinatesStructure(ModelPart& rModelPart, const double tolerance)
{
auto& rLocalMesh = rModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k=0; k<static_cast<int>(rLocalMesh.NumberOfNodes()); ++k)
{
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;
const array_1d<double, 3>& disp = it_node->FastGetSolutionStepValue(DISPLACEMENT);

if (std::fabs(it_node->X() - (it_node->X0() + disp[0])) > tolerance)
{
KRATOS_ERROR << "Node " << it_node->Id() << " X != X0 + deltaX";
}
if (std::fabs(it_node->Y() - (it_node->Y0() + disp[1])) > tolerance)
{
KRATOS_ERROR << "Node " << it_node->Id() << " Y != Y0 + deltaY";
}
if (std::fabs(it_node->Z() - (it_node->Z0() + disp[2])) > tolerance)
{
KRATOS_ERROR << "Node " << it_node->Id() << " Z != Z0 + deltaZ";
}
}
}


void EmbeddedPressureToPositiveFacePressureInterpolator(
ModelPart &rFluidModelPart,
ModelPart &rStructureSkinModelPart)
{
BinBasedFastPointLocator<TDim> bin_based_locator(rFluidModelPart);

bin_based_locator.UpdateSearchDatabase();

Vector N;
Element::Pointer p_elem = nullptr;
#pragma omp parallel for firstprivate(N, p_elem)
for (int i_node = 0; i_node < static_cast<int>(rStructureSkinModelPart.NumberOfNodes()); ++i_node) {
auto it_node = rStructureSkinModelPart.NodesBegin() + i_node;
const bool found = bin_based_locator.FindPointOnMeshSimplified(it_node->Coordinates(), N, p_elem);
if (found) {
const auto &r_geom = p_elem->GetGeometry();
double &r_pres = it_node->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE);
r_pres = 0.0;
for (unsigned int i_node = 0; i_node < r_geom.PointsNumber(); ++i_node) {
r_pres += N[i_node] * r_geom[i_node].FastGetSolutionStepValue(PRESSURE);
}
}
}
}

void CalculateTractionFromPressureValues(
ModelPart& rModelPart,
const Variable<double>& rPressureVariable,
const Variable<array_1d<double,3>>& rTractionVariable,
const bool SwapTractionSign)
{
NormalCalculationUtils().CalculateOnSimplex(rModelPart);

std::function<double(const double, const array_1d<double,3>)> traction_modulus_func;
if (SwapTractionSign)  {
traction_modulus_func = [](const double PosPressure, const array_1d<double,3>& rNormal){return - PosPressure / norm_2(rNormal);};
} else {
traction_modulus_func = [](const double PosPressure, const array_1d<double,3>& rNormal){return PosPressure / norm_2(rNormal);};
}

block_for_each(rModelPart.Nodes(), [&](Node& rNode){
const array_1d<double,3>& r_normal = rNode.FastGetSolutionStepValue(NORMAL);
const double p_pos = rNode.FastGetSolutionStepValue(rPressureVariable);
noalias(rNode.FastGetSolutionStepValue(rTractionVariable)) = traction_modulus_func(p_pos, r_normal) * r_normal;
});

rModelPart.GetCommunicator().SynchronizeVariable(rTractionVariable);
}

void CalculateTractionFromPressureValues(
ModelPart& rModelPart,
const Variable<double>& rPositivePressureVariable,
const Variable<double>& rNegativePressureVariable,
const Variable<array_1d<double,3>>& rTractionVariable,
const bool SwapTractionSign)
{
NormalCalculationUtils().CalculateOnSimplex(rModelPart);

std::function<double(const double, const double, const array_1d<double,3>)> traction_modulus_func;
if (SwapTractionSign)  {
traction_modulus_func = [](const double PosPressure, const double NegPressure, const array_1d<double,3>& rNormal){return (NegPressure - PosPressure) / norm_2(rNormal);};
} else {
traction_modulus_func = [](const double PosPressure, const double NegPressure, const array_1d<double,3>& rNormal){return (PosPressure - NegPressure) / norm_2(rNormal);};
}

block_for_each(rModelPart.Nodes(), [&](Node& rNode){
const array_1d<double,3>& r_normal = rNode.FastGetSolutionStepValue(NORMAL);
const double p_pos = rNode.FastGetSolutionStepValue(rPositivePressureVariable);
const double p_neg = rNode.FastGetSolutionStepValue(rNegativePressureVariable);
noalias(rNode.FastGetSolutionStepValue(rTractionVariable)) = traction_modulus_func(p_pos, p_neg, r_normal) * r_normal;
});

rModelPart.GetCommunicator().SynchronizeVariable(rTractionVariable);
}


protected:

















std::string GetSkinElementName()
{
std::string element_name;
if constexpr (TDim == 2) {
element_name = "Element2D2N";
} else {
element_name = "Element3D3N";
}

return element_name;
}


std::string GetSkinConditionName()
{
if constexpr (TDim == 2) {
return "LineCondition2D2N";
} else {
return "SurfaceCondition3D3N";
}
}


void ComputeConsistentResidual(
ModelPart& rInterfaceModelPart,
const Variable<TValueType>& rOriginalVariable,
const Variable<TValueType>& rModifiedVariable,
const Variable<TValueType>& rErrorStorageVariable)
{
VariableUtils().SetVariable<TValueType>(rErrorStorageVariable, rErrorStorageVariable.Zero(), rInterfaceModelPart.Nodes());

#pragma omp parallel for
for(int i_cond = 0; i_cond < static_cast<int>(rInterfaceModelPart.NumberOfConditions()); ++i_cond) {

auto it_cond = rInterfaceModelPart.ConditionsBegin() + i_cond;

auto& rGeom = it_cond->GetGeometry();
const unsigned int n_nodes = rGeom.PointsNumber();

std::vector<TValueType> cons_res_vect(n_nodes);
for (unsigned int i = 0; i < n_nodes; ++i) {
cons_res_vect[i] = rErrorStorageVariable.Zero();
}

const auto &r_int_pts = rGeom.IntegrationPoints(GeometryData::IntegrationMethod::GI_GAUSS_2);
const auto N_container = rGeom.ShapeFunctionsValues(GeometryData::IntegrationMethod::GI_GAUSS_2);
Vector jac_gauss;
rGeom.DeterminantOfJacobian(jac_gauss, GeometryData::IntegrationMethod::GI_GAUSS_2);

for (unsigned int i_gauss = 0; i_gauss < r_int_pts.size(); ++i_gauss) {
const Vector N_gauss = row(N_container, i_gauss);
const double w_gauss = jac_gauss[i_gauss] * r_int_pts[i_gauss].Weight();

for (unsigned int i_node = 0; i_node < n_nodes; ++i_node) {
for (unsigned int j_node = 0; j_node < n_nodes; ++j_node) {
const double aux_val = w_gauss * N_gauss[i_node] * N_gauss[j_node];
const TValueType value = rGeom[j_node].FastGetSolutionStepValue(rOriginalVariable);
const TValueType value_projected = rGeom[j_node].FastGetSolutionStepValue(rModifiedVariable);
cons_res_vect[i_node] += aux_val * (value - value_projected);
}
}
}

for (unsigned int i_node = 0; i_node < n_nodes; ++i_node) {
rGeom[i_node].SetLock(); 
rGeom[i_node].FastGetSolutionStepValue(rErrorStorageVariable) += cons_res_vect[i_node];
rGeom[i_node].UnSetLock(); 
}

rInterfaceModelPart.GetCommunicator().AssembleCurrentData(rErrorStorageVariable);
}
}


void ComputeNodeByNodeResidual(
ModelPart& rInterfaceModelPart,
const Variable<TValueType>& rOriginalVariable,
const Variable<TValueType>& rModifiedVariable,
const Variable<TValueType>& rErrorStorageVariable)
{
auto& rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k = 0; k < static_cast<int>(rLocalMesh.NumberOfNodes()); ++k) {
ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;
auto &r_error_storage = it_node->FastGetSolutionStepValue(rErrorStorageVariable);
const auto &value_origin = it_node->FastGetSolutionStepValue(rOriginalVariable);
const auto &value_modified = it_node->FastGetSolutionStepValue(rModifiedVariable);
r_error_storage = value_modified - value_origin;
}
}

virtual void SetLocalValue(VectorType& rVector, int LocalRow, double Value) const
{
TSpace::SetValue(rVector,LocalRow,Value);
}

virtual double GetLocalValue(const VectorType& rVector, int LocalRow) const
{
return TSpace::GetValue(rVector,LocalRow);
}















private:





































}; 









} 

#endif 
