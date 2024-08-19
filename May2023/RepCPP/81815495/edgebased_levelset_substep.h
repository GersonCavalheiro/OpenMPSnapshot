
#if !defined(KRATOS_EDGEBASED_LEVELSET_SUBSTEP_FLUID_SOLVER_H_INCLUDED)
#define KRATOS_EDGEBASED_LEVELSET_SUBSTEP_FLUID_SOLVER_H_INCLUDED

#include <string>
#include <iostream>
#include <algorithm>

#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "includes/node.h"
#include "utilities/geometry_utilities.h"
#include "free_surface_application.h"
#include "custom_utilities/edge_data_c2c.h"
#include "utilities/reduction_utilities.h"

namespace Kratos
{
template <unsigned int TDim, class MatrixContainer, class TSparseSpace, class TLinearSolver>
class EdgeBasedLevelSetSubstep
{
public:
typedef EdgesStructureTypeC2C<TDim> CSR_Tuple;
typedef vector<CSR_Tuple> EdgesVectorType;
typedef vector<unsigned int> IndicesVectorType;
typedef vector<array_1d<double, TDim>> CalcVectorType;
typedef vector<double> ValuesVectorType;
typedef typename TSparseSpace::MatrixType TSystemMatrixType;
typedef typename TSparseSpace::VectorType TSystemVectorType;
typedef std::size_t SizeType;
EdgeBasedLevelSetSubstep(MatrixContainer &mr_matrix_container,
ModelPart &mr_model_part,
const double viscosity,
const double density,
bool use_mass_correction,
double stabdt_pressure_factor,
double stabdt_convection_factor,
double tau2_factor,
bool assume_constant_dp)
: mr_matrix_container(mr_matrix_container),
mr_model_part(mr_model_part),
mstabdt_pressure_factor(stabdt_pressure_factor),
mstabdt_convection_factor(stabdt_convection_factor),
mtau2_factor(tau2_factor),
massume_constant_dp(assume_constant_dp)
{
for (ModelPart::NodesContainerType::iterator it = mr_model_part.NodesBegin(); it != mr_model_part.NodesEnd(); it++)
it->FastGetSolutionStepValue(VISCOSITY) = viscosity;

mMolecularViscosity = viscosity;
mRho = density;
mdelta_t_avg = 1000.0;
max_dt = 1.0;
muse_mass_correction = use_mass_correction;
mshock_coeff = 0.7;
mWallLawIsActive = false;
mnumsubsteps = 5;
mmax_dt = 0.0;
mcorner_coefficient = 30.0;
medge_coefficient = 2.0;
std::cout << "Edge based level set substep solver is created" << std::endl;
};
~EdgeBasedLevelSetSubstep(){};

/ ;
}

std::cout << "max angle between normals found in the model = " << max_angle_overall << std::endl;

int edge_size = medge_nodes.size();
#pragma omp parallel for firstprivate(edge_size)
for (int i = 0; i < edge_size; i++)
{
int i_node = medge_nodes[i];
mWallReductionFactor[i_node] = medge_coefficient; 
}
int corner_size = mcorner_nodes.size();
for (int i = 0; i < corner_size; i++)
{
int i_node = mcorner_nodes[i];
mWallReductionFactor[i_node] = mcorner_coefficient; 
}
}
void ActivateClassicalWallResistance(double Ywall)
{
mWallLawIsActive = true;
mY_wall = Ywall;
for (unsigned int i = 0; i < mWallReductionFactor.size(); i++)
mWallReductionFactor[i] = 1.0;
}
double ComputeVolumeVariation()
{
ProcessInfo &CurrentProcessInfo = mr_model_part.GetProcessInfo();
double dt = CurrentProcessInfo[DELTA_TIME];
int inout_size = mInOutBoundaryList.size();
double vol_var = 0.0;

for (int i = 0; i < inout_size; i++)
{
unsigned int i_node = mInOutBoundaryList[i];
double dist = mdistances[i_node];
if (dist <= 0.0)
{
const array_1d<double, TDim> &U_i = mvel_n1[i_node];
const array_1d<double, TDim> &an_i = mInOutNormal[i_node];
double projection_length = 0.0;
for (unsigned int comp = 0; comp < TDim; comp++)
{
projection_length += U_i[comp] * an_i[comp];
}
vol_var += projection_length;
}
}
return -vol_var * dt;
}
double ComputeWetVolume()
{
KRATOS_TRY
mr_matrix_container.FillScalarFromDatabase(DISTANCE, mdistances, mr_model_part.Nodes());
double wet_volume = 0.0;

for (int i = 0; i < static_cast<int>(mdistances.size()); i++)
{
double dist = mdistances[i];
const double m = mr_matrix_container.GetLumpedMass()[i];
double porosity = mEps[i];
if (dist <= 0.0)
{
wet_volume += m / porosity;
}
}
return wet_volume;
KRATOS_CATCH("");
}
double ComputeTotalVolume()
{
KRATOS_TRY
mr_matrix_container.FillScalarFromDatabase(DISTANCE, mdistances, mr_model_part.Nodes());
double volume = 0.0;

for (int i = 0; i < static_cast<int>(mdistances.size()); i++)
{
const double m = mr_matrix_container.GetLumpedMass()[i];
double porosity = mEps[i];
volume += m / porosity;
}
return volume;
KRATOS_CATCH("");
}
void DiscreteVolumeCorrection(double expected_volume, double measured_volume)
{
double volume_error = expected_volume - measured_volume;
if (measured_volume < expected_volume)
{
double layer_volume = 0.0;
std::vector<unsigned int> first_outside;
int n_nodes = mdistances.size();
for (int i_node = 0; i_node < n_nodes; i_node++)
{
double dist = mdistances[i_node];
if (dist > 0.0) 
{
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
if (mdistances[j_neighbour] <= 0.0)
{
const double nodal_mass = 1.0 / mr_matrix_container.GetInvertedMass()[i_node];
if (nodal_mass < volume_error - layer_volume)
{
first_outside.push_back(i_node);
layer_volume += nodal_mass;
break;
}
}
}
}
for (unsigned int i = 0; i < first_outside.size(); i++)
{
unsigned int i_node = first_outside[i];
mdistances[i_node] = -mHavg[i_node];
}
}
}
mr_matrix_container.WriteScalarToDatabase(DISTANCE, mdistances, mr_model_part.Nodes());
}
void SetWallReductionCoefficients(double corner_coefficient, double edge_coefficient)
{
mcorner_coefficient = corner_coefficient;
medge_coefficient = edge_coefficient;
}
void ContinuousVolumeCorrection(double expected_volume, double measured_volume)
{
double volume_error = expected_volume - measured_volume;
if (volume_error == 0.0)
return;
if (measured_volume < expected_volume)
{
double layer_volume = 0.0;
std::vector<unsigned int> first_outside;
int n_nodes = mdistances.size();
for (int i_node = 0; i_node < n_nodes; i_node++)
{
double dist = mdistances[i_node];
bool is_bubble = true;
bool is_first_outside = false;
if (dist > 0.0) 
{
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
if (mdistances[j_neighbour] <= 0.0)
{
is_first_outside = true;
}
else
is_bubble = false;
}
}
if (is_first_outside && !is_bubble)
{
const double nodal_mass = 1.0 / mr_matrix_container.GetInvertedMass()[i_node];
first_outside.push_back(i_node);
layer_volume += nodal_mass;
}
}
if (layer_volume == 0.00)
return;
double ratio = volume_error / layer_volume;
if (ratio > 1.0)
ratio = 1.0;
if (ratio < 0.1) 
return;
double average_layer_h = 0.0;
for (unsigned int i = 0; i < first_outside.size(); i++)
{
unsigned int i_node = first_outside[i];
average_layer_h += mHavg[i_node];
}
average_layer_h /= static_cast<double>(first_outside.size());
for (int i_node = 0; i_node < n_nodes; i_node++)
mdistances[i_node] -= average_layer_h * ratio;
}
mr_matrix_container.WriteScalarToDatabase(DISTANCE, mdistances, mr_model_part.Nodes());

return;
}

void CalculatePorousResistanceLaw(unsigned int res_law)
{
if (res_law == 1)
{

for (ModelPart::NodesContainerType::iterator inode = mr_model_part.NodesBegin();
inode != mr_model_part.NodesEnd();
inode++)
{
const double eps = inode->FastGetSolutionStepValue(POROSITY);
const double d = inode->FastGetSolutionStepValue(DIAMETER);
double &a = inode->FastGetSolutionStepValue(LIN_DARCY_COEF);
double &b = inode->FastGetSolutionStepValue(NONLIN_DARCY_COEF);
if (eps < 1.0)
{
double k_inv = 150.0 * (1.0 - eps) * (1.0 - eps) / (eps * eps * eps * d * d);
a = mViscosity * k_inv;
b = (1.75 / eps) * sqrt(k_inv / (150.0 * eps));
}
else
{
a = 0;
b = 0;
}
}
}
else
{

for (ModelPart::NodesContainerType::iterator inode = mr_model_part.NodesBegin();
inode != mr_model_part.NodesEnd();
inode++)
{
const double eps = inode->FastGetSolutionStepValue(POROSITY);   
double &a = inode->FastGetSolutionStepValue(LIN_DARCY_COEF);    
double &b = inode->FastGetSolutionStepValue(NONLIN_DARCY_COEF); 
if (eps == 1.0)
{
a = 0;
b = 0;
}
}
}
mr_matrix_container.FillScalarFromDatabase(LIN_DARCY_COEF, mA, mr_model_part.Nodes());    
mr_matrix_container.FillScalarFromDatabase(NONLIN_DARCY_COEF, mB, mr_model_part.Nodes()); 
}

private:
double mMolecularViscosity;
double mcorner_coefficient;
double medge_coefficient;
double mmax_dt;
MatrixContainer &mr_matrix_container;
ModelPart &mr_model_part;
int mnumsubsteps;
bool muse_mass_correction;
bool mWallLawIsActive;
double mY_wall;
double mstabdt_pressure_factor;
double mstabdt_convection_factor;
double mtau2_factor;
bool massume_constant_dp;
CalcVectorType mBodyForce;
ValuesVectorType mViscosity;
CalcVectorType mWork, mvel_n, mvel_n1, mx, macc;
ValuesVectorType mPn, mPn1;
ValuesVectorType mdistances;
ValuesVectorType mHmin;
ValuesVectorType mHavg;
CalcVectorType mEdgeDimensions;
CalcVectorType mSlipNormal;
CalcVectorType mInOutNormal;
CalcVectorType mPi, mXi;
bool mFirstStep;
ValuesVectorType mNodalFlag;
ValuesVectorType mWallReductionFactor;
IndicesVectorType mSlipBoundaryList, mPressureOutletList, mFixedVelocities, mInOutBoundaryList, mDistanceBoundaryList;
ValuesVectorType mDistanceValuesList;
CalcVectorType mFixedVelocitiesValues;
ValuesVectorType mTauPressure;
ValuesVectorType mTauConvection;
ValuesVectorType mTau2;
ValuesVectorType mdiv_error;
boost::numeric::ublas::vector<bool> mis_slip;
boost::numeric::ublas::vector<int> mis_visited;
TSystemMatrixType mL;
double mRho;
ValuesVectorType mphi_n;
ValuesVectorType mphi_n1;
CalcVectorType mPiConvection;
ValuesVectorType mBeta;
IndicesVectorType medge_nodes;
CalcVectorType medge_nodes_direction;
IndicesVectorType mcorner_nodes;
ValuesVectorType mEps;
ValuesVectorType mdiag_stiffness;
ValuesVectorType mA;
ValuesVectorType mB;
double mdelta_t_avg;
double max_dt;
double mshock_coeff;
void CalculateNormal2D(ModelPart::ConditionsContainerType::iterator cond_it, array_1d<double, 3> &area_normal)
{
Geometry<Node> &face_geometry = (cond_it)->GetGeometry();
area_normal[0] = face_geometry[1].Y() - face_geometry[0].Y();
area_normal[1] = -(face_geometry[1].X() - face_geometry[0].X());
area_normal[2] = 0.00;
noalias((cond_it)->GetValue(NORMAL)) = area_normal;
}
void CalculateNormal3D(ModelPart::ConditionsContainerType::iterator cond_it, array_1d<double, 3> &area_normal, array_1d<double, 3> &v1, array_1d<double, 3> &v2)
{
Geometry<Node> &face_geometry = (cond_it)->GetGeometry();

v1[0] = face_geometry[1].X() - face_geometry[0].X();
v1[1] = face_geometry[1].Y() - face_geometry[0].Y();
v1[2] = face_geometry[1].Z() - face_geometry[0].Z();

v2[0] = face_geometry[2].X() - face_geometry[0].X();
v2[1] = face_geometry[2].Y() - face_geometry[0].Y();
v2[2] = face_geometry[2].Z() - face_geometry[0].Z();

MathUtils<double>::CrossProduct(area_normal, v1, v2);
area_normal *= -0.5;

noalias((cond_it)->GetValue(NORMAL)) = area_normal;
}
void CalculateEdgeLengths(ModelPart::NodesContainerType &rNodes)
{
KRATOS_TRY
unsigned int n_nodes = rNodes.size();
std::vector<array_1d<double, TDim>> position;
position.resize(n_nodes);
for (typename ModelPart::NodesContainerType::iterator node_it = rNodes.begin(); node_it != rNodes.end(); node_it++)
{
unsigned int i_node = static_cast<unsigned int>(node_it->FastGetSolutionStepValue(AUX_INDEX));
noalias(position[i_node]) = node_it->Coordinates();
}
ValuesVectorType &aaa = mr_matrix_container.GetHmin();
for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
{
mHmin[i_node] = aaa[i_node];

KRATOS_ERROR_IF(aaa[i_node] == 0.0) << "found a 0 hmin on node " << i_node << std::endl;
}
if constexpr (TDim == 2)
{
for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
{
double &h_i = mHavg[i_node];
double &m_i = mr_matrix_container.GetLumpedMass()[i_node];
h_i = sqrt(2.0 * m_i);
}
}
else if constexpr (TDim == 3)
{
for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
{
double &h_i = mHavg[i_node];
double &m_i = mr_matrix_container.GetLumpedMass()[i_node];
h_i = pow(6.0 * m_i, 1.0 / 3.0);
}
}
for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
{
array_1d<double, TDim> &pos_i = position[i_node];
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
array_1d<double, TDim> &pos_j = position[j_neighbour];
array_1d<double, TDim> &l_k = mEdgeDimensions[csr_index];
for (unsigned int comp = 0; comp < TDim; comp++)
l_k[comp] = pos_i[comp] - pos_j[comp];
}
}
KRATOS_CATCH("")
}
void CalculateRHS_convection(
const ValuesVectorType &mphi,
const CalcVectorType &convective_velocity,
ValuesVectorType &rhs,
ValuesVectorType &active_nodes)
{
KRATOS_TRY
int n_nodes = mphi.size();

double stab_low;
double stab_high;
array_1d<double, TDim> a_i;
array_1d<double, TDim> a_j;
#pragma omp parallel for private(stab_low, stab_high, a_i, a_j)
for (int i_node = 0; i_node < n_nodes; i_node++)
{
double &rhs_i = rhs[i_node];
const double &h_i = mHavg[i_node];
const double &phi_i = mphi[i_node];
noalias(a_i) = convective_velocity[i_node];
a_i /= mEps[i_node];
const array_1d<double, TDim> &proj_i = mPiConvection[i_node];
double pi_i = proj_i[0] * a_i[0];
for (unsigned int l_comp = 1; l_comp < TDim; l_comp++)
pi_i += proj_i[l_comp] * a_i[l_comp];
rhs_i = 0.0;
if (active_nodes[i_node] != 0.0)
{
const double &beta = mBeta[i_node];
double norm_a = a_i[0] * a_i[0];
for (unsigned int l_comp = 1; l_comp < TDim; l_comp++)
norm_a += a_i[l_comp] * a_i[l_comp];
norm_a = sqrt(norm_a);
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
if (active_nodes[j_neighbour] != 0.0)
{
const double &phi_j = mphi[j_neighbour];
noalias(a_j) = convective_velocity[j_neighbour];
a_j /= mEps[j_neighbour];
const array_1d<double, TDim> &proj_j = mPiConvection[j_neighbour];
double pi_j = proj_j[0] * a_i[0];
for (unsigned int l_comp = 1; l_comp < TDim; l_comp++)
pi_j += proj_j[l_comp] * a_i[l_comp];
CSR_Tuple &edge_ij = mr_matrix_container.GetEdgeValues()[csr_index];
edge_ij.Sub_ConvectiveContribution(rhs_i, a_i, phi_i, a_j, phi_j);
edge_ij.CalculateConvectionStabilization_LOW(stab_low, a_i, phi_i, a_j, phi_j);
double edge_tau = mTauConvection[i_node];
edge_ij.CalculateConvectionStabilization_HIGH(stab_high, a_i, pi_i, a_j, pi_j);
edge_ij.Sub_StabContribution(rhs_i, edge_tau, 1.0, stab_low, stab_high);
double coeff = 0.5 * mshock_coeff; 
double laplacian_ij = 0.0;
edge_ij.CalculateScalarLaplacian(laplacian_ij);
double capturing = laplacian_ij * (phi_j - phi_i);
double aaa = 0.0;
for (unsigned int k_comp = 0; k_comp < TDim; k_comp++)
for (unsigned int m_comp = 0; m_comp < TDim; m_comp++)
aaa += a_i[k_comp] * a_i[m_comp] * edge_ij.LaplacianIJ(k_comp, m_comp);
if (norm_a > 1e-10)
{
aaa /= (norm_a * norm_a);
double capturing2 = aaa * (phi_j - phi_i);
if (fabs(capturing) > fabs(capturing2))
rhs_i -= coeff * (capturing - capturing2) * beta * norm_a * h_i;
}
}
}
}
}

KRATOS_CATCH("")
}
void CornerDectectionHelper(Geometry<Node> &face_geometry,
const array_1d<double, 3> &face_normal,
const double An,
const GlobalPointersVector<Condition> &neighb,
const unsigned int i1,
const unsigned int i2,
const unsigned int neighb_index,
std::vector<unsigned int> &edge_nodes,
CalcVectorType &cornern_list)
{
double acceptable_angle = 45.0 / 180.0 * 3.1; 
double acceptable_cos = cos(acceptable_angle);
if (face_geometry[i1].Id() < face_geometry[i2].Id()) 
{
const array_1d<double, 3> &neighb_normal = neighb[neighb_index].GetValue(NORMAL);
double neighb_An = norm_2(neighb_normal);
double cos_normal = 1.0 / (An * neighb_An) * inner_prod(face_normal, neighb_normal);
if (cos_normal < acceptable_cos)
{
array_1d<double, 3> edge = face_geometry[i2].Coordinates() - face_geometry[i1].Coordinates();
double temp = norm_2(edge);
edge /= temp;
int index1 = face_geometry[i1].FastGetSolutionStepValue(AUX_INDEX);
int index2 = face_geometry[i2].FastGetSolutionStepValue(AUX_INDEX);
edge_nodes[index1] += 1;
edge_nodes[index2] += 1;
double sign1 = 0.0;
for (unsigned int i = 0; i < edge.size(); i++)
{
sign1 += cornern_list[index1][i] * edge[i];
}

if (sign1 >= 0)
{
for (unsigned int i = 0; i < edge.size(); i++)
cornern_list[index1][i] += edge[i];
}
else
{
for (unsigned int i = 0; i < edge.size(); i++)
cornern_list[index1][i] -= edge[i];
}

double sign2 = inner_prod(cornern_list[index2], edge);
if (sign2 >= 0)
{
for (unsigned int i = 0; i < edge.size(); i++)
cornern_list[index2][i] += edge[i];
}
else
{
for (unsigned int i = 0; i < edge.size(); i++)
cornern_list[index2][i] -= edge[i];
}
}
}
}
void DetectEdges3D(ModelPart::ConditionsContainerType &rConditions)
{
KRATOS_TRY
array_1d<double, 3> area_normal;
unsigned int n_nodes = mNodalFlag.size();
std::vector<unsigned int> temp_edge_nodes(n_nodes);
CalcVectorType temp_cornern_list(n_nodes);
for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
{
temp_edge_nodes[i_node] = 0.0;
noalias(temp_cornern_list[i_node]) = ZeroVector(TDim);
}
for (ModelPart::ConditionsContainerType::iterator cond_it = rConditions.begin(); cond_it != rConditions.end(); cond_it++)
{
Geometry<Node> &face_geometry = cond_it->GetGeometry();
const array_1d<double, 3> &face_normal = cond_it->GetValue(NORMAL);
double An = norm_2(face_normal);
unsigned int current_id = cond_it->Id();
if (cond_it->Is(SLIP)) 
{
const GlobalPointersVector<Condition> &neighb = cond_it->GetValue(NEIGHBOUR_CONDITIONS);
if (neighb[0].Id() != current_id) 
CornerDectectionHelper(face_geometry, face_normal, An, neighb, 1, 2, 0, temp_edge_nodes, temp_cornern_list);
if (neighb[1].Id() != current_id) 
CornerDectectionHelper(face_geometry, face_normal, An, neighb, 2, 0, 1, temp_edge_nodes, temp_cornern_list);
if (neighb[2].Id() != current_id) 
CornerDectectionHelper(face_geometry, face_normal, An, neighb, 0, 1, 2, temp_edge_nodes, temp_cornern_list);
}
}

std::vector<unsigned int> tempmedge_nodes;
std::vector<array_1d<double, TDim>> tempmedge_nodes_direction;
std::vector<unsigned int> tempmcorner_nodes;
for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
{
if (temp_edge_nodes[i_node] == 2) 
{
tempmedge_nodes.push_back(i_node);
array_1d<double, TDim> &node_edge = temp_cornern_list[i_node];
node_edge /= norm_2(node_edge);
tempmedge_nodes_direction.push_back(node_edge);
}
else if (temp_edge_nodes[i_node] > 2)
tempmcorner_nodes.push_back(i_node);
}
medge_nodes.resize(tempmedge_nodes.size(), false);
medge_nodes_direction.resize(tempmedge_nodes_direction.size(), false);
mcorner_nodes.resize(tempmcorner_nodes.size(), false);

IndexPartition<unsigned int>(tempmedge_nodes.size()).for_each([&](unsigned int i){
medge_nodes[i] = tempmedge_nodes[i];
medge_nodes_direction[i] = tempmedge_nodes_direction[i];
});

IndexPartition<unsigned int>(tempmcorner_nodes.size()).for_each([&](unsigned int i){
mcorner_nodes[i] = tempmcorner_nodes[i];
});

for (unsigned int i = 0; i < mcorner_nodes.size(); i++)
{
KRATOS_WATCH(mcorner_nodes[i]);
}
KRATOS_CATCH("")
}

double ComputePorosityCoefficient(const double &vel_norm, const double &eps, const double &a, const double &b)
{
double linear;
double non_linear;
linear = eps * a;
non_linear = eps * b * vel_norm;
return linear + non_linear;
}
void LaplacianSmooth(ValuesVectorType &to_be_smoothed, ValuesVectorType &aux)
{
ModelPart::NodesContainerType &rNodes = mr_model_part.Nodes();
int n_nodes = rNodes.size();

IndexPartition<unsigned int>(n_nodes).for_each([&](unsigned int i_node){
double dist = mdistances[i_node];
double correction = 0.0;
const double &origin_i = to_be_smoothed[i_node];
if (dist <= 0.0) 
{
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
const double &origin_j = to_be_smoothed[j_neighbour];
CSR_Tuple &edge_ij = mr_matrix_container.GetEdgeValues()[csr_index];
double l_ikjk;
edge_ij.CalculateScalarLaplacian(l_ikjk);
correction += l_ikjk * (origin_j - origin_i);
}
}
aux[i_node] = origin_i - correction;
});

IndexPartition<unsigned int>(n_nodes).for_each([&](unsigned int i_node){
to_be_smoothed[i_node] = aux[i_node];
});
}

void ComputeWallResistance(
const CalcVectorType &vel,
ValuesVectorType &diag_stiffness
)
{

double ym = mY_wall;
KRATOS_ERROR_IF(mViscosity[0] == 0) << "it is not possible to use the wall law with 0 viscosity" << std::endl;

int slip_size = mSlipBoundaryList.size();
#pragma omp parallel for firstprivate(slip_size, ym)
for (int i_slip = 0; i_slip < slip_size; i_slip++)
{
unsigned int i_node = mSlipBoundaryList[i_slip];
double dist = mdistances[i_node];
if (dist <= 0.0)
{
double nu = mMolecularViscosity;
const array_1d<double, TDim> &U_i = vel[i_node];
const array_1d<double, TDim> &an_i = mSlipNormal[i_node];

double mod_vel = 0.0;
double area = 0.0;
for (unsigned int comp = 0; comp < TDim; comp++)
{
mod_vel += U_i[comp] * U_i[comp];
area += an_i[comp] * an_i[comp];
}
mod_vel = sqrt(mod_vel);
area = sqrt(area);

diag_stiffness[i_node] = area * nu * mod_vel / (ym)*mWallReductionFactor[i_node];
}
else
{
diag_stiffness[i_node] = 0.0;
}
}
}

void ApplySmagorinsky3D(double MolecularViscosity, double Cs)
{
KRATOS_TRY
ModelPart::NodesContainerType &rNodes = mr_model_part.Nodes();
array_1d<double, TDim> grad_vx;
array_1d<double, TDim> grad_vy;
array_1d<double, TDim> grad_vz;
int n_nodes = rNodes.size();
mr_matrix_container.FillVectorFromDatabase(VELOCITY, mvel_n1, rNodes);
array_1d<double, TDim> stab_high;
#pragma omp parallel for private(grad_vx, grad_vy, grad_vz)
for (int i_node = 0; i_node < n_nodes; i_node++)
{
for (unsigned int comp = 0; comp < TDim; comp++)
{
grad_vx[comp] = 0.0;
grad_vy[comp] = 0.0;
grad_vz[comp] = 0.0;
}
const array_1d<double, TDim> &U_i = mvel_n1[i_node];
const double h = mHmin[i_node];
const double m_inv = mr_matrix_container.GetInvertedMass()[i_node];
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
const array_1d<double, TDim> &U_j = mvel_n1[j_neighbour];
CSR_Tuple &edge_ij = mr_matrix_container.GetEdgeValues()[csr_index];
edge_ij.Add_grad_p(grad_vx, U_i[0], U_j[0]);
edge_ij.Add_grad_p(grad_vy, U_i[1], U_j[1]);
edge_ij.Add_grad_p(grad_vz, U_i[2], U_j[2]);
}
for (unsigned int comp = 0; comp < TDim; comp++)
{
grad_vx[comp] *= m_inv;
grad_vy[comp] *= m_inv;
grad_vz[comp] *= m_inv;
}
grad_vx[0] *= 2.0;
grad_vy[1] *= 2.0;
if constexpr (TDim > 2)
grad_vz[2] *= 2.0;
grad_vx[1] += grad_vy[0];
if constexpr (TDim > 2)
grad_vx[2] += grad_vz[0];
if constexpr (TDim > 2)
grad_vy[2] += grad_vz[1];
grad_vy[0] += grad_vx[1];
grad_vz[0] += grad_vx[2];
grad_vz[1] += grad_vy[2];

double aux = 0.0;
for (unsigned int comp = 0; comp < TDim; comp++)
{
aux += grad_vx[comp] * grad_vx[comp];
aux += grad_vy[comp] * grad_vy[comp];
aux += grad_vz[comp] * grad_vz[comp];
}
aux *= 0.5;
if (aux < 0.0)
aux = 0.0;
double turbulent_viscosity = Cs * h * h * sqrt(aux);
mViscosity[i_node] = turbulent_viscosity + MolecularViscosity;
}
mr_matrix_container.WriteScalarToDatabase(VISCOSITY, mViscosity, rNodes);
KRATOS_CATCH("");
}

void Add_Effective_Inverse_Multiply(
CalcVectorType &destination,
const CalcVectorType &origin1,
const double value,
const ValuesVectorType &mass,
const ValuesVectorType &diag_stiffness,
const CalcVectorType &origin)
{
KRATOS_TRY
int loop_size = destination.size();


IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
array_1d<double, TDim> &dest = destination[i_node];
const double m = mass[i_node];
const double d = diag_stiffness[i_node];
const array_1d<double, TDim> &origin_vec1 = origin1[i_node];
const array_1d<double, TDim> &origin_value = origin[i_node];

for (unsigned int comp = 0; comp < TDim; comp++)
dest[comp] = value / (m + value * d) * (m / value * origin_vec1[comp] + origin_value[comp]);
});

KRATOS_CATCH("")
}

void ComputeConvectiveProjection(
CalcVectorType &mPiConvection,
const ValuesVectorType &mphi_n1,
const ValuesVectorType &mEps,
const CalcVectorType &mvel_n1)
{
int n_nodes = mPiConvection.size();
array_1d<double, TDim> a_i;
array_1d<double, TDim> a_j;
#pragma omp parallel for private(a_i, a_j)
for (int i_node = 0; i_node < n_nodes; i_node++)
{
array_1d<double, TDim> &pi_i = mPiConvection[i_node];
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
pi_i[l_comp] = 0.0;

const double &phi_i = mphi_n1[i_node];
noalias(a_i) = mvel_n1[i_node];
a_i /= mEps[i_node];
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
noalias(a_j) = mvel_n1[j_neighbour];
a_j /= mEps[j_neighbour];
const double &phi_j = mphi_n1[j_neighbour];
CSR_Tuple &edge_ij = mr_matrix_container.GetEdgeValues()[csr_index];
edge_ij.Add_grad_p(pi_i, phi_i, phi_j);
}
const double m_inv = mr_matrix_container.GetInvertedMass()[i_node];
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
pi_i[l_comp] *= m_inv;
}
}

void ComputeLimitor(
CalcVectorType &mPiConvection,
const ValuesVectorType &mphi_n1,
ValuesVectorType &mBeta,
const CalcVectorType &mvel_n1,
const CalcVectorType &mEdgeDimensions)
{
int n_nodes = mPiConvection.size();

IndexPartition<unsigned int>(n_nodes).for_each([&](unsigned int i_node){
const array_1d<double, TDim> &pi_i = mPiConvection[i_node];
const double &p_i = mphi_n1[i_node];
double &beta_i = mBeta[i_node];
beta_i = 0.0;
double n = 0.0;
for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
const double &p_j = mphi_n1[j_neighbour];
const array_1d<double, TDim> &l_k = mEdgeDimensions[csr_index];
const array_1d<double, TDim> &pi_j = mPiConvection[j_neighbour];
double proj = 0.0;
for (unsigned int comp = 0; comp < TDim; comp++)
proj += 0.5 * l_k[comp] * (pi_i[comp] + pi_j[comp]);
double numerator = fabs(fabs(p_j - p_i) - fabs(proj));
double denom = fabs(fabs(p_j - p_i) + 1e-6);
beta_i += numerator / denom;
n += 1.0;
}
beta_i /= n;
if (beta_i > 1.0)
beta_i = 1.0;
});
}
};
} 
#undef SYMM_PRESS
#endif 
