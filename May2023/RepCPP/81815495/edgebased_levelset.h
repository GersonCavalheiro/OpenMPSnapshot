
#if !defined(KRATOS_EDGEBASED_LEVELSET_FLUID_SOLVER_H_INCLUDED)
#define KRATOS_EDGEBASED_LEVELSET_FLUID_SOLVER_H_INCLUDED

#include <string>
#include <iostream>
#include <algorithm>


#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "includes/global_pointer_variables.h"
#include "includes/node.h"
#include "includes/cfd_variables.h"
#include "utilities/geometry_utilities.h"
#include "free_surface_application.h"

namespace Kratos
{

template <unsigned int TDim, class MatrixContainer, class TSparseSpace, class TLinearSolver>
class EdgeBasedLevelSet
{
public:
typedef EdgesStructureType<TDim> CSR_Tuple;
typedef vector<CSR_Tuple> EdgesVectorType;

typedef vector<unsigned int> IndicesVectorType;
typedef vector<array_1d<double, TDim>> CalcVectorType;
typedef vector<double> ValuesVectorType;

typedef typename TSparseSpace::MatrixType TSystemMatrixType;
typedef typename TSparseSpace::VectorType TSystemVectorType;

typedef std::size_t SizeType;


EdgeBasedLevelSet(MatrixContainer &mr_matrix_container,
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
};

~EdgeBasedLevelSet(){};

/
for (unsigned int comp = 0; comp < TDim; comp++)
{
rhs_i[comp] += m_i * porosity_coefficient * str_v_i[comp];
}

for (unsigned int csr_index = mr_matrix_container.GetRowStartIndex()[i_node]; csr_index != mr_matrix_container.GetRowStartIndex()[i_node + 1]; csr_index++)
{
unsigned int j_neighbour = mr_matrix_container.GetColumnIndex()[csr_index];
array_1d<double, TDim> a_j = convective_velocity[j_neighbour];
const array_1d<double, TDim> &U_j = vel[j_neighbour];
const array_1d<double, TDim> &pi_j = mPi[j_neighbour];
const double &p_j = pressure[j_neighbour];
const double &eps_j = mEps[j_neighbour];

a_j /= eps_j;

CSR_Tuple &edge_ij = mr_matrix_container.GetEdgeValues()[csr_index];

edge_ij.Sub_ConvectiveContribution(rhs_i, a_i, U_i, a_j, U_j);

edge_ij.Sub_grad_p(rhs_i, p_i * inverse_rho * eps_i, p_j * inverse_rho * eps_i);

edge_ij.Sub_ViscousContribution(rhs_i, U_i, nu_i, U_j, nu_j);

edge_ij.CalculateConvectionStabilization_LOW(stab_low, a_i, U_i, a_j, U_j);
edge_ij.CalculateConvectionStabilization_HIGH(stab_high, a_i, pi_i, a_j, pi_j);
edge_ij.Sub_StabContribution(rhs_i, edge_tau, 1.0, stab_low, stab_high);
}
}
}

if (mWallLawIsActive == true)
ComputeWallResistance(vel, diag_stiffness);
ModelPart::NodesContainerType &rNodes = mr_model_part.Nodes();
mr_matrix_container.WriteVectorToDatabase(VELOCITY, mvel_n1, rNodes);

KRATOS_CATCH("")
}

/
for (ModelPart::NodesContainerType::iterator inode = mr_model_part.NodesBegin();
inode != mr_model_part.NodesEnd();
inode++)
{
const double eps = inode->FastGetSolutionStepValue(POROSITY);   
const double d = inode->FastGetSolutionStepValue(DIAMETER);     
const double nu = inode->FastGetSolutionStepValue(VISCOSITY);   
double &a = inode->FastGetSolutionStepValue(LIN_DARCY_COEF);    
double &b = inode->FastGetSolutionStepValue(NONLIN_DARCY_COEF); 
if (eps < 1.0)
{
double k_inv = 150.0 * (1.0 - eps) * (1.0 - eps) / (eps * eps * eps * d * d);
a = nu * k_inv;
b = (1.75 / eps) * sqrt(k_inv / (150.0 * eps));
}
else
{
a = 0.0;
b = 0.0;
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
a = 0.0;
b = 0.0;
}
}
}

mr_matrix_container.FillScalarFromDatabase(LIN_DARCY_COEF, mA, mr_model_part.Nodes());    
mr_matrix_container.FillScalarFromDatabase(NONLIN_DARCY_COEF, mB, mr_model_part.Nodes()); 
}

private:
double mMolecularViscosity;
MatrixContainer &mr_matrix_container;
ModelPart &mr_model_part;

bool muse_mass_correction;

bool mWallLawIsActive;
double mY_wall;

double mstabdt_pressure_factor;
double mstabdt_convection_factor;
double mtau2_factor;
bool massume_constant_dp;

ValuesVectorType mViscosity;
CalcVectorType mBodyForce;
CalcVectorType mWork, mvel_n, mvel_n1, mx;
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
IndicesVectorType mSlipBoundaryList, mPressureOutletList, mFixedVelocities, mInOutBoundaryList;
CalcVectorType mFixedVelocitiesValues;

ValuesVectorType mTauPressure;
ValuesVectorType mTauConvection;
ValuesVectorType mTau2;

ValuesVectorType mdiv_error;
std::vector<bool> mis_slip;
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
CalcVectorType mStrVel;

double mdelta_t_avg;
double max_dt;

double mshock_coeff;

/
double mod_uthaw = sqrt(mod_vel * nu / ym);
const double y_plus = ym * mod_uthaw / nu;

if (y_plus > y_plus_incercept)
{
unsigned int it = 0;
double dx = 1e10;

while (fabs(dx) > toll * mod_uthaw && it < itmax)
{
double a = 1.0 / k;
double temp = a * log(ym * mod_uthaw / nu) + B;
double y = mod_uthaw * (temp)-mod_vel;
double y1 = temp + a;
dx = y / y1;
mod_uthaw -= dx;
it = it + 1;
}

if (it == itmax)
std::cout << "attention max number of iterations exceeded in wall law computation" << std::endl;
}
}
else
diag_stiffness[i_node] += 0.0;
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
grad_vz[2] *= 2.0;
grad_vx[1] += grad_vy[0];
grad_vx[2] += grad_vz[0];
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
double turbulent_viscosity = Cs * h * h * sqrt(aux) ;

mViscosity[i_node] = turbulent_viscosity + MolecularViscosity;
}
mr_matrix_container.WriteScalarToDatabase(VISCOSITY, mViscosity, rNodes);
KRATOS_CATCH("");
}

void ApplySmagorinsky2D(double MolecularViscosity, double Cs)
{
KRATOS_TRY
ModelPart::NodesContainerType &rNodes = mr_model_part.Nodes();
array_1d<double, TDim> grad_vx;
array_1d<double, TDim> grad_vy;

int n_nodes = rNodes.size();
mr_matrix_container.FillVectorFromDatabase(VELOCITY, mvel_n1, rNodes);
array_1d<double, TDim> stab_high;

#pragma omp parallel for private(grad_vx, grad_vy)
for (int i_node = 0; i_node < n_nodes; i_node++)
{
for (unsigned int comp = 0; comp < TDim; comp++)
{
grad_vx[comp] = 0.0;
grad_vy[comp] = 0.0;
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
}
for (unsigned int comp = 0; comp < TDim; comp++)
{
grad_vx[comp] *= m_inv;
grad_vy[comp] *= m_inv;
}
grad_vx[0] *= 2.0;
grad_vy[1] *= 2.0;
grad_vx[1] += grad_vy[0];
grad_vy[0] += grad_vx[1];
double aux = 0.0;
for (unsigned int comp = 0; comp < TDim; comp++)
{
aux += grad_vx[comp] * grad_vx[comp];
aux += grad_vy[comp] * grad_vy[comp];
}
aux *= 0.5;
if (aux < 0.0)
aux = 0.0;
double turbulent_viscosity = Cs * h * h * sqrt(aux) ;

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
};
} 
#undef SYMM_PRESS
#endif 
