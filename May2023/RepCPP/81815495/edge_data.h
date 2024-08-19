
#if !defined(KRATOS_EDGE_DATA_H_INCLUDED)
#define KRATOS_EDGE_DATA_H_INCLUDED

#define USE_CONSERVATIVE_FORM_FOR_SCALAR_CONVECTION

#define USE_CONSERVATIVE_FORM_FOR_VECTOR_CONVECTION

#include <string>
#include <iostream>
#include <algorithm>


#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/global_pointer_variables.h"
#include "includes/node.h"
#include "utilities/geometry_utilities.h"
#include "free_surface_application.h"
#include "utilities/openmp_utils.h"

namespace Kratos
{
template <unsigned int TDim>
class EdgesStructureType
{
public:
double Mass = 0.0;
boost::numeric::ublas::bounded_matrix<double, TDim, TDim> LaplacianIJ;
array_1d<double, TDim> Ni_DNj = ZeroVector(TDim);
array_1d<double, TDim> DNi_Nj = ZeroVector(TDim);


inline void Add_Gp(array_1d<double, TDim> &destination, const double &p_i, const double &p_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination[comp] -= Ni_DNj[comp] * p_j - DNi_Nj[comp] * p_i;
}

inline void Sub_Gp(array_1d<double, TDim> &destination, const double &p_i, const double &p_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination[comp] += Ni_DNj[comp] * p_j - DNi_Nj[comp] * p_i;
}


inline void Add_D_v(double &destination,
const array_1d<double, TDim> &v_i,
const array_1d<double, TDim> &v_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination += Ni_DNj[comp] * (v_j[comp] - v_i[comp]);
}

inline void Sub_D_v(double &destination,
const array_1d<double, TDim> &v_i,
const array_1d<double, TDim> &v_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination -= Ni_DNj[comp] * (v_j[comp] - v_i[comp]);
}


inline void Add_grad_p(array_1d<double, TDim> &destination, const double &p_i, const double &p_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination[comp] += Ni_DNj[comp] * (p_j - p_i);
}

inline void Sub_grad_p(array_1d<double, TDim> &destination, const double &p_i, const double &p_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination[comp] -= Ni_DNj[comp] * (p_j - p_i);
}


inline void Add_div_v(double &destination,
const array_1d<double, TDim> &v_i,
const array_1d<double, TDim> &v_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination -= Ni_DNj[comp] * v_j[comp] - DNi_Nj[comp] * v_i[comp];
}

inline void Sub_div_v(double &destination,
const array_1d<double, TDim> &v_i,
const array_1d<double, TDim> &v_j)
{
for (unsigned int comp = 0; comp < TDim; comp++)
destination += Ni_DNj[comp] * v_j[comp] - DNi_Nj[comp] * v_i[comp];
}


inline void CalculateScalarLaplacian(double &l_ij)
{
l_ij = LaplacianIJ(0, 0);
for (unsigned int comp = 1; comp < TDim; comp++)
l_ij += LaplacianIJ(comp, comp);
}

inline void Add_ConvectiveContribution(array_1d<double, TDim> &destination,
const array_1d<double, TDim> &a_i, const array_1d<double, TDim> &U_i,
const array_1d<double, TDim> &a_j, const array_1d<double, TDim> &U_j)
{

#ifdef USE_CONSERVATIVE_FORM_FOR_VECTOR_CONVECTION
double temp = a_i[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
temp += a_i[k_comp] * Ni_DNj[k_comp];
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] += temp * (U_j[l_comp] - U_i[l_comp]);
#else
double aux_i = a_i[0] * Ni_DNj[0];
double aux_j = a_j[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
{
aux_i += a_i[k_comp] * Ni_DNj[k_comp];
aux_j += a_j[k_comp] * Ni_DNj[k_comp];
}
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] += aux_j * U_j[l_comp] - aux_i * U_i[l_comp];
#endif
}

inline void Sub_ConvectiveContribution(array_1d<double, TDim> &destination,
const array_1d<double, TDim> &a_i, const array_1d<double, TDim> &U_i,
const array_1d<double, TDim> &a_j, const array_1d<double, TDim> &U_j)
{

#ifdef USE_CONSERVATIVE_FORM_FOR_VECTOR_CONVECTION
double temp = a_i[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
temp += a_i[k_comp] * Ni_DNj[k_comp];
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] -= temp * (U_j[l_comp] - U_i[l_comp]);
#else
double aux_i = a_i[0] * Ni_DNj[0];
double aux_j = a_j[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
{
aux_i += a_i[k_comp] * Ni_DNj[k_comp];
aux_j += a_j[k_comp] * Ni_DNj[k_comp];
}
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] -= aux_j * U_j[l_comp] - aux_i * U_i[l_comp];
#endif
}

inline void Sub_ConvectiveContribution(double &destination,
const array_1d<double, TDim> &a_i, const double &phi_i,
const array_1d<double, TDim> &a_j, const double &phi_j)
{

#ifdef USE_CONSERVATIVE_FORM_FOR_SCALAR_CONVECTION
double temp = a_i[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
temp += a_i[k_comp] * Ni_DNj[k_comp];

destination -= temp * (phi_j - phi_i);
#else
double aux_i = a_i[0] * Ni_DNj[0];
double aux_j = a_j[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
{
aux_i += a_i[k_comp] * Ni_DNj[k_comp];
aux_j += a_j[k_comp] * Ni_DNj[k_comp];
}
destination -= aux_j * phi_j - aux_i * phi_i;
#endif
}

inline void Add_ConvectiveContribution(double &destination,
const array_1d<double, TDim> &a_i, const double &phi_i,
const array_1d<double, TDim> &a_j, const double &phi_j)
{

#ifdef USE_CONSERVATIVE_FORM_FOR_SCALAR_CONVECTION
double temp = a_i[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
temp += a_i[k_comp] * Ni_DNj[k_comp];

destination += temp * (phi_j - phi_i);
#else
double aux_i = a_i[0] * Ni_DNj[0];
double aux_j = a_j[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
{
aux_i += a_i[k_comp] * Ni_DNj[k_comp];
aux_j += a_j[k_comp] * Ni_DNj[k_comp];
}
destination += aux_j * phi_j - aux_i * phi_i;
#endif
}


inline void CalculateConvectionStabilization_LOW(array_1d<double, TDim> &stab_low,
const array_1d<double, TDim> &a_i, const array_1d<double, TDim> &U_i,
const array_1d<double, TDim> &a_j, const array_1d<double, TDim> &U_j)
{
double conv_stab = 0.0;
for (unsigned int k_comp = 0; k_comp < TDim; k_comp++)
for (unsigned int m_comp = 0; m_comp < TDim; m_comp++)
conv_stab += a_i[k_comp] * a_i[m_comp] * LaplacianIJ(k_comp, m_comp);
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
stab_low[l_comp] = conv_stab * (U_j[l_comp] - U_i[l_comp]);
}

inline void CalculateConvectionStabilization_LOW(double &stab_low,
const array_1d<double, TDim> &a_i, const double &phi_i,
const array_1d<double, TDim> &a_j, const double &phi_j)
{
double conv_stab = 0.0;
for (unsigned int k_comp = 0; k_comp < TDim; k_comp++)
for (unsigned int m_comp = 0; m_comp < TDim; m_comp++)
conv_stab += a_i[k_comp] * a_i[m_comp] * LaplacianIJ(k_comp, m_comp);
stab_low = conv_stab * (phi_j - phi_i);
}

inline void CalculateConvectionStabilization_HIGH(array_1d<double, TDim> &stab_high,
const array_1d<double, TDim> &a_i, const array_1d<double, TDim> &pi_i,
const array_1d<double, TDim> &a_j, const array_1d<double, TDim> &pi_j)
{

#ifdef USE_CONSERVATIVE_FORM_FOR_VECTOR_CONVECTION
double temp = 0.0;
for (unsigned int k_comp = 0; k_comp < TDim; k_comp++)
temp += a_i[k_comp] * Ni_DNj[k_comp];
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
stab_high[l_comp] = -temp * (pi_j[l_comp] - pi_i[l_comp]); 
#else
double aux_i = a_i[0] * Ni_DNj[0];
double aux_j = a_j[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
{
aux_i += a_i[k_comp] * Ni_DNj[k_comp];
aux_j += a_j[k_comp] * Ni_DNj[k_comp];
}
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
stab_high[l_comp] = -(aux_j * pi_j[l_comp] - aux_i * pi_i[l_comp]);
#endif
}

inline void CalculateConvectionStabilization_HIGH(double &stab_high,
const array_1d<double, TDim> &a_i, const double &pi_i,
const array_1d<double, TDim> &a_j, const double &pi_j)
{

#ifdef USE_CONSERVATIVE_FORM_FOR_SCALAR_CONVECTION
double temp = 0.0;
for (unsigned int k_comp = 0; k_comp < TDim; k_comp++)
temp += a_i[k_comp] * Ni_DNj[k_comp];

stab_high = -temp * (pi_j - pi_i); 
#else
double aux_i = a_i[0] * Ni_DNj[0];
double aux_j = a_j[0] * Ni_DNj[0];
for (unsigned int k_comp = 1; k_comp < TDim; k_comp++)
{
aux_i += a_i[k_comp] * Ni_DNj[k_comp];
aux_j += a_j[k_comp] * Ni_DNj[k_comp];
}

stab_high = -(aux_j * pi_j - aux_i * pi_i);
#endif
}

inline void Add_StabContribution(array_1d<double, TDim> &destination,
const double tau, const double beta,
const array_1d<double, TDim> &stab_low, const array_1d<double, TDim> &stab_high)
{
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] += tau * (stab_low[l_comp] - beta * stab_high[l_comp]);
}

inline void Add_StabContribution(double &destination,
const double tau, const double beta,
const double &stab_low, const double &stab_high)
{
destination += tau * (stab_low - beta * stab_high);
}

inline void Sub_StabContribution(array_1d<double, TDim> &destination,
const double tau, const double beta,
const array_1d<double, TDim> &stab_low, const array_1d<double, TDim> &stab_high)
{
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] -= tau * (stab_low[l_comp] - beta * stab_high[l_comp]);
}

inline void Sub_StabContribution(double &destination,
const double tau, const double beta,
const double &stab_low, const double &stab_high)
{
destination -= tau * (stab_low - beta * stab_high);
}


inline void Add_ViscousContribution(array_1d<double, TDim> &destination,
const array_1d<double, TDim> &U_i, const double &nu_i,
const array_1d<double, TDim> &U_j, const double &nu_j)
{
double L = 0.0;
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
L += LaplacianIJ(l_comp, l_comp);

for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] += nu_i * L * (U_j[l_comp] - U_i[l_comp]);
}

inline void Sub_ViscousContribution(array_1d<double, TDim> &destination,
const array_1d<double, TDim> &U_i, const double &nu_i,
const array_1d<double, TDim> &U_j, const double &nu_j)
{
double L = 0.0;
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
L += LaplacianIJ(l_comp, l_comp);

for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
destination[l_comp] -= nu_i * L * (U_j[l_comp] - U_i[l_comp]);
}
};


template <unsigned int TDim, class TSparseSpace>
class MatrixContainer
{
public:
typedef EdgesStructureType<TDim> CSR_Tuple;
typedef vector<CSR_Tuple> EdgesVectorType;
typedef vector<unsigned int> IndicesVectorType;
typedef vector<double> ValuesVectorType;
typedef vector<array_1d<double, TDim>> CalcVectorType;


MatrixContainer(){};

~MatrixContainer(){};


inline unsigned int GetNumberEdges()
{
return mNumberEdges;
}

inline EdgesVectorType &GetEdgeValues()
{
return mNonzeroEdgeValues;
}

inline IndicesVectorType &GetColumnIndex()
{
return mColumnIndex;
}

inline IndicesVectorType &GetRowStartIndex()
{
return mRowStartIndex;
}

inline ValuesVectorType &GetLumpedMass()
{
return mLumpedMassMatrix;
}

inline ValuesVectorType &GetInvertedMass()
{
return mInvertedMassMatrix;
}

inline CalcVectorType &GetDiagGradient()
{
return mDiagGradientMatrix;
}

inline ValuesVectorType &GetHmin()
{
return mHmin;
}


void ConstructCSRVector(ModelPart &model_part)
{
KRATOS_TRY


int n_nodes = model_part.Nodes().size();
mNumberEdges = 0;
int i_node = 0;

for (typename ModelPart::NodesContainerType::iterator node_it = model_part.NodesBegin(); node_it != model_part.NodesEnd(); node_it++)
{
mNumberEdges += (node_it->GetValue(NEIGHBOUR_NODES)).size();

node_it->FastGetSolutionStepValue(AUX_INDEX) = static_cast<double>(i_node++);
}
if (i_node != n_nodes)
KRATOS_WATCH("ERROR - Highest nodal index doesn't coincide with number of nodes!");

mNonzeroEdgeValues.resize(mNumberEdges);
SetToZero(mNonzeroEdgeValues);
mColumnIndex.resize(mNumberEdges);
SetToZero(mColumnIndex);
mRowStartIndex.resize(n_nodes + 1);
SetToZero(mRowStartIndex);
mLumpedMassMatrix.resize(n_nodes);
SetToZero(mLumpedMassMatrix);
mInvertedMassMatrix.resize(n_nodes);
SetToZero(mInvertedMassMatrix);
mDiagGradientMatrix.resize(n_nodes);
SetToZero(mDiagGradientMatrix);
mHmin.resize(n_nodes);
SetToZero(mHmin);


unsigned int row_start_temp = 0;

int number_of_threads = ParallelUtilities::GetNumThreads();
std::vector<int> row_partition(number_of_threads);
OpenMPUtils::DivideInPartitions(model_part.Nodes().size(), number_of_threads, row_partition);

for (int k = 0; k < number_of_threads; k++)
{
#pragma omp parallel
if (OpenMPUtils::ThisThread() == k)
{
for (unsigned int aux_i = static_cast<unsigned int>(row_partition[k]); aux_i < static_cast<unsigned int>(row_partition[k + 1]); aux_i++)
{
typename ModelPart::NodesContainerType::iterator node_it = model_part.NodesBegin() + aux_i;

i_node = static_cast<unsigned int>(node_it->FastGetSolutionStepValue(AUX_INDEX));

GlobalPointersVector<Node> &neighb_nodes = node_it->GetValue(NEIGHBOUR_NODES);

unsigned int n_neighbours = neighb_nodes.size();

std::vector<unsigned int> work_array;
work_array.reserve(n_neighbours);

for (GlobalPointersVector<Node>::iterator neighb_it = neighb_nodes.begin(); neighb_it != neighb_nodes.end(); neighb_it++)
{
work_array.push_back(static_cast<unsigned int>(neighb_it->FastGetSolutionStepValue(AUX_INDEX)));
}
std::sort(work_array.begin(), work_array.end());

mRowStartIndex[i_node] = row_start_temp;
for (unsigned int counter = 0; counter < n_neighbours; counter++)
{
unsigned int j_neighbour = work_array[counter];
unsigned int csr_index = mRowStartIndex[i_node] + counter;

mColumnIndex[csr_index] = j_neighbour;
}
row_start_temp += n_neighbours;
}
}
}
mRowStartIndex[n_nodes] = mNumberEdges;

IndexPartition<unsigned int>(n_nodes).for_each([&](unsigned int i_node){
mHmin[i_node] = 1e10;
});

KRATOS_CATCH("")
}


void BuildCSRData(ModelPart &model_part)
{
KRATOS_TRY


array_1d<double, TDim + 1> N;
boost::numeric::ublas::bounded_matrix<double, TDim + 1, TDim> dN_dx;
double volume;
double weighting_factor = 1.0 / static_cast<double>(TDim + 1);
boost::numeric::ublas::bounded_matrix<double, TDim + 1, TDim + 1> mass_consistent;
array_1d<double, TDim + 1> mass_lumped;
array_1d<unsigned int, TDim + 1> nodal_indices;

array_1d<double, TDim + 1> heights;

for (typename ModelPart::ElementsContainerType::iterator elem_it = model_part.ElementsBegin(); elem_it != model_part.ElementsEnd(); elem_it++)
{

GeometryUtils::CalculateGeometryData(elem_it->GetGeometry(), dN_dx, N, volume);

for (unsigned int ie_node = 0; ie_node <= TDim; ie_node++)
{
heights[ie_node] = dN_dx(ie_node, 0) * dN_dx(ie_node, 0);
for (unsigned int comp = 1; comp < TDim; comp++)
{
heights[ie_node] += dN_dx(ie_node, comp) * dN_dx(ie_node, comp);
}
heights[ie_node] = 1.0 / sqrt(heights[ie_node]);
}

CalculateMassMatrix(mass_consistent, volume);
noalias(mass_lumped) = ZeroVector(TDim + 1);
for (unsigned int ie_node = 0; ie_node <= TDim; ie_node++)
{
for (unsigned int je_node = 0; je_node <= TDim; je_node++)
{
mass_lumped[ie_node] += mass_consistent(ie_node, je_node);
}
}

double weighted_volume = volume * weighting_factor;


for (unsigned int ie_node = 0; ie_node <= TDim; ie_node++)
nodal_indices[ie_node] = static_cast<unsigned int>(elem_it->GetGeometry()[ie_node].FastGetSolutionStepValue(AUX_INDEX));

for (unsigned int ie_node = 0; ie_node <= TDim; ie_node++)
{
if (mHmin[nodal_indices[ie_node]] > heights[ie_node])
mHmin[nodal_indices[ie_node]] = heights[ie_node];

for (unsigned int je_node = 0; je_node <= TDim; je_node++)
{
if (ie_node != je_node)
{
unsigned int csr_index = GetCSRIndex(nodal_indices[ie_node], nodal_indices[je_node]);

mNonzeroEdgeValues[csr_index].Mass += mass_consistent(ie_node, je_node);

boost::numeric::ublas::bounded_matrix<double, TDim, TDim> &laplacian = mNonzeroEdgeValues[csr_index].LaplacianIJ;
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
for (unsigned int k_comp = 0; k_comp < TDim; k_comp++)
laplacian(l_comp, k_comp) += dN_dx(ie_node, l_comp) * dN_dx(je_node, k_comp) * volume;

array_1d<double, TDim> &gradient = mNonzeroEdgeValues[csr_index].Ni_DNj;
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
gradient[l_comp] += dN_dx(je_node, l_comp) * weighted_volume;
array_1d<double, TDim> &transp_gradient = mNonzeroEdgeValues[csr_index].DNi_Nj;
for (unsigned int l_comp = 0; l_comp < TDim; l_comp++)
transp_gradient[l_comp] += dN_dx(ie_node, l_comp) * weighted_volume;
}
}
}

for (unsigned int ie_node = 0; ie_node <= TDim; ie_node++)
mLumpedMassMatrix[nodal_indices[ie_node]] += mass_lumped[ie_node];
for (unsigned int ie_node = 0; ie_node <= TDim; ie_node++)
{
array_1d<double, TDim> &gradient = mDiagGradientMatrix[nodal_indices[ie_node]];
for (unsigned int component = 0; component < TDim; component++)
gradient[component] += dN_dx(ie_node, component) * weighted_volume;
}
}

for (unsigned int inode = 0; inode < mLumpedMassMatrix.size(); inode++)
{
mInvertedMassMatrix[inode] = mLumpedMassMatrix[inode];
}

for (unsigned int inode = 0; inode < mInvertedMassMatrix.size(); inode++)
{
mInvertedMassMatrix[inode] = 1.0 / mInvertedMassMatrix[inode];
}

KRATOS_CATCH("")
}


unsigned int GetCSRIndex(unsigned int NodeI, unsigned int NeighbourJ)
{
KRATOS_TRY

unsigned int csr_index;
for (csr_index = mRowStartIndex[NodeI]; csr_index != mRowStartIndex[NodeI + 1]; csr_index++)
if (mColumnIndex[csr_index] == NeighbourJ)
break;

return csr_index;

KRATOS_CATCH("")
}


CSR_Tuple *GetTuplePointer(unsigned int NodeI, unsigned int NeighbourJ)
{
KRATOS_TRY

unsigned int csr_index;
for (csr_index = mRowStartIndex[NodeI]; csr_index != mRowStartIndex[NodeI + 1]; csr_index++)
if (mColumnIndex[csr_index] == NeighbourJ)
break;

return &mNonzeroEdgeValues[csr_index];

KRATOS_CATCH("")
}


void Clear()
{
KRATOS_TRY

mNonzeroEdgeValues.clear();
mColumnIndex.clear();
mRowStartIndex.clear();
mInvertedMassMatrix.clear();
mLumpedMassMatrix.clear();
mDiagGradientMatrix.clear();
mHmin.clear();

KRATOS_CATCH("")
}

void FillCoordinatesFromDatabase(CalcVectorType &rDestination, ModelPart::NodesContainerType &rNodes)
{

KRATOS_TRY

int n_nodes = rNodes.size();
ModelPart::NodesContainerType::iterator it_begin = rNodes.begin();

#pragma omp parallel for firstprivate(n_nodes, it_begin)
for (int i = 0; i < n_nodes; i++)
{
ModelPart::NodesContainerType::iterator node_it = it_begin + i;

unsigned int i_node = i;

for (unsigned int component = 0; component < TDim; component++)
(rDestination[i_node])[component] = (*node_it)[component];
}

KRATOS_CATCH("");
}


void FillVectorFromDatabase(Variable<array_1d<double, 3>> &rVariable, CalcVectorType &rDestination, ModelPart::NodesContainerType &rNodes)
{

KRATOS_TRY


int n_nodes = rNodes.size();

ModelPart::NodesContainerType::iterator it_begin = rNodes.begin();

unsigned int var_pos = it_begin->pGetVariablesList()->Index(rVariable);

#pragma omp parallel for firstprivate(n_nodes, it_begin, var_pos)
for (int i = 0; i < n_nodes; i++)
{
ModelPart::NodesContainerType::iterator node_it = it_begin + i;

unsigned int i_node = i;

array_1d<double, 3> &vector = node_it->FastGetCurrentSolutionStepValue(rVariable, var_pos);
for (unsigned int component = 0; component < TDim; component++)
(rDestination[i_node])[component] = vector[component];
}

KRATOS_CATCH("");
}

void FillOldVectorFromDatabase(Variable<array_1d<double, 3>> &rVariable, CalcVectorType &rDestination, ModelPart::NodesContainerType &rNodes)
{

KRATOS_TRY

int n_nodes = rNodes.size();

ModelPart::NodesContainerType::iterator it_begin = rNodes.begin();

unsigned int var_pos = it_begin->pGetVariablesList()->Index(rVariable);

#pragma omp parallel for firstprivate(n_nodes, it_begin, var_pos)
for (int i = 0; i < n_nodes; i++)
{
ModelPart::NodesContainerType::iterator node_it = it_begin + i;

unsigned int i_node = i;

array_1d<double, 3> &vector = node_it->FastGetSolutionStepValue(rVariable, 1, var_pos);
for (unsigned int component = 0; component < TDim; component++)
(rDestination[i_node])[component] = vector[component];
}

KRATOS_CATCH("");
}

void FillScalarFromDatabase(Variable<double> &rVariable, ValuesVectorType &rDestination, ModelPart::NodesContainerType &rNodes)
{
KRATOS_TRY

int n_nodes = rNodes.size();

ModelPart::NodesContainerType::iterator it_begin = rNodes.begin();

unsigned int var_pos = it_begin->pGetVariablesList()->Index(rVariable);

#pragma omp parallel for firstprivate(n_nodes, it_begin, var_pos)
for (int i = 0; i < n_nodes; i++)
{
ModelPart::NodesContainerType::iterator node_it = it_begin + i;

unsigned int i_node = i;

double &scalar = node_it->FastGetCurrentSolutionStepValue(rVariable, var_pos);
rDestination[i_node] = scalar;
}

KRATOS_CATCH("");
}

void FillOldScalarFromDatabase(Variable<double> &rVariable, ValuesVectorType &rDestination, ModelPart::NodesContainerType &rNodes)
{
KRATOS_TRY

int n_nodes = rNodes.size();
ModelPart::NodesContainerType::iterator it_begin = rNodes.begin();

unsigned int var_pos = it_begin->pGetVariablesList()->Index(rVariable);

#pragma omp parallel for firstprivate(n_nodes, it_begin, var_pos)
for (int i = 0; i < n_nodes; i++)
{
ModelPart::NodesContainerType::iterator node_it = it_begin + i;

unsigned int i_node = i;

double &scalar = node_it->FastGetSolutionStepValue(rVariable, 1, var_pos);
rDestination[i_node] = scalar;
}

KRATOS_CATCH("");
}

void WriteVectorToDatabase(Variable<array_1d<double, 3>> &rVariable, CalcVectorType &rOrigin, ModelPart::NodesContainerType &rNodes)
{
KRATOS_TRY

int n_nodes = rNodes.size();
ModelPart::NodesContainerType::iterator it_begin = rNodes.begin();

unsigned int var_pos = it_begin->pGetVariablesList()->Index(rVariable);

#pragma omp parallel for firstprivate(n_nodes, it_begin, var_pos)
for (int i = 0; i < n_nodes; i++)
{
ModelPart::NodesContainerType::iterator node_it = it_begin + i;

unsigned int i_node = i;

array_1d<double, 3> &vector = node_it->FastGetCurrentSolutionStepValue(rVariable, var_pos);
for (unsigned int component = 0; component < TDim; component++)
vector[component] = (rOrigin[i_node])[component];
}

KRATOS_CATCH("");
}

void WriteScalarToDatabase(Variable<double> &rVariable, ValuesVectorType &rOrigin, ModelPart::NodesContainerType &rNodes)
{
KRATOS_TRY

int n_nodes = rNodes.size();
ModelPart::NodesContainerType::iterator it_begin = rNodes.begin();

unsigned int var_pos = it_begin->pGetVariablesList()->Index(rVariable);

#pragma omp parallel for firstprivate(n_nodes, it_begin, var_pos)
for (int i = 0; i < n_nodes; i++)
{
ModelPart::NodesContainerType::iterator node_it = it_begin + i;

int i_node = i;

double &scalar = node_it->FastGetCurrentSolutionStepValue(rVariable, var_pos);
scalar = rOrigin[i_node];
}

KRATOS_CATCH("");
}


void Add_Minv_value(
CalcVectorType &destination,
const CalcVectorType &origin1,
const double value,
const ValuesVectorType &Minv_vec,
const CalcVectorType &origin)
{
KRATOS_TRY

int loop_size = destination.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
array_1d<double, TDim> &dest = destination[i_node];
const double m_inv = Minv_vec[i_node];
const array_1d<double, TDim> &origin_vec1 = origin1[i_node];
const array_1d<double, TDim> &origin_value = origin[i_node];

double temp = value * m_inv;
for (unsigned int comp = 0; comp < TDim; comp++)
dest[comp] = origin_vec1[comp] + temp * origin_value[comp];
});

KRATOS_CATCH("")
}

void Add_Minv_value(
ValuesVectorType &destination,
const ValuesVectorType &origin1,
const double value,
const ValuesVectorType &Minv_vec,
const ValuesVectorType &origin)
{
KRATOS_TRY

int loop_size = destination.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
double &dest = destination[i_node];
const double m_inv = Minv_vec[i_node];
const double &origin_vec1 = origin1[i_node];
const double &origin_value = origin[i_node];

double temp = value * m_inv;
dest = origin_vec1 + temp * origin_value;
});

KRATOS_CATCH("")
}


void AllocateAndSetToZero(CalcVectorType &data_vector, int size)
{
data_vector.resize(size);
int loop_size = size;

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
array_1d<double, TDim> &aaa = data_vector[i_node];
for (unsigned int comp = 0; comp < TDim; comp++)
aaa[comp] = 0.0;
});
}

void AllocateAndSetToZero(ValuesVectorType &data_vector, int size)
{
data_vector.resize(size);
int loop_size = size;

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
data_vector[i_node] = 0.0;
});
}


void SetToZero(EdgesVectorType &data_vector)
{
int loop_size = data_vector.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
data_vector[i_node].Mass = 0.0;
noalias(data_vector[i_node].LaplacianIJ) = ZeroMatrix(TDim, TDim);
noalias(data_vector[i_node].Ni_DNj) = ZeroVector(TDim);
noalias(data_vector[i_node].DNi_Nj) = ZeroVector(TDim);
});
}

void SetToZero(IndicesVectorType &data_vector)
{
int loop_size = data_vector.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
data_vector[i_node] = 0.0;
});
}

void SetToZero(CalcVectorType &data_vector)
{
int loop_size = data_vector.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
array_1d<double, TDim> &aaa = data_vector[i_node];
for (unsigned int comp = 0; comp < TDim; comp++)
aaa[comp] = 0.0;
});
}

void SetToZero(ValuesVectorType &data_vector)
{
int loop_size = data_vector.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
data_vector[i_node] = 0.0;
});
}


void AssignVectorToVector(const CalcVectorType &origin,
CalcVectorType &destination)
{
int loop_size = origin.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
const array_1d<double, TDim> &orig = origin[i_node];
array_1d<double, TDim> &dest = destination[i_node];
for (unsigned int comp = 0; comp < TDim; comp++)
dest[comp] = orig[comp];
});
}

void AssignVectorToVector(const ValuesVectorType &origin,
ValuesVectorType &destination)
{
int loop_size = origin.size();

IndexPartition<unsigned int>(loop_size).for_each([&](unsigned int i_node){
destination[i_node] = origin[i_node];
});
}

private:
unsigned int mNumberEdges;

EdgesVectorType mNonzeroEdgeValues;

IndicesVectorType mColumnIndex;

IndicesVectorType mRowStartIndex;

ValuesVectorType mInvertedMassMatrix;

ValuesVectorType mHmin;

ValuesVectorType mLumpedMassMatrix;
CalcVectorType mDiagGradientMatrix;


void CalculateMassMatrix(boost::numeric::ublas::bounded_matrix<double, 3, 3> &mass_consistent, double volume)
{
for (unsigned int i_node = 0; i_node <= TDim; i_node++)
{
mass_consistent(i_node, i_node) = 0.16666666666666666667 * volume; 
double temp = 0.08333333333333333333 * volume; 
for (unsigned int j_neighbour = i_node + 1; j_neighbour <= TDim; j_neighbour++)
{
mass_consistent(i_node, j_neighbour) = temp;
mass_consistent(j_neighbour, i_node) = temp;
}
}
}

void CalculateMassMatrix(boost::numeric::ublas::bounded_matrix<double, 4, 4> &mass_consistent, double volume)
{
for (unsigned int i_node = 0; i_node <= TDim; i_node++)
{
mass_consistent(i_node, i_node) = 0.1 * volume;
double temp = 0.05 * volume;
for (unsigned int j_neighbour = i_node + 1; j_neighbour <= TDim; j_neighbour++)
{
mass_consistent(i_node, j_neighbour) = temp;
mass_consistent(j_neighbour, i_node) = temp;
}
}
}
};

} 

#endif 
