
#if !defined(KRATOS_BFECC_CONVECTION_INCLUDED )
#define  KRATOS_BFECC_CONVECTION_INCLUDED

#define PRESSURE_ON_EULERIAN_MESH
#define USE_FEW_PARTICLES

#include <string>
#include <iostream>
#include <algorithm>


#include "includes/define.h"
#include "includes/model_part.h"
#include "utilities/geometry_utilities.h"
#include "geometries/tetrahedra_3d_4.h"
#include "includes/variables.h"
#include "utilities/timer.h"
#include "utilities/binbased_fast_point_locator.h"
#include "utilities/openmp_utils.h"
#include "processes/compute_nodal_gradient_process.h"
#include "utilities/parallel_utilities.h"
#include "utilities/pointer_communicator.h"
#include "utilities/pointer_map_communicator.h"

namespace Kratos
{

template<std::size_t TDim>
class BFECCConvection
{
public:
KRATOS_CLASS_POINTER_DEFINITION(BFECCConvection<TDim>);

BFECCConvection(
typename BinBasedFastPointLocator<TDim>::Pointer pSearchStructure,
const bool PartialDt = false,
const bool ActivateLimiter = false)
: mpSearchStructure(pSearchStructure), mActivateLimiter(ActivateLimiter)
{
}

~BFECCConvection()
{
}

/

auto& global_pointer = global_pointer_list(j);
auto X_j = coordinate_proxy.Get(global_pointer);

S_plus += std::max(0.0, inner_prod(grad_i, X_i-X_j));
S_minus += std::min(0.0, inner_prod(grad_i, X_i-X_j));
}

mSigmaPlus[i_node] = std::min(1.0, (std::abs(S_minus)+epsilon)/(S_plus+epsilon));
mSigmaMinus[i_node] = std::min(1.0, (S_plus+epsilon)/(std::abs(S_minus)+epsilon));
}
);

IndexPartition<int>(nparticles).for_each(
[&](int i_node){
auto it_node = rModelPart.NodesBegin() + i_node;
const double distance_i = it_node->FastGetSolutionStepValue(rVar);
const auto& X_i = it_node->Coordinates();
const auto& grad_i = it_node->GetValue(DISTANCE_GRADIENT);

double numerator = 0.0;
double denominator = 0.0;

GlobalPointersVector< Node >& global_pointer_list = it_node->GetValue(NEIGHBOUR_NODES);

for (unsigned int j = 0; j< global_pointer_list.size(); ++j)
{



auto& global_pointer = global_pointer_list(j);
auto X_j = coordinate_proxy.Get(global_pointer);
const double distance_j = distance_proxy.Get(global_pointer);

double beta_ij = 1.0;
if (inner_prod(grad_i, X_i-X_j) > 0)
beta_ij = mSigmaPlus[i_node];
else if (inner_prod(grad_i, X_i-X_j) < 0)
beta_ij = mSigmaMinus[i_node];

numerator += beta_ij*(distance_i - distance_j);
denominator += beta_ij*std::abs(distance_i - distance_j);
}

const double fraction = (std::abs(numerator)) / (denominator + epsilon);
mLimiter[i_node] = 1.0 - std::pow(fraction, power);
}
);
}


void ResetBoundaryConditions(ModelPart& rModelPart, const Variable< double >& rVar)
{
KRATOS_TRY

ModelPart::NodesContainerType::iterator inodebegin = rModelPart.NodesBegin();
vector<int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, rModelPart.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;

if (inode->IsFixed(rVar))
{
inode->FastGetSolutionStepValue(rVar)=inode->GetSolutionStepValue(rVar,1);
}
}
}

KRATOS_CATCH("")
}

void CopyScalarVarToPreviousTimeStep(ModelPart& rModelPart, const Variable< double >& rVar)
{
KRATOS_TRY
ModelPart::NodesContainerType::iterator inodebegin = rModelPart.NodesBegin();
vector<int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, rModelPart.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
inode->GetSolutionStepValue(rVar,1) = inode->FastGetSolutionStepValue(rVar);
}
}
KRATOS_CATCH("")
}

protected:
Kratos::Vector mSigmaPlus, mSigmaMinus, mLimiter;

private:
typename BinBasedFastPointLocator<TDim>::Pointer mpSearchStructure;
const bool mActivateLimiter;



};

} 

#endif 
