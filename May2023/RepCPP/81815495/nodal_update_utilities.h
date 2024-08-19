
#if !defined( KRATOS_NODAL_UPDATE_UTILITIES )
#define  KRATOS_NODAL_UPDATE_UTILITIES



#include <set>




#include "includes/define.h"
#include "includes/variables.h"
#include "includes/mesh_moving_variables.h"
#include "includes/fsi_variables.h"
#include "containers/array_1d.h"
#include "includes/model_part.h"
#include "includes/communicator.h"
#include "includes/ublas_interface.h"
#include "utilities/openmp_utils.h"
#include "utilities/variable_utils.h"


namespace Kratos
{





















template <unsigned int TDim>
class NodalUpdateBaseClass
{

public:





KRATOS_CLASS_POINTER_DEFINITION( NodalUpdateBaseClass );





NodalUpdateBaseClass() {}






NodalUpdateBaseClass(const NodalUpdateBaseClass& Other);





virtual ~NodalUpdateBaseClass() {}






virtual void UpdateMeshTimeDerivatives(ModelPart& rModelPart,
const double timeStep) {
KRATOS_ERROR << "Calling the nodal update base class UpdateMeshTimeDerivatives() method. Call the proper time scheme derived one.";
}


virtual void SetMeshTimeDerivativesOnInterface(ModelPart& rInterfaceModelPart) {
auto& rLocalMesh = rInterfaceModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k=0; k<static_cast<int>(rLocalMesh.NumberOfNodes()); ++k)
{
ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;

array_1d<double, 3>& v_node = it_node->FastGetSolutionStepValue(VELOCITY);  
noalias(v_node) = it_node->FastGetSolutionStepValue(MESH_VELOCITY);         
}

rInterfaceModelPart.GetCommunicator().SynchronizeVariable(VELOCITY);
}



protected:


































private:





































}; 



template <unsigned int TDim>
class NodalUpdateNewmark : public NodalUpdateBaseClass<TDim>
{

public:





KRATOS_CLASS_POINTER_DEFINITION( NodalUpdateNewmark );





NodalUpdateNewmark(const double BossakAlpha = -0.3) {
const double bossak_f = 0.0;
const double bossak_beta = 0.25;
const double bossak_gamma = 0.5;

mBossakBeta = std::pow((1.0 + bossak_f - BossakAlpha), 2) * bossak_beta;
mBossakGamma = bossak_gamma + bossak_f - BossakAlpha;
}







NodalUpdateNewmark(const NodalUpdateNewmark& Other);





virtual ~NodalUpdateNewmark() {}






void UpdateMeshTimeDerivatives(ModelPart &rModelPart,
const double timeStep)  override{

auto& rLocalMesh = rModelPart.GetCommunicator().LocalMesh();
ModelPart::NodeIterator local_mesh_nodes_begin = rLocalMesh.NodesBegin();
#pragma omp parallel for firstprivate(local_mesh_nodes_begin)
for(int k = 0; k < static_cast<int>(rLocalMesh.NumberOfNodes()); ++k) {
const ModelPart::NodeIterator it_node = local_mesh_nodes_begin+k;

const array_1d<double, 3>& umesh_n = it_node->FastGetSolutionStepValue(MESH_DISPLACEMENT, 1);  
const array_1d<double, 3>& vmesh_n = it_node->FastGetSolutionStepValue(MESH_VELOCITY, 1);      
const array_1d<double, 3>& amesh_n = it_node->FastGetSolutionStepValue(MESH_ACCELERATION, 1);  

const array_1d<double, 3>& umesh_n1 = it_node->FastGetSolutionStepValue(MESH_DISPLACEMENT);    
array_1d<double, 3>& vmesh_n1 = it_node->FastGetSolutionStepValue(MESH_VELOCITY);              
array_1d<double, 3>& amesh_n1 = it_node->FastGetSolutionStepValue(MESH_ACCELERATION);          

const double const_u = mBossakGamma / (timeStep * mBossakBeta);
const double const_v = 1.0 - mBossakGamma / mBossakBeta;
const double const_a = timeStep * (1.0 - mBossakGamma / (2.0 * mBossakBeta));

for (unsigned int d=0; d<TDim; ++d) {
vmesh_n1[d] = const_u * (umesh_n1[d] - umesh_n[d]) + const_v * vmesh_n[d] + const_a * amesh_n[d];
amesh_n1[d] = (1.0 / (timeStep * mBossakGamma)) * (vmesh_n1[d] - vmesh_n[d]) - ((1 - mBossakGamma) / mBossakGamma) * amesh_n[d];
}
}

rModelPart.GetCommunicator().SynchronizeVariable(MESH_VELOCITY);
rModelPart.GetCommunicator().SynchronizeVariable(MESH_ACCELERATION);

}



protected:









double mBossakBeta;
double mBossakGamma;

























private:





































}; 









} 

#endif 
