
#if !defined(KRATOS_MOVE_PARTICLE_UTILITY_FLUID_PFEM2_TRANSPORT_INCLUDED)
#define  KRATOS_MOVE_PARTICLE_UTILITY_FLUID_PFEM2_TRANSPORT_INCLUDED

#include <string>
#include <iostream>
#include <algorithm>


#include "includes/define.h"
#include "includes/node.h"

#include "includes/dof.h"
#include "includes/variables.h"
#include "containers/array_1d.h"
#include "containers/data_value_container.h"
#include "includes/mesh.h"
#include "utilities/math_utils.h"

#include "utilities/geometry_utilities.h"

#include "includes/model_part.h"


#include "spatial_containers/spatial_containers.h"
#include "spatial_containers/cell.h"
#include "spatial_containers/bins_dynamic_objects.h"

#include "utilities/spatial_containers_configure.h"

#include "geometries/line_2d_2.h"
#include "geometries/triangle_2d_3.h"
#include "geometries/triangle_3d_3.h"
#include "geometries/point.h"

#include "convection_diffusion_application.h"
#include "convection_particle.h"

#include "utilities/openmp_utils.h"
#include "utilities/parallel_utilities.h"
#include "time.h"


namespace Kratos
{
template< unsigned int TDim>
class MoveParticleUtilityScalarTransport
{
public:

typedef SpatialContainersConfigure<TDim>     Configure;
typedef typename Configure::PointType                      PointType;
typedef typename Configure::ContainerType                  ContainerType;
typedef typename Configure::IteratorType                   IteratorType;
typedef typename Configure::ResultContainerType            ResultContainerType;
typedef typename Configure::ResultIteratorType             ResultIteratorType;
typedef PointerVector< Convection_Particle, Convection_Particle*, std::vector<Convection_Particle*> > ParticlePointerVector;

KRATOS_CLASS_POINTER_DEFINITION(MoveParticleUtilityScalarTransport);

MoveParticleUtilityScalarTransport(ModelPart& model_part, int maximum_number_of_particles)
: mr_model_part(model_part) , mmaximum_number_of_particles(maximum_number_of_particles) ,
mUnknownVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetUnknownVariable()) ,
mProjectionVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetProjectionVariable()) ,
mVelocityVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetVelocityVariable()) ,
mMeshVelocityVar((model_part.GetProcessInfo()[CONVECTION_DIFFUSION_SETTINGS])->GetMeshVelocityVariable())
{
std::cout << "initializing moveparticle utility for scalar transport" << std::endl;

Check();
ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
for(unsigned int  ii=0; ii<mr_model_part.Elements().size(); ii++)
{
ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
ielem->SetId(ii+1);
}
mlast_elem_id= (mr_model_part.ElementsEnd()-1)->Id();
int node_id=0;
ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
vector<unsigned int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator pnode = inodebegin+ii;
array_1d<double,3> position_node;
double distance=0.0;
position_node = pnode->Coordinates();
GlobalPointersVector< Node >& rneigh = pnode->GetValue(NEIGHBOUR_NODES);
const double number_of_neighbours = double(rneigh.size());
for( GlobalPointersVector<Node >::iterator inode = rneigh.begin(); inode!=rneigh.end(); inode++)
{
array_1d<double,3> position_difference;
position_difference = inode->Coordinates() - position_node;
double current_distance= sqrt(pow(position_difference[0],2)+pow(position_difference[1],2)+pow(position_difference[2],2));
distance += current_distance / number_of_neighbours;
}
pnode->FastGetSolutionStepValue(MEAN_SIZE)=distance;

node_id=pnode->GetId();
}
}
mlast_node_id=node_id;

vector<unsigned int> element_partition;
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
{
ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;

double mElemSize;
array_1d<double,3> Edge(3,0.0);
Edge = ielem->GetGeometry()[1].Coordinates() - ielem->GetGeometry()[0].Coordinates();
mElemSize = Edge[0]*Edge[0];
for (unsigned int d = 1; d < TDim; d++)
mElemSize += Edge[d]*Edge[d];

for (unsigned int i = 2; i < (TDim+1); i++)
for(unsigned int j = 0; j < i; j++)
{
Edge = ielem->GetGeometry()[i].Coordinates() - ielem->GetGeometry()[j].Coordinates();
double Length = Edge[0]*Edge[0];
for (unsigned int d = 1; d < TDim; d++)
Length += Edge[d]*Edge[d];
if (Length < mElemSize) mElemSize = Length;
}
mElemSize = sqrt(mElemSize);
ielem->GetValue(MEAN_SIZE) = mElemSize;
}
}


BoundedMatrix<double, 5*(1+TDim), 3 > pos;
BoundedMatrix<double, 5*(1+TDim), (1+TDim) > N;

int particle_id=0;
mnelems = mr_model_part.Elements().size();

std::cout << "about to resize vectors" << std::endl;


mparticles_vector.resize(mnelems*mmaximum_number_of_particles);

mnumber_of_particles_in_elems.resize(mnelems);
mnumber_of_particles_in_elems=ZeroVector(mnelems);

mnumber_of_particles_in_elems_aux.resize(mnelems);

mvector_of_particle_pointers_vectors.resize(mnelems);
int i_int=0; 
std::cout << "about to create particles" << std::endl;

moffset=0;
for(unsigned int ii=0; ii<mr_model_part.Elements().size(); ii++)
{
ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
mvector_of_particle_pointers_vectors[ii] = ParticlePointerVector( mmaximum_number_of_particles*2 );
ParticlePointerVector& particle_pointers = mvector_of_particle_pointers_vectors[ii];
int & number_of_particles = mnumber_of_particles_in_elems[ii];
number_of_particles=0;

Geometry< Node >& geom = ielem->GetGeometry();
ComputeGaussPointPositions_initial(geom, pos, N); 
for (unsigned int j = 0; j < pos.size1(); j++)
{
++particle_id;

Convection_Particle& pparticle = mparticles_vector[particle_id-1];
pparticle.X()=pos(j,0);
pparticle.Y()=pos(j,1);
pparticle.Z()=pos(j,2);

pparticle.GetEraseFlag()=false;

float & scalar1= pparticle.GetScalar1();
scalar1=0.0;

for (unsigned int k = 0; k < (TDim+1); k++)
{
scalar1 +=  N(j, k) * geom[k].FastGetSolutionStepValue(mUnknownVar);
}


particle_pointers(j) = &pparticle;
number_of_particles++ ;

}
++i_int;
}

m_nparticles=particle_id; 
KRATOS_WATCH(m_nparticles);
mparticle_printing_tool_initialized=false;
}


virtual ~MoveParticleUtilityScalarTransport()
{}

void MountBin()
{
KRATOS_TRY

ContainerType& rElements           =  mr_model_part.ElementsArray();
IteratorType it_begin              =  rElements.begin();
IteratorType it_end                =  rElements.end();

typename BinsObjectDynamic<Configure>::Pointer paux = typename BinsObjectDynamic<Configure>::Pointer(new BinsObjectDynamic<Configure>(it_begin, it_end  ) );
paux.swap(mpBinsObjectDynamic);

std::cout << "finished mounting Bins" << std::endl;

KRATOS_CATCH("")
}

void MountBin(const double CellSize)
{
KRATOS_TRY

ContainerType& rElements           =  mr_model_part.ElementsArray();
IteratorType it_begin              =  rElements.begin();
IteratorType it_end                =  rElements.end();
typename BinsObjectDynamic<Configure>::Pointer paux = typename BinsObjectDynamic<Configure>::Pointer(new BinsObjectDynamic<Configure>(it_begin, it_end, CellSize ) );
paux.swap(mpBinsObjectDynamic);

KRATOS_INFO("MoveParticleUtilityScalarTransport") << "Finished mounting Bins with cell size: " << CellSize << std::endl;

KRATOS_CATCH("")
}

void CalculateVelOverElemSize()
{
KRATOS_TRY


const double nodal_weight = 1.0/ (1.0 + double (TDim) );

ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
vector<unsigned int> element_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
{
ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
Geometry<Node >& geom = ielem->GetGeometry();

array_1d<double, 3 >vector_mean_velocity=ZeroVector(3);

for (unsigned int i=0; i != (TDim+1) ; i++)
vector_mean_velocity += geom[i].FastGetSolutionStepValue(mVelocityVar);
vector_mean_velocity *= nodal_weight;

const double mean_velocity = sqrt ( pow(vector_mean_velocity[0],2) + pow(vector_mean_velocity[1],2) + pow(vector_mean_velocity[2],2) );
ielem->GetValue(MEAN_VEL_OVER_ELEM_SIZE) = mean_velocity / (ielem->GetValue(MEAN_SIZE));
}
}
KRATOS_CATCH("")
}



void ResetBoundaryConditions()
{
KRATOS_TRY

ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
vector<unsigned int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;

if (inode->IsFixed(mUnknownVar))
{
inode->FastGetSolutionStepValue(mUnknownVar)=inode->GetSolutionStepValue(mUnknownVar,1);
}
}
}

KRATOS_CATCH("")
}

void CalculateDeltaVariables()
{
KRATOS_TRY
ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
vector<unsigned int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
inode->FastGetSolutionStepValue(DELTA_SCALAR1) = inode->FastGetSolutionStepValue(mUnknownVar) - inode->FastGetSolutionStepValue(mProjectionVar) ;
}
}

KRATOS_CATCH("")
}


void CopyScalarVarToPreviousTimeStep(const Variable<double>& OriginVariable,
ModelPart::NodesContainerType& rNodes)
{
KRATOS_TRY
ModelPart::NodesContainerType::iterator inodebegin = rNodes.begin();
vector<unsigned int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, rNodes.size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
inode->GetSolutionStepValue(OriginVariable,1) = inode->FastGetSolutionStepValue(OriginVariable);
}
}
KRATOS_CATCH("")
}


void MoveParticles()
{

KRATOS_TRY

ProcessInfo& CurrentProcessInfo = mr_model_part.GetProcessInfo();

const int offset = moffset; 

bool even_timestep;
if (offset!=0) even_timestep=false;
else even_timestep=true;

const int post_offset = mmaximum_number_of_particles*int(even_timestep);	


double delta_t = CurrentProcessInfo[DELTA_TIME];

array_1d<double,TDim+1> N;
const unsigned int max_results = 10000;


max_nsubsteps = 10;
max_substep_dt=delta_t/double(max_nsubsteps);

vector<unsigned int> element_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();



#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
{

int & number_of_particles = mnumber_of_particles_in_elems[ii]; 

mnumber_of_particles_in_elems_aux[ii]=number_of_particles;
mnumber_of_particles_in_elems[ii]=0;
}
}
std::cout << "convecting particles" << std::endl;

#pragma omp barrier

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{

ResultContainerType results(max_results);

GlobalPointersVector< Element > elements_in_trajectory;
elements_in_trajectory.resize(20);

for(unsigned int ielem=element_partition[kkk]; ielem<element_partition[kkk+1]; ielem++)
{

ModelPart::ElementsContainerType::iterator old_element = ielembegin+ielem;
const int old_element_id = old_element->Id();

ParticlePointerVector& old_element_particle_pointers = mvector_of_particle_pointers_vectors(old_element_id-1);

if ( (results.size()) !=max_results)
results.resize(max_results);

unsigned int number_of_elements_in_trajectory=0; 

for(int ii=0; ii<(mnumber_of_particles_in_elems_aux(ielem)); ii++)
{

Convection_Particle & pparticle = old_element_particle_pointers[offset+ii];

Element::Pointer pcurrent_element( *old_element.base() );
ResultIteratorType result_begin = results.begin();
bool & erase_flag=pparticle.GetEraseFlag();
if (erase_flag==false){
MoveParticle(pparticle,pcurrent_element,elements_in_trajectory,number_of_elements_in_trajectory,result_begin,max_results); 

const int current_element_id = pcurrent_element->Id();

int & number_of_particles_in_current_elem = mnumber_of_particles_in_elems(current_element_id-1);

if (number_of_particles_in_current_elem<mmaximum_number_of_particles && erase_flag==false)
{
{

ParticlePointerVector& current_element_particle_pointers = mvector_of_particle_pointers_vectors(current_element_id-1);

#pragma omp critical
{
if (number_of_particles_in_current_elem<mmaximum_number_of_particles) 
{

current_element_particle_pointers(post_offset+number_of_particles_in_current_elem) = &pparticle;

number_of_particles_in_current_elem++ ;
if (number_of_particles_in_current_elem>mmaximum_number_of_particles)
KRATOS_WATCH("MAL");

}
else
pparticle.GetEraseFlag()=true; 
}
}
}
else
pparticle.GetEraseFlag()=true; 



}
}
}
}


moffset = post_offset;; 

KRATOS_CATCH("")
}

void TransferLagrangianToEulerian() 
{
KRATOS_TRY

const double threshold= 0.0/(double(TDim)+1.0);


std::cout << "projecting info to mesh" << std::endl;


const int offset = moffset; 


ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
vector<unsigned int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
inode->FastGetSolutionStepValue(mProjectionVar)=0.0;
inode->FastGetSolutionStepValue(YP)=0.0;
}

}

vector<unsigned int> element_partition;
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
{
ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;

array_1d<double,3*(TDim+1)> nodes_positions;
array_1d<double,(TDim+1)> nodes_added_scalar1 = ZeroVector((TDim+1));
array_1d<double,(TDim+1)> nodes_addedweights = ZeroVector((TDim+1));

Geometry<Node >& geom = ielem->GetGeometry();

for (int i=0 ; i!=(TDim+1) ; ++i)
{
nodes_positions[i*3+0]=geom[i].X();
nodes_positions[i*3+1]=geom[i].Y();
nodes_positions[i*3+2]=geom[i].Z();
}

int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];


for (int iii=0; iii<number_of_particles_in_elem ; iii++ )
{
if (iii==mmaximum_number_of_particles) 
break;

Convection_Particle & pparticle = element_particle_pointers[offset+iii];

if (pparticle.GetEraseFlag()==false)
{

array_1d<double,3> & position = pparticle.Coordinates();

const float& particle_scalar1 = pparticle.GetScalar1();  

array_1d<double,TDim+1> N;
bool is_found = CalculatePosition(nodes_positions,position[0],position[1],position[2],N);
if (is_found==false) 
{
KRATOS_WATCH(N);
for (int j=0 ; j!=(TDim+1); j++)
if (N[j]<0.0 && N[j]> -1e-5)
N[j]=1e-10;

}

for (int j=0 ; j!=(TDim+1); j++) 
{

double weight=N(j)*N(j);
if (weight<threshold) weight=1e-10;
if (weight<0.0) {KRATOS_WATCH(weight)}
else
{
nodes_addedweights[j]+= weight;

nodes_added_scalar1[j] += weight*particle_scalar1;


}
}
}
}

for (int i=0 ; i!=(TDim+1) ; ++i) {
geom[i].SetLock();
geom[i].FastGetSolutionStepValue(mProjectionVar) +=nodes_added_scalar1[i];
geom[i].FastGetSolutionStepValue(YP) +=nodes_addedweights[i];
geom[i].UnSetLock();
}
}
}

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
double sum_weights = inode->FastGetSolutionStepValue(YP);
if (sum_weights>0.00001)
{
double & height = inode->FastGetSolutionStepValue(mProjectionVar);
height /=sum_weights; 
}

else 
{
inode->FastGetSolutionStepValue(mProjectionVar)=inode->FastGetSolutionStepValue(mUnknownVar,1); 
}
}
}


KRATOS_CATCH("")
}



void TransferLagrangianToEulerianImp() 
{
KRATOS_TRY


std::cout << "projecting info to mesh (semi implicit)" << std::endl;


const int offset = moffset; 


ModelPart::NodesContainerType::iterator inodebegin = mr_model_part.NodesBegin();
vector<unsigned int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
inode->FastGetSolutionStepValue(mProjectionVar)=0.0;
inode->FastGetSolutionStepValue(YP)=0.0;
}

}

vector<unsigned int> element_partition;
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();
#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{

BoundedMatrix<double, TDim+1 , TDim+1  > mass_matrix; 
array_1d<double,(TDim+1)> rhs_scalar1;

array_1d<double,3*(TDim+1)> nodes_positions;
array_1d<double,(TDim+1)> nodes_added_scalar1 = ZeroVector((TDim+1));
array_1d<double,(TDim+1)> nodes_addedweights = ZeroVector((TDim+1));

for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
{
ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;

nodes_added_scalar1 = ZeroVector((TDim+1));   
nodes_addedweights = ZeroVector((TDim+1));     
mass_matrix = ZeroMatrix(TDim+1 , TDim+1 );  
rhs_scalar1 = ZeroVector((TDim+1));         

Geometry<Node >& geom = ielem->GetGeometry();
const double elem_volume = geom.Area();

for (int i=0 ; i!=(TDim+1) ; ++i)  
{
nodes_positions[i*3+0]=geom[i].X();
nodes_positions[i*3+1]=geom[i].Y();
nodes_positions[i*3+2]=geom[i].Z();
}

int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];

for (int iii=0; iii<number_of_particles_in_elem ; iii++ )
{
if (iii==mmaximum_number_of_particles) 
break;

Convection_Particle & pparticle = element_particle_pointers[offset+iii];

if (pparticle.GetEraseFlag()==false)
{

array_1d<double,3> & position = pparticle.Coordinates();

const float& particle_scalar1 = pparticle.GetScalar1();  

array_1d<double,TDim+1> N;
bool is_found = CalculatePosition(nodes_positions,position[0],position[1],position[2],N);
if (is_found==false) 
{
KRATOS_WATCH(N);
for (int j=0 ; j!=(TDim+1); j++)
if (N[j]<0.0 && N[j]> -1e-5)
N[j]=1e-10;

}

for (int j=0 ; j!=(TDim+1); j++) 
{
double weight=N(j);
for (int k=0 ; k!=(TDim+1); k++) 
mass_matrix(j,k) += weight*N(k);

rhs_scalar1[j] += weight * double(particle_scalar1);

if(true)
{
double this_particle_weight = weight*elem_volume/(double(number_of_particles_in_elem))*0.1; 
nodes_addedweights[j]+= this_particle_weight;
nodes_added_scalar1[j] += this_particle_weight*particle_scalar1;
}
}
}
}

BoundedMatrix<double, TDim+1 , TDim+1  > inverse_mass_matrix=ZeroMatrix(TDim+1 , TDim+1);
if constexpr (TDim==3)
InvertMatrix( mass_matrix,  inverse_mass_matrix);
else
InvertMatrix3x3( mass_matrix,  inverse_mass_matrix);

if(number_of_particles_in_elem > static_cast<int>(TDim)*3) 
{
for (int i=0 ; i!=(TDim+1); i++)
{
for (int j=0 ; j!=(TDim+1); j++)
{
nodes_added_scalar1[i] += inverse_mass_matrix(i,j)*rhs_scalar1[j]*elem_volume*(1.0/(double(1+TDim)));

}
}
for (int i=0 ; i!=(TDim+1); i++)
nodes_addedweights[i] += elem_volume*(1.0/(double(1+TDim)));
}


for (int i=0 ; i!=(TDim+1) ; ++i) {
geom[i].SetLock();
geom[i].FastGetSolutionStepValue(mProjectionVar) +=nodes_added_scalar1[i];
geom[i].FastGetSolutionStepValue(YP) +=nodes_addedweights[i];
geom[i].UnSetLock();
}
}
}

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
double sum_weights = inode->FastGetSolutionStepValue(YP);
if (sum_weights>0.00001)
{
double & scalar1 = inode->FastGetSolutionStepValue(mProjectionVar);
scalar1 /=sum_weights; 
}

else 
{
inode->FastGetSolutionStepValue(mProjectionVar)=inode->FastGetSolutionStepValue(mUnknownVar,1);
}
}
}


KRATOS_CATCH("")
}

void CorrectParticlesWithoutMovingUsingDeltaVariables()
{
KRATOS_TRY

const int offset = moffset; 
ModelPart::ElementsContainerType::iterator ielembegin = mr_model_part.ElementsBegin();


vector<unsigned int> element_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mr_model_part.Elements().size(), element_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
for(unsigned int ii=element_partition[kkk]; ii<element_partition[kkk+1]; ii++)
{
ModelPart::ElementsContainerType::iterator ielem = ielembegin+ii;
Element::Pointer pelement(*ielem.base());
Geometry<Node >& geom = ielem->GetGeometry();

int & number_of_particles_in_elem= mnumber_of_particles_in_elems[ii];
ParticlePointerVector&  element_particle_pointers =  mvector_of_particle_pointers_vectors[ii];

for (int iii=0; iii<number_of_particles_in_elem ; iii++ )
{
if (iii>mmaximum_number_of_particles) 
break;

Convection_Particle & pparticle = element_particle_pointers[offset+iii];


bool erase_flag= pparticle.GetEraseFlag();
if (erase_flag==false)
{
CorrectParticleUsingDeltaVariables(pparticle,pelement,geom); 
}


}
}
}
KRATOS_CATCH("")
}

/
for (unsigned int i=0;i!=(neighb_elems.size());i++)
{
if(neighb_elems(i).get()!=nullptr)
{
Geometry<Node >& geom = neighb_elems[i].GetGeometry();
bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],N);
if (is_found_2)
{
pelement=neighb_elems(i)->shared_from_this();
return true;
}
}
}

SizeType results_found = mpBinsObjectDynamic->SearchObjectsInCell(Point{coords}, result_begin, MaxNumberOfResults );

if(results_found>0){
for(SizeType i = 0; i< results_found; i++)
{
Geometry<Node >& geom = (*(result_begin+i))->GetGeometry();

bool is_found = CalculatePosition(geom,coords[0],coords[1],coords[2],N);

if(is_found == true)
{
pelement=Element::Pointer((*(result_begin+i)));
return true;
}
}
}
return false;
}


bool FindNodeOnMesh( array_1d<double,3>& position,
array_1d<double,TDim+1>& N,
Element::Pointer & pelement,
GlobalPointersVector< Element >& elements_in_trajectory,
unsigned int & number_of_elements_in_trajectory,
unsigned int & check_from_element_number,
ResultIteratorType result_begin,
const unsigned int MaxNumberOfResults)
{
typedef std::size_t SizeType;

const array_1d<double,3>& coords = position;
array_1d<double,TDim+1> aux_N;
Geometry<Node >& geom_default = pelement->GetGeometry(); 
bool is_found_1 = CalculatePosition(geom_default,coords[0],coords[1],coords[2],N);
if(is_found_1 == true)
{
return true; 
}

for (unsigned int i=(check_from_element_number);i!=number_of_elements_in_trajectory;i++)
{
Geometry<Node >& geom = elements_in_trajectory[i].GetGeometry();
bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],aux_N);
if (is_found_2)
{
pelement=elements_in_trajectory(i)->shared_from_this();
N=aux_N;
check_from_element_number = i+1 ; 
return true;
}

}

auto& neighb_elems = pelement->GetValue(NEIGHBOUR_ELEMENTS);

for (unsigned int i=0;i!=(neighb_elems.size());i++)
{
if(neighb_elems(i).get()!=nullptr)
{
Geometry<Node >& geom = neighb_elems[i].GetGeometry();
bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],N);
if (is_found_2)
{
pelement=neighb_elems(i)->shared_from_this();
if (number_of_elements_in_trajectory<20)
{
elements_in_trajectory(number_of_elements_in_trajectory)=pelement;
number_of_elements_in_trajectory++;
check_from_element_number = number_of_elements_in_trajectory;  
}
return true;
}
}
}


SizeType results_found = mpBinsObjectDynamic->SearchObjectsInCell(Point{coords}, result_begin, MaxNumberOfResults );

if(results_found>0)
{
for(SizeType i = 0; i< results_found; i++)
{
Geometry<Node >& geom = (*(result_begin+i))->GetGeometry();

bool is_found = CalculatePosition(geom,coords[0],coords[1],coords[2],N);

if(is_found == true)
{
pelement=Element::Pointer((*(result_begin+i)));
if (number_of_elements_in_trajectory<20)
{
elements_in_trajectory(number_of_elements_in_trajectory)=pelement;
number_of_elements_in_trajectory++;
check_from_element_number = number_of_elements_in_trajectory;  
}
return true;
}
}
}

return false;
}




inline bool CalculatePosition(Geometry<Node >&geom,
const double xc, const double yc, const double zc,
array_1d<double, 3 > & N
)
{
double x0 = geom[0].X();
double y0 = geom[0].Y();
double x1 = geom[1].X();
double y1 = geom[1].Y();
double x2 = geom[2].X();
double y2 = geom[2].Y();

double area = CalculateVol(x0, y0, x1, y1, x2, y2);
double inv_area = 0.0;
if (area == 0.0)
{
KRATOS_THROW_ERROR(std::logic_error, "element with zero area found", "");
} else
{
inv_area = 1.0 / area;
}


N[0] = CalculateVol(x1, y1, x2, y2, xc, yc) * inv_area;
N[1] = CalculateVol(x2, y2, x0, y0, xc, yc) * inv_area;
N[2] = CalculateVol(x0, y0, x1, y1, xc, yc) * inv_area;

if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0) 
return true;

return false;
}
inline bool CalculatePosition(const array_1d<double,3*(TDim+1)>& nodes_positions,
const double xc, const double yc, const double zc,
array_1d<double, 3 > & N
)
{
const double& x0 = nodes_positions[0];
const double& y0 = nodes_positions[1];
const double& x1 = nodes_positions[3];
const double& y1 = nodes_positions[4];
const double& x2 = nodes_positions[6];
const double& y2 = nodes_positions[7];

double area = CalculateVol(x0, y0, x1, y1, x2, y2);
double inv_area = 0.0;
if (area == 0.0)
{
KRATOS_THROW_ERROR(std::logic_error, "element with zero area found", "");
} else
{
inv_area = 1.0 / area;
}


N[0] = CalculateVol(x1, y1, x2, y2, xc, yc) * inv_area;
N[1] = CalculateVol(x2, y2, x0, y0, xc, yc) * inv_area;
N[2] = CalculateVol(x0, y0, x1, y1, xc, yc) * inv_area;

if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0) 
return true;

return false;
}



inline bool CalculatePosition(Geometry<Node >&geom,
const double xc, const double yc, const double zc,
array_1d<double, 4 > & N
)
{

double x0 = geom[0].X();
double y0 = geom[0].Y();
double z0 = geom[0].Z();
double x1 = geom[1].X();
double y1 = geom[1].Y();
double z1 = geom[1].Z();
double x2 = geom[2].X();
double y2 = geom[2].Y();
double z2 = geom[2].Z();
double x3 = geom[3].X();
double y3 = geom[3].Y();
double z3 = geom[3].Z();

double vol = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3);

double inv_vol = 0.0;
if (vol < 0.000000000000000000000000000001)
{
KRATOS_THROW_ERROR(std::logic_error, "element with zero vol found", "");
} else
{
inv_vol = 1.0 / vol;
}

N[0] = CalculateVol(x1, y1, z1, x3, y3, z3, x2, y2, z2, xc, yc, zc) * inv_vol;
N[1] = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, xc, yc, zc) * inv_vol;
N[2] = CalculateVol(x3, y3, z3, x1, y1, z1, x0, y0, z0, xc, yc, zc) * inv_vol;
N[3] = CalculateVol(x3, y3, z3, x0, y0, z0, x2, y2, z2, xc, yc, zc) * inv_vol;


if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[3] >= 0.0 &&
N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0 && N[3] <= 1.0)
return true;

return false;
}
inline bool CalculatePosition(const array_1d<double,3*(TDim+1)>& nodes_positions,
const double xc, const double yc, const double zc,
array_1d<double, 4 > & N
)
{

const double& x0 = nodes_positions[0];
const double& y0 = nodes_positions[1];
const double& z0 = nodes_positions[2];
const double& x1 = nodes_positions[3];
const double& y1 = nodes_positions[4];
const double& z1 = nodes_positions[5];
const double& x2 = nodes_positions[6];
const double& y2 = nodes_positions[7];
const double& z2 = nodes_positions[8];
const double& x3 = nodes_positions[9];
const double& y3 = nodes_positions[10];
const double& z3 = nodes_positions[11];

double vol = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3);

double inv_vol = 0.0;
if (vol < 0.000000000000000000000000000001)
{
KRATOS_THROW_ERROR(std::logic_error, "element with zero vol found", "");
} else
{
inv_vol = 1.0 / vol;
}

N[0] = CalculateVol(x1, y1, z1, x3, y3, z3, x2, y2, z2, xc, yc, zc) * inv_vol;
N[1] = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, xc, yc, zc) * inv_vol;
N[2] = CalculateVol(x3, y3, z3, x1, y1, z1, x0, y0, z0, xc, yc, zc) * inv_vol;
N[3] = CalculateVol(x3, y3, z3, x0, y0, z0, x2, y2, z2, xc, yc, zc) * inv_vol;


if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[3] >= 0.0 &&
N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0 && N[3] <= 1.0)
return true;

return false;
}

inline double CalculateVol(const double x0, const double y0,
const double x1, const double y1,
const double x2, const double y2
)
{
return 0.5 * ((x1 - x0)*(y2 - y0)- (y1 - y0)*(x2 - x0));
}

inline double CalculateVol(const double x0, const double y0, const double z0,
const double x1, const double y1, const double z1,
const double x2, const double y2, const double z2,
const double x3, const double y3, const double z3
)
{
double x10 = x1 - x0;
double y10 = y1 - y0;
double z10 = z1 - z0;

double x20 = x2 - x0;
double y20 = y2 - y0;
double z20 = z2 - z0;

double x30 = x3 - x0;
double y30 = y3 - y0;
double z30 = z3 - z0;

double detJ = x10 * y20 * z30 - x10 * y30 * z20 + y10 * z20 * x30 - y10 * x20 * z30 + z10 * x20 * y30 - z10 * y20 * x30;
return detJ * 0.1666666666666666666667;
}



void ComputeGaussPointPositions_4(Geometry< Node >& geom, BoundedMatrix<double, 7, 3 > & pos,BoundedMatrix<double, 7, 3 > & N)
{
double one_third = 1.0 / 3.0;
double one_sixt = 0.15; 
double two_third = 0.7; 

N(0, 0) = one_sixt;
N(0, 1) = one_sixt;
N(0, 2) = two_third;
N(1, 0) = two_third;
N(1, 1) = one_sixt;
N(1, 2) = one_sixt;
N(2, 0) = one_sixt;
N(2, 1) = two_third;
N(2, 2) = one_sixt;
N(3, 0) = one_third;
N(3, 1) = one_third;
N(3, 2) = one_third;

pos(0, 0) = one_sixt * geom[0].X() + one_sixt * geom[1].X() + two_third * geom[2].X();
pos(0, 1) = one_sixt * geom[0].Y() + one_sixt * geom[1].Y() + two_third * geom[2].Y();
pos(0, 2) = one_sixt * geom[0].Z() + one_sixt * geom[1].Z() + two_third * geom[2].Z();

pos(1, 0) = two_third * geom[0].X() + one_sixt * geom[1].X() + one_sixt * geom[2].X();
pos(1, 1) = two_third * geom[0].Y() + one_sixt * geom[1].Y() + one_sixt * geom[2].Y();
pos(1, 2) = two_third * geom[0].Z() + one_sixt * geom[1].Z() + one_sixt * geom[2].Z();

pos(2, 0) = one_sixt * geom[0].X() + two_third * geom[1].X() + one_sixt * geom[2].X();
pos(2, 1) = one_sixt * geom[0].Y() + two_third * geom[1].Y() + one_sixt * geom[2].Y();
pos(2, 2) = one_sixt * geom[0].Z() + two_third * geom[1].Z() + one_sixt * geom[2].Z();

pos(3, 0) = one_third * geom[0].X() + one_third * geom[1].X() + one_third * geom[2].X();
pos(3, 1) = one_third * geom[0].Y() + one_third * geom[1].Y() + one_third * geom[2].Y();
pos(3, 2) = one_third * geom[0].Z() + one_third * geom[1].Z() + one_third * geom[2].Z();

}


void ComputeGaussPointPositionsForPostReseed(Geometry< Node >& geom, BoundedMatrix<double, 7, 3 > & pos,BoundedMatrix<double, 7, 3 > & N) 
{
double one_third = 1.0 / 3.0;
double one_eight = 0.12; 
double three_quarters = 0.76; 

N(0, 0) = one_eight;
N(0, 1) = one_eight;
N(0, 2) = three_quarters;

N(1, 0) = three_quarters;
N(1, 1) = one_eight;
N(1, 2) = one_eight;

N(2, 0) = one_eight;
N(2, 1) = three_quarters;
N(2, 2) = one_eight;

N(3, 0) = one_third;
N(3, 1) = one_third;
N(3, 2) = one_third;

N(4, 0) = one_eight;
N(4, 1) = 0.44;
N(4, 2) = 0.44;

N(5, 0) = 0.44;
N(5, 1) = one_eight;
N(5, 2) = 0.44;

N(6, 0) = 0.44;
N(6, 1) = 0.44;
N(6, 2) = one_eight;


pos(0, 0) = one_eight * geom[0].X() + one_eight * geom[1].X() + three_quarters * geom[2].X();
pos(0, 1) = one_eight * geom[0].Y() + one_eight * geom[1].Y() + three_quarters * geom[2].Y();
pos(0, 2) = one_eight * geom[0].Z() + one_eight * geom[1].Z() + three_quarters * geom[2].Z();

pos(1, 0) = three_quarters * geom[0].X() + one_eight * geom[1].X() + one_eight * geom[2].X();
pos(1, 1) = three_quarters * geom[0].Y() + one_eight * geom[1].Y() + one_eight * geom[2].Y();
pos(1, 2) = three_quarters * geom[0].Z() + one_eight * geom[1].Z() + one_eight * geom[2].Z();

pos(2, 0) = one_eight * geom[0].X() + three_quarters * geom[1].X() + one_eight * geom[2].X();
pos(2, 1) = one_eight * geom[0].Y() + three_quarters * geom[1].Y() + one_eight * geom[2].Y();
pos(2, 2) = one_eight * geom[0].Z() + three_quarters * geom[1].Z() + one_eight * geom[2].Z();

pos(3, 0) = one_third * geom[0].X() + one_third * geom[1].X() + one_third * geom[2].X();
pos(3, 1) = one_third * geom[0].Y() + one_third * geom[1].Y() + one_third * geom[2].Y();
pos(3, 2) = one_third * geom[0].Z() + one_third * geom[1].Z() + one_third * geom[2].Z();

pos(4, 0) = one_eight * geom[0].X() + 0.44 * geom[1].X() + 0.44 * geom[2].X();
pos(4, 1) = one_eight * geom[0].Y() + 0.44 * geom[1].Y() + 0.44 * geom[2].Y();
pos(4, 2) = one_eight * geom[0].Z() + 0.44 * geom[1].Z() + 0.44 * geom[2].Z();

pos(5, 0) = 0.44 * geom[0].X() + one_eight * geom[1].X() + 0.44 * geom[2].X();
pos(5, 1) = 0.44 * geom[0].Y() + one_eight * geom[1].Y() + 0.44 * geom[2].Y();
pos(5, 2) = 0.44 * geom[0].Z() + one_eight * geom[1].Z() + 0.44 * geom[2].Z();

pos(6, 0) = 0.44 * geom[0].X() + 0.44 * geom[1].X() + one_eight * geom[2].X();
pos(6, 1) = 0.44 * geom[0].Y() + 0.44 * geom[1].Y() + one_eight * geom[2].Y();
pos(6, 2) = 0.44 * geom[0].Z() + 0.44 * geom[1].Z() + one_eight * geom[2].Z();




}

void ComputeGaussPointPositionsForPostReseed(Geometry< Node >& geom, BoundedMatrix<double, 9, 3 > & pos,BoundedMatrix<double, 9, 4 > & N) 
{
double one_quarter = 0.25;
double small_fraction = 0.1; 
double big_fraction = 0.7; 
double mid_fraction = 0.3; 

N(0, 0) = big_fraction;
N(0, 1) = small_fraction;
N(0, 2) = small_fraction;
N(0, 3) = small_fraction;

N(1, 0) = small_fraction;
N(1, 1) = big_fraction;
N(1, 2) = small_fraction;
N(1, 3) = small_fraction;

N(2, 0) = small_fraction;
N(2, 1) = small_fraction;
N(2, 2) = big_fraction;
N(2, 3) = small_fraction;

N(3, 0) = small_fraction;
N(3, 1) = small_fraction;
N(3, 2) = small_fraction;
N(3, 3) = big_fraction;

N(4, 0) = one_quarter;
N(4, 1) = one_quarter;
N(4, 2) = one_quarter;
N(4, 3) = one_quarter;

N(5, 0) = small_fraction;
N(5, 1) = mid_fraction;
N(5, 2) = mid_fraction;
N(5, 3) = mid_fraction;

N(6, 0) = mid_fraction;
N(6, 1) = small_fraction;
N(6, 2) = mid_fraction;
N(6, 3) = mid_fraction;

N(7, 0) = mid_fraction;
N(7, 1) = mid_fraction;
N(7, 2) = small_fraction;
N(7, 3) = mid_fraction;

N(8, 0) = mid_fraction;
N(8, 1) = mid_fraction;
N(8, 2) = mid_fraction;
N(8, 3) = small_fraction;

pos=ZeroMatrix(9,3);
for (unsigned int i=0; i!=4; i++) 
{
array_1d<double, 3 > & coordinates = geom[i].Coordinates();
for (unsigned int j=0; j!=9; j++) 
{
for (unsigned int k=0; k!=3; k++) 
pos(j,k) += N(j,i) * coordinates[k];
}
}


}



void ComputeGaussPointPositionsForPreReseed(Geometry< Node >& geom, BoundedMatrix<double, 3, 3 > & pos,BoundedMatrix<double, 3, 3 > & N) 
{

N(0, 0) = 0.5;
N(0, 1) = 0.25;
N(0, 2) = 0.25;

N(1, 0) = 0.25;
N(1, 1) = 0.5;
N(1, 2) = 0.25;

N(2, 0) = 0.25;
N(2, 1) = 0.25;
N(2, 2) = 0.5;

pos(0, 0) = 0.5 * geom[0].X() + 0.25 * geom[1].X() + 0.25 * geom[2].X();
pos(0, 1) = 0.5 * geom[0].Y() + 0.25 * geom[1].Y() + 0.25 * geom[2].Y();
pos(0, 2) = 0.5 * geom[0].Z() + 0.25 * geom[1].Z() + 0.25 * geom[2].Z();

pos(1, 0) = 0.25 * geom[0].X() + 0.5 * geom[1].X() + 0.25 * geom[2].X();
pos(1, 1) = 0.25 * geom[0].Y() + 0.5 * geom[1].Y() + 0.25 * geom[2].Y();
pos(1, 2) = 0.25 * geom[0].Z() + 0.5 * geom[1].Z() + 0.25 * geom[2].Z();

pos(2, 0) = 0.25 * geom[0].X() + 0.25 * geom[1].X() + 0.5 * geom[2].X();
pos(2, 1) = 0.25 * geom[0].Y() + 0.25 * geom[1].Y() + 0.5 * geom[2].Y();
pos(2, 2) = 0.25 * geom[0].Z() + 0.25 * geom[1].Z() + 0.5 * geom[2].Z();

}

void ComputeGaussPointPositionsForPreReseed(Geometry< Node >& geom, BoundedMatrix<double, 4, 3 > & pos,BoundedMatrix<double, 4, 4 > & N) 
{


N(0, 0) = 0.4;
N(0, 1) = 0.2;
N(0, 2) = 0.2;
N(0, 3) = 0.2;

N(1, 0) = 0.2;
N(1, 1) = 0.4;
N(1, 2) = 0.2;
N(1, 3) = 0.2;

N(2, 0) = 0.2;
N(2, 1) = 0.2;
N(2, 2) = 0.4;
N(2, 3) = 0.2;

N(3, 0) = 0.2;
N(3, 1) = 0.2;
N(3, 2) = 0.2;
N(3, 3) = 0.4;

pos=ZeroMatrix(4,3);
for (unsigned int i=0; i!=4; i++) 
{
array_1d<double, 3 > & coordinates = geom[i].Coordinates();
for (unsigned int j=0; j!=4; j++) 
{
for (unsigned int k=0; k!=3; k++) 
pos(j,k) += N(j,i) * coordinates[k];
}
}

}



void ComputeGaussPointPositions_45(Geometry< Node >& geom, BoundedMatrix<double, 45, 3 > & pos,BoundedMatrix<double, 45, 3 > & N)
{
unsigned int counter=0;
for (unsigned int i=0; i!=9;i++)
{
for (unsigned int j=0; j!=(9-i);j++)
{
N(counter,0)=0.05+double(i)*0.1;
N(counter,1)=0.05+double(j)*0.1;
N(counter,2)=1.0 - ( N(counter,1)+ N(counter,0) ) ;
pos(counter, 0) = N(counter,0) * geom[0].X() + N(counter,1) * geom[1].X() + N(counter,2) * geom[2].X();
pos(counter, 1) = N(counter,0) * geom[0].Y() + N(counter,1) * geom[1].Y() + N(counter,2) * geom[2].Y();
pos(counter, 2) = N(counter,0) * geom[0].Z() + N(counter,1) * geom[1].Z() + N(counter,2) * geom[2].Z();
counter++;

}
}

}

void ComputeGaussPointPositions_initial(Geometry< Node >& geom, BoundedMatrix<double, 15, 3 > & pos,BoundedMatrix<double, 15, 3 > & N) 
{
unsigned int counter=0;
for (unsigned int i=0; i!=5;i++)
{
for (unsigned int j=0; j!=(5-i);j++)
{
N(counter,0)=0.05+double(i)*0.2;
N(counter,1)=0.05+double(j)*0.2;
N(counter,2)=1.0 - ( N(counter,1)+ N(counter,0) ) ;
pos(counter, 0) = N(counter,0) * geom[0].X() + N(counter,1) * geom[1].X() + N(counter,2) * geom[2].X();
pos(counter, 1) = N(counter,0) * geom[0].Y() + N(counter,1) * geom[1].Y() + N(counter,2) * geom[2].Y();
pos(counter, 2) = N(counter,0) * geom[0].Z() + N(counter,1) * geom[1].Z() + N(counter,2) * geom[2].Z();
counter++;

}
}

}

void ComputeGaussPointPositions_initial(Geometry< Node >& geom, BoundedMatrix<double, 20, 3 > & pos,BoundedMatrix<double, 20, 4 > & N) 
{
double fraction_increment;
unsigned int counter=0;
for (unsigned int i=0; i!=4;i++) 
{
for (unsigned int j=0; j!=(4-i);j++)
{
for (unsigned int k=0; k!=(4-i-j);k++)
{
N(counter,0)= 0.27 * ( 0.175 + double(i) ) ; 

fraction_increment = 0.27; 

N(counter,1)=fraction_increment * (0.175 + double(j));
N(counter,2)=fraction_increment * (0.175 + double(k));
N(counter,3)=1.0 - ( N(counter,0)+ N(counter,1) + N(counter,2) ) ;
pos(counter, 0) = N(counter,0) * geom[0].X() + N(counter,1) * geom[1].X() + N(counter,2) * geom[2].X() + N(counter,3) * geom[3].X();
pos(counter, 1) = N(counter,0) * geom[0].Y() + N(counter,1) * geom[1].Y() + N(counter,2) * geom[2].Y() + N(counter,3) * geom[3].Y();
pos(counter, 2) = N(counter,0) * geom[0].Z() + N(counter,1) * geom[1].Z() + N(counter,2) * geom[2].Z() + N(counter,3) * geom[3].Z();
counter++;
}

}
}

}

template<class T>
bool InvertMatrix(const T& input, T& inverse)
{
typedef permutation_matrix<std::size_t> pmatrix;

T A(input);

pmatrix pm(A.size1());

int res = lu_factorize(A, pm);
if (res != 0)
return false;

inverse.assign(identity_matrix<double> (A.size1()));

lu_substitute(A, pm, inverse);

return true;
}

bool InvertMatrix3x3(const BoundedMatrix<double, TDim+1 , TDim+1  >& A, BoundedMatrix<double, TDim+1 , TDim+1  >& result)
{
double determinant =    +A(0,0)*(A(1,1)*A(2,2)-A(2,1)*A(1,2))
-A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
+A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
double invdet = 1/determinant;
result(0,0) =  (A(1,1)*A(2,2)-A(2,1)*A(1,2))*invdet;
result(1,0) = -(A(0,1)*A(2,2)-A(0,2)*A(2,1))*invdet;
result(2,0) =  (A(0,1)*A(1,2)-A(0,2)*A(1,1))*invdet;
result(0,1) = -(A(1,0)*A(2,2)-A(1,2)*A(2,0))*invdet;
result(1,1) =  (A(0,0)*A(2,2)-A(0,2)*A(2,0))*invdet;
result(2,1) = -(A(0,0)*A(1,2)-A(1,0)*A(0,2))*invdet;
result(0,2) =  (A(1,0)*A(2,1)-A(2,0)*A(1,1))*invdet;
result(1,2) = -(A(0,0)*A(2,1)-A(2,0)*A(0,1))*invdet;
result(2,2) =  (A(0,0)*A(1,1)-A(1,0)*A(0,1))*invdet;

return true;
}

virtual int Check()
{
KRATOS_TRY
ProcessInfo& rCurrentProcessInfo = mr_model_part.GetProcessInfo();
if (rCurrentProcessInfo.Has(CONVECTION_DIFFUSION_SETTINGS)==false)
KRATOS_THROW_ERROR(std::logic_error, "no CONVECTION_DIFFUSION_SETTINGS in model_part", "");

ConvectionDiffusionSettings::Pointer my_settings = rCurrentProcessInfo.GetValue(CONVECTION_DIFFUSION_SETTINGS);

if(my_settings->IsDefinedUnknownVariable()==true)
{
if (mr_model_part.NodesBegin()->SolutionStepsDataHas(my_settings->GetUnknownVariable()) == false)
KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Unknown Variable defined but not contained in the model part", "");
}
else
KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Unknown Variable not defined!", "");


if(my_settings->IsDefinedProjectionVariable()==true)
{
if (mr_model_part.NodesBegin()->SolutionStepsDataHas(my_settings->GetProjectionVariable()) == false)
KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Projection Variable defined but not contained in the model part", "");
}
else
KRATOS_THROW_ERROR(std::logic_error, "No Projection variable assigned for ConvDiff!", "");


if(my_settings->IsDefinedConvectionVariable()==true)
KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: ConvectionVariable not used. Use VelocityVariable instead", "");

if(my_settings->IsDefinedVelocityVariable()==true)
{
if (mr_model_part.NodesBegin()->SolutionStepsDataHas(my_settings->GetVelocityVariable()) == false)
KRATOS_THROW_ERROR(std::logic_error, "ConvDiffSettings: Velocity Variable defined but not contained in the model part", "");
}
else
KRATOS_THROW_ERROR(std::logic_error, "No Velocity variable assigned for ConvDiff!", "");

if (mr_model_part.NodesBegin()->SolutionStepsDataHas(MEAN_SIZE) == false)
KRATOS_THROW_ERROR(std::logic_error, "Add MEAN_SIZE variable to model part!", "");

if (mr_model_part.NodesBegin()->SolutionStepsDataHas(DELTA_SCALAR1) == false)
KRATOS_THROW_ERROR(std::logic_error, "Add DELTA_SCALAR1 variable to model part!", "");

return 0;

KRATOS_CATCH("")

}




ModelPart& mr_model_part;
int m_nparticles;
int mnelems;
int moffset;
int max_nsubsteps;
double max_substep_dt;
int mmaximum_number_of_particles;
std::vector< Convection_Particle > mparticles_vector; 
int mlast_elem_id;
bool modd_timestep;
bool mparticle_printing_tool_initialized;
unsigned int mfilter_factor;
unsigned int mlast_node_id;

vector<int> mnumber_of_particles_in_elems;
vector<int> mnumber_of_particles_in_elems_aux;
vector<ParticlePointerVector>  mvector_of_particle_pointers_vectors;

typename BinsObjectDynamic<Configure>::Pointer  mpBinsObjectDynamic;

const Variable<double>& mUnknownVar;
const Variable<double>& mProjectionVar;
const Variable<array_1d<double,3> >& mVelocityVar;
const Variable<array_1d<double,3> >& mMeshVelocityVar;

};

}  

#endif 
