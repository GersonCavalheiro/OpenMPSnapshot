
#include "custom_processes/generate_dem_process.h"
#include "includes/define.h"
#include "includes/kratos_flags.h"

namespace Kratos {

GenerateDemProcess::GenerateDemProcess(
ModelPart& rModelPart,
ModelPart& rDemModelPart)
: mrModelPart(rModelPart),
mrDEMModelPart(rDemModelPart)
{
}




void GenerateDemProcess::Execute()
{
auto& r_comm = mrModelPart.GetCommunicator().GetDataCommunicator();
FindGlobalNodalNeighboursProcess nodal_neigh_process(r_comm, mrModelPart);
nodal_neigh_process.Execute();

const auto it_element_begin = mrModelPart.ElementsBegin();
const int max_id_FEM_nodes = this->GetMaximumFEMId();

mIsDynamic = it_element_begin->GetGeometry()[0].SolutionStepsDataHas(VELOCITY);

for (int i = 0; i < static_cast<int>(mrModelPart.Elements().size()); i++) {
auto it_elem = it_element_begin + i;
auto& r_geom = it_elem->GetGeometry();

bool is_active = true;
if (it_elem->IsDefined(ACTIVE))
is_active = it_elem->Is(ACTIVE);
const bool dem_generated = it_elem->GetValue(DEM_GENERATED);

if (!is_active && !dem_generated) {
auto p_DEM_properties = mrDEMModelPart.pGetProperties(1);
const auto& r_node0 = r_geom[0];
const auto& r_node1 = r_geom[1];
const auto& r_node2 = r_geom[2];
const double dist01 = this->CalculateDistanceBetweenNodes(r_node0, r_node1);
const double dist02 = this->CalculateDistanceBetweenNodes(r_node0, r_node2);
const double dist12 = this->CalculateDistanceBetweenNodes(r_node1, r_node2);

for (int i = 0; i < r_geom.size(); i++) {
auto& r_node = r_geom[i];
if (!r_node.GetValue(IS_DEM)) {
auto& r_neigh_nodes = r_node.GetValue(NEIGHBOUR_NODES);
Vector potential_radii(r_neigh_nodes.size());
Vector distances(r_neigh_nodes.size());

bool has_dem_neigh = false;
for (int neigh = 0; neigh < r_neigh_nodes.size(); neigh++) {
auto& r_neighbour = r_neigh_nodes[neigh];
const double dist_between_nodes = this->CalculateDistanceBetweenNodes(r_node, r_neighbour);
distances(neigh) = dist_between_nodes;
if (r_neighbour.GetValue(IS_DEM)) {
has_dem_neigh = true;
potential_radii(neigh) = dist_between_nodes - r_neighbour.GetValue(RADIUS);
if (potential_radii(neigh) < 0.0 || potential_radii(neigh) / dist_between_nodes < 0.2) { 
const double new_radius = dist_between_nodes*0.5;
auto& pDEM_particle = r_neighbour.GetValue(DEM_PARTICLE_POINTER);
auto& r_radius_neigh_old = pDEM_particle->GetGeometry()[0].GetSolutionStepValue(RADIUS);
pDEM_particle->SetRadius(new_radius);
r_radius_neigh_old = new_radius;
r_neighbour.SetValue(RADIUS, new_radius);
potential_radii(neigh) = new_radius;
}
} else {
potential_radii(neigh) = 0.0;
}
}
double radius;
if (has_dem_neigh) {
radius = this->GetMinimumValue(potential_radii);
} else {
radius = this->GetMinimumValue(distances)*0.5;
}
const array_1d<double,3>& r_coordinates = r_node.Coordinates();
const int id = this->GetMaximumDEMId() + 1;

if (mrDEMModelPart.Elements().size() == 0)
this->CreateDEMParticle(id + max_id_FEM_nodes, r_coordinates, p_DEM_properties, 0.8*radius, r_node);
else
this->CreateDEMParticle(id, r_coordinates, p_DEM_properties,0.8* radius, r_node);
}
}
it_elem->SetValue(DEM_GENERATED, true);
it_elem->Set(TO_ERASE, true);
} else if (!is_active && dem_generated)
it_elem->Set(TO_ERASE, true);
}
}




void GenerateDemProcess::CreateDEMParticle(
const int Id,
const array_1d<double, 3> Coordinates,
const Properties::Pointer pProperties,
const double Radius,
NodeType& rNode
)
{
auto &r_process_info = mrModelPart.GetProcessInfo();
std::string sphere_type;
if (r_process_info[DEMFEM_CONTACT])
sphere_type = "PolyhedronSkinSphericParticle3D";
else
sphere_type = "SphericParticle3D";

auto spheric_particle = mParticleCreator.CreateSphericParticleRaw(mrDEMModelPart, Id, Coordinates, pProperties, Radius, sphere_type);
rNode.SetValue(IS_DEM, true);
rNode.SetValue(RADIUS, Radius);
rNode.SetValue(DEM_PARTICLE_POINTER, spheric_particle);

if (mIsDynamic) {
const array_1d<double, 3>& r_vel = rNode.FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3>& r_particle_vel = spheric_particle->GetGeometry()[0].FastGetSolutionStepValue(VELOCITY);
noalias(r_particle_vel) = r_vel;
}
const array_1d<double, 3>& r_displ = rNode.FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3>& r_particle_displ = spheric_particle->GetGeometry()[0].FastGetSolutionStepValue(DISPLACEMENT);
noalias(r_particle_displ) = r_displ;
}




double GenerateDemProcess::CalculateDistanceBetweenNodes(
const NodeType& rNode1,
const NodeType& rNode2
)
{
const auto& node_1_coord = rNode1.Coordinates();
const auto& node_2_coord = rNode2.Coordinates();

const double X1 = node_1_coord[0];
const double X2 = node_2_coord[0];
const double Y1 = node_1_coord[1];
const double Y2 = node_2_coord[1];
const double Z1 = node_1_coord[2];
const double Z2 = node_2_coord[2];
return std::sqrt(std::pow(X1-X2, 2) + std::pow(Y1-Y2, 2) + std::pow(Z1-Z2, 2));
}




double GenerateDemProcess::GetMinimumValue(
const Vector& rValues
)
{ 
double aux = 1.0e10;
for (int i = 0; i < rValues.size(); i++)
if (aux > rValues[i] && rValues[i] != 0.0)
aux = rValues[i];
return aux;
}




int GenerateDemProcess::GetMaximumDEMId()
{
const auto it_DEM_begin = mrDEMModelPart.ElementsBegin();
const int num_threads = OpenMPUtils::GetNumThreads();
std::vector<int> max_vector(num_threads, 0.0);

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrDEMModelPart.Elements().size()); i++) {
auto it_DEM = it_DEM_begin + i;
auto& r_geometry = it_DEM->GetGeometry();
const int DEM_id = r_geometry[0].Id();

const int thread_id = OpenMPUtils::ThisThread();

if (DEM_id > max_vector[thread_id])
max_vector[thread_id] = DEM_id;
}
return *std::max_element(max_vector.begin(), max_vector.end());
}




int GenerateDemProcess::GetMaximumFEMId()
{
const auto it_FEM_node_begin = mrModelPart.NodesBegin();
const int num_threads = OpenMPUtils::GetNumThreads();
std::vector<int> max_vector(num_threads, 0);

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(mrModelPart.Nodes().size()); i++) {
auto it_FEM_node = it_FEM_node_begin + i;
const int FEM_node_id = it_FEM_node->Id();

const int thread_id = OpenMPUtils::ThisThread();
if (FEM_node_id > max_vector[thread_id])
max_vector[thread_id] = FEM_node_id;
}
return *std::max_element(max_vector.begin(), max_vector.end());
}

}  
