
#include "hole_cutting_utility.h"

namespace Kratos {

template <int TDim>
void ChimeraHoleCuttingUtility::CreateHoleAfterDistance(ModelPart& rModelPart,
ModelPart& rHoleModelPart,
ModelPart& rHoleBoundaryModelPart,
const double Distance)
{
KRATOS_TRY;
ChimeraHoleCuttingUtility::RemoveOutOfDomainElements<TDim>(
rModelPart, rHoleModelPart, ChimeraHoleCuttingUtility::Domain::MAIN_BACKGROUND,
Distance, ChimeraHoleCuttingUtility::SideToExtract::INSIDE);
ChimeraHoleCuttingUtility::ExtractBoundaryMesh<TDim>(rHoleModelPart, rHoleBoundaryModelPart);
KRATOS_CATCH("");
}

template <int TDim>
void ChimeraHoleCuttingUtility::RemoveOutOfDomainElements(
ModelPart& rModelPart,
ModelPart& rRemovedModelPart,
const ChimeraHoleCuttingUtility::Domain DomainType,
const double OverLapDistance,
const ChimeraHoleCuttingUtility::SideToExtract Side)
{
KRATOS_TRY;
std::vector<IndexType> vector_of_node_ids;
vector_of_node_ids.reserve(rModelPart.NumberOfNodes());
std::vector<IndexType> vector_of_elem_ids;
vector_of_elem_ids.reserve(rModelPart.NumberOfElements());
auto elem_begin = rModelPart.Elements().ptr_begin();
auto elem_end = rModelPart.Elements().ptr_end();

for (auto p_elem=elem_begin; p_elem != elem_end; ++p_elem){
bool is_elem_outside = true;
Geometry<Node>& geom = (*p_elem)->GetGeometry();
int num_nodes_outside = 0;

for (auto& node : geom) {
double nodal_distance = node.FastGetSolutionStepValue(CHIMERA_DISTANCE);

nodal_distance = nodal_distance * DomainType;
if (nodal_distance < -1 * OverLapDistance) {
++num_nodes_outside;
is_elem_outside = is_elem_outside && true;
vector_of_node_ids.push_back(node.Id()); 
}
else {
is_elem_outside = is_elem_outside && false;
}
}


if (num_nodes_outside > 0) {
(*p_elem)->Set(ACTIVE, false);
if (Side == ChimeraHoleCuttingUtility::SideToExtract::INSIDE)
rRemovedModelPart.AddElement(*p_elem);
for (auto& node : geom) {
node.FastGetSolutionStepValue(VELOCITY_X, 0) = 0.0;
node.FastGetSolutionStepValue(VELOCITY_Y, 0) = 0.0;
if constexpr (TDim > 2)
node.FastGetSolutionStepValue(VELOCITY_Z, 0) = 0.0;
node.FastGetSolutionStepValue(PRESSURE, 0) = 0.0;
node.FastGetSolutionStepValue(VELOCITY_X, 1) = 0.0;
node.FastGetSolutionStepValue(VELOCITY_Y, 1) = 0.0;
if constexpr (TDim > 2)
node.FastGetSolutionStepValue(VELOCITY_Z, 1) = 0.0;
node.FastGetSolutionStepValue(PRESSURE, 1) = 0.0;
if (Side == ChimeraHoleCuttingUtility::SideToExtract::INSIDE)
vector_of_node_ids.push_back(node.Id());
}
}
else {
if (Side == ChimeraHoleCuttingUtility::SideToExtract::OUTSIDE) {
rRemovedModelPart.AddElement(*p_elem);
for (auto& node : geom)
vector_of_node_ids.push_back(node.Id());
}
}
}

std::set<IndexType> s(vector_of_node_ids.begin(), vector_of_node_ids.end());
vector_of_node_ids.assign(s.begin(), s.end());

if (rRemovedModelPart.IsSubModelPart()) {
rRemovedModelPart.AddNodes(vector_of_node_ids);
}
else {
auto& r_nodes = rModelPart.Nodes();
for (const auto& i_node_id : vector_of_node_ids) {
auto& p_node = r_nodes(i_node_id);
rRemovedModelPart.AddNode(p_node);
}
}

KRATOS_CATCH("");
}
template <int TDim>
void ChimeraHoleCuttingUtility::ExtractBoundaryMesh(ModelPart& rVolumeModelPart,
ModelPart& rExtractedBoundaryModelPart,
const ChimeraHoleCuttingUtility::SideToExtract GetInternal)
{
KRATOS_TRY;

struct KeyComparator {
bool operator()(const vector<IndexType>& lhs, const vector<IndexType>& rhs) const
{
if (lhs.size() != rhs.size())
return false;
for (IndexType i = 0; i < lhs.size(); i++)
if (lhs[i] != rhs[i])
return false;
return true;
}
};

struct KeyHasher {
IndexType operator()(const vector<int>& k) const
{
IndexType seed = 0.0;
std::hash<int> hasher;
for (IndexType i = 0; i < k.size(); i++)
seed ^= hasher(k[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
return seed;
}
};

if (rVolumeModelPart.NumberOfElements() == 0)
return; 

IndexType n_nodes = rVolumeModelPart.ElementsBegin()->GetGeometry().size();
KRATOS_ERROR_IF(!(n_nodes != 3 || n_nodes != 4))
<< "Hole cutting process is only supported for tetrahedral and "
"triangular elements"
<< std::endl;

typedef std::unordered_map<vector<IndexType>, IndexType, KeyHasher, KeyComparator> hashmap;
typedef std::unordered_map<vector<IndexType>, vector<IndexType>, KeyHasher, KeyComparator> hashmap_vec;

hashmap n_faces_map;
const int num_elements = static_cast<int>(rVolumeModelPart.NumberOfElements());
const auto elements_begin = rVolumeModelPart.ElementsBegin();
#pragma omp parallel for
for (int i_e = 0; i_e < num_elements; ++i_e) {
auto i_element = elements_begin + i_e;
Element::GeometryType::GeometriesArrayType faces;
if constexpr (TDim == 2)
faces = i_element->GetGeometry().GenerateEdges();
else if constexpr (TDim == 3)
faces = i_element->GetGeometry().GenerateFaces();

for (IndexType i_face = 0; i_face < faces.size(); i_face++) {
vector<IndexType> ids(faces[i_face].size());

for (IndexType i = 0; i < faces[i_face].size(); i++)
ids[i] = faces[i_face][i].Id();

std::sort(ids.begin(), ids.end());

#pragma omp critical
n_faces_map[ids] += 1;
}
}
hashmap_vec ordered_skin_face_nodes_map;

#pragma omp parallel for
for (int i_e = 0; i_e < num_elements; ++i_e) {
auto i_element = elements_begin + i_e;
Element::GeometryType::GeometriesArrayType faces;
if constexpr (TDim == 2)
faces = i_element->GetGeometry().GenerateEdges();
else if constexpr (TDim == 3)
faces = i_element->GetGeometry().GenerateFaces();

for (IndexType i_face = 0; i_face < faces.size(); i_face++) {
vector<IndexType> ids(faces[i_face].size());
vector<IndexType> unsorted_ids(faces[i_face].size());

for (IndexType i = 0; i < faces[i_face].size(); i++) {
ids[i] = faces[i_face][i].Id();
unsorted_ids[i] = faces[i_face][i].Id();
}

std::sort(ids.begin(), ids.end());
#pragma omp critical
{
if (n_faces_map[ids] == 1)
ordered_skin_face_nodes_map[ids] = unsorted_ids;
}
}
}
IndexType id_condition = 1;
Condition const& r_ref_triangle_condition =
KratosComponents<Condition>::Get("SurfaceCondition3D3N"); 
Condition const& r_ref_line_condition =
KratosComponents<Condition>::Get("LineCondition2D2N"); 
Properties::Pointer properties =
rExtractedBoundaryModelPart.rProperties()(0);

std::vector<IndexType> vector_of_node_ids;
for (typename hashmap::const_iterator it = n_faces_map.begin();
it != n_faces_map.end(); ++it) {
if (it->second == 1) {
if (it->first.size() == 2) {
vector<IndexType> original_nodes_order =
ordered_skin_face_nodes_map[it->first];

Node::Pointer pnode1 =
rVolumeModelPart.Nodes()(original_nodes_order[0]);
Node::Pointer pnode2 =
rVolumeModelPart.Nodes()(original_nodes_order[1]);

vector_of_node_ids.push_back(original_nodes_order[0]);
vector_of_node_ids.push_back(original_nodes_order[1]);

Line2D2<Node> line1(pnode1, pnode2);
Condition::Pointer p_condition1 =
r_ref_line_condition.Create(id_condition++, line1, properties);
rExtractedBoundaryModelPart.Conditions().push_back(p_condition1);
}
if (it->first.size() == 3) {
vector<IndexType> original_nodes_order =
ordered_skin_face_nodes_map[it->first];
Node::Pointer pnode1 =
rVolumeModelPart.Nodes()(original_nodes_order[0]);
Node::Pointer pnode2 =
rVolumeModelPart.Nodes()(original_nodes_order[1]);
Node::Pointer pnode3 =
rVolumeModelPart.Nodes()(original_nodes_order[2]);

vector_of_node_ids.push_back(original_nodes_order[0]);
vector_of_node_ids.push_back(original_nodes_order[1]);
vector_of_node_ids.push_back(original_nodes_order[2]);

Triangle3D3<Node> triangle1(pnode1, pnode2, pnode3);
Condition::Pointer p_condition1 = r_ref_triangle_condition.Create(
id_condition++, triangle1, properties);
rExtractedBoundaryModelPart.Conditions().push_back(p_condition1);
}
if (it->first.size() == 4) {
vector<IndexType> original_nodes_order =
ordered_skin_face_nodes_map[it->first];

Node::Pointer pnode1 =
rVolumeModelPart.Nodes()(original_nodes_order[0]);
Node::Pointer pnode2 =
rVolumeModelPart.Nodes()(original_nodes_order[1]);
Node::Pointer pnode3 =
rVolumeModelPart.Nodes()(original_nodes_order[2]);
Node::Pointer pnode4 =
rVolumeModelPart.Nodes()(original_nodes_order[3]);
vector_of_node_ids.push_back(original_nodes_order[0]);
vector_of_node_ids.push_back(original_nodes_order[1]);
vector_of_node_ids.push_back(original_nodes_order[2]);
vector_of_node_ids.push_back(original_nodes_order[3]);

Triangle3D3<Node> triangle1(pnode1, pnode2, pnode3);
Condition::Pointer p_condition1 = r_ref_triangle_condition.Create(
id_condition++, triangle1, properties);
rExtractedBoundaryModelPart.Conditions().push_back(p_condition1);

Triangle3D3<Node> triangle2(pnode1, pnode3, pnode4);
Condition::Pointer p_condition2 = r_ref_triangle_condition.Create(
id_condition++, triangle2, properties);
rExtractedBoundaryModelPart.Conditions().push_back(p_condition2);
}
}
}

std::set<IndexType> sort_set(vector_of_node_ids.begin(), vector_of_node_ids.end());
vector_of_node_ids.assign(sort_set.begin(), sort_set.end());

for (const auto& i_node_id : vector_of_node_ids) {
Node::Pointer pnode = rVolumeModelPart.Nodes()(i_node_id);
rExtractedBoundaryModelPart.AddNode(pnode);
}

const int num_nodes = static_cast<int>(rExtractedBoundaryModelPart.NumberOfNodes());
const auto nodes_begin = rExtractedBoundaryModelPart.NodesBegin();

#pragma omp parallel for
for (int i_n = 0; i_n < num_nodes; ++i_n) {
auto i_node = nodes_begin + i_n;
i_node->Set(TO_ERASE, false);
}

const int num_conditions =
static_cast<int>(rExtractedBoundaryModelPart.NumberOfConditions());
const auto conditions_begin = rExtractedBoundaryModelPart.ConditionsBegin();

#pragma omp parallel for
for (int i_c = 0; i_c < num_conditions; ++i_c) {
auto i_condition = conditions_begin + i_c;
i_condition->Set(TO_ERASE, false);
}

for (auto& i_condition : rExtractedBoundaryModelPart.Conditions()) {
auto& geo = i_condition.GetGeometry();
bool is_internal = true;
for (const auto& node : geo)
is_internal = is_internal && node.GetValue(CHIMERA_INTERNAL_BOUNDARY);
if (is_internal) {
if (GetInternal == ChimeraHoleCuttingUtility::SideToExtract::OUTSIDE) {
i_condition.Set(TO_ERASE);
for (auto& node : geo)
node.Set(TO_ERASE);
}
}
else {
if (GetInternal == ChimeraHoleCuttingUtility::SideToExtract::INSIDE) {
i_condition.Set(TO_ERASE);
for (auto& node : geo)
node.Set(TO_ERASE);
}
}
}

rExtractedBoundaryModelPart.RemoveConditions(TO_ERASE);
rExtractedBoundaryModelPart.RemoveNodes(TO_ERASE);
KRATOS_CATCH("");
}

template void ChimeraHoleCuttingUtility::RemoveOutOfDomainElements<2>(
ModelPart& rModelPart,
ModelPart& rRemovedModelPart,
const ChimeraHoleCuttingUtility::Domain DomainType,
const double OverLapDistance,
const ChimeraHoleCuttingUtility::SideToExtract Side);

template void ChimeraHoleCuttingUtility::RemoveOutOfDomainElements<3>(
ModelPart& rModelPart,
ModelPart& rRemovedModelPart,
const ChimeraHoleCuttingUtility::Domain DomainType,
const double OverLapDistance,
const ChimeraHoleCuttingUtility::SideToExtract Side);

template void ChimeraHoleCuttingUtility::ExtractBoundaryMesh<2>(
ModelPart& rVolumeModelPart,
ModelPart& rExtractedBoundaryModelPart,
const ChimeraHoleCuttingUtility::SideToExtract GetInternal);
template void ChimeraHoleCuttingUtility::ExtractBoundaryMesh<3>(
ModelPart& rVolumeModelPart,
ModelPart& rExtractedBoundaryModelPart,
const ChimeraHoleCuttingUtility::SideToExtract GetInternal);

template void ChimeraHoleCuttingUtility::CreateHoleAfterDistance<2>(ModelPart& rModelPart,
ModelPart& rHoleModelPart,
ModelPart& rHoleBoundaryModelPart,
const double Distance);
template void ChimeraHoleCuttingUtility::CreateHoleAfterDistance<3>(ModelPart& rModelPart,
ModelPart& rHoleModelPart,
ModelPart& rHoleBoundaryModelPart,
const double Distance);

} 
