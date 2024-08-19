


#if !defined(KRATOS_CALCULATE_DISTANCE_CONDITION_PROCESS_H_INCLUDED )
#define  KRATOS_CALCULATE_DISTANCE_CONDITION_PROCESS_H_INCLUDED



#include <string>
#include <iostream>




#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "spatial_containers/octree_binary.h"
#include "utilities/spatial_containers_configure.h"
#include "utilities/timer.h"
#include "utilities/math_utils.h"
#include "utilities/geometry_utilities.h"
#include "geometries/triangle_3d_3.h"
#include "utilities/body_normal_calculation_utils.h"
#include "utilities/parallel_utilities.h"


namespace Kratos
{

class DistanceSpatialContainersConditionConfigure
{
public:

class CellNodeData
{
double mDistance;
double mCoordinates[3];
std::size_t mId;
public:
double& Distance(){return mDistance;}
double& X() {return mCoordinates[0];}
double& Y() {return mCoordinates[1];}
double& Z() {return mCoordinates[2];}
double& operator[](int i) {return mCoordinates[i];}
std::size_t& Id(){return mId;}
};





enum { Dimension = 3,
DIMENSION = 3,
MAX_LEVEL = 12,
MIN_LEVEL = 2    
};

typedef Point                                               PointType;  
typedef std::vector<double>::iterator                       DistanceIteratorType;
typedef PointerVectorSet<
GeometricalObject::Pointer,
IndexedObject,
std::less<typename IndexedObject::result_type>,
std::equal_to<typename IndexedObject::result_type>,
Kratos::shared_ptr<typename GeometricalObject::Pointer>,
std::vector< Kratos::shared_ptr<typename GeometricalObject::Pointer> >
>  ContainerType;
typedef ContainerType::value_type                           PointerType;
typedef ContainerType::iterator                             IteratorType;
typedef PointerVectorSet<
GeometricalObject::Pointer,
IndexedObject,
std::less<typename IndexedObject::result_type>,
std::equal_to<typename IndexedObject::result_type>,
Kratos::shared_ptr<typename GeometricalObject::Pointer>,
std::vector< Kratos::shared_ptr<typename GeometricalObject::Pointer> >
>  ResultContainerType;
typedef ResultContainerType::value_type                     ResultPointerType;
typedef ResultContainerType::iterator                       ResultIteratorType;

typedef GeometricalObject::Pointer                          pointer_type;
typedef CellNodeData                                        cell_node_data_type;
typedef std::vector<CellNodeData*> data_type;

typedef std::vector<PointerType>::iterator                  PointerTypeIterator;




KRATOS_CLASS_POINTER_DEFINITION(DistanceSpatialContainersConditionConfigure);


DistanceSpatialContainersConditionConfigure() {}

virtual ~DistanceSpatialContainersConditionConfigure() {}





static data_type* AllocateData() {
return new data_type(27, (CellNodeData*)NULL);
}

static void CopyData(data_type* source, data_type* destination) {
*destination = *source;
}

static void DeleteData(data_type* data) {
delete data;
}
class CalculateSignedDistanceTo3DConditionSkinProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(CalculateSignedDistanceTo3DConditionSkinProcess);

typedef DistanceSpatialContainersConditionConfigure ConfigurationType;
typedef OctreeBinaryCell<ConfigurationType> CellType;
typedef OctreeBinary<CellType> OctreeType;
typedef ConfigurationType::cell_node_data_type CellNodeDataType;
typedef Point PointType;  
typedef OctreeType::cell_type::object_container_type object_container_type;
typedef struct{
array_1d<double,3>  Coordinates;
array_1d<double,3>  StructElemNormal;
}IntersectionNodeStruct;
typedef struct{
std::vector<IntersectionNodeStruct> IntNodes;
}TetEdgeStruct;



CalculateSignedDistanceTo3DConditionSkinProcess(ModelPart& rThisModelPartStruc, ModelPart& rThisModelPartFluid)
: mrSkinModelPart(rThisModelPartStruc), mrBodyModelPart(rThisModelPartStruc), mrFluidModelPart(rThisModelPartFluid)
{
}

~CalculateSignedDistanceTo3DConditionSkinProcess() override
{
}



void operator()()
{
Execute();
}





GenerateOctree();

DistanceFluidStructure();

CalculateDistance2(); 

















KRATOS_CATCH("");
}


if ( IsIntersectionOnCorner( NewIntersectionNode , EdgeNode1 , EdgeNode2) )
NumberIntersectionsOnTetCorner++;
else
{

array_1d<double,3> emb_vel=(*i_StructCondition)->GetGeometry()[0].FastGetSolutionStepValue(VELOCITY);
emb_vel+=(*i_StructCondition)->GetGeometry()[1].FastGetSolutionStepValue(VELOCITY);
emb_vel+=(*i_StructCondition)->GetGeometry()[2].FastGetSolutionStepValue(VELOCITY);


i_fluidElement->GetValue(EMBEDDED_VELOCITY)+=emb_vel;
intersection_counter++;

}
}

}


}

}
}

}

if( NewTetEdge.IntNodes.size() > 0 )
IntersectedTetEdges.push_back(NewTetEdge);
}


void GenerateOctree()
{
Timer::Start("Generating Octree");

double low[3];
double high[3];

for (int i = 0 ; i < 3; i++)
{
low[i] = high[i] = mrFluidModelPart.NodesBegin()->Coordinates()[i];
}

for(ModelPart::NodeIterator i_node = mrFluidModelPart.NodesBegin();
i_node != mrFluidModelPart.NodesEnd();
i_node++)
{
const array_1d<double,3>& r_coordinates = i_node->Coordinates();
for (int i = 0 ; i < 3; i++)
{
low[i]  = r_coordinates[i] < low[i]  ? r_coordinates[i] : low[i];
high[i] = r_coordinates[i] > high[i] ? r_coordinates[i] : high[i];
}
}
mOctree.SetBoundingBox(low,high);


for(ModelPart::NodeIterator i_node = mrSkinModelPart.NodesBegin();
i_node != mrSkinModelPart.NodesEnd();
i_node++)
{
double temp_point[3];
temp_point[0] = i_node->X();
temp_point[1] = i_node->Y();
temp_point[2] = i_node->Z();
mOctree.Insert(temp_point);
}


for(ModelPart::ConditionIterator i_cond = mrSkinModelPart.ConditionsBegin();
i_cond != mrSkinModelPart.ConditionsEnd();
i_cond++)
{
mOctree.Insert(*(i_cond).base());
}

Timer::Stop("Generating Octree");



}





void GenerateNodes()
{
Timer::Start("Generating Nodes");
std::vector<OctreeType::cell_type*> all_leaves;
mOctree.GetAllLeavesVector(all_leaves);

IndexPartition<std::size_t>(all_leaves.size()).for_each([&](std::size_t Index){
*(all_leaves[Index]->pGetDataPointer()) = ConfigurationType::AllocateData();
});

std::size_t last_id = mrBodyModelPart.NumberOfNodes() + 1;
for (std::size_t i = 0; i < all_leaves.size(); i++)
{
CellType* cell = all_leaves[i];
GenerateCellNode(cell, last_id);
}

Timer::Stop("Generating Nodes");

}

void GenerateCellNode(CellType* pCell, std::size_t& LastId)
{
for (int i_pos=0; i_pos < 8; i_pos++) 
{
ConfigurationType::cell_node_data_type* p_node = (*(pCell->pGetData()))[i_pos];
if(p_node == 0)
{
(*(pCell->pGetData()))[i_pos] = new ConfigurationType::cell_node_data_type;

(*(pCell->pGetData()))[i_pos]->Id() = LastId++;

mOctreeNodes.push_back((*(pCell->pGetData()))[i_pos]);

SetNodeInNeighbours(pCell,i_pos,(*(pCell->pGetData()))[i_pos]);
}

}
}

void SetNodeInNeighbours(CellType* pCell, int Position, CellNodeDataType* pNode)
{
CellType::key_type point_key[3];
pCell->GetKey(Position, point_key);

for (std::size_t i_direction = 0; i_direction < 8; i_direction++) {
CellType::key_type neighbour_key[3];
if (pCell->GetNeighbourKey(Position, i_direction, neighbour_key)) {
CellType* neighbour_cell = mOctree.pGetCell(neighbour_key);
if (!neighbour_cell || (neighbour_cell == pCell))
continue;

std::size_t position = neighbour_cell->GetLocalPosition(point_key);
if((*neighbour_cell->pGetData())[position])
{
std::cout << "ERROR!! Bad Position calculated!!!!!!!!!!! position :" << position << std::endl;
continue;
}

(*neighbour_cell->pGetData())[position] = pNode;
}
}
}


void CalculateDistance2()
{
Timer::Start("Calculate Distances2");
ModelPart::NodesContainerType::ContainerType& nodes = mrFluidModelPart.NodesArray();
int nodes_size = nodes.size();

std::vector<CellType*> leaves;

mOctree.GetAllLeavesVector(leaves);


IndexPartition<std::size_t>(nodes_size).for_each([&](std::size_t Index){
CalculateNodeDistance(*(nodes[Index]));
});

Timer::Stop("Calculate Distances2");

}










void CalculateDistance()
{
Timer::Start("Calculate Distances");
ConfigurationType::data_type& nodes = mOctreeNodes;
int nodes_size = nodes.size();

IndexPartition<std::size_t>(nodes_size).for_each([&](std::size_t Index){
nodes[Index]->Distance() = 1.00;
});

std::vector<CellType*> leaves;

mOctree.GetAllLeavesVector(leaves);
int leaves_size = leaves.size();

for(int i = 0 ; i < leaves_size ; i++)
CalculateNotEmptyLeavesDistance(leaves[i]);

for(int i_direction = 0 ; i_direction < 1 ; i_direction++)
{

for(int i = 0 ; i < nodes_size ; i++)
{
if(nodes[i]->X() < 1.00 && nodes[i]->Y() < 1.00 && nodes[i]->Z() < 1.00)
CalculateDistance(*(nodes[i]), i_direction);
}
}
Timer::Stop("Calculate Distances");

}

void CalculateDistance(CellNodeDataType& rNode, int i_direction)
{
double coords[3] = {rNode.X(), rNode.Y(), rNode.Z()};


typedef Element::GeometryType triangle_type;
typedef std::vector<std::pair<double, triangle_type*> > intersections_container_type;

intersections_container_type intersections;
ConfigurationType::data_type nodes_array;


const double epsilon = 1e-12;

double distance = 1.0;

double ray[3] = {coords[0], coords[1], coords[2]};
ray[i_direction] = 0; 

GetIntersectionsAndNodes(ray, i_direction, intersections, nodes_array);
for (std::size_t i_node = 0; i_node < nodes_array.size() ; i_node++)
{
double coord = (*nodes_array[i_node])[i_direction];

int ray_color= 1;
std::vector<std::pair<double, Element::GeometryType*> >::iterator i_intersection = intersections.begin();
while (i_intersection != intersections.end()) {
double d = coord - i_intersection->first;
if (d > epsilon) {

ray_color = -ray_color;
distance = d;
} else if (d > -epsilon) {
distance = 0.00;
break;
} else {
if(distance > -d)
distance = -d;
break;
}

i_intersection++;
}

distance *= ray_color;

double& node_distance = nodes_array[i_node]->Distance();
if(fabs(distance) < fabs(node_distance))
node_distance = distance;
else if (distance*node_distance < 0.00) 
node_distance = -node_distance;


}
}

void CalculateNotEmptyLeavesDistance(CellType* pCell)
{
typedef OctreeType::cell_type::object_container_type object_container_type;

object_container_type* objects = (pCell->pGetObjects());

if (objects->empty())
return;


for (int i_pos=0; i_pos < 8; i_pos++) 
{
double distance = 1.00; 

for(object_container_type::iterator i_object = objects->begin(); i_object != objects->end(); i_object++)
{
CellType::key_type keys[3];
pCell->GetKey(i_pos,keys);

double cell_point[3];
mOctree.CalculateCoordinates(keys,cell_point);

double d = GeometryUtils::PointDistanceToTriangle3D((*i_object)->GetGeometry()[0], (*i_object)->GetGeometry()[1], (*i_object)->GetGeometry()[2], Point(cell_point[0], cell_point[1], cell_point[2]));

if(d < distance)
distance = d;
}

double& node_distance = (*(pCell->pGetData()))[i_pos]->Distance();
if(distance < node_distance)
node_distance = distance;

}

}


void CalculateNodeDistance(Node& rNode)
{
double coord[3] = {rNode.X(), rNode.Y(), rNode.Z()};
double distance = DistancePositionInSpace(coord);
double& node_distance =  rNode.GetSolutionStepValue(DISTANCE);

if(fabs(node_distance) > fabs(distance))
node_distance = distance;
else if (distance*node_distance < 0.00) 
node_distance = -node_distance;
}







double DistancePositionInSpace(double* coords)
{

typedef Element::GeometryType triangle_type;
typedef std::vector<std::pair<double, triangle_type*> > intersections_container_type;

intersections_container_type intersections;

const int dimension = 3;
const double epsilon = 1e-12;

double distances[3] = {1.0, 1.0, 1.0};

for (int i_direction = 0; i_direction < dimension; i_direction++)
{
double ray[3] = {coords[0], coords[1], coords[2]};
mOctree.NormalizeCoordinates(ray);
ray[i_direction] = 0; 

GetIntersections(ray, i_direction, intersections);



int ray_color= 1;
std::vector<std::pair<double, Element::GeometryType*> >::iterator i_intersection = intersections.begin();
while (i_intersection != intersections.end()) {
double d = coords[i_direction] - i_intersection->first;
if (d > epsilon) {

ray_color = -ray_color;
distances[i_direction] = d;
} else if (d > -epsilon) {
distances[i_direction] = 0.00;
break;
} else {
if(distances[i_direction] > -d)
distances[i_direction] = -d;
break;
}

i_intersection++;
}

distances[i_direction] *= ray_color;
}


double distance = (fabs(distances[0]) > fabs(distances[1])) ? distances[1] : distances[0];
distance = (fabs(distance) > fabs(distances[2])) ? distances[2] : distance;

return distance;

}


void GetIntersectionsAndNodes(double* ray, int direction, std::vector<std::pair<double,Element::GeometryType*> >& intersections, ConfigurationType::data_type& rNodesArray)
{

const double epsilon = 1.00e-12;

intersections.clear();

OctreeType* octree = &mOctree;

OctreeType::key_type ray_key[3] = {octree->CalcKeyNormalized(ray[0]), octree->CalcKeyNormalized(ray[1]), octree->CalcKeyNormalized(ray[2])};
OctreeType::key_type cell_key[3];

ray_key[direction] = 0;
OctreeType::cell_type* cell = octree->pGetCell(ray_key);

while (cell) {
std::size_t position = cell->GetLocalPosition(ray_key); 
OctreeType::key_type node_key[3];
cell->GetKey(position, node_key);
if((node_key[0] == ray_key[0]) && (node_key[1] == ray_key[1]) && (node_key[2] == ray_key[2]))
{
if(cell->pGetData())
{
if(cell->pGetData()->size() > position)
{
CellNodeDataType* p_node = (*cell->pGetData())[position];
if(p_node)
{
rNodesArray.push_back(p_node);
}
}
else
KRATOS_WATCH(cell->pGetData()->size())
}
}


GetCellIntersections(cell, ray, ray_key, direction, intersections);




if (cell->GetNeighbourKey(1 + direction * 2, cell_key)) {
ray_key[direction] = cell_key[direction];
cell = octree->pGetCell(ray_key);
ray_key[direction] -= 1 ;
} else
cell = NULL;
}



if (!intersections.empty()) {
std::sort(intersections.begin(), intersections.end());
std::vector<std::pair<double, Element::GeometryType*> >::iterator i_begin = intersections.begin();
std::vector<std::pair<double, Element::GeometryType*> >::iterator i_intersection = intersections.begin();
while (++i_begin != intersections.end()) {
if (fabs(i_begin->first - i_intersection->first) > epsilon) 
*(++i_intersection) = *i_begin;
}
intersections.resize((++i_intersection) - intersections.begin());

}
}

void GetIntersections(double* ray, int direction, std::vector<std::pair<double,Element::GeometryType*> >& intersections)
{

const double epsilon = 1.00e-12;

intersections.clear();

OctreeType* octree = &mOctree;

OctreeType::key_type ray_key[3] = {octree->CalcKeyNormalized(ray[0]), octree->CalcKeyNormalized(ray[1]), octree->CalcKeyNormalized(ray[2])};
OctreeType::key_type cell_key[3];

OctreeType::cell_type* cell = octree->pGetCell(ray_key);

while (cell) {
GetCellIntersections(cell, ray, ray_key, direction, intersections);
if (cell->GetNeighbourKey(1 + direction * 2, cell_key)) {
ray_key[direction] = cell_key[direction];
cell = octree->pGetCell(ray_key);
ray_key[direction] -= 1 ;
} else
cell = NULL;
}


if (!intersections.empty()) {
std::sort(intersections.begin(), intersections.end());
std::vector<std::pair<double, Element::GeometryType*> >::iterator i_begin = intersections.begin();
std::vector<std::pair<double, Element::GeometryType*> >::iterator i_intersection = intersections.begin();
while (++i_begin != intersections.end()) {
if (fabs(i_begin->first - i_intersection->first) > epsilon) 
*(++i_intersection) = *i_begin;
}
intersections.resize((++i_intersection) - intersections.begin());

}
}

int GetCellIntersections(OctreeType::cell_type* cell, double* ray,
OctreeType::key_type* ray_key, int direction,
std::vector<std::pair<double, Element::GeometryType*> >& intersections)  {

typedef OctreeType::cell_type::object_container_type object_container_type;

object_container_type* objects = (cell->pGetObjects());

if (objects->empty())
return 0;

double ray_point1[3] = {ray[0], ray[1], ray[2]};
double ray_point2[3] = {ray[0], ray[1], ray[2]};
double normalized_coordinate;
mOctree.CalculateCoordinateNormalized(ray_key[direction], normalized_coordinate);
ray_point1[direction] = normalized_coordinate;
ray_point2[direction] = ray_point1[direction] + mOctree.CalcSizeNormalized(cell);

mOctree.ScaleBackToOriginalCoordinate(ray_point1);
mOctree.ScaleBackToOriginalCoordinate(ray_point2);

for (object_container_type::iterator i_object = objects->begin(); i_object != objects->end(); i_object++) {
double intersection[3]={0.00,0.00,0.00};

int is_intersected = IntersectionTriangleSegment((*i_object)->GetGeometry(), ray_point1, ray_point2, intersection); 

if (is_intersected == 1) 
intersections.push_back(std::pair<double, Element::GeometryType*>(intersection[direction], &((*i_object)->GetGeometry())));
}

return 0;
}

int IntersectionTriangleSegment(Element::GeometryType& rGeometry, double* RayPoint1, double* RayPoint2, double* IntersectionPoint)
{

const double epsilon = 1.00e-12;

array_1d<double,3>    u, v, n;             
array_1d<double,3>    dir, w0, w;          
double     r, a, b;             


u = rGeometry[1] - rGeometry[0];
v = rGeometry[2] - rGeometry[0];

MathUtils<double>::CrossProduct(n, u, v);             

if (norm_2(n) == 0)            
return -1;                 

for(int i = 0 ; i < 3 ; i++)
{
dir[i] = RayPoint2[i] - RayPoint1[i];             
w0[i] = RayPoint1[i] - rGeometry[0][i];
}

a = -inner_prod(n,w0);
b = inner_prod(n,dir);

if (fabs(b) < epsilon) {     
if (a == 0)                
return 2;
else return 0;             
}

r = a / b;
if (r < 0.0)                   
return 0;                  

for(int i = 0 ; i < 3 ; i++)
IntersectionPoint[i]  = RayPoint1[i] + r * dir[i];           

double    uu, uv, vv, wu, wv, D;
uu = inner_prod(u,u);
uv = inner_prod(u,v);
vv = inner_prod(v,v);


for(int i = 0 ; i < 3 ; i++)
w[i] = IntersectionPoint[i] - rGeometry[0][i];


wu = inner_prod(w,u);
wv = inner_prod(w,v);
D = uv * uv - uu * vv;

double s, t;
s = (uv * wv - vv * wu) / D;
if (s < 0.0 - epsilon || s > 1.0 + epsilon)        
return 0;
t = (uv * wu - uu * wv) / D;
if (t < 0.0 - epsilon || (s + t) > 1.0 + epsilon)  
return 0;

return 1;                      

}








std::string Info() const override
{
return "CalculateSignedDistanceTo3DConditionSkinProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "CalculateSignedDistanceTo3DConditionSkinProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}

void PrintGiDMesh(std::ostream & rOStream) const {
std::vector<CellType*> leaves;

mOctree.GetAllLeavesVector(leaves);

std::cout << "writing " << leaves.size() << " leaves" << std::endl;
rOStream << "MESH \"leaves\" dimension 3 ElemType Hexahedra Nnode 8" << std::endl;
rOStream << "# color 96 96 96" << std::endl;
rOStream << "Coordinates" << std::endl;
rOStream << "# node number coordinate_x coordinate_y coordinate_z  " << std::endl;

for(ConfigurationType::data_type::const_iterator i_node = mOctreeNodes.begin() ; i_node != mOctreeNodes.end() ; i_node++)
{
rOStream << (*i_node)->Id() << "  " << (*i_node)->X() << "  " << (*i_node)->Y() << "  " << (*i_node)->Z() << std::endl;
}
std::cout << "Nodes written..." << std::endl;
rOStream << "end coordinates" << std::endl;
rOStream << "Elements" << std::endl;
rOStream << "# element node_1 node_2 node_3 material_number" << std::endl;

for (std::size_t i = 0; i < leaves.size(); i++) {
if ((leaves[i]->pGetData()))
{
ConfigurationType::data_type& nodes = (*(leaves[i]->pGetData()));

rOStream << i + 1;
for(int j = 0 ; j < 8 ; j++)
rOStream << "  " << nodes[j]->Id();
rOStream << std::endl;
}
}
rOStream << "end elements" << std::endl;

}

void PrintGiDResults(std::ostream & rOStream) const {
std::vector<CellType*> leaves;

mOctree.GetAllLeavesVector(leaves);

rOStream << "GiD Post Results File 1.0" << std::endl << std::endl;

rOStream << "Result \"Distance\" \"Kratos\" 1 Scalar OnNodes" << std::endl;

rOStream << "Values" << std::endl;

for(ConfigurationType::data_type::const_iterator i_node = mOctreeNodes.begin() ; i_node != mOctreeNodes.end() ; i_node++)
{
rOStream << (*i_node)->Id() << "  " << (*i_node)->Distance() << std::endl;
}
rOStream << "End Values" << std::endl;

}




protected:















private:


ModelPart& mrSkinModelPart;
ModelPart& mrBodyModelPart;
ModelPart& mrFluidModelPart;

ConfigurationType::data_type mOctreeNodes;

OctreeType mOctree;

static const double epsilon;










CalculateSignedDistanceTo3DConditionSkinProcess& operator=(CalculateSignedDistanceTo3DConditionSkinProcess const& rOther);




}; 






inline std::istream& operator >> (std::istream& rIStream,
CalculateSignedDistanceTo3DConditionSkinProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const CalculateSignedDistanceTo3DConditionSkinProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

const double CalculateSignedDistanceTo3DConditionSkinProcess::epsilon = 1e-12;


}  

#endif 


