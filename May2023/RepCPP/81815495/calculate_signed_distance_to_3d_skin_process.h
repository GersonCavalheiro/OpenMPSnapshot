

#if !defined(KRATOS_CALCULATE_DISTANCE_PROCESS_H_INCLUDED )
#define  KRATOS_CALCULATE_DISTANCE_PROCESS_H_INCLUDED



#include <string>
#include <iostream> 
#include <ctime>



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
#include "geometries/quadrilateral_3d_4.h"
#include "utilities/body_normal_calculation_utils.h"
#include "includes/kratos_flags.h"
#include "utilities/binbased_fast_point_locator.h"
#include "utilities/binbased_nodes_in_element_locator.h"
#include "processes/calculate_distance_to_skin_process.h"

#ifdef _OPENMP
#include "omp.h"
#endif

using namespace boost::numeric::ublas;


namespace Kratos
{

class DistanceSpatialContainersConfigure
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

typedef Point                                           PointType;  
typedef std::vector<double>::iterator                   DistanceIteratorType;
typedef ModelPart::ElementsContainerType::ContainerType ContainerType;
typedef ContainerType::value_type                       PointerType;
typedef ContainerType::iterator                         IteratorType;
typedef ModelPart::ElementsContainerType::ContainerType ResultContainerType;
typedef ResultContainerType::value_type                 ResultPointerType;
typedef ResultContainerType::iterator                   ResultIteratorType;

typedef Element::Pointer                                        pointer_type;
typedef CellNodeData                cell_node_data_type;
typedef std::vector<CellNodeData*> data_type;

typedef std::vector<PointerType>::iterator             PointerTypeIterator;




KRATOS_CLASS_POINTER_DEFINITION(DistanceSpatialContainersConfigure);


DistanceSpatialContainersConfigure() {}

virtual ~DistanceSpatialContainersConfigure() {}





static data_type* AllocateData() {
return new data_type(27, (CellNodeData*)NULL);
}

static void CopyData(data_type* source, data_type* destination) {
*destination = *source;
}

static void DeleteData(data_type* data) {
delete data;
}

static inline void CalculateBoundingBox(const PointerType& rObject, PointType& rLowPoint, PointType& rHighPoint)
{
rHighPoint = rObject->GetGeometry().GetPoint(0);
rLowPoint  = rObject->GetGeometry().GetPoint(0);

for (unsigned int point = 0; point<rObject->GetGeometry().PointsNumber(); point++)
{
for(std::size_t i = 0; i<3; i++)
{
rLowPoint[i]  =  (rLowPoint[i]  >  rObject->GetGeometry().GetPoint(point)[i] ) ?  rObject->GetGeometry().GetPoint(point)[i] : rLowPoint[i];
rHighPoint[i] =  (rHighPoint[i] <  rObject->GetGeometry().GetPoint(point)[i] ) ?  rObject->GetGeometry().GetPoint(point)[i] : rHighPoint[i];
}
}
}

static inline void GetBoundingBox(const PointerType rObject, double* rLowPoint, double* rHighPoint)
{

for(std::size_t i = 0; i<3; i++)
{
rLowPoint[i]  =  rObject->GetGeometry().GetPoint(0)[i];
rHighPoint[i] =  rObject->GetGeometry().GetPoint(0)[i];
}

for (unsigned int point = 0; point<rObject->GetGeometry().PointsNumber(); point++)
{
for(std::size_t i = 0; i<3; i++)
{
rLowPoint[i]  =  (rLowPoint[i]  >  rObject->GetGeometry().GetPoint(point)[i] ) ?  rObject->GetGeometry().GetPoint(point)[i] : rLowPoint[i];
rHighPoint[i] =  (rHighPoint[i] <  rObject->GetGeometry().GetPoint(point)[i] ) ?  rObject->GetGeometry().GetPoint(point)[i] : rHighPoint[i];
}
}
}

static inline bool Intersection(const PointerType& rObj_1, const PointerType& rObj_2)
{
Element::GeometryType& geom_1 = rObj_1->GetGeometry();
Element::GeometryType& geom_2 = rObj_2->GetGeometry();
return  geom_1.HasIntersection(geom_2);

}


static inline bool  IntersectionBox(const PointerType& rObject,  const PointType& rLowPoint, const PointType& rHighPoint)
{
return rObject->GetGeometry().HasIntersection(rLowPoint, rHighPoint);
}


static  inline bool  IsIntersected(const Element::Pointer rObject, double Tolerance, const double* rLowPoint, const double* rHighPoint)
{
Point low_point(rLowPoint[0] - Tolerance, rLowPoint[1] - Tolerance, rLowPoint[2] - Tolerance);
Point high_point(rHighPoint[0] + Tolerance, rHighPoint[1] + Tolerance, rHighPoint[2] + Tolerance);

KRATOS_THROW_ERROR(std::logic_error, "Not Implemented method", "")
}







virtual std::string Info() const
{
return " Spatial Containers Configure";
}

virtual void PrintInfo(std::ostream& rOStream) const {}

virtual void PrintData(std::ostream& rOStream) const {}



protected:

private:

DistanceSpatialContainersConfigure& operator=(DistanceSpatialContainersConfigure const& rOther);

DistanceSpatialContainersConfigure(DistanceSpatialContainersConfigure const& rOther);


}; 








class CalculateSignedDistanceTo3DSkinProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(CalculateSignedDistanceTo3DSkinProcess);

typedef DistanceSpatialContainersConfigure ConfigurationType;
typedef OctreeBinaryCell<ConfigurationType> CellType;
typedef OctreeBinary<CellType> OctreeType;
typedef ConfigurationType::cell_node_data_type CellNodeDataType;
typedef Point PointType;  
typedef OctreeType::cell_type::object_container_type object_container_type;
typedef struct{
array_1d<double,3>  Coordinates;
array_1d<double,3>  StructElemNormal;
unsigned int EdgeNode1;
unsigned int EdgeNode2;
}IntersectionNodeStruct;
typedef struct{
std::vector<IntersectionNodeStruct> IntNodes;
}TetEdgeStruct;



CalculateSignedDistanceTo3DSkinProcess(ModelPart& rThisModelPartStruc, ModelPart& rThisModelPartFluid)
: mrSkinModelPart(rThisModelPartStruc), mrBodyModelPart(rThisModelPartStruc), mrFluidModelPart(rThisModelPartFluid)
{
}

~CalculateSignedDistanceTo3DSkinProcess() override
{
}



void operator()()
{
Execute();
}



void MappingPressureToStructure(BinBasedFastPointLocator<3>& node_locator)
{
Vector N;
const int max_results = 10000;
BinBasedFastPointLocator<3>::ResultContainerType results(max_results);
const int n_structure_nodes = mrSkinModelPart.Nodes().size();

#pragma omp parallel for firstprivate(results,N)
for (int i = 0; i < n_structure_nodes; i++)
{
ModelPart::NodesContainerType::iterator iparticle = mrSkinModelPart.NodesBegin() + i;
Node ::Pointer p_structure_node = *(iparticle.base());
p_structure_node->Set(VISITED, false);
}
for (int i = 0; i < n_structure_nodes; i++)
{
ModelPart::NodesContainerType::iterator iparticle = mrSkinModelPart.NodesBegin() + i;
Node ::Pointer p_structure_node = *(iparticle.base());
BinBasedFastPointLocator<3>::ResultIteratorType result_begin = results.begin();
Element::Pointer pElement;

bool is_found = node_locator.FindPointOnMesh(p_structure_node->Coordinates(), N, pElement, result_begin, max_results);

if (is_found == true)
{
array_1d<double,4> nodalPressures;
const Vector& ElementalDistances = pElement->GetValue(ELEMENTAL_DISTANCES);

Geometry<Node >& geom = pElement->GetGeometry();

for(unsigned int j=0; j<geom.size(); j++)
{
nodalPressures[j] = geom[j].FastGetSolutionStepValue(PRESSURE);
}

if(pElement->GetValue(SPLIT_ELEMENT)==true)
{
array_1d<double,4> Npos,Nneg;

ComputeDiscontinuousInterpolation((*p_structure_node),pElement->GetGeometry(),ElementalDistances,Npos,Nneg);

double p_positive_structure = inner_prod(nodalPressures,Npos);
double p_negative_structure = inner_prod(nodalPressures,Nneg);

p_structure_node->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE) = p_positive_structure;
p_structure_node->FastGetSolutionStepValue(NEGATIVE_FACE_PRESSURE) = p_negative_structure;
p_structure_node->Set(VISITED);
}
else
{
double p = inner_prod(nodalPressures,N);
p_structure_node->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE) = p;
p_structure_node->FastGetSolutionStepValue(NEGATIVE_FACE_PRESSURE) = p;
p_structure_node->Set(VISITED);
}
}
}
int n_bad_nodes=0;
for (int i = 0; i < n_structure_nodes; i++)
{
ModelPart::NodesContainerType::iterator iparticle = mrSkinModelPart.NodesBegin() + i;
Node ::Pointer p_structure_node = *(iparticle.base());
if (p_structure_node->IsNot(VISITED))
n_bad_nodes++;
}
while (n_bad_nodes >= 1.0) {
int n_bad_nodes_backup = n_bad_nodes;

for (int i = 0; i < n_structure_nodes; i++) {
ModelPart::NodesContainerType::iterator iparticle = mrSkinModelPart.NodesBegin() + i;
Node ::Pointer p_structure_node = *(iparticle.base());

if (p_structure_node->IsNot(VISITED)) {
int n_good_neighbors = 0;
double pos_pres = 0.0;
double neg_pres = 0.0;
GlobalPointersVector< Node >& neighours = p_structure_node->GetValue(NEIGHBOUR_NODES);

for (GlobalPointersVector< Node >::iterator j = neighours.begin(); j != neighours.end(); j++) {
if (j->Is(VISITED)) {
n_good_neighbors++;
pos_pres += j->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE);
neg_pres += j->FastGetSolutionStepValue(NEGATIVE_FACE_PRESSURE);
}
}
if (n_good_neighbors != 0) {
pos_pres /= n_good_neighbors;
neg_pres /= n_good_neighbors;
p_structure_node->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE) = pos_pres;
p_structure_node->FastGetSolutionStepValue(NEGATIVE_FACE_PRESSURE) = neg_pres;
p_structure_node->Set(VISITED);
n_bad_nodes--;
}
}
}

if(n_bad_nodes == n_bad_nodes_backup) break; 



}
for (int i = 0; i < n_structure_nodes; i++)
{
ModelPart::NodesContainerType::iterator iparticle = mrSkinModelPart.NodesBegin() + i;
Node ::Pointer p_structure_node = *(iparticle.base());

double pos_pressure=p_structure_node->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE);
double neg_pressure=p_structure_node->FastGetSolutionStepValue(NEGATIVE_FACE_PRESSURE);

GlobalPointersVector< Node >& neighours = p_structure_node->GetValue(NEIGHBOUR_NODES);

if (neighours.size()>=1.0)
{			    
double av_pos_pres=0.0;
double av_neg_pres=0.0;
for( GlobalPointersVector< Node >::iterator j = neighours.begin();
j != neighours.end(); j++)
{

av_pos_pres+=j->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE);
av_neg_pres+=j->FastGetSolutionStepValue(NEGATIVE_FACE_PRESSURE);

}
av_pos_pres/=neighours.size();
av_neg_pres/=neighours.size();

if (fabs(pos_pressure)>3.0*fabs(av_pos_pres))
{
p_structure_node->FastGetSolutionStepValue(POSITIVE_FACE_PRESSURE) = av_pos_pres;
}
if (fabs(neg_pressure)>3.0*fabs(av_neg_pres))
{
p_structure_node->FastGetSolutionStepValue(NEGATIVE_FACE_PRESSURE) = av_neg_pres;
}				    	

}
}



}

double PointDistanceToPlane( Point&            planeBasePoint,
array_1d<double, 3>& planeNormal,
Point&            ToPoint)
{
array_1d<double,3> planeToPointVec = ToPoint - planeBasePoint;

const double sn = inner_prod(planeToPointVec,planeNormal);
const double sd = inner_prod(planeNormal,planeNormal);
double DistanceToPlane = sn / sqrt(sd);

if( fabs(DistanceToPlane) < epsilon )
DistanceToPlane = 0;

return DistanceToPlane;
}

void AvoidZeroDistances( ModelPart::ElementsContainerType::iterator& Element,
array_1d<double,4>&                         ElementalDistances)
{
double dist_limit = 1e-5;

for(unsigned int i_node = 0; i_node < 4; i_node++)
{
double & di = ElementalDistances[i_node];
if(fabs(di) < dist_limit)
{
if(di >= 0) di = dist_limit;
else di = -dist_limit;
}
}

}

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

mpOctree->NormalizeCoordinates(ray);
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

void GetIntersectionsAndNodes(double* ray, int direction, std::vector<std::pair<double,Element::GeometryType*> >& intersections, DistanceSpatialContainersConfigure::data_type& rNodesArray)
{

const double epsilon = 1.00e-12;

intersections.clear();

OctreeType* octree = mpOctree.get();

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

OctreeType* octree = mpOctree.get();

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
mpOctree->CalculateCoordinateNormalized(ray_key[direction], normalized_coordinate);
ray_point1[direction] = normalized_coordinate;
ray_point2[direction] = ray_point1[direction] + mpOctree->CalcSizeNormalized(cell);

mpOctree->ScaleBackToOriginalCoordinate(ray_point1);
mpOctree->ScaleBackToOriginalCoordinate(ray_point2);

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

double triangle_origin_distance = -inner_prod(n, rGeometry[0]);
Point ray_point_1, ray_point_2;

for(int i = 0 ; i < 3 ; i++)
{
dir[i] = RayPoint2[i] - RayPoint1[i];             
w0[i] = RayPoint1[i] - rGeometry[0][i];
ray_point_1[i] = RayPoint1[i];
ray_point_2[i] = RayPoint2[i];
}

double sign_distance_1 = inner_prod(n, ray_point_1) + triangle_origin_distance;
double sign_distance_2 = inner_prod(n, ray_point_2) + triangle_origin_distance;

if (sign_distance_1*sign_distance_2 > epsilon) 
return 0;
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
return "CalculateSignedDistanceTo3DSkinProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "CalculateSignedDistanceTo3DSkinProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}

void PrintGiDMesh(std::ostream & rOStream) const {
std::vector<CellType*> leaves;

mpOctree->GetAllLeavesVector(leaves);

std::cout << "writing " << leaves.size() << " leaves" << std::endl;
rOStream << "MESH \"leaves\" dimension 3 ElemType Hexahedra Nnode 8" << std::endl;
rOStream << "# color 96 96 96" << std::endl;
rOStream << "Coordinates" << std::endl;
rOStream << "# node number coordinate_x coordinate_y coordinate_z  " << std::endl;

for(DistanceSpatialContainersConfigure::data_type::const_iterator i_node = mOctreeNodes.begin() ; i_node != mOctreeNodes.end() ; i_node++)
{
rOStream << (*i_node)->Id() << "  " << (*i_node)->X() << "  " << (*i_node)->Y() << "  " << (*i_node)->Z() << std::endl;
}
std::cout << "Nodes written..." << std::endl;
rOStream << "end coordinates" << std::endl;
rOStream << "Elements" << std::endl;
rOStream << "# Element node_1 node_2 node_3 material_number" << std::endl;

for (std::size_t i = 0; i < leaves.size(); i++) {
if ((leaves[i]->pGetData()))
{
DistanceSpatialContainersConfigure::data_type& nodes = (*(leaves[i]->pGetData()));

rOStream << i + 1;
for(int j = 0 ; j < 8 ; j++)
rOStream << "  " << nodes[j]->Id();
rOStream << std::endl;
}
}
rOStream << "end Elements" << std::endl;

}

void PrintGiDResults(std::ostream & rOStream) const {
std::vector<CellType*> leaves;

mpOctree->GetAllLeavesVector(leaves);

rOStream << "GiD Post Results File 1.0" << std::endl << std::endl;

rOStream << "Result \"Distance\" \"Kratos\" 1 Scalar OnNodes" << std::endl;

rOStream << "Values" << std::endl;

for(DistanceSpatialContainersConfigure::data_type::const_iterator i_node = mOctreeNodes.begin() ; i_node != mOctreeNodes.end() ; i_node++)
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

DistanceSpatialContainersConfigure::data_type mOctreeNodes;

Kratos::shared_ptr<OctreeType> mpOctree;

static const double epsilon;




static inline void EigenVectors(const Matrix& A, Matrix& vectors, Vector& lambda, double zero_tolerance =1e-9, int max_iterations = 10)
{
Matrix Help= A;

for(int i=0; i<3; i++)
for(int j=0; j<3; j++)
Help(i,j)= Help(i,j);


vectors.resize(Help.size1(),Help.size2(),false);

lambda.resize(Help.size1(),false);

Matrix HelpDummy(Help.size1(),Help.size2());

bool is_converged = false;

Matrix unity=ZeroMatrix(Help.size1(),Help.size2());

for(unsigned int i=0; i< Help.size1(); i++)
unity(i,i)= 1.0;

Matrix V= unity;

Matrix VDummy(Help.size1(),Help.size2());

Matrix Rotation(Help.size1(),Help.size2());


for(int iterations=0; iterations<max_iterations; iterations++)
{

is_converged= true;

double a= 0.0;

unsigned int index1= 0;

unsigned int index2= 1;

for(unsigned int i=0; i< Help.size1(); i++)
{
for(unsigned int j=(i+1); j< Help.size2(); j++)
{
if((fabs(Help(i,j)) > a ) && (fabs(Help(i,j)) > zero_tolerance))
{
a= fabs(Help(i,j));

index1= i;
index2= j;

is_converged= false;
}
}
}


if(is_converged)
break;


double gamma= (Help(index2,index2)-Help(index1,index1))/(2*Help(index1,index2));

double u=1.0;

if(fabs(gamma) > zero_tolerance && fabs(gamma)< (1/zero_tolerance))
{
u= gamma/fabs(gamma)*1.0/(fabs(gamma)+sqrt(1.0+gamma*gamma));
}
else
{
if  (fabs(gamma)>= (1.0/zero_tolerance))
u= 0.5/gamma;
}

double c= 1.0/(sqrt(1.0+u*u));

double s= c*u;

double teta= s/(1.0+c);

HelpDummy= Help;

HelpDummy(index2,index2)= Help(index2,index2)+u*Help(index1,index2);
HelpDummy(index1,index1)= Help(index1,index1)-u*Help(index1,index2);
HelpDummy(index1,index2)= 0.0;
HelpDummy(index2,index1)= 0.0;

for(unsigned int i=0; i<Help.size1(); i++)
{
if((i!= index1) && (i!= index2))
{
HelpDummy(index2,i)=Help(index2,i)+s*(Help(index1,i)- teta*Help(index2,i));
HelpDummy(i,index2)=Help(index2,i)+s*(Help(index1,i)- teta*Help(index2,i));

HelpDummy(index1,i)=Help(index1,i)-s*(Help(index2,i)+ teta*Help(index1,i));
HelpDummy(i,index1)=Help(index1,i)-s*(Help(index2,i)+ teta*Help(index1,i));
}
}


Help= HelpDummy;

Rotation =unity;
Rotation(index2,index1)=-s;
Rotation(index1,index2)=s;
Rotation(index1,index1)=c;
Rotation(index2,index2)=c;


VDummy = ZeroMatrix(Help.size1(), Help.size2());

for(unsigned int i=0; i< Help.size1(); i++)
{
for(unsigned int j=0; j< Help.size1(); j++)
{
for(unsigned int k=0; k< Help.size1(); k++)
{
VDummy(i,j) += V(i,k)*Rotation(k,j);
}
}
}
V= VDummy;
}

if(!(is_converged))
{
std::cout<<"########################################################"<<std::endl;
std::cout<<"Max_Iterations exceed in Jacobi-Seidel-Iteration (eigenvectors)"<<std::endl;
std::cout<<"########################################################"<<std::endl;
}

for(unsigned int i=0; i< Help.size1(); i++)
{
for(unsigned int j=0; j< Help.size1(); j++)
{
vectors(i,j)= V(j,i);
}
}

for(unsigned int i=0; i<Help.size1(); i++)
lambda(i)= Help(i,i);

return;
}


inline void CreatePartition(unsigned int number_of_threads, const int number_of_rows, DenseVector<unsigned int>& partitions)
{
partitions.resize(number_of_threads + 1);
int partition_size = number_of_rows / number_of_threads;
partitions[0] = 0;
partitions[number_of_threads] = number_of_rows;
for (unsigned int i = 1; i < number_of_threads; i++)
partitions[i] = partitions[i - 1] + partition_size;
}










CalculateSignedDistanceTo3DSkinProcess& operator=(CalculateSignedDistanceTo3DSkinProcess const& rOther);




}; 






inline std::istream& operator >> (std::istream& rIStream,
CalculateSignedDistanceTo3DSkinProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const CalculateSignedDistanceTo3DSkinProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

const double CalculateSignedDistanceTo3DSkinProcess::epsilon = 1e-18;


}  

#endif 


