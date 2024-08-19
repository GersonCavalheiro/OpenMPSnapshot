
#pragma once



#include "includes/define.h"
#include "geometries/point.h"
#include "containers/pointer_vector.h"
#include "utilities/math_utils.h"
#include "utilities/geometrical_projection_utilities.h"

namespace Kratos
{







class KRATOS_API(KRATOS_CORE) IntersectionUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION( IntersectionUtilities );




IntersectionUtilities(){}

virtual ~IntersectionUtilities(){}



template <class TGeometryType>
static int ComputeTriangleLineIntersection(
const TGeometryType& rTriangleGeometry,
const array_1d<double,3>& rLinePoint1,
const array_1d<double,3>& rLinePoint2,
array_1d<double,3>& rIntersectionPoint,
const double epsilon = 1e-12) {


const array_1d<double,3> u = rTriangleGeometry[1] - rTriangleGeometry[0];
const array_1d<double,3> v = rTriangleGeometry[2] - rTriangleGeometry[0];
array_1d<double,3> n;
MathUtils<double>::CrossProduct<array_1d<double,3>,array_1d<double,3>,array_1d<double,3>>(n,u,v);

if (MathUtils<double>::Norm3(n) < epsilon){
return -1;
}

const array_1d<double,3> dir = rLinePoint2 - rLinePoint1; 
const array_1d<double,3> w_0 = rLinePoint1 - rTriangleGeometry[0];
const double a = -inner_prod(n,w_0);
const double b = inner_prod(n,dir);

if (std::abs(b) < epsilon){
if (a == 0.0){
return 2;    
} else {
return 0;    
}
}

const double r = a / b;
if (r < 0.0){
return 0;    
} else if (r > 1.0) {
return 0;    
}

rIntersectionPoint = rLinePoint1 + r*dir;

if (PointInTriangle(rTriangleGeometry[0], rTriangleGeometry[1], rTriangleGeometry[2], rIntersectionPoint)) {
return 1;
}
return 0;
}


template <class TGeometryType>
static bool TriangleLineIntersection2D(
const TGeometryType& rTriangle,
const array_1d<double,3>& rPoint0,
const array_1d<double,3>& rPoint1)
{
return TriangleLineIntersection2D(rTriangle[0], rTriangle[1], rTriangle[2], rPoint0, rPoint1);
}


static bool TriangleLineIntersection2D(
const array_1d<double,3>& rVert0,
const array_1d<double,3>& rVert1,
const array_1d<double,3>& rVert2,
const array_1d<double,3>& rPoint0,
const array_1d<double,3>& rPoint1)
{
array_1d<double,3> int_point;
if (ComputeLineLineIntersection(rVert0, rVert1, rPoint0, rPoint1, int_point)) return true;
if (ComputeLineLineIntersection(rVert1, rVert2, rPoint0, rPoint1, int_point)) return true;
if (ComputeLineLineIntersection(rVert2, rVert0, rPoint0, rPoint1, int_point)) return true;

if (PointInTriangle(rVert0, rVert1, rVert2, rPoint0)) return true;

return false;
}


static bool PointInTriangle(
const array_1d<double,3>& rVert0,
const array_1d<double,3>& rVert1,
const array_1d<double,3>& rVert2,
const array_1d<double,3>& rPoint,
const double Tolerance = std::numeric_limits<double>::epsilon())
{
const array_1d<double,3> u = rVert1 - rVert0;
const array_1d<double,3> v = rVert2 - rVert0;
const array_1d<double,3> w = rPoint - rVert0;

const double uu = inner_prod(u, u);
const double uv = inner_prod(u, v);
const double vv = inner_prod(v, v);
const double wu = inner_prod(w, u);
const double wv = inner_prod(w, v);
const double denom = uv * uv - uu * vv;

const double xi  = (uv * wv - vv * wu) / denom;
const double eta = (uv * wu - uu * wv) / denom;

if (xi < -Tolerance) return false;
if (eta < -Tolerance) return false;
if (xi + eta > 1.0 + Tolerance) return false;
return true;
}


template <class TGeometryType, class TCoordinatesType>
static int ComputeTriangleLineIntersectionInTheSamePlane( 
const TGeometryType& rTriangleGeometry,
const TCoordinatesType& rLinePoint1,
const TCoordinatesType& rLinePoint2,
TCoordinatesType& rIntersectionPoint1,
TCoordinatesType& rIntersectionPoint2,
int& rSolution,
const double Epsilon = 1e-12
) 
{
auto is_inside_projected = [&Epsilon] (auto& rGeometry, const TCoordinatesType& rPoint) -> bool {
const Point point_to_project(rPoint);
Point point_projected;
const double distance = GeometricalProjectionUtilities::FastProjectOnLine(rGeometry, point_to_project, point_projected);

if (std::abs(distance) > Epsilon * rGeometry.Length()) {
return false;
}
array_1d<double, 3> local_coordinates;
return rGeometry.IsInside(point_projected, local_coordinates);
};

for (auto& r_edge : rTriangleGeometry.GenerateEdges()) {
const auto& r_edge_point_1 = r_edge[0].Coordinates();
const auto& r_edge_point_2 = r_edge[1].Coordinates();
array_1d<double, 3> intersection_point_1, intersection_point_2;
const auto check_1 = ComputeLineLineIntersection(rLinePoint1, rLinePoint2, r_edge_point_1, r_edge_point_2, intersection_point_1, Epsilon);
const auto check_2 = ComputeLineLineIntersection(r_edge_point_1, r_edge_point_2, rLinePoint1, rLinePoint2, intersection_point_2, Epsilon);
if (check_1 == 0 && check_2 == 0) continue; 
array_1d<double, 3> intersection_point = check_1 != 0 ? intersection_point_1 : intersection_point_2;
if (check_1 == 2 || check_2 == 2) { 
array_1d<double, 3> vector_line = r_edge_point_2 - r_edge_point_1;
vector_line /= norm_2(vector_line);
array_1d<double, 3> diff_coor_1 = rLinePoint1 - r_edge_point_1;
const double diff_coor_1_norm = norm_2(diff_coor_1);
if (diff_coor_1_norm > std::numeric_limits<double>::epsilon()) {
diff_coor_1 /= diff_coor_1_norm;
} else {
diff_coor_1 = rLinePoint1 - r_edge_point_2;
diff_coor_1 /= norm_2(diff_coor_1);
}
array_1d<double, 3> diff_coor_2 = rLinePoint2 - r_edge_point_1;
const double diff_coor_2_norm = norm_2(diff_coor_2);
if (diff_coor_2_norm > std::numeric_limits<double>::epsilon()) {
diff_coor_2 /= diff_coor_2_norm;
} else {
diff_coor_2 = rLinePoint2 - r_edge_point_2;
diff_coor_2 /= norm_2(diff_coor_2);
}
const double diff1m = norm_2(diff_coor_1 - vector_line);
const double diff1p = norm_2(diff_coor_1 + vector_line);
const double diff2m = norm_2(diff_coor_2 - vector_line);
const double diff2p = norm_2(diff_coor_2 + vector_line);

if ((diff1m < Epsilon || diff1p < Epsilon) && (diff2m < Epsilon || diff2p < Epsilon)) {
if (is_inside_projected(r_edge, rLinePoint1)) { 
if (rSolution == 0) {
noalias(rIntersectionPoint1) = rLinePoint1;
rSolution = 2;
} else {
if (norm_2(rIntersectionPoint1 - rLinePoint1) > Epsilon) { 
noalias(rIntersectionPoint2) = rLinePoint1;
rSolution = 1;
break;
}
}
} else { 
if (rSolution == 0) {
noalias(rIntersectionPoint1) = norm_2(r_edge_point_1 - rLinePoint1) <  norm_2(r_edge_point_2 - rLinePoint1) ? r_edge_point_1 : r_edge_point_2;
rSolution = 2;
} else {
noalias(intersection_point) = norm_2(r_edge_point_1 - rLinePoint1) <  norm_2(r_edge_point_2 - rLinePoint1) ? r_edge_point_1 : r_edge_point_2;
if (norm_2(rIntersectionPoint1 - intersection_point) > Epsilon) { 
noalias(rIntersectionPoint2) = intersection_point;
rSolution = 1;
break;
}
}
}
if (rSolution == 2) {
if (is_inside_projected(r_edge, rLinePoint2)) { 
if (norm_2(rIntersectionPoint1 - rLinePoint2) > Epsilon) { 
noalias(rIntersectionPoint2) = rLinePoint2;
rSolution = 1;
break;
}
} else { 
noalias(intersection_point) = norm_2(r_edge_point_1 - rLinePoint2) <  norm_2(r_edge_point_2 - rLinePoint2) ? r_edge_point_1 : r_edge_point_2;
if (norm_2(rIntersectionPoint1 - intersection_point) > Epsilon) { 
noalias(rIntersectionPoint2) = intersection_point;
rSolution = 1;
break;
}
}
} else { 
break;
}
}
} else { 
if (rSolution == 0) {
noalias(rIntersectionPoint1) = intersection_point;
rSolution = 2;
} else {
if (norm_2(rIntersectionPoint1 - intersection_point) > Epsilon) { 
noalias(rIntersectionPoint2) = intersection_point;
rSolution = 1;
break;
}
}
}
}

return rSolution;
}


template <class TGeometryType, class TCoordinatesType, bool TConsiderInsidePoints = true>
static int ComputeTetrahedraLineIntersection(
const TGeometryType& rTetrahedraGeometry,
const TCoordinatesType& rLinePoint1,
const TCoordinatesType& rLinePoint2,
TCoordinatesType& rIntersectionPoint1,
TCoordinatesType& rIntersectionPoint2,
const double Epsilon = 1e-12
) 
{
int solution = 0;
for (auto& r_face : rTetrahedraGeometry.GenerateFaces()) {
array_1d<double,3> intersection_point;
const int face_solution = ComputeTriangleLineIntersection(r_face, rLinePoint1, rLinePoint2, intersection_point, Epsilon);
if (face_solution == 1) { 
if (solution == 0) {
noalias(rIntersectionPoint1) = intersection_point;
solution = 2;
} else {
if (norm_2(rIntersectionPoint1 - intersection_point) > Epsilon) { 
noalias(rIntersectionPoint2) = intersection_point;
solution = 1;
break;
}
}
} else if (face_solution == 2) { 
ComputeTriangleLineIntersectionInTheSamePlane(r_face, rLinePoint1, rLinePoint2, rIntersectionPoint1, rIntersectionPoint2, solution, Epsilon);
if (solution == 1) break;
}
}

if constexpr (TConsiderInsidePoints) {
if (solution == 0) {
array_1d<double,3> local_coordinates;
if (rTetrahedraGeometry.IsInside(rLinePoint1, local_coordinates)) {
noalias(rIntersectionPoint1) = rLinePoint1;
solution = 4;
}
if (rTetrahedraGeometry.IsInside(rLinePoint2, local_coordinates)) {
if (solution == 0) {
noalias(rIntersectionPoint1) = rLinePoint2;
solution = 4;
} else {
noalias(rIntersectionPoint2) = rLinePoint2;
solution = 3;
}
}
} else if (solution == 2) {
array_1d<double,3> local_coordinates;
if (rTetrahedraGeometry.IsInside(rLinePoint1, local_coordinates)) {
if (norm_2(rIntersectionPoint1 - rLinePoint1) > Epsilon) { 
noalias(rIntersectionPoint2) = rLinePoint1;
solution = 4;
}
} 
if (solution == 2) {
if (rTetrahedraGeometry.IsInside(rLinePoint2, local_coordinates)) {
if (norm_2(rIntersectionPoint1 - rLinePoint2) > Epsilon) {  
noalias(rIntersectionPoint2) = rLinePoint2;
solution = 4;
}
}
}
}
}

if (solution == 2) {
int index_node = -1;
for (int i_node = 0; i_node < 4; ++i_node) {
if (norm_2(rTetrahedraGeometry[i_node].Coordinates() - rIntersectionPoint1) < Epsilon) {
index_node = i_node;
break;
}
}
if (index_node > -1) {
return index_node + 5;
}
}

return solution;
}


template<class TGeometryType>
static PointerVector<Point> ComputeShortestLineBetweenTwoLines(
const TGeometryType& rSegment1,
const TGeometryType& rSegment2
)  
{
const double zero_tolerance = std::numeric_limits<double>::epsilon();

KRATOS_ERROR_IF_NOT((rSegment1.GetGeometryFamily() == GeometryData::KratosGeometryFamily::Kratos_Linear && rSegment1.PointsNumber() == 2)) << "The first geometry type is not correct, it is suppossed to be a linear line" << std::endl;
KRATOS_ERROR_IF_NOT((rSegment2.GetGeometryFamily() == GeometryData::KratosGeometryFamily::Kratos_Linear && rSegment2.PointsNumber() == 2)) << "The second geometry type is not correct, it is suppossed to be a linear line" << std::endl;

auto resulting_line = PointerVector<Point>();

array_1d<double, 3> p13,p43,p21;
double d1343,d4321,d1321,d4343,d2121;
double mua, mub;
double numer,denom;

const Point& p1 = rSegment1[0];
const Point& p2 = rSegment1[1];
const Point& p3 = rSegment2[0];
const Point& p4 = rSegment2[1];

p13[0] = p1.X() - p3.X();
p13[1] = p1.Y() - p3.Y();
p13[2] = p1.Z() - p3.Z();

p43[0] = p4.X() - p3.X();
p43[1] = p4.Y() - p3.Y();
p43[2] = p4.Z() - p3.Z();
if (std::abs(p43[0]) < zero_tolerance && std::abs(p43[1]) < zero_tolerance && std::abs(p43[2]) < zero_tolerance)
return resulting_line;

p21[0] = p2.X() - p1.X();
p21[1] = p2.Y() - p1.Y();
p21[2] = p2.Z() - p1.Z();
if (std::abs(p21[0]) < zero_tolerance && std::abs(p21[1]) < zero_tolerance && std::abs(p21[2]) < zero_tolerance)
return resulting_line;

d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

denom = d2121 * d4343 - d4321 * d4321;
auto pa = Kratos::make_shared<Point>(0.0, 0.0, 0.0);
auto pb = Kratos::make_shared<Point>(0.0, 0.0, 0.0);
if (std::abs(denom) < zero_tolerance) { 
Point projected_point;
array_1d<double,3> local_coords;
GeometricalProjectionUtilities::FastProjectOnLine(rSegment2, rSegment1[0], projected_point);
if (rSegment2.IsInside(projected_point, local_coords)) {
pa->Coordinates() = rSegment1[0].Coordinates();
pb->Coordinates() = projected_point;
} else {
GeometricalProjectionUtilities::FastProjectOnLine(rSegment2, rSegment1[1], projected_point);
if (rSegment2.IsInside(projected_point, local_coords)) {
pa->Coordinates() = rSegment1[1].Coordinates();
pb->Coordinates() = projected_point;
} else { 
GeometricalProjectionUtilities::FastProjectOnLine(rSegment1, rSegment2[0], projected_point);
if (rSegment1.IsInside(projected_point, local_coords)) {
pa->Coordinates() = rSegment2[0].Coordinates();
pb->Coordinates() = projected_point;
} else {
GeometricalProjectionUtilities::FastProjectOnLine(rSegment1, rSegment2[1], projected_point);
if (rSegment1.IsInside(projected_point, local_coords)) {
pa->Coordinates() = rSegment2[1].Coordinates();
pb->Coordinates() = projected_point;
} else { 
return resulting_line;
}
}
}
}
} else {
numer = d1343 * d4321 - d1321 * d4343;

mua = numer / denom;
mub = (d1343 + d4321 * mua) / d4343;

pa->X() = p1.X() + mua * p21[0];
pa->Y() = p1.Y() + mua * p21[1];
pa->Z() = p1.Z() + mua * p21[2];
pb->X() = p3.X() + mub * p43[0];
pb->Y() = p3.Y() + mub * p43[1];
pb->Z() = p3.Z() + mub * p43[2];
}

resulting_line.push_back(pa);
resulting_line.push_back(pb);
return resulting_line;
}


template <class TGeometryType>
static int ComputeLineLineIntersection(
const TGeometryType& rLineGeometry,
const array_1d<double,3>& rLinePoint0,
const array_1d<double,3>& rLinePoint1,
array_1d<double,3>& rIntersectionPoint,
const double epsilon = 1e-12)
{
return ComputeLineLineIntersection(
rLineGeometry[0], rLineGeometry[1], rLinePoint0, rLinePoint1, rIntersectionPoint, epsilon);
}


static int ComputeLineLineIntersection(
const array_1d<double,3>& rLine1Point0,
const array_1d<double,3>& rLine1Point1,
const array_1d<double,3>& rLine2Point0,
const array_1d<double,3>& rLine2Point1,
array_1d<double,3>& rIntersectionPoint,
const double epsilon = 1e-12)
{
const array_1d<double,3> r = rLine1Point1 - rLine1Point0;
const array_1d<double,3> s = rLine2Point1 - rLine2Point0;
const array_1d<double,3> q_p = rLine2Point0 - rLine1Point0;        

const double aux_1 = CrossProd2D(r,s);
const double aux_2 = CrossProd2D(q_p,r);
const double aux_3 = CrossProd2D(q_p,s);

if (std::abs(aux_1) < epsilon && std::abs(aux_2) < epsilon){
const double aux_4 = inner_prod(r,r);
const double aux_5 = inner_prod(s,r);
const double t_0 = inner_prod(q_p,r)/aux_4;
const double t_1 = t_0 + aux_5/aux_4;
if (aux_5 < 0.0){
if (t_1 >= 0.0 && t_0 <= 1.0){
return 2;    
}
} else {
if (t_0 >= 0.0 && t_1 <= 1.0){
return 2;    
}
}
} else if (std::abs(aux_1) < epsilon && std::abs(aux_2) > epsilon){
return 0;
} else if (std::abs(aux_1) > epsilon){
const double u = aux_2/aux_1;
const double t = aux_3/aux_1;
if (((u >= 0.0) && (u <= 1.0)) && ((t >= 0.0) && (t <= 1.0))){
rIntersectionPoint = rLine2Point0 + u*s;
if (u < epsilon || (1.0 - u) < epsilon) {
return 3;
} else {
return 1;
}
}
}
return 0;
}


static int ComputePlaneLineIntersection(
const array_1d<double,3>& rPlaneBasePoint,
const array_1d<double,3>& rPlaneNormal,
const array_1d<double,3>& rLinePoint1,
const array_1d<double,3>& rLinePoint2,
array_1d<double,3>& rIntersectionPoint,
const double epsilon = 1e-12)
{

const array_1d<double,3> line_dir = rLinePoint2 - rLinePoint1;

const double a = inner_prod(rPlaneNormal,( rPlaneBasePoint - rLinePoint1 ));
const double b = inner_prod(rPlaneNormal,line_dir);
if (std::abs(b) < epsilon){
if (std::abs(a) < epsilon){
return 2;    
} else {
return 0;    
}
}

const double r = a / b;
if (r < 0.0){
return 0;    
} else if (r > 1.0) {
return 0;    
}
rIntersectionPoint = rLinePoint1 + r * line_dir;

return 1;
}


static int ComputeLineBoxIntersection(
const array_1d<double,3> &rBoxPoint0,
const array_1d<double,3> &rBoxPoint1,
const array_1d<double,3> &rLinePoint0,
const array_1d<double,3> &rLinePoint1)
{
array_1d<double,3> intersection_point = ZeroVector(3);

if (rLinePoint1[0] < rBoxPoint0[0] && rLinePoint0[0] < rBoxPoint0[0]) return false;
if (rLinePoint1[0] > rBoxPoint1[0] && rLinePoint0[0] > rBoxPoint1[0]) return false;
if (rLinePoint1[1] < rBoxPoint0[1] && rLinePoint0[1] < rBoxPoint0[1]) return false;
if (rLinePoint1[1] > rBoxPoint1[1] && rLinePoint0[1] > rBoxPoint1[1]) return false;
if (rLinePoint1[2] < rBoxPoint0[2] && rLinePoint0[2] < rBoxPoint0[2]) return false;
if (rLinePoint1[2] > rBoxPoint1[2] && rLinePoint0[2] > rBoxPoint1[2]) return false;
if (rLinePoint0[0] > rBoxPoint0[0] && rLinePoint0[0] < rBoxPoint1[0] &&
rLinePoint0[1] > rBoxPoint0[1] && rLinePoint0[1] < rBoxPoint1[1] &&
rLinePoint0[2] > rBoxPoint0[2] && rLinePoint0[2] < rBoxPoint1[2]) {
return true;
}
if ((GetLineBoxIntersection(rLinePoint0[0]-rBoxPoint0[0], rLinePoint1[0]-rBoxPoint0[0], rLinePoint0, rLinePoint1, intersection_point) && InBox(intersection_point, rBoxPoint0, rBoxPoint1, 1 )) ||
(GetLineBoxIntersection(rLinePoint0[1]-rBoxPoint0[1], rLinePoint1[1]-rBoxPoint0[1], rLinePoint0, rLinePoint1, intersection_point) && InBox(intersection_point, rBoxPoint0, rBoxPoint1, 2 )) ||
(GetLineBoxIntersection(rLinePoint0[2]-rBoxPoint0[2], rLinePoint1[2]-rBoxPoint0[2], rLinePoint0, rLinePoint1, intersection_point) && InBox(intersection_point, rBoxPoint0, rBoxPoint1, 3 )) ||
(GetLineBoxIntersection(rLinePoint0[0]-rBoxPoint1[0], rLinePoint1[0]-rBoxPoint1[0], rLinePoint0, rLinePoint1, intersection_point) && InBox(intersection_point, rBoxPoint0, rBoxPoint1, 1 )) ||
(GetLineBoxIntersection(rLinePoint0[1]-rBoxPoint1[1], rLinePoint1[1]-rBoxPoint1[1], rLinePoint0, rLinePoint1, intersection_point) && InBox(intersection_point, rBoxPoint0, rBoxPoint1, 2 )) ||
(GetLineBoxIntersection(rLinePoint0[2]-rBoxPoint1[2], rLinePoint1[2]-rBoxPoint1[2], rLinePoint0, rLinePoint1, intersection_point) && InBox(intersection_point, rBoxPoint0, rBoxPoint1, 3 ))){
return true;
}

return false;
}




private:







static inline double CrossProd2D(const array_1d<double,3> &a, const array_1d<double,3> &b){
return (a(0)*b(1) - a(1)*b(0));
}

static inline int GetLineBoxIntersection(
const double Dist1,
const double Dist2,
const array_1d<double,3> &rPoint1,
const array_1d<double,3> &rPoint2,
array_1d<double,3> &rIntersectionPoint)
{
if ((Dist1 * Dist2) >= 0.0){
return 0;
}
if (std::abs(Dist1-Dist2) < 1e-12){
return 0;
}
rIntersectionPoint = rPoint1 + (rPoint2-rPoint1)*(-Dist1/(Dist2-Dist1));
return 1;
}

static inline int InBox(
const array_1d<double,3> &rIntersectionPoint,
const array_1d<double,3> &rBoxPoint0,
const array_1d<double,3> &rBoxPoint1,
const unsigned int Axis)
{
if ( Axis==1 && rIntersectionPoint[2] > rBoxPoint0[2] && rIntersectionPoint[2] < rBoxPoint1[2] && rIntersectionPoint[1] > rBoxPoint0[1] && rIntersectionPoint[1] < rBoxPoint1[1]) return 1;
if ( Axis==2 && rIntersectionPoint[2] > rBoxPoint0[2] && rIntersectionPoint[2] < rBoxPoint1[2] && rIntersectionPoint[0] > rBoxPoint0[0] && rIntersectionPoint[0] < rBoxPoint1[0]) return 1;
if ( Axis==3 && rIntersectionPoint[0] > rBoxPoint0[0] && rIntersectionPoint[0] < rBoxPoint1[0] && rIntersectionPoint[1] > rBoxPoint0[1] && rIntersectionPoint[1] < rBoxPoint1[1]) return 1;
return 0;
}





}; 



}  