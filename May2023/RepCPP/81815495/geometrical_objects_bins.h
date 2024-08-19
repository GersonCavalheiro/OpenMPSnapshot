
#pragma once

#include <unordered_set>


#include "geometries/bounding_box.h"
#include "geometries/point.h"
#include "spatial_containers/spatial_search_result.h"

namespace Kratos
{


class GeometricalObject; 


class KRATOS_API(KRATOS_CORE) GeometricalObjectsBins
{
public:

KRATOS_CLASS_POINTER_DEFINITION(GeometricalObjectsBins);

using CellType = std::vector<GeometricalObject*>;
using ResultType = SpatialSearchResult<GeometricalObject>;



template<typename TIteratorType>
GeometricalObjectsBins(
TIteratorType GeometricalObjectsBegin,
TIteratorType GeometricalObjectsEnd
)
{
const std::size_t number_of_objects = std::distance(GeometricalObjectsBegin, GeometricalObjectsEnd);
if (number_of_objects > 0){
mBoundingBox.Set(GeometricalObjectsBegin->GetGeometry().begin(), GeometricalObjectsBegin->GetGeometry().end());
for (TIteratorType i_object = GeometricalObjectsBegin ; i_object != GeometricalObjectsEnd ; i_object++){
mBoundingBox.Extend(i_object->GetGeometry().begin() , i_object->GetGeometry().end());
}
}
mBoundingBox.Extend(Tolerance);
CalculateCellSize(number_of_objects);
mCells.resize(GetTotalNumberOfCells());
AddObjectsToCells(GeometricalObjectsBegin, GeometricalObjectsEnd);
}


template<typename TContainer>
GeometricalObjectsBins(TContainer& rGeometricalObjectsVector)
: GeometricalObjectsBins(rGeometricalObjectsVector.begin(), rGeometricalObjectsVector.end())
{
}

virtual ~GeometricalObjectsBins(){}





CellType& GetCell(
const std::size_t I,
const std::size_t J,
const std::size_t K
);


BoundingBox<Point> GetCellBoundingBox(
const std::size_t I,
const std::size_t J,
const std::size_t K
);


void SearchInRadius(
const Point& rPoint,
const double Radius,
std::vector<ResultType>& rResults
);


template<typename TPointIteratorType>
void SearchInRadius(
TPointIteratorType itPointBegin,
TPointIteratorType itPointEnd,
const double Radius,
std::vector<std::vector<ResultType>>& rResults
)
{
const std::size_t number_of_points = std::distance(itPointBegin, itPointEnd);
rResults.resize(number_of_points);
for (auto it_point = itPointBegin ; it_point != itPointEnd ; it_point++){
SearchInRadius(*it_point, Radius, rResults[it_point - itPointBegin]);
}
}


ResultType SearchNearestInRadius(
const Point& rPoint,
const double Radius
);


template<typename TPointIteratorType>
std::vector<ResultType> SearchNearestInRadius(
TPointIteratorType itPointBegin,
TPointIteratorType itPointEnd,
const double Radius
)
{
std::vector<ResultType> results;
const std::size_t number_of_points = std::distance(itPointBegin, itPointEnd);
results.resize(number_of_points);
for (auto it_point = itPointBegin ; it_point != itPointEnd ; it_point++){
results[it_point - itPointBegin] = SearchNearestInRadius(*it_point, Radius);
}
return results;
}


ResultType SearchNearest(const Point& rPoint);


template<typename TPointIteratorType>
std::vector<ResultType> SearchNearest(
TPointIteratorType itPointBegin,
TPointIteratorType itPointEnd
)
{
std::vector<ResultType> results;
const std::size_t number_of_points = std::distance(itPointBegin, itPointEnd);
results.resize(number_of_points);
for (auto it_point = itPointBegin ; it_point != itPointEnd ; it_point++){
results[it_point - itPointBegin] = SearchNearest(*it_point);
}
return results;
}


ResultType SearchIsInside(const Point& rPoint);


template<typename TPointIteratorType>
std::vector<ResultType> SearchIsInside(
TPointIteratorType itPointBegin,
TPointIteratorType itPointEnd
)
{
std::vector<ResultType> results;
const std::size_t number_of_points = std::distance(itPointBegin, itPointEnd);
results.resize(number_of_points);
for (auto it_point = itPointBegin ; it_point != itPointEnd ; it_point++){
results[it_point - itPointBegin] = SearchIsInside(*it_point);
}
return results;
}



const BoundingBox<Point>& GetBoundingBox() const {
return mBoundingBox;
}


const array_1d<double, 3>& GetCellSizes(){
return mCellSizes;
}


const array_1d<std::size_t, 3>& GetNumberOfCells(){
return mNumberOfCells;
}


std::size_t GetTotalNumberOfCells(){
return mNumberOfCells[0] * mNumberOfCells[1] * mNumberOfCells[2];
}




virtual std::string Info() const
{
std::stringstream buffer;
buffer << "GeometricalObjectsBins" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const {rOStream << "GeometricalObjectsBins";}

virtual void PrintData(std::ostream& rOStream) const {}


protected:

GeometricalObjectsBins() = default;


static constexpr unsigned int Dimension = 3;    
static constexpr double Tolerance = 1e-12;      


BoundingBox<Point> mBoundingBox;                 
array_1d<std::size_t, Dimension> mNumberOfCells; 
array_1d<double, 3>  mCellSizes;                 
array_1d<double, 3>  mInverseOfCellSize;         
std::vector<CellType> mCells;                    






void CalculateCellSize(const std::size_t NumberOfCells);


template<typename TIteratorType>
void AddObjectsToCells(
TIteratorType GeometricalObjectsBegin,
TIteratorType GeometricalObjectsEnd
)
{
for(auto i_geometrical_object = GeometricalObjectsBegin ; i_geometrical_object != GeometricalObjectsEnd ; i_geometrical_object++){
array_1d<std::size_t, 3> min_position(3,0);
array_1d<std::size_t, 3> max_position(3,0);
CalculateMinMaxPositions(i_geometrical_object->GetGeometry(), min_position, max_position);
for(std::size_t k = min_position[2] ; k < max_position[2] ; k++){
for(std::size_t j = min_position[1] ; j < max_position[1] ; j++){
for(std::size_t i = min_position[0] ; i < max_position[0] ; i++){
auto cell_bounding_box = GetCellBoundingBox(i,j,k);
if(IsIntersected(i_geometrical_object->GetGeometry(), cell_bounding_box, Tolerance)){
GetCell(i,j,k).push_back(&(*i_geometrical_object));
}
}
}
}
}
}

private:



template<typename TGeometryType>
void CalculateMinMaxPositions(
const TGeometryType& rGeometry,
array_1d<std::size_t, 3>& rMinPosition,
array_1d<std::size_t, 3>& rMaxPosition
)
{
if(rGeometry.empty())
return;

BoundingBox<Point> bounding_box(rGeometry.begin(), rGeometry.end());

for(unsigned int i = 0; i < 3; i++ ) {
rMinPosition[i] = CalculatePosition( bounding_box.GetMinPoint()[i], i );
rMaxPosition[i] = CalculatePosition( bounding_box.GetMaxPoint()[i], i ) + 1;
}
}


std::size_t CalculatePosition(
const double Coordinate,
const int ThisDimension
) const;


template<typename TGeometryType>
static inline bool IsIntersected(
TGeometryType& rGeometry,
const BoundingBox<Point>& rBox,
const double ThisTolerance
)
{
Point low_point_tolerance;
Point high_point_tolerance;

for(unsigned int i = 0; i<3; i++) {
low_point_tolerance[i]  =  rBox.GetMinPoint()[i] - ThisTolerance;
high_point_tolerance[i] =  rBox.GetMaxPoint()[i] + ThisTolerance;
}

return rGeometry.HasIntersection(low_point_tolerance,high_point_tolerance);
}


void SearchInRadiusInCell(
const CellType& rCell,
const Point& rPoint,
const double Radius,
std::unordered_set<GeometricalObject*>& rResults
);


void SearchNearestInCell(
const CellType& rCell,
const Point& rPoint,
ResultType& rResult,
const double MaxRadius
);


void SearchIsInsideInCell(
const CellType& rCell,
const Point& rPoint,
ResultType& rResult
);






GeometricalObjectsBins& operator=(GeometricalObjectsBins const& rOther) = delete;

GeometricalObjectsBins(GeometricalObjectsBins const& rOther) = delete;


}; 









}  