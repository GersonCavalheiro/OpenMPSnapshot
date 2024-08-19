
#pragma once



#include "includes/define.h"
#include "includes/node.h"

#include "spatial_containers/spatial_containers.h"
#include "spatial_containers/cell.h"
#include "spatial_containers/bins_dynamic_objects.h"

#include "utilities/spatial_containers_configure.h"

namespace Kratos
{






template< SizeType TDim, class TConfigureType = SpatialContainersConfigure<TDim> >
class BinBasedFastPointLocator
{
public:

typedef TConfigureType ConfigureType;

typedef typename ConfigureType::PointType PointType;
typedef typename ConfigureType::EntityType EntityType;
typedef typename ConfigureType::ContainerType ContainerType;
typedef typename ConfigureType::IteratorType IteratorType;
typedef typename ConfigureType::ResultContainerType ResultContainerType;
typedef typename ConfigureType::ResultIteratorType ResultIteratorType;

typedef BinsObjectDynamic<ConfigureType> BinsType;
typedef typename BinsObjectDynamic<ConfigureType>::CoordinateType BinsCoordinateType;
typedef typename BinsObjectDynamic<ConfigureType>::PointType BinsPointType;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

typedef std::size_t SizeType;

typedef std::size_t IndexType;

KRATOS_CLASS_POINTER_DEFINITION(BinBasedFastPointLocator);



explicit BinBasedFastPointLocator(ModelPart& rModelPart)
: mrModelPart(rModelPart)
{
}

virtual ~BinBasedFastPointLocator() = default;

BinBasedFastPointLocator(BinBasedFastPointLocator const& rOther)
: mrModelPart(rOther.mrModelPart)
{
auto paux = typename BinsType::Pointer(new BinsType(*rOther.mpBinsObjectDynamic));
paux.swap(mpBinsObjectDynamic);
}




void UpdateSearchDatabase()
{
KRATOS_TRY

ContainerType entities_array;
GetContainer(mrModelPart, entities_array);
IteratorType it_begin = entities_array.begin();
IteratorType it_end = entities_array.end();

auto paux = typename BinsType::Pointer(new BinsType(it_begin, it_end));
paux.swap(mpBinsObjectDynamic);

KRATOS_CATCH("")
}


void UpdateSearchDatabaseAssignedSize(const BinsCoordinateType CellSize)
{
KRATOS_TRY

ContainerType entities_array;
GetContainer(mrModelPart, entities_array);
IteratorType it_begin = entities_array.begin();
IteratorType it_end = entities_array.end();

auto paux = typename BinsType::Pointer(new BinsType(it_begin, it_end, CellSize));
paux.swap(mpBinsObjectDynamic);

KRATOS_CATCH("")
}


KRATOS_DEPRECATED_MESSAGE("This is legacy version (using array instead of vector for shape function)") bool FindPointOnMesh(
const array_1d<double, 3 >& rCoordinates,
array_1d<double, TDim + 1 >& rNShapeFunction,
typename EntityType::Pointer& pEntity,
ResultIteratorType ItResultBegin,
const SizeType MaxNumberOfResults = 1000,
const double Tolerance = 1.0e-5
)
{
SizeType results_found = mpBinsObjectDynamic->SearchObjectsInCell(BinsPointType{rCoordinates}, ItResultBegin, MaxNumberOfResults);

if (results_found > 0) {
for (IndexType i = 0; i < results_found; i++) {
GeometryType& r_geom = (*(ItResultBegin + i))->GetGeometry();

array_1d<double, 3> point_local_coordinates;
Vector shape_function;
const bool is_found = LocalIsInside(r_geom, rCoordinates, point_local_coordinates, Tolerance);
r_geom.ShapeFunctionsValues(shape_function, point_local_coordinates);
noalias(rNShapeFunction) = shape_function;

if (is_found) {
pEntity = (*(ItResultBegin + i));
return true;
}
}
}

pEntity = nullptr;
return false;
}


bool FindPointOnMesh(
const array_1d<double, 3 >& rCoordinates,
Vector& rNShapeFunction,
typename EntityType::Pointer& pEntity,
ResultIteratorType ItResultBegin,
const SizeType MaxNumberOfResults = 1000,
const double Tolerance = 1.0e-5
)
{
const SizeType results_found = mpBinsObjectDynamic->SearchObjectsInCell(BinsPointType{rCoordinates}, ItResultBegin, MaxNumberOfResults);

if (results_found > 0) {
for (IndexType i = 0; i < static_cast<IndexType>(results_found); i++) {

GeometryType& r_geom = (*(ItResultBegin + i))->GetGeometry();

array_1d<double, 3> point_local_coordinates;
const bool is_found = LocalIsInside(r_geom, rCoordinates, point_local_coordinates, Tolerance);
r_geom.ShapeFunctionsValues(rNShapeFunction, point_local_coordinates);

if (is_found) {
pEntity = (*(ItResultBegin + i));
return true;
}
}
}

pEntity = nullptr;
return false;
}


bool FindPointOnMeshSimplified(
const array_1d<double, 3 >& rCoordinates,
Vector& rNShapeFunction,
typename EntityType::Pointer& pEntity,
const SizeType MaxNumberOfResults = 1000,
const double Tolerance = 1.0e-5
)
{
ResultContainerType results(MaxNumberOfResults);

const bool is_found = FindPointOnMesh(rCoordinates, rNShapeFunction, pEntity, results.begin(), MaxNumberOfResults, Tolerance);

return is_found;
}




protected:





virtual bool LocalIsInside(
const GeometryType& rGeometry,
const GeometryType::CoordinatesArrayType& rPointGlobalCoordinates,
GeometryType::CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
return rGeometry.IsInside(rPointGlobalCoordinates, rResult, Tolerance);
}




private:


ModelPart& mrModelPart; 

typename BinsType::Pointer mpBinsObjectDynamic; 




static inline void GetContainer(
ModelPart& rModelPart,
PointerVectorSet<Element, IndexedObject>::ContainerType& rContainerArray
)
{
rContainerArray = rModelPart.ElementsArray();
}


static inline void GetContainer(
ModelPart& rModelPart,
PointerVectorSet<Condition, IndexedObject>::ContainerType& rContainerArray
)
{
rContainerArray = rModelPart.ConditionsArray();
}




};

} 
