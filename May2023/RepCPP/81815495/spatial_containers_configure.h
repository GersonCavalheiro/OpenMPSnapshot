
#pragma once

#include <string>
#include <iostream>
#include <cmath>



namespace Kratos
{


using SizeType = std::size_t;





template <SizeType TDimension, class TEntity = Element>
class SpatialContainersConfigure
{
public:

using PointType = Point;

using NodeType = Node;

using GeometryType = Geometry<NodeType>;

using DistanceIteratorType = std::vector<double>::iterator;

using EntityType = TEntity;

using ContainerType = typename PointerVectorSet<TEntity, IndexedObject>::ContainerType;
using PointerType = typename ContainerType::value_type;
using IteratorType = typename ContainerType::iterator;
using ResultContainerType = typename PointerVectorSet<TEntity, IndexedObject>::ContainerType;
using ResultPointerType = typename ResultContainerType::value_type;
using ResultIteratorType = typename ResultContainerType::iterator;

static constexpr std::size_t Dimension = TDimension;

static constexpr std::size_t DIMENSION = TDimension;

static constexpr std::size_t MAX_LEVEL = 16;

static constexpr std::size_t MIN_LEVEL = 2;

KRATOS_CLASS_POINTER_DEFINITION(SpatialContainersConfigure);



SpatialContainersConfigure() {}

virtual ~SpatialContainersConfigure() {}




static inline void CalculateBoundingBox(
const PointerType& rObject,
PointType& rLowPoint,
PointType& rHighPoint
)
{
rHighPoint = rObject->GetGeometry().GetPoint(0);
rLowPoint  = rObject->GetGeometry().GetPoint(0);
for (unsigned int point = 0; point<rObject->GetGeometry().PointsNumber(); point++) {
for(unsigned int i = 0; i<TDimension; i++) {
rLowPoint[i]  =  (rLowPoint[i]  >  rObject->GetGeometry().GetPoint(point)[i] ) ?  rObject->GetGeometry().GetPoint(point)[i] : rLowPoint[i];
rHighPoint[i] =  (rHighPoint[i] <  rObject->GetGeometry().GetPoint(point)[i] ) ?  rObject->GetGeometry().GetPoint(point)[i] : rHighPoint[i];
}
}
}


static inline void CalculateBoundingBox(
const PointerType& rObject,
PointType& rLowPoint,
PointType& rHighPoint,
const double Radius
)
{
(void)Radius;
CalculateBoundingBox(rObject, rLowPoint, rHighPoint);
}


static inline bool Intersection(
const PointerType& rObj_1,
const PointerType& rObj_2
)
{
GeometryType& r_geom_1 = rObj_1->GetGeometry();
GeometryType& r_geom_2 = rObj_2->GetGeometry();
return r_geom_1.HasIntersection(r_geom_2);
}


static inline bool Intersection(
const PointerType& rObj_1,
const PointerType& rObj_2,
const double Radius
)
{
(void)Radius;
return Intersection(rObj_1, rObj_2);
}


static inline bool IntersectionBox(
const PointerType& rObject,
const PointType& rLowPoint,
const PointType& rHighPoint
)
{
return rObject->GetGeometry().HasIntersection(rLowPoint, rHighPoint);
}


static inline bool IntersectionBox(
const PointerType& rObject,
const PointType& rLowPoint,
const PointType& rHighPoint,
const double Radius
)
{
(void)Radius;
return IntersectionBox(rObject, rLowPoint, rHighPoint);
}


static inline void Distance(
const PointerType& rObj_1,
const PointerType& rObj_2,
double& rDistance
)
{

}




virtual std::string Info() const
{
return " Spatial Containers Configure";
}

virtual void PrintInfo(std::ostream& rOStream) const {}

virtual void PrintData(std::ostream& rOStream) const {}


protected:







private:







SpatialContainersConfigure& operator=(SpatialContainersConfigure const& rOther);

SpatialContainersConfigure(SpatialContainersConfigure const& rOther);


}; 




template <std::size_t TDimension>
inline std::istream& operator >> (std::istream& rIStream,
SpatialContainersConfigure<TDimension> & rThis)
{
return rIStream;
}

template <std::size_t TDimension>
inline std::ostream& operator << (std::ostream& rOStream,
const SpatialContainersConfigure<TDimension>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
