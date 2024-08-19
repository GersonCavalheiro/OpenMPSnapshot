
#pragma once

#include <string>
#include <iostream>
#include <cmath>

#include "spatial_containers/spatial_search.h"

namespace Kratos {




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) NodeConfigureForNodeSearch {
public:
KRATOS_CLASS_POINTER_DEFINITION(NodeConfigureForNodeSearch);


static constexpr auto Epsilon   = std::numeric_limits<double>::epsilon();
static constexpr auto Dimension = 3;

typedef SpatialSearch                                           SearchType;

typedef SearchType::PointType                                   PointType;
typedef SearchType::NodesContainerType::ContainerType           ContainerType;
typedef SearchType::NodesContainerType                          NodesContainerType;

typedef SearchType::NodeType                                    NodeType;
typedef ContainerType::value_type                               PointerType;
typedef ContainerType::iterator                                 IteratorType;

typedef SearchType::NodesContainerType::ContainerType           ResultContainerType;

typedef ResultContainerType::iterator                           ResultIteratorType;
typedef std::vector<double>::iterator                           DistanceIteratorType;


NodeConfigureForNodeSearch(){}

virtual ~NodeConfigureForNodeSearch(){}



static inline void CalculateBoundingBox(const PointerType& rObject, PointType& rLowPoint, PointType& rHighPoint)
{
rHighPoint = rLowPoint  = *rObject;
}


static inline void CalculateBoundingBox(const PointerType& rObject, PointType& rLowPoint, PointType& rHighPoint, const double Radius)
{
auto radiusExtension = PointType(Radius, Radius, Radius);

rLowPoint  = PointType{*rObject - radiusExtension};
rHighPoint = PointType{*rObject + radiusExtension};
}


static inline void CalculateCenter(const PointerType& rObject, PointType& rCenter)
{
rCenter  = *rObject;
}


static inline bool Intersection(const PointerType& rObj_1, const PointerType& rObj_2, const double Radius)
{
double distance;
Distance(rObj_1, rObj_2, distance);

if( distance > Epsilon + Radius){
return false;
}

return true;
}


static inline bool  IntersectionBox(const PointerType& rObject,  const PointType& rLowPoint, const PointType& rHighPoint)
{
for(std::size_t i = 0; i < Dimension; i++) {
if( (*rObject)[i] < rLowPoint[i] - Epsilon || (*rObject)[i] > rHighPoint[i] + Epsilon) {
return false;
}
}

return true;
}


static inline bool  IntersectionBox(const PointerType& rObject,  const PointType& rLowPoint, const PointType& rHighPoint, const double Radius)
{
for(std::size_t i = 0; i < Dimension; i++) {
if( ((*rObject)[i] + Radius) < rLowPoint[i] - Epsilon || ((*rObject)[i] - Radius) > rHighPoint[i] + Epsilon) {
return false;
}
}

return true;
}


static inline void Distance(const PointerType& rObj_1, const PointerType& rObj_2, double& distance)
{
double pwdDistance = 0.0f;

for(std::size_t i = 0; i < Dimension; i++) {
pwdDistance += std::pow((*rObj_1)[i] - (*rObj_2)[i], 2);
}

distance = std::sqrt(pwdDistance);
}

virtual std::string Info() const {return " Spatial Containers Configure for Nodes to perform a Node Search"; }

virtual void PrintInfo(std::ostream& rOStream) const {}

virtual void PrintData(std::ostream& rOStream) const {}

private:


NodeConfigureForNodeSearch& operator=(NodeConfigureForNodeSearch const& rOther);

NodeConfigureForNodeSearch(NodeConfigureForNodeSearch const& rOther);


}; 




inline std::istream& operator >> (std::istream& rIStream, NodeConfigureForNodeSearch& rThis){
return rIStream;
}

inline std::ostream& operator << (std::ostream& rOStream, const NodeConfigureForNodeSearch& rThis){
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}   