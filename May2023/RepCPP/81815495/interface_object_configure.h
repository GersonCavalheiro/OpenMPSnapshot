
#pragma once


#include "custom_searching/interface_object.h"

namespace Kratos
{



class InterfaceObjectConfigure
{
public:

KRATOS_CLASS_POINTER_DEFINITION(InterfaceObjectConfigure);


static constexpr auto Epsilon   = std::numeric_limits<double>::epsilon();
static constexpr auto Dimension = 3;


typedef Point          PointType;


typedef InterfaceObject                     ObjectType;
typedef PointerVectorSet <
ObjectType,
IndexedObject
>                                           ObjectContainerType;
typedef ObjectType::Pointer                 PointerType;

typedef ObjectContainerType::ContainerType  ContainerType;
typedef ObjectContainerType::ContainerType  ResultContainerType;

typedef ContainerType::iterator             IteratorType;
typedef ResultContainerType::iterator       ResultIteratorType;
typedef std::vector<double>::iterator       DistanceIteratorType;

typedef double                              CoordinateType;
typedef Tvector<CoordinateType, Dimension>   CoordinateArray;


InterfaceObjectConfigure() {};

virtual ~InterfaceObjectConfigure() {}



static inline void CalculateBoundingBox(const PointerType& rObject, PointType& rLowPoint, PointType& rHighPoint)
{
rHighPoint = rLowPoint = *rObject;
}


static inline void CalculateBoundingBox(const PointerType& rObject, PointType& rLowPoint, PointType& rHighPoint, const double& Radius)
{
auto radiusExtension = PointType(Radius, Radius, Radius);

rLowPoint  = PointType{*rObject - radiusExtension};
rHighPoint = PointType{*rObject + radiusExtension};
}


static inline void CalculateCenter(const PointerType& rObject, PointType& rCentralPoint)
{
rCentralPoint = *rObject;
}


static inline bool Intersection(const PointerType& rObj_1, const PointerType& rObj_2)
{
for(std::size_t i = 0; i < Dimension; i++)
{
if(std::fabs((*rObj_1)[i] - (*rObj_2)[i]) > Epsilon)
{
return false;
}
}

return true;
}


static inline bool Intersection(const PointerType& rObj_1, const PointerType& rObj_2, double Radius)
{

double pwdDistance = 0.0f;

for(std::size_t i = 0; i < Dimension; i++)
{
pwdDistance += std::pow((*rObj_1)[i] - (*rObj_2)[i], 2);
}

if (std::sqrt(pwdDistance) > Epsilon + Radius)
{
return false;
}
else
{
return true;
}
}


static inline bool IntersectionBox(const PointerType& rObject, const PointType& rLowPoint, const PointType& rHighPoint)
{
for(std::size_t i = 0; i < Dimension; i++)
{
if( (*rObject)[i] < rLowPoint[i] - Epsilon || (*rObject)[i] > rHighPoint[i] + Epsilon)
{
return false;
}
}

return true;
}


static inline bool IntersectionBox(const PointerType& rObject, const PointType& rLowPoint, const PointType& rHighPoint, const double& Radius)
{
for(std::size_t i = 0; i < Dimension; i++)
{
if( ((*rObject)[i] + Radius) < rLowPoint[i] - Epsilon || ((*rObject)[i] - Radius) > rHighPoint[i] + Epsilon)
{
return false;
}
}

return true;
}


static inline void Distance(const PointerType& rObj_1, const PointerType& rObj_2, double& distance)
{
double pwdDistance = 0.0f;

for(std::size_t i = 0; i < Dimension; i++)
{
pwdDistance += std::pow((*rObj_1)[i] - (*rObj_2)[i], 2);
}

distance = std::sqrt(pwdDistance);
}


static inline double GetObjectRadius(const PointerType& rObject, const double& Radius)
{
return 0.0f;
}


virtual std::string Info() const
{
return "Spatial Containers Configure for 'Points'";
}

virtual std::string Data() const
{
return "Dimension: " + std::to_string(Dimension);
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << Info() << std::endl;
}

virtual void PrintData(std::ostream& rOStream) const
{
rOStream << Data() << Dimension << std::endl;
}


private:

InterfaceObjectConfigure& operator=(InterfaceObjectConfigure const& rOther);

InterfaceObjectConfigure(InterfaceObjectConfigure const& rOther);


}; 



inline std::istream& operator >> (std::istream& rIStream, InterfaceObjectConfigure& rThis)
{
return rIStream;
}

inline std::ostream& operator << (std::ostream& rOStream, const InterfaceObjectConfigure& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


} 
