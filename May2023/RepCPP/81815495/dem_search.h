
#pragma once

#include "includes/define.h"

#include <string>
#include <iostream>

#include "spatial_containers/spatial_search.h"

namespace Kratos
{








template<std::size_t dim, class T>
class PointDistance2
{
public:
inline double operator()( T const& p1, T const& p2 )
{
double dist = 0.0;

double tmp1 = p1[0] - p2[0];
double tmp2 = p1[1] - p2[1];
double tmp3 = p1[2] - p2[2];

dist += tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;


return dist;
}
};

template<std::size_t Dimension>
class RadiusPoint
{
public:
RadiusPoint() {}
virtual ~RadiusPoint(){}

void Initialize(SpatialSearch::ElementPointerType baseElem)
{
for(std::size_t i = 0; i < Dimension; i++)
coord[i] = baseElem->GetGeometry()[0][i];

pNaseElem = baseElem;

}

void Initialize(SpatialSearch::ElementPointerType baseElem, double Radius)
{
for(std::size_t i = 0; i < Dimension; i++)
coord[i] = baseElem->GetGeometry()[0][i];

pNaseElem = baseElem;

}

public:

double       mRadius;

double       coord[Dimension];

double       & operator[](std::size_t i)       {return coord[i];}
double const & operator[](std::size_t i) const {return coord[i];}

SpatialSearch::ElementPointerType pNaseElem;

void operator=(Point const& Other){
for(std::size_t i = 0; i < Dimension; i++)
coord[i] = Other[i];
}
};

template< std::size_t Dimension >
std::ostream & operator<<( std::ostream& rOut, RadiusPoint<Dimension> & rPoint){
for(std::size_t i = 0 ; i < Dimension ; i++)
rOut << rPoint[i] << " ";
return rOut;
}

template< std::size_t Dimension >
std::istream & operator>>( std::istream& rIn, RadiusPoint<Dimension> & rPoint){
for(std::size_t i = 0 ; i < Dimension ; i++)
rIn >> rPoint[i];

return rIn;
}

template< class TDerived >
class DEMSearch : public SpatialSearch
{
public:

KRATOS_CLASS_POINTER_DEFINITION(DEMSearch);

typedef RadiusPoint<Dimension>        PointType;
typedef PointType*                    PtrPointType;
typedef std::vector<PtrPointType>*    PointVector;

PointVector       searchPoints;


using SpatialSearch::SearchElementsInRadiusExclusive;
using SpatialSearch::SearchElementsInRadiusInclusive;
using SpatialSearch::SearchNodesInRadiusExclusive;
using SpatialSearch::SearchNodesInRadiusInclusive;
using SpatialSearch::SearchConditionsOverElementsInRadiusExclusive;
using SpatialSearch::SearchConditionsOverElementsInRadiusInclusive;
using SpatialSearch::SearchElementsOverConditionsInRadiusExclusive;
using SpatialSearch::SearchElementsOverConditionsInRadiusInclusive;


DEMSearch(const double domain_min_x = 0.0, const double domain_min_y = 0.0, const double domain_min_z = 0.0,
const double domain_max_x = -1.0, const double domain_max_y = -1.0, const double domain_max_z = -1.0)
{
mDomainMin[0] = domain_min_x;
mDomainMin[1] = domain_min_y;
mDomainMin[2] = domain_min_z;
mDomainMax[0] = domain_max_x;
mDomainMax[1] = domain_max_y;
mDomainMax[2] = domain_max_z;
TDerived::ElementConfigureType::SetDomain(domain_min_x, domain_min_y, domain_min_z, domain_max_x, domain_max_y, domain_max_z);
TDerived::NodeConfigureType::SetDomain(domain_min_x, domain_min_y, domain_min_z, domain_max_x, domain_max_y, domain_max_z);
mDomainPeriodicity = TDerived::ElementConfigureType::GetDomainPeriodicity();
searchPoints = new std::vector<PtrPointType>(0);
}

virtual ~DEMSearch(){
delete searchPoints;
}



void SearchElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType & Radius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance ) override
{
static_cast<TDerived*>(this)->SearchElementsInRadiusExclusiveImplementation(StructureElements,InputElements,Radius,rResults,rResultsDistance);
}

void SearchElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType & Radius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance )
{
static_cast<TDerived*>(this)->SearchElementsInRadiusInclusiveImplementation(StructureElements,InputElements,Radius,rResults,rResultsDistance);
}

void SearchElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType & Radius,
VectorResultElementsContainerType& rResults ) override
{
static_cast<TDerived*>(this)->SearchElementsInRadiusExclusiveImplementation(StructureElements,InputElements,Radius,rResults);
}

void SearchElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType & Radius,
VectorResultElementsContainerType& rResults )
{
static_cast<TDerived*>(this)->SearchElementsInRadiusInclusiveImplementation(StructureElements,InputElements,Radius,rResults);
}

void SearchNodesInRadiusExclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType & Radius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance ) override
{
static_cast<TDerived*>(this)->SearchNodesInRadiusExclusiveImplementation(StructureNodes,InputNodes,Radius,rResults,rResultsDistance);
}

void SearchNodesInRadiusInclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType & Radius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance ) override
{
static_cast<TDerived*>(this)->SearchNodesInRadiusInclusiveImplementation(StructureNodes,InputNodes,Radius,rResults,rResultsDistance);
}

void SearchNodesInRadiusExclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType & Radius,
VectorResultNodesContainerType& rResults ) override
{
static_cast<TDerived*>(this)->SearchNodesInRadiusExclusiveImplementation(StructureNodes,InputNodes,Radius,rResults);
}

void SearchNodesInRadiusInclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType & Radius,
VectorResultNodesContainerType& rResults ) override
{
static_cast<TDerived*>(this)->SearchNodesInRadiusInclusiveImplementation(StructureNodes,InputNodes,Radius,rResults);
}

void SearchConditionsOverElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
ConditionsContainerType const& InputConditions,
const RadiusArrayType & Radius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance )
{
static_cast<TDerived*>(this)->SearchGeometricalInRadiusExclusiveImplementation(StructureElements,InputConditions,Radius,rResults,rResultsDistance);
}

void SearchConditionsOverElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
ConditionsContainerType const& InputConditions,
const RadiusArrayType & Radius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance )
{
static_cast<TDerived*>(this)->SearchGeometricalInRadiusInclusiveImplementation(StructureElements,InputConditions,Radius,rResults,rResultsDistance);
}

void SearchElementsOverConditionsInRadiusExclusive (
ConditionsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType & Radius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance ) override
{
static_cast<TDerived*>(this)->SearchGeometricalInRadiusExclusiveImplementation(StructureElements,InputElements,Radius,rResults,rResultsDistance);
}

void SearchElementsOverConditionsInRadiusInclusive (
ConditionsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType & Radius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance ) override
{
static_cast<TDerived*>(this)->SearchGeometricalInRadiusInclusiveImplementation(StructureElements,InputElements,Radius,rResults,rResultsDistance);
}






virtual std::string Info() const override
{
std::stringstream buffer;
buffer << "DemSearch" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const override {rOStream << "DemSearch";}

virtual void PrintData(std::ostream& rOStream) const override {}





protected:








bool mDomainPeriodicity;
array_1d<double, 3> mDomainMin;
array_1d<double, 3> mDomainMax;





private:












DEMSearch& operator=(DEMSearch const& rOther)
{
return *this;
}

DEMSearch(DEMSearch const& rOther)
{
*this = rOther;
}



}; 









}  
