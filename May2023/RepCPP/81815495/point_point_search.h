
#if !defined(KRATOS_POINT_POINT_SEARCH_H_INCLUDED)
#define  KRATOS_POINT_POINT_SEARCH_H_INCLUDED

#include <string>
#include <iostream>

#include "includes/define.h"

#include "utilities/openmp_utils.h"

#include "spatial_containers/spatial_search.h"
#include "point_configure.h"
#include "spatial_containers/bins_dynamic_objects.h"
#include "spatial_containers/bins_dynamic.h"



#include "utilities/timer.h"
#ifdef CUSTOMTIMER
#define KRATOS_TIMER_START(t) Timer::Start(t);
#define KRATOS_TIMER_STOP(t) Timer::Stop(t);
#else
#define KRATOS_TIMER_START(t)
#define KRATOS_TIMER_STOP(t)
#endif

namespace Kratos
{








class PointPointSearch: public SpatialSearch
{
public:

KRATOS_CLASS_POINTER_DEFINITION(PointPointSearch);

typedef PointType*                                  PointPointerType;
typedef std::vector<PointPointerType>*              PointVector;
typedef std::vector<PointPointerType>::iterator     PointIterator;

typedef double*                                     DistanceVector;
typedef double*                                     DistanceIterator;

typedef PointConfigure<3>                           PointConfigureType;
typedef BinsObjectDynamic<PointConfigureType>       PointBinsType;
typedef PointerVectorSet<Point, IndexedObject>      PointSetType;



PointPointSearch(){}

~PointPointSearch(){}

void SearchPointsImplementation(
NodesContainerType const& r_nodes,
NodesContainerType const& r_nodes_to_find,
RadiusArrayType const& radius,
VectorResultNodesContainerType& r_results,
VectorDistanceType& r_results_distances)
{
KRATOS_TRY

int max_n_of_neigh_nodes = r_nodes_to_find.size();

NodesContainerType::ContainerType& nodes         = const_cast <NodesContainerType::ContainerType&> (r_nodes.GetContainer());
NodesContainerType::ContainerType& nodes_to_find = const_cast <NodesContainerType::ContainerType&> (r_nodes_to_find.GetContainer());

PointSetType::ContainerType nodes_temp;
PointSetType::ContainerType nodes_to_find_temp;

std::map<Point::Pointer, Node::Pointer> map_point_to_node;

nodes_temp.reserve(nodes.size());

for (NodesContainerType::ContainerType::iterator it = nodes.begin(); it != nodes.end(); ++it){
auto p_point = std::make_shared<Point>((*it)->Coordinates());
nodes_temp.push_back(p_point);
}

nodes_to_find_temp.reserve(nodes_to_find.size());

for (auto it = nodes_to_find.begin(); it != nodes_to_find.end(); ++it){
auto p_point = std::make_shared<Point>((*it)->Coordinates());
nodes_to_find_temp.push_back(p_point);
map_point_to_node[p_point] = *it; / ]);
}

r_results_distances[i].insert(r_results_distances[i].begin(), local_results_distances.begin(), local_results_distances.begin() + n_of_results);
}
}

KRATOS_CATCH("")
}

virtual std::string Info() const override
{
std::stringstream buffer;
buffer << "PointPointSearch" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const override {rOStream << "PointPointSearch";}

virtual void PrintData(std::ostream& rOStream) const override {}





protected:
















private:













PointPointSearch& operator=(PointPointSearch const& rOther)
{
return *this;
}

PointPointSearch(PointPointSearch const& rOther)
{
*this = rOther;
}


}; 


}  

#endif 


