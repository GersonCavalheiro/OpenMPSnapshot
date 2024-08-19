
#pragma once



#include "geometries/point.h"
#include "includes/kratos_parameters.h"
#include "spatial_containers/spatial_search.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{




enum class SpatialContainer
{
KDTree,
Octree,
BinsStatic,
BinsDynamic
};




template<class TObject>
class PointObject
: public Point
{
public:


typedef Point BaseType;

KRATOS_CLASS_POINTER_DEFINITION( PointObject );


PointObject():
BaseType()
{
}


PointObject(typename TObject::Pointer pObject):
mpObject(pObject)
{
UpdatePoint();
}



void UpdatePoint();



typename TObject::Pointer pGetObject()
{
return mpObject;
}


void pSetObject(typename TObject::Pointer pObject)
{
mpObject = pObject;
UpdatePoint();
}

private:

typename TObject::Pointer mpObject = nullptr;


}; 


template<SpatialContainer TSearchBackend>
class KRATOS_API(KRATOS_CORE) SpecializedSpatialSearch
: public SpatialSearch
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SpecializedSpatialSearch);

using BaseType = SpatialSearch;

using BaseType::PointType;

using BaseType::ElementsContainerType;
using BaseType::ElementType;
using BaseType::ElementPointerType;
using BaseType::ResultElementsContainerType;
using BaseType::VectorResultElementsContainerType;

using BaseType::NodesContainerType;
using BaseType::NodeType;
using BaseType::NodePointerType;
using BaseType::ResultNodesContainerType;
using BaseType::VectorResultNodesContainerType;

using BaseType::ConditionsContainerType;
using BaseType::ConditionType;
using BaseType::ConditionPointerType;
using BaseType::ResultConditionsContainerType;
using BaseType::VectorResultConditionsContainerType;

using BaseType::RadiusArrayType;
using BaseType::DistanceType;
using BaseType::VectorDistanceType;

using BaseType::ResultIteratorType;



SpecializedSpatialSearch()
{
mParameters = GetDefaultParameters();
}


SpecializedSpatialSearch(Parameters ThisParameters)
: mParameters(ThisParameters)
{
const Parameters default_parameters = GetDefaultParameters();

mParameters.RecursivelyValidateAndAssignDefaults(default_parameters);
}

~SpecializedSpatialSearch() override = default;




void SearchElementsInRadiusExclusive(
const ElementsContainerType& rStructureElements,
const ElementsContainerType& rInputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
) override;


void SearchElementsInRadiusExclusive(
const ElementsContainerType& rStructureElements,
const ElementsContainerType& rInputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults
) override;


void SearchNodesInRadiusExclusive(
const NodesContainerType& rStructureNodes,
const NodesContainerType& rInputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
) override;


void SearchNodesInRadiusExclusive(
const NodesContainerType& rStructureNodes,
const NodesContainerType& rInputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
) override;


void SearchConditionsInRadiusExclusive(
const ConditionsContainerType& rStructureConditions,
const ConditionsContainerType& rInputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance
) override;


void SearchConditionsInRadiusExclusive(
const ConditionsContainerType& rStructureConditions,
const ConditionsContainerType& rInputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults
) override;


std::string Info() const override
{
std::stringstream buffer;
buffer << "SpecializedSpatialSearch" ;

return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SpecializedSpatialSearch";
}

void PrintData(std::ostream& rOStream) const override
{

}


protected:


Parameters mParameters; 



Parameters GetDefaultParameters() const;

private:




template<class TContainer, class TResultType>
std::vector<typename PointObject<typename TContainer::value_type>::Pointer> PrepareSearch(
const TContainer& rStructure,
const TContainer& rInput,
TResultType& rResults,
VectorDistanceType& rResultsDistance
)
{
using ObjectType = typename TContainer::value_type;
using PointType = PointObject<ObjectType>;
using PointTypePointer = typename PointType::Pointer;
using PointVector = std::vector<PointTypePointer>;

PointVector points;
const std::size_t structure_size = rStructure.size();
points.reserve(structure_size);
const auto it_begin = rStructure.begin();
for (std::size_t i = 0; i < structure_size; ++i) {
auto it = it_begin + i;
points.push_back(PointTypePointer(new PointType(*(it.base()))));
}

const std::size_t input_size = rInput.size();
if (rResults.size() != input_size) {
rResults.resize(input_size);
}
if (rResultsDistance.size() != input_size) {
rResultsDistance.resize(input_size);
}

return points;
}


template<class TContainer, class TSpatialContainer, class TResultType>
void ParallelSearch(
const TContainer& rInput,
const RadiusArrayType& rRadius,
TSpatialContainer& rSearch,
TResultType& rResults,
VectorDistanceType& rResultsDistance
)
{
using ObjectType = typename TContainer::value_type;
using PointType = PointObject<ObjectType>;
using PointTypePointer = typename PointType::Pointer;
using PointVector = std::vector<PointTypePointer>;
using DistanceVector = std::vector<double>;
const std::size_t input_size = rInput.size();

const int allocation_size = mParameters["allocation_size"].GetInt();

IndexPartition<std::size_t>(input_size).for_each([&](std::size_t i) {
auto it = rInput.begin() + i;
PointType aux_point(*(it.base()));
PointVector results(allocation_size);
DistanceVector results_distances(allocation_size);
const std::size_t number_of_results = rSearch.SearchInRadius(aux_point, rRadius[i], results.begin(), results_distances.begin(), allocation_size);
if (number_of_results > 0) {
auto& r_results = rResults[i];
auto& r_results_distance = rResultsDistance[i];
r_results.reserve(number_of_results);
r_results_distance.reserve(number_of_results);
for (std::size_t j = 0; j < number_of_results; ++j) {
auto p_point = results[j];
r_results.push_back(p_point->pGetObject());
r_results_distance.push_back(results_distances[j]);
}
}
});
}


SpecializedSpatialSearch& operator=(SpecializedSpatialSearch const& rOther)
{
return *this;
}

SpecializedSpatialSearch(SpecializedSpatialSearch const& rOther)
{
*this = rOther;
}

}; 



template<SpatialContainer TSearchBackend>
inline std::ostream& operator << (std::ostream& rOStream,
const SpecializedSpatialSearch<TSearchBackend>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}  