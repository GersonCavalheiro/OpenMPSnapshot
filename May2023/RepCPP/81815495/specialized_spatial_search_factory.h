
#pragma once



#include "spatial_containers/specialized_spatial_search.h"

namespace Kratos
{






class SpecializedSpatialSearchFactory 
: public SpatialSearch
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SpecializedSpatialSearchFactory);


SpecializedSpatialSearchFactory()
{
Parameters default_parameters = GetDefaultParameters();
Parameters search_parameters(default_parameters["search_parameters"].WriteJsonString());
mpSpatialSearch = SpatialSearch::Pointer(new SpecializedSpatialSearch<SpatialContainer::KDTree>(search_parameters));
}

SpecializedSpatialSearchFactory(Parameters ThisParameters)
{
Parameters default_parameters = GetDefaultParameters();
ThisParameters.RecursivelyValidateAndAssignDefaults(default_parameters);
const std::string& r_container_type = ThisParameters["container_type"].GetString();
Parameters search_parameters(ThisParameters["search_parameters"].WriteJsonString());
if (r_container_type == "KDTree" || r_container_type == "kd_tree") {
mpSpatialSearch = SpatialSearch::Pointer(new SpecializedSpatialSearch<SpatialContainer::KDTree>(search_parameters));
} else if (r_container_type == "Octree" || r_container_type == "octree") {
mpSpatialSearch = SpatialSearch::Pointer(new SpecializedSpatialSearch<SpatialContainer::Octree>(search_parameters));
} else if (r_container_type == "BinsStatic" || r_container_type == "bins_static") {
mpSpatialSearch = SpatialSearch::Pointer(new SpecializedSpatialSearch<SpatialContainer::BinsStatic>(search_parameters));
} else if (r_container_type == "BinsDynamic" || r_container_type == "bins_dynamic") {
mpSpatialSearch = SpatialSearch::Pointer(new SpecializedSpatialSearch<SpatialContainer::BinsDynamic>(search_parameters));
} else {
KRATOS_ERROR << "Unknown container type: " << r_container_type << std::endl;
}
}

~SpecializedSpatialSearchFactory() override = default;



void SearchElementsInRadiusExclusive(
const ElementsContainerType& rStructureElements,
const ElementsContainerType& rInputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
) override
{
mpSpatialSearch->SearchElementsInRadiusExclusive(rStructureElements, rInputElements, rRadius, rResults, rResultsDistance);
}


void SearchElementsInRadiusExclusive(
const ElementsContainerType& rStructureElements,
const ElementsContainerType& rInputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults
) override
{
mpSpatialSearch->SearchElementsInRadiusExclusive(rStructureElements, rInputElements, rRadius, rResults);
}


void SearchNodesInRadiusExclusive(
const NodesContainerType& rStructureNodes,
const NodesContainerType& rInputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
) override
{
mpSpatialSearch->SearchNodesInRadiusExclusive(rStructureNodes, rInputNodes, rRadius, rResults, rResultsDistance);
}


void SearchNodesInRadiusExclusive(
const NodesContainerType& rStructureNodes,
const NodesContainerType& rInputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
) override
{
mpSpatialSearch->SearchNodesInRadiusExclusive(rStructureNodes, rInputNodes, rRadius, rResults);
}


void SearchConditionsInRadiusExclusive(
const ConditionsContainerType& rStructureConditions,
const ConditionsContainerType& rInputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance
) override
{
mpSpatialSearch->SearchConditionsInRadiusExclusive(rStructureConditions, rInputConditions, rRadius, rResults, rResultsDistance);
}


void SearchConditionsInRadiusExclusive(
const ConditionsContainerType& rStructureConditions,
const ConditionsContainerType& rInputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults
) override
{
mpSpatialSearch->SearchConditionsInRadiusExclusive(rStructureConditions, rInputConditions, rRadius, rResults);
}


std::string Info() const override
{
std::stringstream buffer;
buffer << "SpecializedSpatialSearchFactory" ;

return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SpecializedSpatialSearchFactory";
}

void PrintData(std::ostream& rOStream) const override
{

}

private:

SpatialSearch::Pointer mpSpatialSearch = nullptr; 



Parameters GetDefaultParameters() const
{
Parameters default_parameters = Parameters(R"(
{   "container_type"  : "KDTree",
"search_parameters" : {
"bucket_size"     : 4,
"allocation_size" : 1000
}
})" );

return default_parameters;
}

}; 



inline std::ostream& operator << (std::ostream& rOStream,
const SpecializedSpatialSearchFactory& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}  