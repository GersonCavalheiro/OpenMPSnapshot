
#pragma once



#include "includes/model_part.h"

namespace Kratos
{






class KRATOS_API(KRATOS_CORE) SpatialSearch
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SpatialSearch);

static constexpr std::size_t Dimension = 3;

static constexpr std::size_t MAX_LEVEL = 16;

static constexpr std::size_t MIN_LEVEL = 2;

using PointType = Point;

using ElementsContainerType = ModelPart::ElementsContainerType;
using ElementType = ModelPart::ElementType;
using ElementPointerType = ModelPart::ElementType::Pointer;
using ResultElementsContainerType = ElementsContainerType::ContainerType;
using VectorResultElementsContainerType = std::vector<ResultElementsContainerType>;

using NodesContainerType = ModelPart::NodesContainerType;
using NodeType = ModelPart::NodeType;
using NodePointerType = ModelPart::NodeType::Pointer;
using ResultNodesContainerType = NodesContainerType::ContainerType;
using VectorResultNodesContainerType = std::vector<ResultNodesContainerType>;

using ConditionsContainerType = ModelPart::ConditionsContainerType;
using ConditionType = ModelPart::ConditionType;
using ConditionPointerType = ModelPart::ConditionType::Pointer;
using ResultConditionsContainerType = ConditionsContainerType::ContainerType;
using VectorResultConditionsContainerType = std::vector<ResultConditionsContainerType>;

using RadiusArrayType = std::vector<double>;
using DistanceType = std::vector<double>;
using VectorDistanceType = std::vector<DistanceType>;

using ResultIteratorType = ElementsContainerType::ContainerType::iterator;


SpatialSearch(){}

virtual ~SpatialSearch(){}



/
virtual void SearchElementsInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsInRadiusExclusive (
ModelPart& rModelPart,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchElementsInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsInRadiusInclusive (
ModelPart& rModelPart,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchElementsInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults
);


virtual void SearchElementsInRadiusExclusive (
ModelPart& rModelPart,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults
);


virtual void SearchElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults
);


virtual void SearchElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults
);

/
virtual void SearchElementsInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchElementsInRadiusInclusive (
ModelPart& rModelPart,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);

/
virtual void SearchNodesInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchNodesInRadiusExclusive (
ModelPart& rModelPart,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchNodesInRadiusExclusive (
NodesContainerType const& StructureNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchNodesInRadiusExclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchNodesInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchNodesInRadiusInclusive (
ModelPart& rModelPart,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchNodesInRadiusInclusive (
NodesContainerType const& StructureNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchNodesInRadiusInclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


/
virtual void SearchNodesInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchNodesInRadiusExclusive (
ModelPart& rModelPart,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchNodesInRadiusExclusive (
NodesContainerType const& StructureNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchNodesInRadiusExclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);

/
virtual void SearchNodesInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchNodesInRadiusInclusive (
ModelPart& rModelPart,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchNodesInRadiusInclusive (
NodesContainerType const& StructureNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchNodesInRadiusInclusive (
NodesContainerType const& StructureNodes,
NodesContainerType const& InputNodes,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);

/
virtual void SearchConditionsInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsInRadiusExclusive (
ModelPart& rModelPart,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsInRadiusExclusive (
ConditionsContainerType const& StructureConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsInRadiusExclusive (
ConditionsContainerType const& StructureConditions,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchConditionsInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsInRadiusInclusive (
ModelPart& rModelPart,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsInRadiusInclusive (
ConditionsContainerType const& StructureConditions,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsInRadiusInclusive (
ConditionsContainerType const& StructureConditions,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchConditionsInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults
);


virtual void SearchConditionsInRadiusExclusive (
ModelPart& rModelPart,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults
);


virtual void SearchConditionsInRadiusExclusive (
ConditionsContainerType const& StructureConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults
);


virtual void SearchConditionsInRadiusExclusive (
ConditionsContainerType const& StructureConditions,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultConditionsContainerType& rResults
);

/
virtual void SearchConditionsInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchConditionsInRadiusInclusive (
ModelPart& rModelPart,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchConditionsInRadiusInclusive (
ConditionsContainerType const& StructureConditions,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);


virtual void SearchConditionsInRadiusInclusive (
ConditionsContainerType const& StructureConditions,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultNodesContainerType& rResults
);

/
virtual void SearchConditionsOverElementsInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsOverElementsInRadiusExclusive (
ModelPart& rModelPart,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsOverElementsInRadiusExclusive (
ElementsContainerType const& StructureElements,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchConditionsOverElementsInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsOverElementsInRadiusInclusive (
ModelPart& rModelPart,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchConditionsOverElementsInRadiusInclusive (
ElementsContainerType const& StructureElements,
ConditionsContainerType const& InputConditions,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchElementsOverConditionsInRadiusExclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsOverConditionsInRadiusExclusive (
ModelPart& rModelPart,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsOverConditionsInRadiusExclusive (
ConditionsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);

/
virtual void SearchElementsOverConditionsInRadiusInclusive (
ModelPart& rModelPart,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsOverConditionsInRadiusInclusive (
ModelPart& rModelPart,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);


virtual void SearchElementsOverConditionsInRadiusInclusive (
ConditionsContainerType const& StructureElements,
ElementsContainerType const& InputElements,
const RadiusArrayType& rRadius,
VectorResultElementsContainerType& rResults,
VectorDistanceType& rResultsDistance
);




virtual std::string Info() const
{
std::stringstream buffer;
buffer << "SpatialSearch" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const {rOStream << "SpatialSearch";}

virtual void PrintData(std::ostream& rOStream) const {}


protected:







private:







SpatialSearch& operator=(SpatialSearch const& rOther)
{
return *this;
}

SpatialSearch(SpatialSearch const& rOther)
{
*this = rOther;
}

}; 



inline std::ostream& operator << (std::ostream& rOStream, 
const SpatialSearch& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


