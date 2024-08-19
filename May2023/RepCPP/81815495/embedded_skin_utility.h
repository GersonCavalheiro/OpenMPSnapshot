
#if !defined(KRATOS_GENERATE_EMBEDDED_SKIN_UTILITY_H_INCLUDED )
#define  KRATOS_GENERATE_EMBEDDED_SKIN_UTILITY_H_INCLUDED


#include <string>
#include <iostream>




#include "includes/define.h"
#include "includes/model_part.h"
#include "geometries/geometry_data.h"
#include "modified_shape_functions/modified_shape_functions.h"
#include "utilities/binbased_fast_point_locator.h"
#include "utilities/divide_geometry.h"
#include "utilities/math_utils.h"


namespace Kratos
{







template<std::size_t TDim>
class KRATOS_API(KRATOS_CORE) EmbeddedSkinUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION(EmbeddedSkinUtility);

typedef std::unordered_map<
Node::Pointer,
std::tuple< const Element::Pointer, const unsigned int >,
SharedPointerHasher<Node::Pointer>,
SharedPointerComparator<Node::Pointer> > EdgeNodesMapType;


enum LevelSetTypeEnum
{
Continuous = 1,
Discontinuous = 2
};


EmbeddedSkinUtility(
ModelPart &rModelPart,
ModelPart &rSkinModelPart,
const std::string LevelSetType = "continuous",
const std::vector<std::string>& InterpolatedSkinVariables = {}) :
mrModelPart(rModelPart),
mrSkinModelPart(rSkinModelPart),
mLevelSetType(LevelSetType == "continuous" ? Continuous : Discontinuous),
mrConditionPrototype(KratosComponents<Condition>::Get(this->GetConditionType())),
mInterpolatedSkinVariables(InterpolatedSkinVariables) {};

virtual ~EmbeddedSkinUtility() = default;





void GenerateSkin();


void InterpolateMeshVariableToSkin(
const Variable<double> &rMeshVariable,
const Variable<double> &rSkinVariable);


void InterpolateMeshVariableToSkin(
const Variable<array_1d<double,3>> &rMeshVariable,
const Variable<array_1d<double,3>> &rSkinVariable);


void InterpolateDiscontinuousMeshVariableToSkin(
const Variable<double> &rMeshVariable,
const Variable<double> &rSkinVariable,
const std::string &rInterfaceSide);


void InterpolateDiscontinuousMeshVariableToSkin(
const Variable<array_1d<double,3>> &rMeshVariable,
const Variable<array_1d<double,3>> &rSkinVariable,
const std::string &rInterfaceSide);









private:



ModelPart &mrModelPart;
ModelPart &mrSkinModelPart;
EdgeNodesMapType mEdgeNodesMap;
const LevelSetTypeEnum mLevelSetType;
const Condition &mrConditionPrototype;
std::vector<std::string> mInterpolatedSkinVariables;





void Clear();


void ComputeElementSkin(
const Element::Pointer pElement,
const Vector &rNodalDistances,
unsigned int &rTempNodeId,
unsigned int &rTempCondId,
Properties::Pointer pCondProp,
ModelPart::NodesContainerType &rNewNodesVect,
ModelPart::ConditionsContainerType &rNewCondsVect);


bool inline ElementIsSplit(
const Geometry<Node> &rGeometry,
const Vector &rNodalDistances);


template<class TVarType>
void InterpolateMeshVariableToSkinSpecialization(
const Variable<TVarType> &rMeshVariable,
const Variable<TVarType> &rSkinVariable,
const std::string &rInterfaceSide = "positive")
{
KRATOS_ERROR_IF((mrModelPart.NodesBegin())->SolutionStepsDataHas(rMeshVariable) == false)
<< "Mesh model part solution step data missing variable: " << rMeshVariable << std::endl;
KRATOS_ERROR_IF((mrSkinModelPart.NodesBegin())->SolutionStepsDataHas(rSkinVariable) == false)
<< "Generated skin model part solution step data missing variable: " << rSkinVariable << std::endl;

KRATOS_ERROR_IF(mrModelPart.NumberOfElements() == 0) << "Mesh model part has no elements.";

unsigned int i_edge;
Element::Pointer p_elem;
#pragma omp parallel for private (i_edge, p_elem)
for (int i_node = 0; i_node < static_cast<int>(mrSkinModelPart.NumberOfNodes()); ++i_node) {
auto it_node = mrSkinModelPart.NodesBegin() + i_node;
Node::Pointer p_node = &(*it_node);

const auto i_node_info = mEdgeNodesMap.find(p_node);
if (i_node_info != mEdgeNodesMap.end()){
std::tie(p_elem, i_edge) = std::get<1>(*i_node_info);

const auto p_elem_geom = p_elem->pGetGeometry();
const auto elem_dist = this->SetDistancesVector(*p_elem);
const auto p_mod_sh_func = pCreateModifiedShapeFunctions(p_elem_geom, elem_dist);

const auto edge_sh_func = this->GetModifiedShapeFunctionsValuesOnEdge(
p_mod_sh_func,
rInterfaceSide);

const auto edge_N = row(edge_sh_func, i_edge);
const auto &r_elem_geom = p_elem->GetGeometry();
auto &r_value = it_node->FastGetSolutionStepValue(rSkinVariable);
r_value = rSkinVariable.Zero();
for (unsigned int i_elem_node = 0; i_elem_node < r_elem_geom.PointsNumber(); ++i_elem_node) {
r_value += edge_N[i_elem_node] * r_elem_geom[i_elem_node].FastGetSolutionStepValue(rMeshVariable);
}
} else{
KRATOS_ERROR << "Intersected edge node " << it_node->Id() << " not found in intersected edges nodes map" << std::endl;
}
}
};


void RenumberAndAddSkinEntities(
const ModelPart::NodesContainerType &rNewNodesVect,
const ModelPart::ConditionsContainerType &rNewCondsVect);


const Vector SetDistancesVector(const Element &rElement);


DivideGeometry<Node>::Pointer SetDivideGeometryUtility(
const Geometry<Node> &rGeometry,
const Vector &rNodalDistances);


Geometry< Node >::Pointer pCreateNewConditionGeometry(
const GeometryData::KratosGeometryType &rOriginGeometryType,
const Condition::NodesArrayType &rNewNodesArray);


Condition::Pointer pCreateNewCondition(
const GeometryData::KratosGeometryType &rOriginGeometryType,
const Condition::NodesArrayType &rNewNodesArray,
const unsigned int &rConditionId,
const Properties::Pointer pConditionProperties);


static const std::string GetConditionType();


Properties::Pointer SetSkinEntitiesProperties();


ModifiedShapeFunctions::UniquePointer pCreateModifiedShapeFunctions(
const Geometry<Node>::Pointer pGeometry,
const Vector& rNodalDistances);


Matrix GetModifiedShapeFunctionsValues(
const ModifiedShapeFunctions::UniquePointer &rpModifiedShapeFunctions,
const std::string &rInterfaceSide) const;


Matrix GetModifiedShapeFunctionsValuesOnEdge(
const ModifiedShapeFunctions::UniquePointer &rpModifiedShapeFunctions,
const std::string &rInterfaceSide) const;






EmbeddedSkinUtility& operator=(EmbeddedSkinUtility const& rOther) = delete;

EmbeddedSkinUtility(EmbeddedSkinUtility const& rOther) = delete;

}; 




}  

#endif 
