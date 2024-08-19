
#pragma once



#include "containers/model.h"
#include "includes/define.h"
#include "includes/key_hash.h"
#include "modified_shape_functions/modified_shape_functions.h"

namespace Kratos
{







class KRATOS_API(KRATOS_CORE) ShiftedBoundaryMeshlessInterfaceUtility
{
public:


enum class ExtensionOperator
{
MLS,
RBF,
GradientBased
};

using IndexType = ModelPart::IndexType;

using NodeType = ModelPart::NodeType;

using GeometryType = ModelPart::GeometryType;

using ShapeFunctionsGradientsType = GeometryType::ShapeFunctionsGradientsType;

using ModifiedShapeFunctionsFactoryType = std::function<ModifiedShapeFunctions::UniquePointer(const GeometryType::Pointer, const Vector&)>;

using MeshlessShapeFunctionsFunctionType = std::function<void(const Matrix&, const array_1d<double,3>&, const double, Vector&)>;

using MLSShapeFunctionsAndGradientsFunctionType = std::function<void(const Matrix&, const array_1d<double,3>&, const double, Vector&, Matrix&)>;

using ElementSizeFunctionType = std::function<double(const GeometryType&)>;

using NodesCloudSetType = std::unordered_set<NodeType::Pointer, SharedPointerHasher<NodeType::Pointer>, SharedPointerComparator<NodeType::Pointer>>;

using CloudDataVectorType = DenseVector<std::pair<NodeType::Pointer, double>>;

using NodesCloudMapType = std::unordered_map<NodeType::Pointer, CloudDataVectorType, SharedPointerHasher<NodeType::Pointer>, SharedPointerComparator<NodeType::Pointer>>;


KRATOS_CLASS_POINTER_DEFINITION(ShiftedBoundaryMeshlessInterfaceUtility);


ShiftedBoundaryMeshlessInterfaceUtility(
Model& rModel,
Parameters ThisParameters);

ShiftedBoundaryMeshlessInterfaceUtility(ShiftedBoundaryMeshlessInterfaceUtility const& rOther) = delete;


ShiftedBoundaryMeshlessInterfaceUtility& operator=(ShiftedBoundaryMeshlessInterfaceUtility const& rOther) = delete;


void CalculateExtensionOperator();



const Parameters GetDefaultParameters() const;


std::string Info() const
{
return "ShiftedBoundaryMeshlessInterfaceUtility";
}

void PrintInfo(std::ostream& rOStream) const
{
rOStream << "ShiftedBoundaryMeshlessInterfaceUtility";
}

void PrintData(std::ostream& rOStream) const
{
}


private:


ModelPart* mpModelPart = nullptr;
ModelPart* mpBoundarySubModelPart = nullptr;

bool mConformingBasis;

const Variable<double>* mpLevelSetVariable;

ExtensionOperator mExtensionOperator;

std::size_t mMLSExtensionOperatorOrder;

const Condition* mpConditionPrototype;




void CalculateGradientBasedConformingExtensionBasis();


void CalculateMeshlessBasedConformingExtensionBasis();


void CalculateMeshlessBasedNonConformingExtensionBasis();


void SetInterfaceFlags();


MLSShapeFunctionsAndGradientsFunctionType GetMLSShapeFunctionsAndGradientsFunction() const;


MeshlessShapeFunctionsFunctionType GetMLSShapeFunctionsFunction() const;


MeshlessShapeFunctionsFunctionType GetRBFShapeFunctionsFunction() const;


ElementSizeFunctionType GetElementSizeFunction(const GeometryType& rGeometry);


void SetSplitElementSupportCloud(
const Element& rSplitElement,
PointerVector<NodeType>& rCloudNodes,
Matrix& rCloudCoordinates);


void SetNegativeNodeSupportCloud(
const NodeType& rNegativeNode,
PointerVector<NodeType>& rCloudNodes,
Matrix& rCloudCoordinates);


double CalculateKernelRadius(
const Matrix& rCloudCoordinates,
const array_1d<double,3>& rOrigin);


std::size_t GetRequiredNumberOfPoints();


std::unordered_map<std::size_t, std::map<std::size_t, Vector>> SetSurrogateBoundaryNodalGradientWeights();




}; 

} 
