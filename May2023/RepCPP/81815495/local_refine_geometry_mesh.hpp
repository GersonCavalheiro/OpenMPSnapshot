
#pragma once





#include "includes/variables.h"
#include "includes/deprecated_variables.h"
#include "includes/model_part.h"
#include "includes/global_pointer_variables.h"
#include "geometries/geometry.h"

namespace Kratos
{
class KRATOS_API(MESHING_APPLICATION) LocalRefineGeometryMesh
{
public:

typedef ModelPart::NodesContainerType NodesArrayType;
typedef ModelPart::ElementsContainerType ElementsArrayType;
typedef ModelPart::ConditionsContainerType ConditionsArrayType;
typedef std::vector<Matrix> Matrix_Order_Tensor;
typedef std::vector<Vector> Vector_Order_Tensor;
typedef std::vector<Vector_Order_Tensor> Node_Vector_Order_Tensor;
typedef Node PointType;
typedef Node ::Pointer PointPointerType;
typedef std::vector<PointType::Pointer> PointVector;
typedef PointVector::iterator PointIterator;


LocalRefineGeometryMesh(ModelPart& model_part) : mModelPart(model_part)
{

}

virtual ~LocalRefineGeometryMesh()
= default;





virtual void LocalRefineMesh(
bool refine_on_reference,
bool interpolate_internal_variables
);



virtual void CSRRowMatrix(
ModelPart& this_model_part,
compressed_matrix<int>& Coord
);



virtual void SearchEdgeToBeRefined(
ModelPart& this_model_part,
compressed_matrix<int>& Coord
);



virtual void CreateListOfNewNodes(
ModelPart& this_model_part,
compressed_matrix<int>& Coord,
std::vector<int> &List_New_Nodes,
std::vector<array_1d<int, 2 > >& Position_Node
);



virtual void CalculateCoordinateAndInsertNewNodes(
ModelPart& this_model_part,
const std::vector<array_1d<int, 2 > >& Position_Node,
const std::vector<int> &List_New_Nodes
);



virtual void EraseOldElementAndCreateNewElement(
ModelPart& this_model_part,
const compressed_matrix<int>& Coord,
PointerVector< Element >& New_Elements,
bool interpolate_internal_variables
);



virtual void EraseOldConditionsAndCreateNew(
ModelPart& this_model_part,
const compressed_matrix<int>& Coord
);



virtual void CalculateEdges(
Element::GeometryType& geom,
const compressed_matrix<int>& Coord,
int* edge_ids,
std::vector<int> & aux
);



virtual void RenumeringElementsAndNodes(
ModelPart& this_model_part,
PointerVector< Element >& New_Elements
);



inline void CreatePartition(
unsigned int number_of_threads,
const int number_of_rows,
vector<unsigned int>& partitions
);



template<typename TGeometricalObjectPointerType>
void InterpolateInteralVariables(
const int& number_elem,
const TGeometricalObjectPointerType father_elem,
TGeometricalObjectPointerType child_elem,
const ProcessInfo& rCurrentProcessInfo
)
{
std::vector<Vector> values;
father_elem->CalculateOnIntegrationPoints(INTERNAL_VARIABLES, values, rCurrentProcessInfo);
child_elem->SetValuesOnIntegrationPoints(INTERNAL_VARIABLES, values, rCurrentProcessInfo);
}

virtual void UpdateSubModelPartNodes(ModelPart &rModelPart);

virtual void ResetFatherNodes(ModelPart &rModelPart);

protected:


ModelPart& mModelPart;       
int mCurrentRefinementLevel; 


template<typename TIteratorType>
void SearchEdgeToBeRefinedGeneric(
TIteratorType GeometricalObjectsBegin,
TIteratorType GeometricalObjectsEnd,
compressed_matrix<int>& rCoord
)
{
KRATOS_TRY;

for (TIteratorType it = GeometricalObjectsBegin; it != GeometricalObjectsEnd; ++it) {
if (it->GetValue(SPLIT_ELEMENT)) {
auto& r_geom = it->GetGeometry(); 
for (unsigned int i = 0; i < r_geom.size(); i++) {
int index_i = r_geom[i].Id() - 1;
for (unsigned int j = 0; j < r_geom.size(); j++) {
int index_j = r_geom[j].Id() - 1;
if (index_j > index_i)
{
rCoord(index_i, index_j) = -2;
}
}
}
}
}

KRATOS_CATCH("");
}





private:







};

} 
