
#pragma once




#include "utilities/parallel_utilities.h"
#include "utilities/variable_utils.h"

#include "custom_utilities/local_refine_tetrahedra_mesh_parallel_to_boundaries.hpp"

namespace Kratos
{
class LocalRefineTetrahedraMeshOnlyOnBoundaries : public LocalRefineTetrahedraMeshParallelToBoundaries
{
public:


explicit LocalRefineTetrahedraMeshOnlyOnBoundaries(ModelPart& rModelPart)
: LocalRefineTetrahedraMeshParallelToBoundaries(rModelPart)
{
}

~LocalRefineTetrahedraMeshOnlyOnBoundaries() 
= default;

void SearchEdgeToBeRefined(
ModelPart& rThisModelPart,
compressed_matrix<int>& rCoord
) override
{
KRATOS_TRY;
for (auto& r_elem: rThisModelPart.Elements()) {
if (r_elem.GetValue(SPLIT_ELEMENT)) {
Element::GeometryType& r_geom = r_elem.GetGeometry(); 
for (unsigned int i = 0; i < r_geom.size(); i++) {
int index_i = r_geom[i].Id() - 1;
bool is_boundary_i = r_geom[i].Is(BOUNDARY);
for (unsigned int j = 0; j < r_geom.size(); j++) {
int index_j = r_geom[j].Id() - 1;
bool is_boundary_j = r_geom[j].Is(BOUNDARY);
if (index_j > index_i && (is_boundary_j&&is_boundary_i)) {
rCoord(index_i, index_j) = -2;
}
}
}
}
}


KRATOS_CATCH("");
}


protected:






private:







};

} 
