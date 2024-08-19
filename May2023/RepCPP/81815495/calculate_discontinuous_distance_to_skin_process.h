
#if !defined(KRATOS_CALCULATE_DISCONTINUOUS_DISTANCE_TO_SKIN_PROCESS_H_INCLUDED )
#define  KRATOS_CALCULATE_DISCONTINUOUS_DISTANCE_TO_SKIN_PROCESS_H_INCLUDED

#include <string>
#include <iostream>


#include "geometries/plane_3d.h"
#include "includes/checks.h"
#include "processes/process.h"
#include "processes/find_intersected_geometrical_objects_process.h"
#include "utilities/variable_utils.h"
#include "utilities/pointer_communicator.h"

namespace Kratos
{


class KRATOS_API(KRATOS_CORE) CalculateDiscontinuousDistanceToSkinProcessFlags
{
public:
KRATOS_DEFINE_LOCAL_FLAG(CALCULATE_ELEMENTAL_EDGE_DISTANCES); 
KRATOS_DEFINE_LOCAL_FLAG(CALCULATE_ELEMENTAL_EDGE_DISTANCES_EXTRAPOLATED); 
KRATOS_DEFINE_LOCAL_FLAG(USE_POSITIVE_EPSILON_FOR_ZERO_VALUES); 
};


template<std::size_t TDim = 3>
class KRATOS_API(KRATOS_CORE) CalculateDiscontinuousDistanceToSkinProcess : public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(CalculateDiscontinuousDistanceToSkinProcess);


CalculateDiscontinuousDistanceToSkinProcess(
ModelPart& rVolumePart,
ModelPart& rSkinPart);

CalculateDiscontinuousDistanceToSkinProcess(
ModelPart& rVolumePart,
ModelPart& rSkinPart,
const Flags rOptions);

CalculateDiscontinuousDistanceToSkinProcess(
ModelPart& rVolumePart,
ModelPart& rSkinPart,
Parameters rParameters);

~CalculateDiscontinuousDistanceToSkinProcess() override;


CalculateDiscontinuousDistanceToSkinProcess() = delete;

CalculateDiscontinuousDistanceToSkinProcess(CalculateDiscontinuousDistanceToSkinProcess const& rOther) = delete;

CalculateDiscontinuousDistanceToSkinProcess& operator=(CalculateDiscontinuousDistanceToSkinProcess const& rOther) = delete;

FindIntersectedGeometricalObjectsProcess mFindIntersectedObjectsProcess;



virtual void Initialize();


virtual void FindIntersections();


virtual std::vector<PointerVector<GeometricalObject>>& GetIntersections();


virtual void CalculateDistances(std::vector<PointerVector<GeometricalObject>>& rIntersectedObjects);


void Clear() override;


void Execute() override;


void CalculateEmbeddedVariableFromSkin(
const Variable<double> &rVariable,
const Variable<double> &rEmbeddedVariable);


void CalculateEmbeddedVariableFromSkin(
const Variable<array_1d<double,3>> &rVariable,
const Variable<array_1d<double,3>> &rEmbeddedVariable);


const Parameters GetDefaultParameters() const override;




std::string Info() const override;

void PrintInfo(std::ostream& rOStream) const override;

void PrintData(std::ostream& rOStream) const override;

protected:

const Variable<Vector>* mpElementalDistancesVariable = &ELEMENTAL_DISTANCES;




Plane3D SetIntersectionPlane(const std::vector<array_1d<double,3>> &rIntPtsVector);


double CalculateCharacteristicLength();

private:

ModelPart& mrSkinPart;
ModelPart& mrVolumePart;

Flags mOptions;

static const std::size_t mNumNodes = TDim + 1;
static const std::size_t mNumEdges = (TDim == 2) ? 3 : 6;

const double mZeroToleranceMultiplier = 1e3;
bool mDetectedZeroDistanceValues = false;
bool mAreNeighboursComputed = false;
bool mCalculateElementalEdgeDistances = false;
bool mCalculateElementalEdgeDistancesExtrapolated = false;
bool mUsePositiveEpsilonForZeroValues = true;


const Variable<Vector>* mpElementalEdgeDistancesVariable = &ELEMENTAL_EDGE_DISTANCES;
const Variable<Vector>* mpElementalEdgeDistancesExtrapolatedVariable = &ELEMENTAL_EDGE_DISTANCES_EXTRAPOLATED;
const Variable<array_1d<double, 3>>* mpEmbeddedVelocityVariable = &EMBEDDED_VELOCITY;



void CalculateElementalDistances(
Element& rElement1,
PointerVector<GeometricalObject>& rIntersectedObjects);


void CalculateElementalAndEdgeDistances(
Element& rElement1,
PointerVector<GeometricalObject>& rIntersectedObjects);


unsigned int ComputeEdgesIntersections(
Element& rElement1,
const PointerVector<GeometricalObject>& rIntersectedObjects,
const Element::GeometryType::GeometriesArrayType& rEdgesContainer,
array_1d<double,mNumEdges> &rCutEdgesRatioVector,
array_1d<double,mNumEdges> &rCutExtraEdgesRatioVector,
std::vector<array_1d <double,3> > &rIntersectionPointsArray);


int ComputeEdgeIntersection(
const Element::GeometryType& rIntObjGeometry,
const Element::NodeType& rEdgePoint1,
const Element::NodeType& rEdgePoint2,
Point& rIntersectionPoint);


bool CheckIfPointIsRepeated(
const array_1d<double,3>& rIntersectionPoint,
const std::vector<array_1d<double,3>>&  rIntersectionPointsVector,
const double& rEdgeTolerance);


void ComputeIntersectionNormal(
const Element::GeometryType& rGeometry,
const Vector& rElementalDistances,
array_1d<double,3> &rNormal);


void ComputeIntersectionPlaneElementalDistances(
Element& rElement,
const PointerVector<GeometricalObject>& rIntersectedObjects,
const std::vector<array_1d<double,3>>& rIntersectionPointsCoordinates);


void ComputePlaneApproximation(
const Element& rElement1,
const std::vector< array_1d<double,3> >& rPointsCoord,
array_1d<double,3>& rPlaneBasePointCoords,
array_1d<double,3>& rPlaneNormal);



void ComputeElementalDistancesFromPlaneApproximation(
Element& rElement,
Vector& rElementalDistances,
const std::vector<array_1d<double,3>>& rPointVector);


void ReplaceZeroDistances(Vector& rElementalDistances);


void CorrectDistanceOrientation(
const Element::GeometryType& rGeometry,
const PointerVector<GeometricalObject>& rIntersectedObjects,
Vector& rElementalDistances);


void inline ComputeIntersectionNormalFromGeometry(
const Element::GeometryType &rGeometry,
array_1d<double,3> &rIntObjNormal);


void ComputeExtrapolatedEdgesIntersectionsIfIncised(
const Element& rElement,
const Element::GeometryType::GeometriesArrayType& rEdgesContainer,
unsigned int &rNumCutEdges,
array_1d<double,mNumEdges>& rCutEdgesRatioVector,
array_1d<double,3> &rExtraGeomNormal,
array_1d<double,mNumEdges>& rCutExtraEdgesRatioVector);


void ComputeExtrapolatedGeometryIntersections(
const Element& rElement,
const Element::GeometryType::GeometriesArrayType& rEdgesContainer,
unsigned int& rNumCutEdges,
array_1d<double,mNumEdges>& rCutEdgesRatioVector,
array_1d<double,3>& rExtraGeomNormal,
array_1d<double,mNumEdges>& rCutExtraEdgesRatioVector);


void ComputeElementalDistancesFromEdgeRatios(
Element& rElement,
const PointerVector<GeometricalObject>& rIntersectedObjects,
const Element::GeometryType::GeometriesArrayType& rEdgesContainer,
const array_1d<double,mNumEdges> &rCutEdgesRatioVector,
const array_1d<double,mNumEdges> &rCutExtraEdgesRatioVector);


void ConvertRatiosToIntersectionPoints(
const Element::GeometryType& rGeometry,
const Element::GeometryType::GeometriesArrayType& rEdgesContainer,
const array_1d<double,mNumEdges> &rEdgeRatiosVector,
std::vector<array_1d <double,3> > &rIntersectionPointsVector);


double ConvertIntersectionPointToEdgeRatio(
const Geometry<Node >& rEdge,
const array_1d<double,3>& rIntersectionPoint);


array_1d<double,3> ConvertEdgeRatioToIntersectionPoint(
const Geometry<Node >& rEdge,
const double& rEdgeRatio);


bool CheckIfCutEdgesShareNode(
const Element& rElement,
const Element::GeometryType::GeometriesArrayType& rEdgesContainer,
const array_1d<double,mNumEdges>& rCutEdgesRatioVector) const;


template<class TVarType>
void CalculateEmbeddedVariableFromSkinSpecialization(
const Variable<TVarType> &rVariable,
const Variable<TVarType> &rEmbeddedVariable)
{
const auto &r_int_obj_vect= this->GetIntersections();
const int n_elems = mrVolumePart.NumberOfElements();

KRATOS_ERROR_IF((mrSkinPart.NodesBegin())->SolutionStepsDataHas(rVariable) == false)
<< "Skin model part solution step data missing variable: " << rVariable << std::endl;

VariableUtils().SetNonHistoricalVariableToZero(rEmbeddedVariable, mrVolumePart.Elements());

#pragma omp parallel for schedule(dynamic)
for (int i_elem = 0; i_elem < n_elems; ++i_elem) {
if (r_int_obj_vect[i_elem].size() != 0) {
unsigned int n_int_edges = 0;
auto it_elem = mrVolumePart.ElementsBegin() + i_elem;
auto &r_geom = it_elem->GetGeometry();
const auto edges = r_geom.GenerateEdges();

for (unsigned int i_edge = 0; i_edge < r_geom.EdgesNumber(); ++i_edge) {
unsigned int n_int_obj = 0;
TVarType i_edge_val = rEmbeddedVariable.Zero();

for (auto &r_int_obj : r_int_obj_vect[i_elem]) {
Point intersection_point;
const int is_intersected = this->ComputeEdgeIntersection(
r_int_obj.GetGeometry(),
edges[i_edge][0],
edges[i_edge][1],
intersection_point);

if (is_intersected == 1) {
n_int_obj++;
array_1d<double,3> local_coords;
r_int_obj.GetGeometry().PointLocalCoordinates(local_coords, intersection_point);
Vector int_obj_N;
r_int_obj.GetGeometry().ShapeFunctionsValues(int_obj_N, local_coords);
for (unsigned int i_node = 0; i_node < r_int_obj.GetGeometry().PointsNumber(); ++i_node) {
i_edge_val += r_int_obj.GetGeometry()[i_node].FastGetSolutionStepValue(rVariable) * int_obj_N[i_node];
}
}
}

if (n_int_obj != 0) {
n_int_edges++;
it_elem->GetValue(rEmbeddedVariable) += i_edge_val / n_int_obj;
}
}

if (n_int_edges != 0) {
it_elem->GetValue(rEmbeddedVariable) /= n_int_edges;
}
}
}
};


void SetToSplitFlag(
Element& rElement,
const double ZeroTolerance);


void CheckAndCorrectEdgeDistances();


GlobalPointerCommunicator<Element>::Pointer CreatePointerCommunicator();

}; 


inline std::istream& operator >> (
std::istream& rIStream,
CalculateDiscontinuousDistanceToSkinProcess<>& rThis);

inline std::ostream& operator << (
std::ostream& rOStream,
const CalculateDiscontinuousDistanceToSkinProcess<>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}  

#endif 
