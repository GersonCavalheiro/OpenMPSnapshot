
#if !defined(KRATOS_MORTAR_UTILITIES)
#define KRATOS_MORTAR_UTILITIES

#include <numeric>
#include <unordered_map>


#include "includes/variables.h"
#include "includes/node.h"
#include "geometries/geometry.h"

namespace Kratos
{





class ModelPart; 


struct MortarUtilitiesSettings
{
constexpr static bool SaveAsHistoricalVariable = true;
constexpr static bool SaveAsNonHistoricalVariable = false;
};


namespace MortarUtilities
{

typedef Node                                              NodeType;
typedef Point                                               PointType;
typedef PointType::CoordinatesArrayType          CoordinatesArrayType;

typedef Geometry<NodeType>                               GeometryType;
typedef Geometry<PointType>                         GeometryPointType;

typedef std::size_t                                         IndexType;

typedef std::size_t                                          SizeType;

typedef std::unordered_map<IndexType, IndexType>               IntMap;



bool KRATOS_API(KRATOS_CORE) LengthCheck(
const GeometryPointType& rGeometryLine,
const double Tolerance = 1.0e-6
);


bool KRATOS_API(KRATOS_CORE) HeronCheck(const GeometryPointType& rGeometryTriangle);


bool KRATOS_API(KRATOS_CORE) HeronCheck(
const PointType& rPointOrig1,
const PointType& rPointOrig2,
const PointType& rPointOrig3
);


void KRATOS_API(KRATOS_CORE) RotatePoint(
PointType& rPointToRotate,
const PointType& rPointReferenceRotation,
const array_1d<double, 3>& rSlaveTangentXi,
const array_1d<double, 3>& rSlaveTangentEta,
const bool Inversed
);


array_1d<double,3> KRATOS_API(KRATOS_CORE) GaussPointUnitNormal(
const Vector& rN,
const GeometryType& rGeometry
);


template <typename TType>
std::vector<std::size_t> SortIndexes(const std::vector<TType> &rThisVector) {
std::vector<std::size_t> idx(rThisVector.size());
iota(idx.begin(), idx.end(), 0);

std::sort(idx.begin(), idx.end(),
[&rThisVector](std::size_t i1, std::size_t i2) {return rThisVector[i1] < rThisVector[i2];});

return idx;
}


void KRATOS_API(KRATOS_CORE) ComputeNodesMeanNormalModelPart(
ModelPart& rModelPart,
const bool ComputeConditions = true
);


void KRATOS_API(KRATOS_CORE) ComputeNodesTangentModelPart(
ModelPart& rModelPart,
const Variable<array_1d<double, 3>>* pSlipVariable = NULL,
const double SlipCoefficient = 1.0,
const bool SlipAlways = false
);


void KRATOS_API(KRATOS_CORE) ComputeNodesTangentFromNormalModelPart(ModelPart& rModelPart);


void KRATOS_API(KRATOS_CORE) ComputeTangentsFromNormal(
NodeType& rNode,
const array_1d<double, 3>& rNormal,
const std::size_t Dimension = 3
);


void KRATOS_API(KRATOS_CORE) ComputeTangentNodeWithLMAndSlip(
NodeType& rNode,
const std::size_t StepLM = 0,
const Variable<array_1d<double, 3>>* pSlipVariable = NULL,
const double SlipCoefficient = 1.0,
const std::size_t Dimension = 3
);


void KRATOS_API(KRATOS_CORE) ComputeTangentNodeWithSlip(
NodeType& rNode,
const std::size_t StepLM = 0,
const Variable<array_1d<double, 3>>* pSlipVariable = NULL,
const double SlipCoefficient = 1.0,
const std::size_t Dimension = 3
);


template<class TContainerType>
void InvertNormalForFlag(
TContainerType& rContainer,
const Flags Flag
)
{
bool to_invert = false;
const auto it_cont_begin = rContainer.begin();
#pragma omp parallel for firstprivate(to_invert)
for(int i = 0; i < static_cast<int>(rContainer.size()); ++i) {
auto it_cont = it_cont_begin + i;
to_invert = Flag == Flags() ? true : it_cont->IsDefined(Flag) ? it_cont->Is(Flag) : false;

if (to_invert) {
GeometryType& r_geometry = it_cont->GetGeometry();

auto& data_geom = r_geometry.GetContainer();
std::reverse(data_geom.begin(), data_geom.end());
}
}
}


template<class TContainerType>
void InvertNormal(TContainerType& rContainer)
{
InvertNormalForFlag(rContainer, Flags());
}



template< SizeType TDim, SizeType TNumNodes>
BoundedMatrix<double, TNumNodes, TDim> GetCoordinates(
const GeometryType& rGeometry,
const bool Current = true,
const IndexType Step = 0
) {

BoundedMatrix<double, TNumNodes, TDim> coordinates;
array_1d<double, 3> coord;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
if (Current) {
coord = rGeometry[i_node].Coordinates();
} else {
coord = rGeometry[i_node].GetInitialPosition();

if (Step > 0)
coord += rGeometry[i_node].FastGetSolutionStepValue(DISPLACEMENT, Step);
}

for (IndexType i_dof = 0; i_dof < TDim; ++i_dof)
coordinates(i_node, i_dof) = coord[i_dof];
}

return coordinates;
}


template< SizeType TNumNodes, SizeType TDim>
BoundedMatrix<double, TNumNodes, TDim> ComputeTangentMatrix(const GeometryType& rGeometry)
{
BoundedMatrix<double, TNumNodes, TDim> tangent_matrix;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
const auto& r_node = rGeometry[i_node];
const auto& r_tangent = r_node.GetValue(TANGENT_XI);
for (std::size_t i_dof = 0; i_dof < TDim; ++i_dof) {
tangent_matrix(i_node, i_dof) = r_tangent[i_dof];
}
}

return tangent_matrix;
}


template< SizeType TNumNodes, class TVarType = Variable<double>>
array_1d<double, TNumNodes> GetVariableVector(
const GeometryType& rGeometry,
const TVarType& rVariable,
const IndexType Step
) {

array_1d<double, TNumNodes> var_vector;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node)
var_vector[i_node] = rGeometry[i_node].FastGetSolutionStepValue(rVariable, Step);

return var_vector;
}


template< SizeType TNumNodes, class TVarType = Variable<double> >
BoundedMatrix<double, TNumNodes, 1> GetVariableVectorMatrix(
const GeometryType& rGeometry,
const TVarType& rVariable,
const unsigned int Step
) {

BoundedMatrix<double, TNumNodes, 1> var_vector;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node)
var_vector(i_node, 0) = rGeometry[i_node].FastGetSolutionStepValue(rVariable, Step);

return var_vector;
}


template< SizeType TNumNodes, class TVarType = Variable<double> >
array_1d<double, TNumNodes> GetVariableVector(
const GeometryType& rGeometry,
const TVarType& rVariable
) {

array_1d<double, TNumNodes> var_vector;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node)
var_vector[i_node] = rGeometry[i_node].GetValue(rVariable);

return var_vector;
}


template< SizeType TNumNodes, class TVarType = Variable<double> >
BoundedMatrix<double, TNumNodes, 1> GetVariableVectorMatrix(
const GeometryType& rGeometry,
const TVarType& rVariable
) {

BoundedMatrix<double, TNumNodes, 1> var_vector;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node)
var_vector(i_node, 0) = rGeometry[i_node].GetValue(rVariable);

return var_vector;
}


template< SizeType TDim, SizeType TNumNodes>
BoundedMatrix<double, TNumNodes, TDim> GetVariableMatrix(
const GeometryType& rGeometry,
const Variable<array_1d<double,3> >& rVariable,
const unsigned int Step
) {

BoundedMatrix<double, TNumNodes, TDim> var_matrix;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
const array_1d<double, 3>& value = rGeometry[i_node].FastGetSolutionStepValue(rVariable, Step);
for (IndexType i_dof = 0; i_dof < TDim; ++i_dof)
var_matrix(i_node, i_dof) = value[i_dof];
}

return var_matrix;
}


template< SizeType TDim, SizeType TNumNodes>
BoundedMatrix<double, TNumNodes, TDim> GetVariableMatrix(
const GeometryType& rGeometry,
const Variable<array_1d<double,3> >& rVariable
) {

BoundedMatrix<double, TNumNodes, TDim> var_matrix;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
const array_1d<double, 3>& value = rGeometry[i_node].GetValue(rVariable);
for (IndexType i_dof = 0; i_dof < TDim; ++i_dof)
var_matrix(i_node, i_dof) = value[i_dof];
}

return var_matrix;
}


template< SizeType TDim, SizeType TNumNodes>
BoundedMatrix<double, TNumNodes, TDim> GetAbsMatrix(const BoundedMatrix<double, TNumNodes, TDim>& rInputMatrix) {

BoundedMatrix<double, TNumNodes, TDim> AbsMatrix;

for (IndexType i_node = 0; i_node < TNumNodes; ++i_node) {
for (IndexType i_dof = 0; i_dof < TDim; ++i_dof)
AbsMatrix(i_node, i_dof) = std::abs(rInputMatrix(i_node, i_dof));
}

return AbsMatrix;
}


template< SizeType TDim, class TVarType>
unsigned int SizeToCompute()
{
if (typeid(TVarType) == typeid(Variable<array_1d<double, 3>>))
return TDim;

return 1;
}


template< class TVarType, bool THistorical>
void KRATOS_API(KRATOS_CORE) ResetValue(
ModelPart& rThisModelPart,
const TVarType& rThisVariable
);


template< class TVarType>
void KRATOS_API(KRATOS_CORE) ResetAuxiliarValue(ModelPart& rThisModelPart);


template< class TVarType>
const std::string KRATOS_API(KRATOS_CORE) GetAuxiliarVariable();


template< class TVarType>
double KRATOS_API(KRATOS_CORE) GetAuxiliarValue(
NodeType& rThisNode,
const std::size_t iSize
);


template< class TVarType, bool THistorical>
void KRATOS_API(KRATOS_CORE) MatrixValue(
const GeometryType& rThisGeometry,
const TVarType& rThisVariable,
Matrix& rThisValue
);


template< class TVarType, bool THistorical>
void KRATOS_API(KRATOS_CORE) AddValue(
GeometryType& rThisGeometry,
const TVarType& rThisVariable,
const Matrix& rThisValue
);


template< class TVarType, bool THistorical>
void KRATOS_API(KRATOS_CORE) AddAreaWeightedNodalValue(
NodeType& rThisNode,
const TVarType& rThisVariable,
const double RefArea = 1.0,
const double Tolerance = 1.0e-4
);


template< class TVarType, bool THistorical>
void KRATOS_API(KRATOS_CORE) UpdateDatabase(
ModelPart& rThisModelPart,
const TVarType& rThisVariable,
Vector& rDx,
const std::size_t Index,
IntMap& rConectivityDatabase
);
};
} 
#endif 
