
#pragma once

#include <unordered_set>


#include "meshing_application_variables.h"
#include "processes/process.h"


#include "includes/model_part.h"
#include "includes/key_hash.h"
#include "includes/kratos_parameters.h"


#include "utilities/binbased_fast_point_locator.h"


#include "spatial_containers/spatial_containers.h" 

namespace Kratos
{


typedef std::size_t SizeType;
typedef std::size_t IndexType;



namespace NodalInterpolationFunctions
{

void KRATOS_API(MESHING_APPLICATION) GetListNonHistoricalVariables(
const ModelPart& rModelPart,
std::unordered_set<std::string>& rVariableList
);
};



class PointBoundary
: public Point
{
public:

typedef Point BaseType;

KRATOS_CLASS_POINTER_DEFINITION( PointBoundary );


PointBoundary():
BaseType(),
mpOriginCond(nullptr)
{}

PointBoundary(const array_1d<double, 3>& Coords)
:BaseType(Coords),
mpOriginCond(nullptr)
{}

PointBoundary(Condition::Pointer pCond):
mpOriginCond(pCond)
{
UpdatePoint();
}

PointBoundary(
const array_1d<double, 3>& Coords,
Condition::Pointer pCond
):
BaseType(Coords),
mpOriginCond(pCond)
{}

PointBoundary(const PointBoundary& rhs):
BaseType(rhs),
mpOriginCond(rhs.mpOriginCond)
{
}

~PointBoundary() override= default;



BaseType GetPoint()
{
BaseType Point(this->Coordinates());
return Point;
}


void SetPoint(const BaseType Point)
{
this->Coordinates() = Point.Coordinates();
}


void SetCondition(Condition::Pointer pCond)
{
mpOriginCond = pCond;
}


Condition::Pointer GetCondition()
{
KRATOS_DEBUG_ERROR_IF(mpOriginCond.get() == nullptr) << "Condition no initialized in the PointBoundary class" << std::endl;
return mpOriginCond;
}


void Check()
{
KRATOS_TRY;

auto aux_coord = Kratos::make_shared<array_1d<double, 3>>(this->Coordinates());
KRATOS_ERROR_IF(!aux_coord) << "Coordinates no initialized in the PointBoundary class" << std::endl;
KRATOS_ERROR_IF(mpOriginCond.get() == nullptr) << "Condition no initialized in the PointBoundary class" << std::endl;

KRATOS_CATCH("");
}


void UpdatePoint()
{
noalias(this->Coordinates()) = mpOriginCond->GetGeometry().Center().Coordinates();
}

private:
Condition::Pointer mpOriginCond; 

}; 


template<SizeType TDim>
class KRATOS_API(MESHING_APPLICATION) NodalValuesInterpolationProcess
: public Process
{
public:

typedef ModelPart::NodesContainerType                    NodesArrayType;
typedef ModelPart::ElementsContainerType              ElementsArrayType;
typedef ModelPart::ConditionsContainerType          ConditionsArrayType;
typedef Node                                                NodeType;
typedef Geometry<NodeType>                                 GeometryType;
typedef Point                                                 PointType;
typedef PointType::CoordinatesArrayType            CoordinatesArrayType;

typedef PointBoundary                                 PointBoundaryType;
typedef PointBoundaryType::Pointer                     PointTypePointer;
typedef std::vector<PointTypePointer>                       PointVector;
typedef PointVector::iterator                             PointIterator;
typedef std::vector<double>                              DistanceVector;
typedef DistanceVector::iterator                       DistanceIterator;

typedef Bucket< 3ul, PointBoundaryType, PointVector, PointTypePointer, PointIterator, DistanceIterator > BucketType;
typedef Tree< KDTreePartition<BucketType> > KDTreeType;

KRATOS_CLASS_POINTER_DEFINITION( NodalValuesInterpolationProcess );



enum class FrameworkEulerLagrange {EULERIAN = 0, LAGRANGIAN = 1, ALE = 2};



NodalValuesInterpolationProcess(
ModelPart& rOriginMainModelPart,
ModelPart& rDestinationMainModelPart,
Parameters ThisParameters = Parameters(R"({})")
);

~NodalValuesInterpolationProcess() override= default;;


void operator()()
{
Execute();
}



void Execute() override;


const Parameters GetDefaultParameters() const override;







std::string Info() const override
{
return "NodalValuesInterpolationProcess";
}




void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}


private:


ModelPart& mrOriginMainModelPart;               
ModelPart& mrDestinationMainModelPart;          
Parameters mThisParameters;                     
std::unordered_set<std::string> mListVariables; 




static inline FrameworkEulerLagrange ConvertFramework(const std::string& Str)
{
if(Str == "Lagrangian" || Str == "LAGRANGIAN")
return FrameworkEulerLagrange::LAGRANGIAN;
else if(Str == "Eulerian" || Str == "EULERIAN")
return FrameworkEulerLagrange::EULERIAN;
else if(Str == "ALE")
return FrameworkEulerLagrange::ALE;
else
return FrameworkEulerLagrange::EULERIAN;
}


template<class TEntity>
void CalculateData(
NodeType::Pointer pNode,
const typename TEntity::Pointer& pEntity,
const Vector& rShapeFunctions
)
{
GeometryType& r_geometry = pEntity->GetGeometry();
const SizeType number_of_nodes = r_geometry.size();

double aux_coeff = 0.0;
for (IndexType i = 0; i < number_of_nodes; ++i) {
aux_coeff += rShapeFunctions[i];
}
for (auto& r_variable_name : mListVariables) {
if (KratosComponents<Variable<double>>::Has(r_variable_name)) {
const Variable<double>& r_variable = KratosComponents<Variable<double>>::Get(r_variable_name);
if (std::abs(aux_coeff) > std::numeric_limits<double>::epsilon()) {
aux_coeff = 1.0/aux_coeff;
double aux_value = 0.0;
for (IndexType i = 0; i < number_of_nodes; ++i) {
if (r_geometry[i].Has(r_variable)) {
aux_value += rShapeFunctions[i] * r_geometry[i].GetValue(r_variable);
}
}
pNode->SetValue(r_variable, aux_coeff * aux_value);
}
} else if (KratosComponents<Variable<array_1d<double, 3>>>::Has(r_variable_name)) {
const Variable<array_1d<double, 3>>& r_variable = KratosComponents<Variable<array_1d<double, 3>>>::Get(r_variable_name);
if (std::abs(aux_coeff) > std::numeric_limits<double>::epsilon()) {
aux_coeff = 1.0/aux_coeff;
array_1d<double, 3> aux_value = ZeroVector(3);
for (IndexType i = 0; i < number_of_nodes; ++i) {
if (r_geometry[i].Has(r_variable)) {
aux_value += rShapeFunctions[i] * r_geometry[i].GetValue(r_variable);
}
}
pNode->SetValue(r_variable, aux_coeff * aux_value);
}
} else if (KratosComponents<Variable<Vector>>::Has(r_variable_name)) {
const Variable<Vector>& r_variable = KratosComponents<Variable<Vector>>::Get(r_variable_name);
if (std::abs(aux_coeff) > std::numeric_limits<double>::epsilon()) {
aux_coeff = 1.0/aux_coeff;
Vector aux_value = ZeroVector(r_geometry[0].GetValue(r_variable).size());
for (IndexType i = 0; i < number_of_nodes; ++i) {
if (r_geometry[i].Has(r_variable)) {
aux_value += rShapeFunctions[i] * r_geometry[i].GetValue(r_variable);
}
}
pNode->SetValue(r_variable, aux_coeff * aux_value);
}
} else if (KratosComponents<Variable<Matrix>>::Has(r_variable_name)) {
const Variable<Matrix>& r_variable = KratosComponents<Variable<Matrix>>::Get(r_variable_name);
if (std::abs(aux_coeff) > std::numeric_limits<double>::epsilon()) {
aux_coeff = 1.0/aux_coeff;
Matrix aux_value = ZeroMatrix(r_geometry[0].GetValue(r_variable).size1(), r_geometry[0].GetValue(r_variable).size2());
for (IndexType i = 0; i < number_of_nodes; ++i) {
if (r_geometry[i].Has(r_variable)) {
aux_value += rShapeFunctions[i] * r_geometry[i].GetValue(r_variable);
}
}
pNode->SetValue(r_variable, aux_coeff * aux_value);
}
}
}
}


template<class TEntity>
void CalculateStepData(
NodeType::Pointer pNode,
const typename TEntity::Pointer& pEntity,
const Vector& rShapeFunctions,
const IndexType Step
)
{
double* step_data = pNode->SolutionStepData().Data(Step);
for (int j = 0; j < mThisParameters["step_data_size"].GetInt(); ++j)
step_data[j] = 0;

GeometryType& geom = pEntity->GetGeometry();
const SizeType number_of_nodes = geom.size();
for (IndexType i = 0; i < number_of_nodes; ++i) {
const double* nodal_data = geom[i].SolutionStepData().Data(Step);
for (int j = 0; j < mThisParameters["step_data_size"].GetInt(); ++j) {
step_data[j] += rShapeFunctions[i] * nodal_data[j];
}
}
}


void GenerateBoundary(const std::string& rAuxiliarNameModelPart);


void GenerateBoundaryFromElements(
ModelPart& rModelPart,
const std::string& rAuxiliarNameModelPart
);


void ExtrapolateValues(
const std::string& rAuxiliarNameModelPart,
std::vector<NodeType::Pointer>& rToExtrapolateNodes
);


void ComputeNormalSkin(ModelPart& rModelPart);





}; 














}  
