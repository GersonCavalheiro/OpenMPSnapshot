

#if !defined(KRATOS_VARIATIONAL_DISTANCE_CALCULATION_PROCESS_INCLUDED )
#define  KRATOS_VARIATIONAL_DISTANCE_CALCULATION_PROCESS_INCLUDED

#include <string>
#include <iostream>
#include <algorithm>


#include "includes/define.h"
#include "containers/model.h"
#include "includes/kratos_flags.h"
#include "elements/distance_calculation_element_simplex.h"
#include "linear_solvers/linear_solver.h"
#include "processes/process.h"
#include "modeler/connectivity_preserve_modeler.h"
#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"
#include "solving_strategies/schemes/residualbased_incrementalupdate_static_scheme.h"
#include "solving_strategies/strategies/residualbased_linear_strategy.h"
#include "utilities/variable_utils.h"

namespace Kratos
{







template< unsigned int TDim, class TSparseSpace, class TDenseSpace, class TLinearSolver >
class VariationalDistanceCalculationProcess : public Process
{
public:

KRATOS_DEFINE_LOCAL_FLAG(PERFORM_STEP1);
KRATOS_DEFINE_LOCAL_FLAG(DO_EXPENSIVE_CHECKS);
KRATOS_DEFINE_LOCAL_FLAG(CALCULATE_EXACT_DISTANCES_TO_PLANE);


typedef Scheme< TSparseSpace,  TDenseSpace > SchemeType;
typedef typename SchemeType::Pointer SchemePointerType;
typedef typename BuilderAndSolver<TSparseSpace,TDenseSpace,TLinearSolver>::Pointer BuilderSolverPointerType;
typedef ImplicitSolvingStrategy< TSparseSpace, TDenseSpace, TLinearSolver > SolvingStrategyType;


KRATOS_CLASS_POINTER_DEFINITION(VariationalDistanceCalculationProcess);




VariationalDistanceCalculationProcess(
ModelPart& rBaseModelPart,
typename TLinearSolver::Pointer pLinearSolver,
unsigned int MaxIterations = 10,
Flags Options = CALCULATE_EXACT_DISTANCES_TO_PLANE.AsFalse(),
std::string AuxPartName = "RedistanceCalculationPart",
double Coefficient1 = 0.01,
double Coefficient2 = 0.1)
:
mDistancePartIsInitialized(false),
mMaxIterations(MaxIterations),
mrModel( rBaseModelPart.GetModel() ),
mrBaseModelPart (rBaseModelPart),
mOptions( Options ),
mAuxModelPartName( AuxPartName ),
mCoefficient1(Coefficient1),
mCoefficient2(Coefficient2)
{
KRATOS_TRY

ValidateInput();

ReGenerateDistanceModelPart(rBaseModelPart);

auto p_builder_solver = Kratos::make_shared<ResidualBasedBlockBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> >(pLinearSolver);

InitializeSolutionStrategy(p_builder_solver);

KRATOS_CATCH("")
}


VariationalDistanceCalculationProcess(
ModelPart& rBaseModelPart,
typename TLinearSolver::Pointer pLinearSolver,
BuilderSolverPointerType pBuilderAndSolver,
unsigned int MaxIterations = 10,
Flags Options = CALCULATE_EXACT_DISTANCES_TO_PLANE.AsFalse(),
std::string AuxPartName = "RedistanceCalculationPart",
double Coefficient1 = 0.01,
double Coefficient2 = 0.1)
:
mDistancePartIsInitialized(false),
mMaxIterations(MaxIterations),
mrModel( rBaseModelPart.GetModel() ),
mrBaseModelPart (rBaseModelPart),
mOptions( Options ),
mAuxModelPartName( AuxPartName ),
mCoefficient1(Coefficient1),
mCoefficient2(Coefficient2)
{
KRATOS_TRY

ValidateInput();

ReGenerateDistanceModelPart(rBaseModelPart);

InitializeSolutionStrategy(pBuilderAndSolver);

KRATOS_CATCH("")
}

~VariationalDistanceCalculationProcess() override
{
Clear();
};


void operator()()
{
Execute();
}


void Execute() override
{
KRATOS_TRY;

if(mDistancePartIsInitialized == false){
ReGenerateDistanceModelPart(mrBaseModelPart);
}

ModelPart& r_distance_model_part = mrModel.GetModelPart( mAuxModelPartName );

r_distance_model_part.pGetProcessInfo()->SetValue(FRACTIONAL_STEP,1);

const int nnodes = static_cast<int>(r_distance_model_part.NumberOfNodes());

block_for_each(r_distance_model_part.Nodes(), [](Node& rNode){
double& d = rNode.FastGetSolutionStepValue(DISTANCE);

rNode.Free(DISTANCE);
rNode.Set(BLOCKED, false);

rNode.SetValue(DISTANCE, d);

if(d == 0){
d = 1.0e-15;
rNode.Set(BLOCKED, true);
rNode.Fix(DISTANCE);
} else {
if(d > 0.0){
d = 1.0e15; 
} else {
d = -1.0e15;
}
}
});

block_for_each(r_distance_model_part.Elements(), [this](Element& rElem){
array_1d<double,TDim+1> distances;
auto& geom = rElem.GetGeometry();

for(unsigned int i=0; i<TDim+1; i++){
distances[i] = geom[i].GetValue(DISTANCE);
}

const array_1d<double,TDim+1> original_distances = distances;

if(this->IsSplit(distances)){
if (mOptions.Is(CALCULATE_EXACT_DISTANCES_TO_PLANE)) {
GeometryUtils::CalculateExactDistancesToPlane(geom, distances);
}
else {
if constexpr (TDim==3){
GeometryUtils::CalculateTetrahedraDistances(geom, distances);
}
else {
GeometryUtils::CalculateTriangleDistances(geom, distances);
}
}

for(unsigned int i = 0; i < TDim+1; ++i){
if(original_distances[i] < 0){
distances[i] = -distances[i];
}
}

for(unsigned int i = 0; i < TDim+1; ++i){
double &d = geom[i].FastGetSolutionStepValue(DISTANCE);
geom[i].SetLock();
if(std::abs(d) > std::abs(distances[i])){
d = distances[i];
}
geom[i].Set(BLOCKED, true);
geom[i].Fix(DISTANCE);
geom[i].UnSetLock();
}
}
});

this->SynchronizeFixity();
this->SynchronizeDistance();

double max_dist = 0.0;
double min_dist = 0.0;
for(int i_node = 0; i_node < nnodes; ++i_node){
auto it_node = r_distance_model_part.NodesBegin() + i_node;
if(it_node->IsFixed(DISTANCE)){
const double& d = it_node->FastGetSolutionStepValue(DISTANCE);
if(d > max_dist){
max_dist = d;
}
if(d < min_dist){
min_dist = d;
}
}
}

const auto &r_communicator = r_distance_model_part.GetCommunicator().GetDataCommunicator();
max_dist = r_communicator.MaxAll(max_dist);
min_dist = r_communicator.MinAll(min_dist);

block_for_each(r_distance_model_part.Nodes(), [&min_dist, &max_dist](Node& rNode){
if(!rNode.IsFixed(DISTANCE)){
double& d = rNode.FastGetSolutionStepValue(DISTANCE);
if(d>0){
d = max_dist;
} else {
d = min_dist;
}
}
});
mpSolvingStrategy->Solve();

r_distance_model_part.pGetProcessInfo()->SetValue(FRACTIONAL_STEP,2);
for(unsigned int it = 0; it<mMaxIterations; it++){
mpSolvingStrategy->Solve();
}

VariableUtils().ApplyFixity(DISTANCE, false, r_distance_model_part.Nodes());
VariableUtils().SetFlag(BOUNDARY, false, r_distance_model_part.Nodes());
VariableUtils().SetFlag(BLOCKED, false, r_distance_model_part.Nodes());

KRATOS_CATCH("")
}

void Clear() override
{
if(mrModel.HasModelPart( mAuxModelPartName ))
mrModel.DeleteModelPart( mAuxModelPartName );
mDistancePartIsInitialized = false;

mpSolvingStrategy->Clear();

}




std::string Info() const override
{
return "VariationalDistanceCalculationProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "VariationalDistanceCalculationProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:



bool mDistancePartIsInitialized;
unsigned int mMaxIterations;

Model& mrModel;
ModelPart& mrBaseModelPart;
Flags mOptions;
std::string mAuxModelPartName;

double mCoefficient1;
double mCoefficient2;

typename SolvingStrategyType::UniquePointer mpSolvingStrategy;



void ValidateInput()
{
const DataCommunicator& r_comm = mrBaseModelPart.GetCommunicator().GetDataCommunicator();
int num_elements = mrBaseModelPart.NumberOfElements();
int num_nodes = mrBaseModelPart.NumberOfNodes();

if (num_elements > 0)
{
const auto geometry_family = mrBaseModelPart.ElementsBegin()->GetGeometry().GetGeometryFamily();
KRATOS_ERROR_IF( (TDim == 2) && (geometry_family != GeometryData::KratosGeometryFamily::Kratos_Triangle) )
<< "In 2D the element type is expected to be a triangle." << std::endl;
KRATOS_ERROR_IF( (TDim == 3) && (geometry_family != GeometryData::KratosGeometryFamily::Kratos_Tetrahedra) )
<< "In 3D the element type is expected to be a tetrahedron" << std::endl;
}

KRATOS_ERROR_IF(r_comm.SumAll(num_nodes) == 0) << "The model part has no nodes." << std::endl;
KRATOS_ERROR_IF(r_comm.SumAll(num_elements) == 0) << "The model Part has no elements." << std::endl;

VariableUtils().CheckVariableExists<Variable<double > >(DISTANCE, mrBaseModelPart.Nodes());
}

void InitializeSolutionStrategy(BuilderSolverPointerType pBuilderAndSolver)
{
auto p_scheme = Kratos::make_shared< ResidualBasedIncrementalUpdateStaticScheme< TSparseSpace,TDenseSpace > >();

ModelPart& r_distance_model_part = mrModel.GetModelPart( mAuxModelPartName );

bool CalculateReactions = false;
bool ReformDofAtEachIteration = false;
bool CalculateNormDxFlag = false;

mpSolvingStrategy = Kratos::make_unique<ResidualBasedLinearStrategy<TSparseSpace, TDenseSpace, TLinearSolver> >(
r_distance_model_part,
p_scheme,
pBuilderAndSolver,
CalculateReactions,
ReformDofAtEachIteration,
CalculateNormDxFlag);

mpSolvingStrategy->Check();
}

virtual void ReGenerateDistanceModelPart(ModelPart& rBaseModelPart)
{
KRATOS_TRY

if(mrModel.HasModelPart( mAuxModelPartName ))
mrModel.DeleteModelPart( mAuxModelPartName );

VariableUtils().AddDof<Variable<double> >(DISTANCE, rBaseModelPart);

ModelPart& r_distance_model_part = mrModel.CreateModelPart( mAuxModelPartName );

Element::Pointer p_distance_element = Kratos::make_intrusive<DistanceCalculationElementSimplex<TDim> >();

r_distance_model_part.GetNodalSolutionStepVariablesList() = rBaseModelPart.GetNodalSolutionStepVariablesList();

ConnectivityPreserveModeler modeler;
modeler.GenerateModelPart(rBaseModelPart, r_distance_model_part, *p_distance_element);

VariableUtils().SetFlag<ModelPart::NodesContainerType>(BOUNDARY, false, r_distance_model_part.Nodes());
for (auto it_cond = rBaseModelPart.ConditionsBegin(); it_cond != rBaseModelPart.ConditionsEnd(); ++it_cond){
Geometry< Node >& geom = it_cond->GetGeometry();
for(unsigned int i=0; i<geom.size(); i++){
geom[i].Set(BOUNDARY,true);
}
}

rBaseModelPart.GetCommunicator().SynchronizeOrNodalFlags(BOUNDARY);

r_distance_model_part.GetProcessInfo().SetValue(VARIATIONAL_REDISTANCE_COEFFICIENT_FIRST, mCoefficient1);
r_distance_model_part.GetProcessInfo().SetValue(VARIATIONAL_REDISTANCE_COEFFICIENT_SECOND, mCoefficient2);

mDistancePartIsInitialized = true;

KRATOS_CATCH("")
}




private:




bool IsSplit(const array_1d<double,TDim+1> &rDistances){
unsigned int positives = 0, negatives = 0;

for(unsigned int i = 0; i < TDim+1; ++i){
if(rDistances[i] >= 0){
++positives;
} else {
++negatives;
}
}

if (positives > 0 && negatives > 0){
return true;
}

return false;
}

void SynchronizeDistance(){
ModelPart& r_distance_model_part = mrModel.GetModelPart( mAuxModelPartName );
auto &r_communicator = r_distance_model_part.GetCommunicator();

if(r_communicator.TotalProcesses() != 1){
int nnodes = static_cast<int>(r_distance_model_part.NumberOfNodes());

#pragma omp parallel for
for(int i_node = 0; i_node < nnodes; ++i_node){
auto it_node = r_distance_model_part.NodesBegin() + i_node;
it_node->FastGetSolutionStepValue(DISTANCE) = std::abs(it_node->FastGetSolutionStepValue(DISTANCE));
}

r_communicator.SynchronizeCurrentDataToMin(DISTANCE);

#pragma omp parallel for
for(int i_node = 0; i_node < nnodes; ++i_node){
auto it_node = r_distance_model_part.NodesBegin() + i_node;
if(it_node->GetValue(DISTANCE) < 0.0){
it_node->FastGetSolutionStepValue(DISTANCE) = -it_node->FastGetSolutionStepValue(DISTANCE);
}
}
}
}

void SynchronizeFixity(){
ModelPart& r_distance_model_part = mrModel.GetModelPart( mAuxModelPartName );
auto &r_communicator = r_distance_model_part.GetCommunicator();

if(r_communicator.TotalProcesses() != 1){
int nnodes = static_cast<int>(r_distance_model_part.NumberOfNodes());

r_communicator.SynchronizeOrNodalFlags(BLOCKED);

#pragma omp parallel for
for(int i_node = 0; i_node < nnodes; ++i_node){
auto it_node = r_distance_model_part.NodesBegin() + i_node;
if (it_node->Is(BLOCKED)){
it_node->Fix(DISTANCE);
}
}
}
}




VariationalDistanceCalculationProcess& operator=(VariationalDistanceCalculationProcess const& rOther);


}; 

template< unsigned int TDim,class TSparseSpace, class TDenseSpace, class TLinearSolver >
const Kratos::Flags VariationalDistanceCalculationProcess<TDim,TSparseSpace,TDenseSpace,TLinearSolver>::PERFORM_STEP1(Kratos::Flags::Create(0));

template< unsigned int TDim,class TSparseSpace, class TDenseSpace, class TLinearSolver >
const Kratos::Flags VariationalDistanceCalculationProcess<TDim,TSparseSpace,TDenseSpace,TLinearSolver>::DO_EXPENSIVE_CHECKS(Kratos::Flags::Create(1));

template< unsigned int TDim,class TSparseSpace, class TDenseSpace, class TLinearSolver >
const Kratos::Flags VariationalDistanceCalculationProcess<TDim,TSparseSpace,TDenseSpace,TLinearSolver>::CALCULATE_EXACT_DISTANCES_TO_PLANE(Kratos::Flags::Create(2));






template< unsigned int TDim, class TSparseSpace, class TDenseSpace, class TLinearSolver>
inline std::istream& operator >> (std::istream& rIStream,
VariationalDistanceCalculationProcess<TDim,TSparseSpace,TDenseSpace,TLinearSolver>& rThis);

template< unsigned int TDim, class TSparseSpace, class TDenseSpace, class TLinearSolver>
inline std::ostream& operator << (std::ostream& rOStream,
const VariationalDistanceCalculationProcess<TDim,TSparseSpace,TDenseSpace,TLinearSolver>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
