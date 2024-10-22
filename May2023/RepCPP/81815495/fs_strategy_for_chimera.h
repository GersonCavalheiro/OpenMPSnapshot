
#ifndef KRATOS_FS_STRATEGY_FOR_CHIMERA_H
#define KRATOS_FS_STRATEGY_FOR_CHIMERA_H

#include "includes/define.h"
#include "utilities/openmp_utils.h"
#include "custom_strategies/strategies/fractional_step_strategy.h"
#include "chimera_application_variables.h"
#include "custom_utilities/fractional_step_settings_for_chimera.h"



namespace Kratos {













template<class TSparseSpace,
class TDenseSpace,
class TLinearSolver
>
class FractionalStepStrategyForChimera : public FractionalStepStrategy<TSparseSpace,TDenseSpace,TLinearSolver>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(FractionalStepStrategyForChimera);

typedef FractionalStepStrategy<TSparseSpace, TDenseSpace, TLinearSolver> BaseType;

typedef FractionalStepSettingsForChimera<TSparseSpace,TDenseSpace,TLinearSolver> SolverSettingsType;


FractionalStepStrategyForChimera(ModelPart& rModelPart,
SolverSettingsType& rSolverConfig,
bool PredictorCorrector):
BaseType(rModelPart,rSolverConfig,PredictorCorrector)
{
KRATOS_WARNING("FractionalStepStrategyForChimera") << "This constructor is deprecated. Use the one with the \'CalculateReactionsFlag\' instead." << std::endl;
this->InitializeStrategy(rSolverConfig,PredictorCorrector);
}

FractionalStepStrategyForChimera(
ModelPart &rModelPart,
SolverSettingsType &rSolverConfig,
bool PredictorCorrector,
bool CalculateReactionsFlag)
: BaseType(rModelPart, rSolverConfig, PredictorCorrector, CalculateReactionsFlag)
{
this->InitializeStrategy(rSolverConfig,PredictorCorrector);
}

~FractionalStepStrategyForChimera() = default;


FractionalStepStrategyForChimera& operator=(FractionalStepStrategyForChimera const& rOther) = delete;

FractionalStepStrategyForChimera(FractionalStepStrategyForChimera const& rOther) = delete;











std::string Info() const override
{
std::stringstream buffer;
buffer << "FractionalStepStrategyForChimera" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override {rOStream << "FractionalStepStrategyForChimera";}

void PrintData(std::ostream& rOStream) const override {}





protected:










std::tuple<bool,double> SolveStep() override
{

double start_solve_time = OpenMPUtils::GetCurrentTime();
ModelPart& r_model_part = BaseType::GetModelPart();

r_model_part.GetProcessInfo().SetValue(FRACTIONAL_STEP,1);

bool converged = false;

for(std::size_t it = 0; it < BaseType::mMaxVelocityIter; ++it)
{
KRATOS_INFO("FRACTIONAL STEP :: ")<<it+1<<std::endl;
r_model_part.GetProcessInfo().SetValue(FRACTIONAL_STEP,1);
double norm_dv = BaseType::mpMomentumStrategy->Solve();

converged = BaseType::CheckFractionalStepConvergence(norm_dv);

if (converged)
{
KRATOS_INFO_IF("FractionalStepStrategyForChimera ", BaseType::GetEchoLevel() > 0 )<<
"Fractional velocity converged in " << it+1 << " iterations." << std::endl;
break;
}
}

KRATOS_INFO_IF("FractionalStepStrategyForChimera ", (BaseType::GetEchoLevel() > 0) && !converged)<<
"Fractional velocity iterations did not converge "<< std::endl;

r_model_part.GetProcessInfo().SetValue(FRACTIONAL_STEP,4);
ComputeSplitOssProjections(r_model_part);

r_model_part.GetProcessInfo().SetValue(FRACTIONAL_STEP,5);

#pragma omp parallel
{
ModelPart::NodeIterator nodes_begin;
ModelPart::NodeIterator nodes_end;
OpenMPUtils::PartitionedIterators(r_model_part.Nodes(),nodes_begin,nodes_end);

for (ModelPart::NodeIterator it_node = nodes_begin; it_node != nodes_end; ++it_node)
{
const double old_press = it_node->FastGetSolutionStepValue(PRESSURE);
it_node->FastGetSolutionStepValue(PRESSURE_OLD_IT) = -old_press;
}
}

KRATOS_INFO_IF("FractionalStepStrategyForChimera ", BaseType::GetEchoLevel() > 0 )<<
"Calculating Pressure."<< std::endl;
double norm_dp = BaseType::mpPressureStrategy->Solve();

#pragma omp parallel
{
ModelPart::NodeIterator nodes_begin;
ModelPart::NodeIterator nodes_end;
OpenMPUtils::PartitionedIterators(r_model_part.Nodes(),nodes_begin,nodes_end);

for (ModelPart::NodeIterator it_node = nodes_begin; it_node != nodes_end; ++it_node)
it_node->FastGetSolutionStepValue(PRESSURE_OLD_IT) += it_node->FastGetSolutionStepValue(PRESSURE);

}

KRATOS_INFO_IF("FractionalStepStrategyForChimera ", BaseType::GetEchoLevel() > 0 )<<"Updating Velocity." << std::endl;
r_model_part.GetProcessInfo().SetValue(FRACTIONAL_STEP,6);
CalculateEndOfStepVelocity();

for (std::vector<Process::Pointer>::iterator iExtraSteps = BaseType::mExtraIterationSteps.begin();
iExtraSteps != BaseType::mExtraIterationSteps.end(); ++iExtraSteps)
(*iExtraSteps)->Execute();

const double stop_solve_time = OpenMPUtils::GetCurrentTime();
KRATOS_INFO_IF("FractionalStepStrategyForChimera", BaseType::GetEchoLevel() >= 1) << "Time for solving step : " << stop_solve_time - start_solve_time << std::endl;

return std::make_tuple(converged, norm_dp);
}



void ComputeSplitOssProjections(ModelPart& rModelPart) override
{
const array_1d<double,3> zero(3,0.0);

array_1d<double,3> out(3,0.0);

#pragma omp parallel
{
ModelPart::NodeIterator nodes_begin;
ModelPart::NodeIterator nodes_end;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),nodes_begin,nodes_end);

for ( ModelPart::NodeIterator it_node = nodes_begin; it_node != nodes_end; ++it_node )
{
it_node->FastGetSolutionStepValue(CONV_PROJ) = zero;
it_node->FastGetSolutionStepValue(PRESS_PROJ) = zero;
it_node->FastGetSolutionStepValue(DIVPROJ) = 0.0;
it_node->FastGetSolutionStepValue(NODAL_AREA) = 0.0;
}
}

#pragma omp parallel
{
ModelPart::ElementIterator elem_begin;
ModelPart::ElementIterator elem_end;
OpenMPUtils::PartitionedIterators(rModelPart.Elements(),elem_begin,elem_end);

for ( ModelPart::ElementIterator it_elem = elem_begin; it_elem != elem_end; ++it_elem )
{
it_elem->Calculate(CONV_PROJ,out,rModelPart.GetProcessInfo());
}
}

rModelPart.GetCommunicator().AssembleCurrentData(CONV_PROJ);
rModelPart.GetCommunicator().AssembleCurrentData(PRESS_PROJ);
rModelPart.GetCommunicator().AssembleCurrentData(DIVPROJ);
rModelPart.GetCommunicator().AssembleCurrentData(NODAL_AREA);

ChimeraProjectionCorrection(rModelPart);
#pragma omp parallel
{
ModelPart::NodeIterator nodes_begin;
ModelPart::NodeIterator nodes_end;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(),nodes_begin,nodes_end);

for ( ModelPart::NodeIterator it_node = nodes_begin; it_node != nodes_end; ++it_node )
{
const double nodal_area = it_node->FastGetSolutionStepValue(NODAL_AREA);
if( nodal_area > mAreaTolerance )
{
it_node->FastGetSolutionStepValue(CONV_PROJ) /= nodal_area;
it_node->FastGetSolutionStepValue(PRESS_PROJ) /= nodal_area;
it_node->FastGetSolutionStepValue(DIVPROJ) /= nodal_area;
}
}
}


auto &r_pre_modelpart = rModelPart.GetSubModelPart(rModelPart.Name()+"fs_pressure_model_part");
const auto& r_constraints_container = r_pre_modelpart.MasterSlaveConstraints();
for(const auto& constraint : r_constraints_container)
{
const auto& master_dofs = constraint.GetMasterDofsVector();
const auto& slave_dofs = constraint.GetSlaveDofsVector();
ModelPart::MatrixType r_relation_matrix;
ModelPart::VectorType r_constant_vector;
constraint.CalculateLocalSystem(r_relation_matrix,r_constant_vector,rModelPart.GetProcessInfo());

IndexType slave_i = 0;
for(const auto& slave_dof : slave_dofs)
{
const auto slave_node_id = slave_dof->Id(); 
auto& r_slave_node = rModelPart.Nodes()[slave_node_id];
IndexType master_j = 0;
for(const auto& master_dof : master_dofs)
{
const auto master_node_id = master_dof->Id();
const double weight = r_relation_matrix(slave_i, master_j);
auto& r_master_node = rModelPart.Nodes()[master_node_id];
auto& conv_proj = r_slave_node.FastGetSolutionStepValue(CONV_PROJ);
auto& pres_proj = r_slave_node.FastGetSolutionStepValue(PRESS_PROJ);
auto& dive_proj = r_slave_node.FastGetSolutionStepValue(DIVPROJ);
auto& nodal_area = r_slave_node.FastGetSolutionStepValue(NODAL_AREA);
conv_proj += (r_master_node.FastGetSolutionStepValue(CONV_PROJ))*weight;
pres_proj += (r_master_node.FastGetSolutionStepValue(PRESS_PROJ))*weight;
dive_proj += (r_master_node.FastGetSolutionStepValue(DIVPROJ))*weight;
nodal_area += (r_master_node.FastGetSolutionStepValue(NODAL_AREA))*weight;

++master_j;
}
++slave_i;
}
}
}

void CalculateEndOfStepVelocity() override
{
ModelPart& r_model_part = BaseType::GetModelPart();

const array_1d<double,3> zero(3,0.0);
array_1d<double,3> out(3,0.0);

#pragma omp parallel
{
ModelPart::NodeIterator nodes_begin;
ModelPart::NodeIterator nodes_end;
OpenMPUtils::PartitionedIterators(r_model_part.Nodes(),nodes_begin,nodes_end);

for ( ModelPart::NodeIterator it_node = nodes_begin; it_node != nodes_end; ++it_node )
{
it_node->FastGetSolutionStepValue(FRACT_VEL) = zero;
}
}

#pragma omp parallel
{
ModelPart::ElementIterator elem_begin;
ModelPart::ElementIterator elem_end;
OpenMPUtils::PartitionedIterators(r_model_part.Elements(),elem_begin,elem_end);

for ( ModelPart::ElementIterator it_elem = elem_begin; it_elem != elem_end; ++it_elem )
{
it_elem->Calculate(VELOCITY,out,r_model_part.GetProcessInfo());
}
}

r_model_part.GetCommunicator().AssembleCurrentData(FRACT_VEL);

if (BaseType::mUseSlipConditions)
BaseType::EnforceSlipCondition(SLIP);

if (BaseType::mDomainSize == 2) InterpolateVelocity<2>(r_model_part);
if (BaseType::mDomainSize == 3) InterpolateVelocity<3>(r_model_part);

}

void ChimeraProjectionCorrection(ModelPart& rModelPart)
{
auto &r_pre_modelpart = rModelPart.GetSubModelPart(rModelPart.Name()+"fs_pressure_model_part");
const auto& r_constraints_container = r_pre_modelpart.MasterSlaveConstraints();
for(const auto& constraint : r_constraints_container)
{
const auto& slave_dofs = constraint.GetSlaveDofsVector();
for(const auto& slave_dof : slave_dofs)
{
const auto slave_node_id = slave_dof->Id(); 
auto& r_slave_node = rModelPart.Nodes()[slave_node_id];
r_slave_node.GetValue(NODAL_AREA)= 0;
r_slave_node.GetValue(CONV_PROJ)= array_1d<double,3>(3,0.0);
r_slave_node.GetValue(PRESS_PROJ)= array_1d<double,3>(3,0.0);
r_slave_node.GetValue(DIVPROJ)= 0 ;
}
}

for(const auto& constraint : r_constraints_container)
{
const auto& master_dofs = constraint.GetMasterDofsVector();
const auto& slave_dofs = constraint.GetSlaveDofsVector();
ModelPart::MatrixType r_relation_matrix;
ModelPart::VectorType r_constant_vector;
constraint.CalculateLocalSystem(r_relation_matrix,r_constant_vector,rModelPart.GetProcessInfo());

IndexType slave_i = 0;
for(const auto& slave_dof : slave_dofs)
{
const IndexType slave_node_id = slave_dof->Id(); 
auto& r_slave_node = rModelPart.Nodes()[slave_node_id];
IndexType master_j = 0;
for(const auto& master_dof : master_dofs)
{
const IndexType master_node_id = master_dof->Id();
const double weight = r_relation_matrix(slave_i, master_j);
auto& r_master_node = rModelPart.Nodes()[master_node_id];

r_slave_node.GetValue(NODAL_AREA) +=(r_master_node.FastGetSolutionStepValue(NODAL_AREA))*weight;
r_slave_node.GetValue(CONV_PROJ) +=(r_master_node.FastGetSolutionStepValue(CONV_PROJ))*weight;
r_slave_node.GetValue(PRESS_PROJ) +=(r_master_node.FastGetSolutionStepValue(PRESS_PROJ))*weight;
r_slave_node.GetValue(DIVPROJ) +=(r_master_node.FastGetSolutionStepValue(DIVPROJ))*weight;

++master_j;
}
++slave_i;
}
}

rModelPart.GetCommunicator().AssembleNonHistoricalData(NODAL_AREA);
rModelPart.GetCommunicator().AssembleNonHistoricalData(CONV_PROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(PRESS_PROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(DIVPROJ);

for (auto it_node = rModelPart.NodesBegin(); it_node != rModelPart.NodesEnd(); it_node++)
{
if (it_node->GetValue(NODAL_AREA) > mAreaTolerance)
{
it_node->FastGetSolutionStepValue(NODAL_AREA) = it_node->GetValue(NODAL_AREA);
it_node->FastGetSolutionStepValue(CONV_PROJ) = it_node->GetValue(CONV_PROJ);
it_node->FastGetSolutionStepValue(PRESS_PROJ) = it_node->GetValue(PRESS_PROJ);
it_node->FastGetSolutionStepValue(DIVPROJ) = it_node->GetValue(DIVPROJ);
it_node->GetValue(NODAL_AREA) = 0.0;
it_node->GetValue(CONV_PROJ) = array_1d<double,3>(3,0.0);
it_node->GetValue(PRESS_PROJ) = array_1d<double,3>(3,0.0);
it_node->GetValue(DIVPROJ) = 0.0;
}
}
}

void ChimeraVelocityCorrection(ModelPart& rModelPart)
{
auto &r_pre_modelpart = rModelPart.GetSubModelPart(rModelPart.Name()+"fs_pressure_model_part");
const auto& r_constraints_container = r_pre_modelpart.MasterSlaveConstraints();
for(const auto& constraint : r_constraints_container)
{
const auto& slave_dofs = constraint.GetSlaveDofsVector();
for(const auto& slave_dof : slave_dofs)
{
const auto slave_node_id = slave_dof->Id(); 
auto& r_slave_node = rModelPart.Nodes()[slave_node_id];
r_slave_node.FastGetSolutionStepValue(FRACT_VEL_X)=0;
r_slave_node.FastGetSolutionStepValue(FRACT_VEL_Y)=0;
}
}

for(const auto& constraint : r_constraints_container)
{
const auto& master_dofs = constraint.GetMasterDofsVector();
const auto& slave_dofs = constraint.GetSlaveDofsVector();
ModelPart::MatrixType r_relation_matrix;
ModelPart::VectorType r_constant_vector;
constraint.CalculateLocalSystem(r_relation_matrix,r_constant_vector,rModelPart.GetProcessInfo());

IndexType slave_i = 0;
for(const auto& slave_dof : slave_dofs)
{
const auto slave_node_id = slave_dof->Id(); 
auto& r_slave_node = rModelPart.Nodes()[slave_node_id];
IndexType master_j = 0;
for(const auto& master_dof : master_dofs)
{
const auto master_node_id = master_dof->Id();
const double weight = r_relation_matrix(slave_i, master_j);
auto& r_master_node = rModelPart.Nodes()[master_node_id];

r_slave_node.GetValue(FRACT_VEL) +=(r_master_node.FastGetSolutionStepValue(FRACT_VEL))*weight;

++master_j;
}
++slave_i;
}
}

rModelPart.GetCommunicator().AssembleNonHistoricalData(FRACT_VEL);

for (typename ModelPart::NodeIterator it_node = rModelPart.NodesBegin(); it_node != rModelPart.NodesEnd(); it_node++)
{
array_1d<double,3>& r_delta_vel = it_node->GetValue(FRACT_VEL);
if ( r_delta_vel[0]*r_delta_vel[0] + r_delta_vel[1]*r_delta_vel[1] + r_delta_vel[2]*r_delta_vel[2] != 0.0)
{
it_node->FastGetSolutionStepValue(FRACT_VEL) = it_node->GetValue(FRACT_VEL);
r_delta_vel = array_1d<double,3>(3,0.0);
}
}
}









private:


const double mAreaTolerance=1E-12;




template <int TDim>
void InterpolateVelocity(ModelPart& rModelPart)
{
#pragma omp parallel
{
ModelPart::NodeIterator nodes_begin;
ModelPart::NodeIterator nodes_end;
OpenMPUtils::PartitionedIterators(rModelPart.Nodes(), nodes_begin, nodes_end);

for (ModelPart::NodeIterator it_node = nodes_begin; it_node != nodes_end; ++it_node) {
const double NodalArea = it_node->FastGetSolutionStepValue(NODAL_AREA);
if (NodalArea > mAreaTolerance) {
if (!it_node->IsFixed(VELOCITY_X))
it_node->FastGetSolutionStepValue(VELOCITY_X) +=
it_node->FastGetSolutionStepValue(FRACT_VEL_X) / NodalArea;
if (!it_node->IsFixed(VELOCITY_Y))
it_node->FastGetSolutionStepValue(VELOCITY_Y) +=
it_node->FastGetSolutionStepValue(FRACT_VEL_Y) / NodalArea;
if constexpr (TDim > 2)
if (!it_node->IsFixed(VELOCITY_Z))
it_node->FastGetSolutionStepValue(VELOCITY_Z) +=
it_node->FastGetSolutionStepValue(FRACT_VEL_Z) / NodalArea;
}
}
}

auto& r_pre_modelpart =
rModelPart.GetSubModelPart(rModelPart.Name()+"fs_pressure_model_part");
const auto& r_constraints_container = r_pre_modelpart.MasterSlaveConstraints();
for (const auto& constraint : r_constraints_container) {
const auto& slave_dofs = constraint.GetSlaveDofsVector();
for (const auto& slave_dof : slave_dofs) {
const auto slave_node_id =
slave_dof->Id(); 
auto& r_slave_node = rModelPart.Nodes()[slave_node_id];
r_slave_node.FastGetSolutionStepValue(VELOCITY_X) = 0;
r_slave_node.FastGetSolutionStepValue(VELOCITY_Y) = 0;
if constexpr (TDim > 2)
r_slave_node.FastGetSolutionStepValue(VELOCITY_Z) = 0;
}
}

for (const auto& constraint : r_constraints_container) {
const auto& master_dofs = constraint.GetMasterDofsVector();
const auto& slave_dofs = constraint.GetSlaveDofsVector();
ModelPart::MatrixType r_relation_matrix;
ModelPart::VectorType r_constant_vector;
constraint.CalculateLocalSystem(r_relation_matrix, r_constant_vector,
rModelPart.GetProcessInfo());

IndexType slave_i = 0;
for (const auto& slave_dof : slave_dofs) {
const auto slave_node_id =
slave_dof->Id(); 
auto& r_slave_node = rModelPart.Nodes()[slave_node_id];
IndexType master_j = 0;
for (const auto& master_dof : master_dofs) {
const auto master_node_id = master_dof->Id();
const double weight = r_relation_matrix(slave_i, master_j);
auto& r_master_node = rModelPart.Nodes()[master_node_id];

r_slave_node.FastGetSolutionStepValue(VELOCITY_X) +=
(r_master_node.FastGetSolutionStepValue(VELOCITY_X)) * weight;
r_slave_node.FastGetSolutionStepValue(VELOCITY_Y) +=
(r_master_node.FastGetSolutionStepValue(VELOCITY_Y)) * weight;
if constexpr (TDim > 2)
r_slave_node.FastGetSolutionStepValue(VELOCITY_Z) +=
(r_master_node.FastGetSolutionStepValue(VELOCITY_Z)) * weight;

++master_j;
}
++slave_i;
}
}
}

void InitializeStrategy(SolverSettingsType& rSolverConfig,
bool PredictorCorrector)
{
KRATOS_TRY;

BaseType::mTimeOrder = rSolverConfig.GetTimeOrder();

BaseType::Check();


BaseType::mDomainSize = rSolverConfig.GetDomainSize();

BaseType::mPredictorCorrector = PredictorCorrector;

BaseType::mUseSlipConditions = rSolverConfig.UseSlipConditions();

BaseType::mReformDofSet = rSolverConfig.GetReformDofSet();

BaseType::SetEchoLevel(rSolverConfig.GetEchoLevel());

bool HaveVelStrategy = rSolverConfig.FindStrategy(SolverSettingsType::Velocity,BaseType::mpMomentumStrategy);

if (HaveVelStrategy)
{
rSolverConfig.FindTolerance(SolverSettingsType::Velocity,BaseType::mVelocityTolerance);
rSolverConfig.FindMaxIter(SolverSettingsType::Velocity,BaseType::mMaxVelocityIter);
KRATOS_INFO("FractionalStepStrategyForChimera ")<<"Velcoity strategy successfully set !"<<std::endl;
}
else
{
KRATOS_THROW_ERROR(std::runtime_error,"FS_Strategy error: No Velocity strategy defined in FractionalStepSettings","");
}

bool HavePressStrategy = rSolverConfig.FindStrategy(SolverSettingsType::Pressure,BaseType::mpPressureStrategy);

if (HavePressStrategy)
{
rSolverConfig.FindTolerance(SolverSettingsType::Pressure,BaseType::mPressureTolerance);
rSolverConfig.FindMaxIter(SolverSettingsType::Pressure,BaseType::mMaxPressureIter);

KRATOS_INFO("FractionalStepStrategyForChimera ")<<"Pressure strategy successfully set !"<<std::endl;
}
else
{
KRATOS_THROW_ERROR(std::runtime_error,"FS_Strategy error: No Pressure strategy defined in FractionalStepSettings","");
}

BaseType::Check();

KRATOS_CATCH("");
}








}; 





} 

#endif 
