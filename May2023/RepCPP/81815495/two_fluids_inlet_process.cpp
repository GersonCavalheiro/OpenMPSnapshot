


#include "utilities/openmp_utils.h"
#include "utilities/variable_utils.h"

#include "two_fluids_inlet_process.h"
#include "fluid_dynamics_application_variables.h"


namespace Kratos
{



TwoFluidsInletProcess::TwoFluidsInletProcess(
ModelPart& rModelPart,
Parameters& rParameters,
Process::Pointer pDistanceProcess )
: Process(), mrInletModelPart(rModelPart) {

Parameters default_parameters( R"(
{
"interface_normal"          : [0.0,1.0,0.0],
"point_on_interface"        : [0.0,0.25,0.0],
"inlet_transition_radius"   : 0.05
}  )" );

rParameters.ValidateAndAssignDefaults(default_parameters);

ModelPart& r_root_model_part = mrInletModelPart.GetRootModelPart();

mInterfaceNormal = rParameters["interface_normal"].GetVector();
mInterfacePoint = rParameters["point_on_interface"].GetVector();
mInletRadius = rParameters["inlet_transition_radius"].GetDouble();

if ( norm_2( mInterfaceNormal ) > 1.0e-7 ){
mInterfaceNormal /= norm_2( mInterfaceNormal );
} else {
KRATOS_ERROR << "Error thrown in TwoFluidsInletProcess: 'interface_normal' in 'interface_settings' must not have a norm of 0.0." << std::endl;
}
if ( mInletRadius < 1.0e-7 ){
KRATOS_ERROR << "Error thrown in TwoFluidsInletProcess: 'inlet_transition_radius' in 'interface_settings' must not have a value smaller or equal 0.0." << std::endl;
}

VariableUtils().SetFlag( INLET, true, mrInletModelPart.Nodes() );
VariableUtils().SetFlag( INLET, true, mrInletModelPart.Conditions() );


const int buffer_size = r_root_model_part.GetBufferSize();
KRATOS_ERROR_IF( buffer_size < 2 ) << "TwoFluidsInletProcess: There is no space for an intermediate storage" << std::endl;

#pragma omp parallel for
for (int i_node = 0; i_node < static_cast<int>( r_root_model_part.NumberOfNodes() ); ++i_node){
auto it_node = r_root_model_part.NodesBegin() + i_node;
it_node->GetSolutionStepValue(DISTANCE, (buffer_size-1) ) = it_node->GetSolutionStepValue(DISTANCE, 0);
}

VariableUtils().SetVariable( DISTANCE, 1.0, r_root_model_part.Nodes() );
VariableUtils().SetVariable( DISTANCE, -1.0e-7, mrInletModelPart.Nodes() );

pDistanceProcess->Execute();

const double scaling_factor = 1.0 / mInletRadius;
#pragma omp parallel for
for (int i_node = 0; i_node < static_cast<int>( r_root_model_part.NumberOfNodes() ); ++i_node){
auto it_node = r_root_model_part.NodesBegin() + i_node;
auto& r_dist = it_node->GetSolutionStepValue(DISTANCE, 0);

if ( (mInletRadius - r_dist) >= 0 ){
r_dist = scaling_factor * (mInletRadius - r_dist);
} else {
r_dist = 0.0;
}
}

VariableUtils().SaveVariable<Variable<double>>( DISTANCE, AUX_DISTANCE, r_root_model_part.Nodes() );

#pragma omp parallel for
for (int i_node = 0; i_node < static_cast<int>( r_root_model_part.NumberOfNodes() ); ++i_node){
auto it_node = r_root_model_part.NodesBegin() + i_node;
it_node->GetSolutionStepValue(DISTANCE, 0) = it_node->GetSolutionStepValue(DISTANCE, (buffer_size-1) );
}

mrInletModelPart.CreateSubModelPart("fluid_1_inlet");
mrInletModelPart.CreateSubModelPart("fluid_2_inlet");
ModelPart& r_fluid_1_inlet = mrInletModelPart.GetSubModelPart("fluid_1_inlet");
ModelPart& r_fluid_2_inlet = mrInletModelPart.GetSubModelPart("fluid_2_inlet");

std::vector<IndexType> index_node_fluid1;
std::vector<IndexType> index_node_fluid2;
for (int i_node = 0; i_node < static_cast<int>( mrInletModelPart.NumberOfNodes() ); ++i_node){
auto it_node = mrInletModelPart.NodesBegin() + i_node;
const double inlet_dist = inner_prod( ( it_node->Coordinates() - mInterfacePoint ), mInterfaceNormal );
if (inlet_dist <= 0.0){
index_node_fluid1.push_back( it_node->GetId() );
} else {
index_node_fluid2.push_back( it_node->GetId() );
}
}
r_fluid_1_inlet.AddNodes( index_node_fluid1 );
r_fluid_2_inlet.AddNodes( index_node_fluid2 );

std::vector<IndexType> index_cond_fluid1;
std::vector<IndexType> index_cond_fluid2;
for (int i_cond = 0; i_cond < static_cast<int>( mrInletModelPart.NumberOfConditions() ); ++i_cond){
auto it_cond = mrInletModelPart.ConditionsBegin() + i_cond;
unsigned int pos_counter = 0;
unsigned int neg_counter = 0;
for (int i_node = 0; i_node < static_cast<int>(it_cond->GetGeometry().PointsNumber()); i_node++){
const Node& rNode = (it_cond->GetGeometry())[i_node];
const double inlet_dist = inner_prod( ( rNode.Coordinates() - mInterfacePoint ), mInterfaceNormal );

if ( inlet_dist > 0 ){ pos_counter++; }
if ( inlet_dist <= 0 ){ neg_counter++; }
}
if( pos_counter > 0 ){
index_cond_fluid2.push_back( it_cond->GetId() );
}
if( neg_counter > 0 ){
index_cond_fluid1.push_back( it_cond->GetId() );
}
}
r_fluid_1_inlet.AddConditions( index_cond_fluid1 );
r_fluid_2_inlet.AddConditions( index_cond_fluid2 );

r_root_model_part.GetCommunicator().GetDataCommunicator().Barrier();
}



void TwoFluidsInletProcess::SmoothDistanceField(){

ModelPart& r_root_model_part = mrInletModelPart.GetRootModelPart();

for (int i_node = 0; i_node < static_cast<int>( r_root_model_part.NumberOfNodes() ); ++i_node){

auto it_node = r_root_model_part.NodesBegin() + i_node;

if ( std::abs(it_node->GetValue(AUX_DISTANCE)) > 1.0e-5 ){

const double& r_inlet_dist = inner_prod( ( it_node->Coordinates() - mInterfacePoint ), mInterfaceNormal );

double& r_domain_dist = it_node->FastGetSolutionStepValue( DISTANCE, 0 );
const double& r_weighting_factor_inlet_field = it_node->GetValue(AUX_DISTANCE);

r_domain_dist = r_weighting_factor_inlet_field * r_inlet_dist + ( 1.0 - r_weighting_factor_inlet_field ) * r_domain_dist;
}

}

}

};  
