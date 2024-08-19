


#include "utilities/openmp_utils.h"
#include "processes/find_nodal_h_process.h"
#include "custom_utilities/fluid_element_utilities.h"

#include "mass_conservation_check_process.h"


namespace Kratos
{



MassConservationCheckProcess::MassConservationCheckProcess(
ModelPart& rModelPart,
const bool PerformCorrections,
const int CorrectionFreq,
const bool WriteToLogFile,
const std::string& LogFileName)
: Process(), mrModelPart(rModelPart) {

mCorrectionFreq = CorrectionFreq;
mWriteToLogFile = WriteToLogFile;
mLogFileName = LogFileName;
mPerformCorrections = PerformCorrections;
}


MassConservationCheckProcess::MassConservationCheckProcess(
ModelPart& rModelPart,
Parameters& rParameters)
: Process(), mrModelPart(rModelPart) {

Parameters default_parameters( R"(
{
"model_part_name"                        : "default_model_part_name",
"perform_corrections"                    : true,
"correction_frequency_in_time_steps"     : 20,
"write_to_log_file"                      : true,
"log_file_name"                          : "mass_conservation.log"
}  )" );

rParameters.ValidateAndAssignDefaults(default_parameters);

mCorrectionFreq = rParameters["correction_frequency_in_time_steps"].GetInt();
mWriteToLogFile = rParameters["write_to_log_file"].GetBool();
mPerformCorrections = rParameters["perform_corrections"].GetBool();
mLogFileName = rParameters["log_file_name"].GetString();
}


std::string MassConservationCheckProcess::Initialize(){

double pos_vol = 0.0;
double neg_vol = 0.0;
double inter_area = 0.0;

ComputeVolumesAndInterface( pos_vol, neg_vol, inter_area );

this->mInitialPositiveVolume = pos_vol;
this->mInitialNegativeVolume = neg_vol;
this->mTheoreticalNegativeVolume = neg_vol;

std::string output_line =   "------ Initial values ----------------- \n";
output_line +=              "  positive volume (air)   = " + std::to_string(this->mInitialPositiveVolume) + "\n";
output_line +=              "  negative volume (water) = " + std::to_string(this->mInitialNegativeVolume) + "\n";
output_line +=              "  fluid interface (area)  = " + std::to_string(inter_area) + "\n\n";
output_line +=              "------ Time step values --------------- \n";
output_line +=              "sim_time \t\twater_vol \t\tair_vol \t\twater_err \t\tnet_inf \t\t";
output_line +=              "net_outf \t\tint_area \t\tcorr_shift \n";

return output_line;
}


std::string MassConservationCheckProcess::ExecuteInTimeStep(){

double pos_vol = 0.0;
double neg_vol = 0.0;
double inter_area = 0;
ComputeVolumesAndInterface( pos_vol, neg_vol, inter_area );
double net_inflow_inlet = ComputeFlowOverBoundary(INLET);
double net_inflow_outlet = ComputeFlowOverBoundary(OUTLET);

const double current_time = mrModelPart.GetProcessInfo()[TIME];
const double current_dt = mrModelPart.GetProcessInfo()[DELTA_TIME];
mQNet2 = mQNet1;
mQNet1 = mQNet0;
mQNet0 = net_inflow_inlet + net_inflow_outlet;

mTheoreticalNegativeVolume += current_dt * ( 5.0*mQNet0 + 8.0*mQNet1 - 1.0*mQNet2 ) / 12.0;

const double water_volume_error = mTheoreticalNegativeVolume - neg_vol;

double shift_for_correction = 0.0;
if ( mPerformCorrections && mrModelPart.GetProcessInfo()[STEP] % mCorrectionFreq == 0 && inter_area > 1e-7){
shift_for_correction = - water_volume_error / inter_area;
ShiftDistanceField( shift_for_correction );
}

std::string output_line_timestep =  std::to_string(current_time) + "\t\t";
output_line_timestep +=             std::to_string(neg_vol) + "\t\t";
output_line_timestep +=             std::to_string(pos_vol) + "\t\t";
output_line_timestep +=             std::to_string(water_volume_error) + "\t\t";
output_line_timestep +=             std::to_string(net_inflow_inlet) + "\t\t";
output_line_timestep +=             std::to_string(net_inflow_outlet) + "\t\t";
output_line_timestep +=             std::to_string(inter_area) + "\t\t";
output_line_timestep +=             std::to_string(shift_for_correction) +"\n";
return output_line_timestep;
}



void MassConservationCheckProcess::ComputeVolumesAndInterface( double& positiveVolume, double& negativeVolume, double& interfaceArea ){

double pos_vol = 0.0;
double neg_vol = 0.0;
double int_area = 0.0;

#pragma omp parallel for reduction(+: pos_vol, neg_vol, int_area)
for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem){
const auto it_elem = mrModelPart.ElementsBegin() + i_elem;

Matrix shape_functions;
GeometryType::ShapeFunctionsGradientsType shape_derivatives;

auto& rGeom = it_elem->GetGeometry();
unsigned int pt_count_pos = 0;
unsigned int pt_count_neg = 0;

for (unsigned int pt = 0; pt < rGeom.Points().size(); pt++){
if ( rGeom[pt].FastGetSolutionStepValue(DISTANCE) > 0.0 ){
pt_count_pos++;
} else {
pt_count_neg++;
}
}

if ( pt_count_pos == rGeom.PointsNumber() ){
pos_vol += it_elem->pGetGeometry()->DomainSize();
}
else if ( pt_count_neg == rGeom.PointsNumber() ){
neg_vol += it_elem->pGetGeometry()->DomainSize();
}
else if ( 0 < pt_count_neg && 0 < pt_count_pos ){
Kratos::unique_ptr<ModifiedShapeFunctions> p_modified_sh_func = nullptr;
Vector w_gauss_pos_side(3, 0.0);
Vector w_gauss_neg_side(3, 0.0);
Vector w_gauss_interface(3, 0.0);

Vector Distance( rGeom.PointsNumber(), 0.0 );
for (unsigned int i = 0; i < rGeom.PointsNumber(); i++){
double& r_dist = rGeom[i].FastGetSolutionStepValue(DISTANCE);
if (std::abs(r_dist) < 1.0e-12) {
const double aux_dist = 1.0e-6* rGeom[i].GetValue(NODAL_H);
if (r_dist > 0.0) {
#pragma omp critical
r_dist = aux_dist;
} else {
#pragma omp critical
r_dist = -aux_dist;
}
}
Distance[i] = rGeom[i].FastGetSolutionStepValue(DISTANCE);
}

if ( rGeom.PointsNumber() == 3 ){ p_modified_sh_func = Kratos::make_unique<Triangle2D3ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else if ( rGeom.PointsNumber() == 4 ){ p_modified_sh_func = Kratos::make_unique<Tetrahedra3D4ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else { KRATOS_ERROR << "The process can not be applied on this kind of element" << std::endl; }

p_modified_sh_func->ComputePositiveSideShapeFunctionsAndGradientsValues(
shape_functions,                    
shape_derivatives,                  
w_gauss_pos_side,                   
GeometryData::IntegrationMethod::GI_GAUSS_1);          

for ( unsigned int i = 0; i < w_gauss_pos_side.size(); i++){
pos_vol += w_gauss_pos_side[i];
}

p_modified_sh_func->ComputeNegativeSideShapeFunctionsAndGradientsValues(
shape_functions,                    
shape_derivatives,                  
w_gauss_neg_side,                   
GeometryData::IntegrationMethod::GI_GAUSS_1);          

for ( unsigned int i = 0; i < w_gauss_neg_side.size(); i++){
neg_vol += w_gauss_neg_side[i];
}

p_modified_sh_func->ComputeInterfacePositiveSideShapeFunctionsAndGradientsValues(
shape_functions,                    
shape_derivatives,                  
w_gauss_interface,                  
GeometryData::IntegrationMethod::GI_GAUSS_1);          

for ( unsigned int i = 0; i < w_gauss_interface.size(); i++){
int_area += std::abs( w_gauss_interface[i] );
}
}
}
const auto& r_comm = mrModelPart.GetCommunicator().GetDataCommunicator();   
std::vector<double> local_data{pos_vol, neg_vol, int_area};
std::vector<double> remote_sum{0, 0, 0};
r_comm.SumAll(local_data, remote_sum);    

positiveVolume = remote_sum[0];
negativeVolume = remote_sum[1];
interfaceArea = remote_sum[2];
}


double MassConservationCheckProcess::ComputeInterfaceArea(){

double int_area = 0.0;

#pragma omp parallel for reduction(+: int_area)
for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem){
const auto it_elem = mrModelPart.ElementsBegin() + i_elem;

Matrix shape_functions;
GeometryType::ShapeFunctionsGradientsType shape_derivatives;

const auto rGeom = it_elem->GetGeometry();
unsigned int pt_count_pos = 0;
unsigned int pt_count_neg = 0;

for (unsigned int pt = 0; pt < rGeom.Points().size(); pt++){
if ( rGeom[pt].FastGetSolutionStepValue(DISTANCE) > 0.0 ){
pt_count_pos++;
} else {
pt_count_neg++;
}
}

if ( pt_count_pos == rGeom.PointsNumber() || pt_count_neg == rGeom.PointsNumber() ){
continue;
}
else {
Kratos::unique_ptr<ModifiedShapeFunctions> p_modified_sh_func = nullptr;
Vector w_gauss_interface(3, 0.0);

Vector Distance( rGeom.PointsNumber(), 0.0 );
for (unsigned int i = 0; i < rGeom.PointsNumber(); i++){
Distance[i] = rGeom[i].FastGetSolutionStepValue(DISTANCE);
}

if ( rGeom.PointsNumber() == 3 ){ p_modified_sh_func = Kratos::make_unique<Triangle2D3ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else if ( rGeom.PointsNumber() == 4 ){ p_modified_sh_func = Kratos::make_unique<Tetrahedra3D4ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else { KRATOS_ERROR << "The process can not be applied on this kind of element" << std::endl; }

p_modified_sh_func->ComputeInterfacePositiveSideShapeFunctionsAndGradientsValues(
shape_functions,                    
shape_derivatives,                  
w_gauss_interface,                  
GeometryData::IntegrationMethod::GI_GAUSS_1);          

for ( unsigned int i = 0; i < w_gauss_interface.size(); i++){
int_area += std::abs( w_gauss_interface[i] );
}
}
}

return int_area;
}



double MassConservationCheckProcess::ComputeNegativeVolume(){

double neg_vol = 0.0;

#pragma omp parallel for reduction(+: neg_vol)
for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem){
const auto it_elem = mrModelPart.ElementsBegin() + i_elem;

Matrix shape_functions;
GeometryType::ShapeFunctionsGradientsType shape_derivatives;
const auto rGeom = it_elem->GetGeometry();
unsigned int pt_count_pos = 0;
unsigned int pt_count_neg = 0;

for (unsigned int pt = 0; pt < rGeom.Points().size(); pt++){
if ( rGeom[pt].FastGetSolutionStepValue(DISTANCE) > 0.0 ){
pt_count_pos++;
} else {
pt_count_neg++;
}
}

if ( pt_count_pos == rGeom.PointsNumber() ){
continue;
}
else if ( pt_count_neg == rGeom.PointsNumber() ){
neg_vol += it_elem->pGetGeometry()->DomainSize();
}
else {
Kratos::unique_ptr<ModifiedShapeFunctions> p_modified_sh_func = nullptr;
Vector w_gauss_neg_side(3, 0.0);

Vector Distance( rGeom.PointsNumber(), 0.0 );
for (unsigned int i = 0; i < rGeom.PointsNumber(); i++){
Distance[i] = rGeom[i].FastGetSolutionStepValue(DISTANCE);
}

if ( rGeom.PointsNumber() == 3 ){ p_modified_sh_func = Kratos::make_unique<Triangle2D3ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else if ( rGeom.PointsNumber() == 4 ){ p_modified_sh_func = Kratos::make_unique<Tetrahedra3D4ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else { KRATOS_ERROR << "The process can not be applied on this kind of element" << std::endl; }

p_modified_sh_func->ComputeNegativeSideShapeFunctionsAndGradientsValues(
shape_functions,                    
shape_derivatives,                  
w_gauss_neg_side,                   
GeometryData::IntegrationMethod::GI_GAUSS_1);          

for ( unsigned int i = 0; i < w_gauss_neg_side.size(); i++){
neg_vol += w_gauss_neg_side[i];
}
}
}

return neg_vol;
}


double MassConservationCheckProcess::ComputePositiveVolume(){

double pos_vol = 0.0;

#pragma omp parallel for reduction(+: pos_vol)
for (int i_elem = 0; i_elem < static_cast<int>(mrModelPart.NumberOfElements()); ++i_elem){
const auto it_elem = mrModelPart.ElementsBegin() + i_elem;

Matrix shape_functions;
GeometryType::ShapeFunctionsGradientsType shape_derivatives;
const auto rGeom = it_elem->GetGeometry();
unsigned int pt_count_pos = 0;
unsigned int pt_count_neg = 0;

for (unsigned int pt = 0; pt < rGeom.Points().size(); pt++){
if ( rGeom[pt].FastGetSolutionStepValue(DISTANCE) > 0.0 ){
pt_count_pos++;
} else {
pt_count_neg++;
}
}

if ( pt_count_pos == rGeom.PointsNumber() ){
pos_vol += it_elem->pGetGeometry()->DomainSize();;
}
else if ( pt_count_neg == rGeom.PointsNumber() ){
continue;
}
else {
Kratos::unique_ptr<ModifiedShapeFunctions> p_modified_sh_func = nullptr;
Vector w_gauss_pos_side(3, 0.0);

Vector Distance( rGeom.PointsNumber(), 0.0 );
for (unsigned int i = 0; i < rGeom.PointsNumber(); i++){
Distance[i] = rGeom[i].FastGetSolutionStepValue(DISTANCE);
}

if ( rGeom.PointsNumber() == 3 ){ p_modified_sh_func = Kratos::make_unique<Triangle2D3ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else if ( rGeom.PointsNumber() == 4 ){ p_modified_sh_func = Kratos::make_unique<Tetrahedra3D4ModifiedShapeFunctions>(it_elem->pGetGeometry(), Distance); }
else { KRATOS_ERROR << "The process can not be applied on this kind of element" << std::endl; }

p_modified_sh_func->ComputePositiveSideShapeFunctionsAndGradientsValues(
shape_functions,                    
shape_derivatives,                  
w_gauss_pos_side,                   
GeometryData::IntegrationMethod::GI_GAUSS_1);          

for ( unsigned int i = 0; i < w_gauss_pos_side.size(); i++){
pos_vol += w_gauss_pos_side[i];
}
}
}

return pos_vol;
}


double MassConservationCheckProcess::ComputeFlowOverBoundary( const Kratos::Flags boundaryFlag ){

double inflow_over_boundary = 0.0;
const double epsilon = 1.0e-8;

#pragma omp parallel for reduction(+: inflow_over_boundary)
for (int i_cond = 0; i_cond < static_cast<int>(mrModelPart.NumberOfConditions()); ++i_cond){

const auto p_condition = mrModelPart.ConditionsBegin() + i_cond;

if ( p_condition->Is( boundaryFlag ) ){

const auto& rGeom = p_condition->GetGeometry();
const int dim = rGeom.PointsNumber();
Vector distance( rGeom.PointsNumber(), 0.0 );

unsigned int neg_count = 0;
unsigned int pos_count = 0;
for (unsigned int i = 0; i < rGeom.PointsNumber(); i++){
distance[i] = p_condition->GetGeometry()[i].FastGetSolutionStepValue( DISTANCE );
if ( rGeom[i].FastGetSolutionStepValue( DISTANCE ) > 0.0 ){
pos_count++;
} else {
neg_count++;
}
}

if ( pos_count == rGeom.PointsNumber() ){ continue; }

if (dim == 2){      

array_1d<double, 3> normal;
this->CalculateNormal2D( normal, rGeom );
if( norm_2( normal ) < epsilon ){ continue; }
else { normal /= norm_2( normal ); }

if ( neg_count == rGeom.PointsNumber() ){
const auto& IntegrationPoints = rGeom.IntegrationPoints(GeometryData::IntegrationMethod::GI_GAUSS_2);
const unsigned int num_gauss = IntegrationPoints.size();
Vector gauss_pts_det_jabobian = ZeroVector(num_gauss);
rGeom.DeterminantOfJacobian(gauss_pts_det_jabobian, GeometryData::IntegrationMethod::GI_GAUSS_2);
const Matrix n_container = rGeom.ShapeFunctionsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 );

for (unsigned int i_gauss = 0; i_gauss < num_gauss; i_gauss++){
const auto& N = row(n_container, i_gauss);
double const w_gauss = gauss_pts_det_jabobian[i_gauss] * IntegrationPoints[i_gauss].Weight();
array_1d<double,3> interpolated_velocity = ZeroVector(3);
for (unsigned int n_node = 0; n_node < rGeom.PointsNumber(); n_node++){
noalias( interpolated_velocity ) += N[n_node] * rGeom[n_node].FastGetSolutionStepValue(VELOCITY);
}
inflow_over_boundary -= w_gauss * inner_prod( normal, interpolated_velocity );
}

} else if ( neg_count < rGeom.PointsNumber() && pos_count < rGeom.PointsNumber() ){

array_1d<double, 3> aux_velocity1, aux_velocity2;

Line3D2<IndexedPoint>::Pointer p_aux_line = nullptr;
GenerateAuxLine( rGeom, distance, p_aux_line, aux_velocity1, aux_velocity2 );

const auto& IntegrationPoints = p_aux_line->IntegrationPoints( GeometryData::IntegrationMethod::GI_GAUSS_2 );
const unsigned int num_gauss = IntegrationPoints.size();
Vector gauss_pts_det_jabobian = ZeroVector(num_gauss);
p_aux_line->DeterminantOfJacobian(gauss_pts_det_jabobian, GeometryData::IntegrationMethod::GI_GAUSS_2);
const Matrix n_container = p_aux_line->ShapeFunctionsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 );

for (unsigned int i_gauss = 0; i_gauss < num_gauss; i_gauss++){
const auto& N = row(n_container, i_gauss);
double const w_gauss = gauss_pts_det_jabobian[i_gauss] * IntegrationPoints[i_gauss].Weight();
const array_1d<double,3> interpolatedVelocity = N[0] * aux_velocity1 + N[1] * aux_velocity2;
inflow_over_boundary -= std::abs( w_gauss ) * inner_prod( normal, interpolatedVelocity );
}
}
}

else if (dim == 3){

array_1d<double, 3> normal;
this->CalculateNormal3D( normal, rGeom);
if( norm_2( normal ) < epsilon ){ continue; }
else { normal /= norm_2( normal ); }

if ( neg_count == rGeom.PointsNumber() ){

const GeometryType::IntegrationPointsArrayType& IntegrationPoints = rGeom.IntegrationPoints(GeometryData::IntegrationMethod::GI_GAUSS_2);
const unsigned int num_gauss = IntegrationPoints.size();
Vector gauss_pts_det_jabobian = ZeroVector(num_gauss);
rGeom.DeterminantOfJacobian(gauss_pts_det_jabobian, GeometryData::IntegrationMethod::GI_GAUSS_2);
const Matrix n_container = rGeom.ShapeFunctionsValues( GeometryData::IntegrationMethod::GI_GAUSS_2 );

for (unsigned int i_gauss = 0; i_gauss < num_gauss; i_gauss++){
const auto& N = row(n_container, i_gauss);
double const wGauss = gauss_pts_det_jabobian[i_gauss] * IntegrationPoints[i_gauss].Weight();
array_1d<double,3> interpolated_velocity = ZeroVector(3);
for (unsigned int n_node = 0; n_node < rGeom.PointsNumber(); n_node++){
noalias( interpolated_velocity ) += N[n_node] * rGeom[n_node].FastGetSolutionStepValue(VELOCITY);
}
inflow_over_boundary -= wGauss * inner_prod( normal, interpolated_velocity );
}

} else if ( neg_count < rGeom.PointsNumber() && pos_count < rGeom.PointsNumber() ){

Matrix r_shape_functions;
GeometryType::ShapeFunctionsGradientsType r_shape_derivatives;
Vector w_gauss_neg_side;

const auto aux_2D_triangle = GenerateAuxTriangle( rGeom );
const auto p_modified_sh_func = Kratos::make_unique<Triangle2D3ModifiedShapeFunctions>( aux_2D_triangle, distance);

p_modified_sh_func->ComputeNegativeSideShapeFunctionsAndGradientsValues(
r_shape_functions,                  
r_shape_derivatives,                
w_gauss_neg_side,                   
GeometryData::IntegrationMethod::GI_GAUSS_2);          

for ( unsigned int i_gauss = 0; i_gauss < w_gauss_neg_side.size(); i_gauss++){
const array_1d<double,3>& N = row(r_shape_functions, i_gauss);
array_1d<double,3> interpolated_velocity = ZeroVector(3);
for (unsigned int n_node = 0; n_node < rGeom.PointsNumber(); n_node++){
noalias( interpolated_velocity ) += N[n_node] * rGeom[n_node].FastGetSolutionStepValue(VELOCITY);
}
inflow_over_boundary -= std::abs( w_gauss_neg_side[i_gauss] ) * inner_prod( normal, interpolated_velocity );
}
}
}
}
}
const auto& r_comm = mrModelPart.GetCommunicator().GetDataCommunicator();
inflow_over_boundary = r_comm.SumAll(inflow_over_boundary);
return inflow_over_boundary;
}



void MassConservationCheckProcess::ShiftDistanceField( double deltaDist ){

ModelPart::NodesContainerType rNodes = mrModelPart.Nodes();
#pragma omp parallel for
for(int count = 0; count < static_cast<int>(rNodes.size()); count++){
ModelPart::NodesContainerType::iterator i_node = rNodes.begin() + count;
i_node->FastGetSolutionStepValue( DISTANCE ) += deltaDist;
if (mrModelPart.GetProcessInfo()[MOMENTUM_CORRECTION]){
i_node->GetValue( DISTANCE_CORRECTION ) = -deltaDist;}
}
}



void MassConservationCheckProcess::CalculateNormal2D(array_1d<double,3>& An, const Geometry<Node >& pGeometry){

An[0] =   pGeometry[1].Y() - pGeometry[0].Y();
An[1] = - (pGeometry[1].X() - pGeometry[0].X());
An[2] =    0.0;
}



void MassConservationCheckProcess::CalculateNormal3D(array_1d<double,3>& An, const Geometry<Node >& pGeometry){

array_1d<double,3> v1,v2;
v1[0] = pGeometry[1].X() - pGeometry[0].X();
v1[1] = pGeometry[1].Y() - pGeometry[0].Y();
v1[2] = pGeometry[1].Z() - pGeometry[0].Z();

v2[0] = pGeometry[2].X() - pGeometry[0].X();
v2[1] = pGeometry[2].Y() - pGeometry[0].Y();
v2[2] = pGeometry[2].Z() - pGeometry[0].Z();

MathUtils<double>::CrossProduct(An,v1,v2);
An *= 0.5;
}



Triangle2D3<Node>::Pointer MassConservationCheckProcess::GenerateAuxTriangle( const Geometry<Node >& rGeom ){


array_1d<double,3> vec_u = rGeom[1].Coordinates() - rGeom[0].Coordinates();
vec_u /= norm_2( vec_u );

array_1d<double,3> vec_w;
MathUtils<double>::CrossProduct(vec_w, vec_u, ( rGeom[2].Coordinates() - rGeom[0].Coordinates() ) );
vec_w /= norm_2( vec_w );

array_1d<double,3> vec_v;
MathUtils<double>::CrossProduct(vec_v, vec_u, vec_w );

Matrix rot_mat = ZeroMatrix(3,3);
for (unsigned int i = 0; i < 3; i++){
rot_mat(0,i) = vec_u[i];
rot_mat(1,i) = vec_v[i];
rot_mat(2,i) = vec_w[i];
}

array_1d<double,3> coord1_transformed = prod( rot_mat, rGeom[0].Coordinates() );
array_1d<double,3> coord2_transformed = prod( rot_mat, rGeom[1].Coordinates() );
array_1d<double,3> coord3_transformed = prod( rot_mat, rGeom[2].Coordinates() );
KRATOS_DEBUG_ERROR_IF_NOT( std::abs(coord1_transformed[2] - coord2_transformed[2])<1.0e-7 &&
std::abs(coord1_transformed[2] - coord3_transformed[2])<1.0e-7 );

Node::Pointer node1 = Kratos::make_intrusive<Kratos::Node>( mrModelPart.Nodes().size() + 2, coord1_transformed[0], coord1_transformed[1] );
Node::Pointer node2 = Kratos::make_intrusive<Kratos::Node>( mrModelPart.Nodes().size() + 3, coord2_transformed[0], coord2_transformed[1] );
Node::Pointer node3 = Kratos::make_intrusive<Kratos::Node>( mrModelPart.Nodes().size() + 4, coord3_transformed[0], coord3_transformed[1] );

Triangle2D3<Node>::Pointer aux_2D_triangle = Kratos::make_shared< Triangle2D3<Node > >( node1, node2, node3 );
return aux_2D_triangle;
}



void MassConservationCheckProcess::GenerateAuxLine( const Geometry<Node >& rGeom,
const Vector& distance,
Line3D2<IndexedPoint>::Pointer& p_aux_line,
array_1d<double, 3>& aux_velocity1,
array_1d<double, 3>& aux_velocity2 ){

const double aux_node_rel_location = std::abs ( distance[0] /( distance[1]-distance[0] ));
Vector n_cut = ZeroVector(2);
n_cut[0] = 1.0 - aux_node_rel_location;
n_cut[1] = aux_node_rel_location;

PointerVectorSet<IndexedPoint, IndexedObject> aux_point_container;
aux_point_container.reserve(2);
array_1d<double, 3> aux_point1_coords, aux_point2_coords;

IndexedPoint::Pointer paux_point1 = nullptr;
IndexedPoint::Pointer paux_point2 = nullptr;
for (unsigned int i_node = 0; i_node < rGeom.PointsNumber(); i_node++){
if ( rGeom[i_node].FastGetSolutionStepValue( DISTANCE ) > 0.0 ){
aux_point1_coords[0] = n_cut[0] * rGeom[0].X() + n_cut[1] * rGeom[1].X();
aux_point1_coords[1] = n_cut[0] * rGeom[0].Y() + n_cut[1] * rGeom[1].Y();
aux_point1_coords[2] = n_cut[0] * rGeom[0].Z() + n_cut[1] * rGeom[1].Z();
paux_point1 = Kratos::make_shared<IndexedPoint>(aux_point1_coords, mrModelPart.NumberOfNodes()+1);
aux_velocity1 = n_cut[0] * rGeom[0].FastGetSolutionStepValue( VELOCITY ) + n_cut[1] * rGeom[1].FastGetSolutionStepValue( VELOCITY );
} else {
aux_point2_coords[0] = rGeom[i_node].X();
aux_point2_coords[1] = rGeom[i_node].Y();
aux_point2_coords[2] = rGeom[i_node].Z();
paux_point2 = Kratos::make_shared<IndexedPoint>(aux_point2_coords, mrModelPart.NumberOfNodes()+2);
aux_velocity2 = rGeom[i_node].FastGetSolutionStepValue( VELOCITY );
}
}

p_aux_line = Kratos::make_shared< Line3D2 < IndexedPoint > >( paux_point1, paux_point2 );
}



};  
