



#include "custom_processes/weighted_divergence_calculation_process.h"


namespace Kratos
{


WeightedDivergenceCalculationProcess::WeightedDivergenceCalculationProcess(
ModelPart& rModelPart,
Parameters ThisParameters):
Process(),
mrModelPart(rModelPart)
{

Parameters default_parameters = Parameters(R"(
{
"time_coefficient"  : 0.2
})"
);

ThisParameters.ValidateAndAssignDefaults(default_parameters);

mTimeCoefficient = ThisParameters["time_coefficient"].GetDouble();
}

std::string WeightedDivergenceCalculationProcess::Info() const
{
return "WeightedDivergenceCalculationProcess";
}

void WeightedDivergenceCalculationProcess::PrintInfo(std::ostream& rOStream) const
{
rOStream << "WeightedDivergenceCalculationProcess";
}

void WeightedDivergenceCalculationProcess::PrintData(std::ostream& rOStream) const
{
this->PrintInfo(rOStream);
}

void WeightedDivergenceCalculationProcess::ExecuteInitialize()
{
KRATOS_TRY;

auto& r_nodes_array = mrModelPart.Nodes();
VariableUtils().SetNonHistoricalVariableToZero(AVERAGED_DIVERGENCE, r_nodes_array);

KRATOS_CATCH("");
}

void WeightedDivergenceCalculationProcess::ExecuteFinalizeSolutionStep()
{
KRATOS_TRY;

const auto& r_current_process_info = mrModelPart.GetProcessInfo();
const auto& r_previous_process_info = r_current_process_info.GetPreviousTimeStepInfo();
const double& time_step_previous = r_previous_process_info[TIME];
const double& final_time = r_current_process_info[END_TIME];
const std::size_t dimension = mrModelPart.GetProcessInfo()[DOMAIN_SIZE];

if (time_step_previous >= mTimeCoefficient * final_time) {
KRATOS_ERROR_IF(dimension < 2 || dimension > 3) << "Inconsinstent dimension to execute the process. Dimension =" << dimension << " but needed dimension = 2 or dimension = 3." << std::endl;
KRATOS_ERROR_IF(mrModelPart.NumberOfElements() == 0) << "the number of elements in the domain is zero. weighted divergence calculation cannot be applied"<< std::endl;
const unsigned int number_elements = mrModelPart.NumberOfElements();

GeometryData::ShapeFunctionsGradientsType DN_DX;

#pragma omp parallel for firstprivate(DN_DX)
for(int i_elem = 0; i_elem < static_cast<int>(number_elements); ++i_elem) {
auto it_elem = mrModelPart.ElementsBegin() + i_elem;
auto& r_geometry = it_elem->GetGeometry();

const std::size_t number_nodes_element = r_geometry.PointsNumber();

Vector values_x(number_nodes_element);
Vector values_y(number_nodes_element);
Vector values_z(number_nodes_element);
for(int i_node=0; i_node < static_cast<int>(number_nodes_element); ++i_node){
const auto &r_velocity = r_geometry[i_node].FastGetSolutionStepValue(VELOCITY);
values_x[i_node] = r_velocity[0];
values_y[i_node] = r_velocity[1];
if (dimension == 3) {
values_z[i_node] = r_velocity[2];
}
}

const auto& r_integration_method = r_geometry.GetDefaultIntegrationMethod(); 
const auto& r_integration_points = r_geometry.IntegrationPoints(r_integration_method);
const std::size_t number_of_integration_points = r_integration_points.size(); 

double divergence_current = 0;
double velocity_seminorm_current = 0;

Vector detJ0;
r_geometry.ShapeFunctionsIntegrationPointsGradients(DN_DX, detJ0, r_integration_method);

for ( IndexType point_number = 0; point_number < number_of_integration_points; ++point_number ){

Vector grad_x;
Vector grad_y;
Vector grad_z;
grad_x = prod(trans(DN_DX[point_number]), values_x);
grad_y = prod(trans(DN_DX[point_number]), values_y);
if (dimension == 3) {
grad_z = prod(trans(DN_DX[point_number]), values_z);
}

const double aux_current_divergence = ComputeAuxiliaryElementDivergence(grad_x, grad_y, grad_z);
const double aux_current_velocity_seminorm = ComputeAuxiliaryElementVelocitySeminorm(grad_x, grad_y, grad_z);
const double gauss_point_volume = r_integration_points[point_number].Weight() * detJ0[point_number];
divergence_current += std::pow(aux_current_divergence,2) * gauss_point_volume;
velocity_seminorm_current += aux_current_velocity_seminorm * gauss_point_volume;
}

const double divergence_old = it_elem->GetValue(AVERAGED_DIVERGENCE);
const double velocity_seminorm_old = it_elem->GetValue(VELOCITY_H1_SEMINORM);

auto divergence_current_avg = ComputeWeightedTimeAverage(divergence_old, divergence_current);
it_elem->SetValue(AVERAGED_DIVERGENCE,divergence_current_avg);

auto velocity_seminorm_current_avg = ComputeWeightedTimeAverage(velocity_seminorm_old, velocity_seminorm_current);
it_elem->SetValue(VELOCITY_H1_SEMINORM,velocity_seminorm_current_avg);
}
}

KRATOS_CATCH("");
}

double WeightedDivergenceCalculationProcess::ComputeWeightedTimeAverage(const double& old_average, const double& current_value)
{
const auto& r_current_process_info = mrModelPart.GetProcessInfo();
const double& time_step_current  = r_current_process_info[TIME];
const auto& r_previous_process_info = r_current_process_info.GetPreviousTimeStepInfo();
const double& time_step_previous = r_previous_process_info[TIME];
const double& final_time = r_current_process_info[END_TIME];

const double new_average = std::sqrt(((time_step_previous-mTimeCoefficient*final_time) * std::pow(old_average,2) + (time_step_current - time_step_previous) * current_value) / (time_step_current - mTimeCoefficient*final_time));
return new_average;
}

double WeightedDivergenceCalculationProcess::ComputeAuxiliaryElementDivergence(Vector& grad_x, Vector& grad_y, Vector& grad_z)
{
const std::size_t dimension = mrModelPart.GetProcessInfo()[DOMAIN_SIZE];
double aux_current_divergence;
if (dimension == 2) {
aux_current_divergence = grad_x[0] + grad_y[1];
}
else if (dimension == 3) {
aux_current_divergence = grad_x[0] + grad_y[1] + grad_z[2];
}
return aux_current_divergence;
}

double WeightedDivergenceCalculationProcess::ComputeAuxiliaryElementVelocitySeminorm(Vector& grad_x, Vector& grad_y, Vector& grad_z)
{
const std::size_t dimension = mrModelPart.GetProcessInfo()[DOMAIN_SIZE];
double aux_current_velocity_seminorm;
if (dimension == 2) {
aux_current_velocity_seminorm = inner_prod(grad_x, grad_x) + inner_prod(grad_y, grad_y);
}
else if (dimension == 3) {
aux_current_velocity_seminorm = inner_prod(grad_x, grad_x) + inner_prod(grad_y, grad_y) + inner_prod(grad_z,grad_z);
}
return aux_current_velocity_seminorm;
}



inline std::ostream& operator << (
std::ostream& rOStream,
const WeightedDivergenceCalculationProcess& rThis)
{
rThis.PrintData(rOStream);
return rOStream;
}

}; 
