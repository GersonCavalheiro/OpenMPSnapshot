
#ifndef MASS_OPT_RESPONSE_H
#define MASS_OPT_RESPONSE_H



#include "custom_responses/response.h"


namespace Kratos
{








class KRATOS_API(OPTIMIZATION_APPLICATION) MassOptResponse : public Response
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MassOptResponse);


MassOptResponse(std::string ResponseName, Model& rModel, Parameters& ResponseSettings )
: Response(ResponseName,"mass",rModel, ResponseSettings){
for(long unsigned int i=0;i<mrResponseSettings["control_types"].size();i++){
auto control_type = mrResponseSettings["control_types"][i].GetString();
if(control_type=="shape"){
std::string gradient_mode = mrResponseSettings["gradient_settings"]["gradient_mode"].GetString();
if (gradient_mode.compare("finite_differencing") == 0)
{
double delta = mrResponseSettings["gradient_settings"]["step_size"].GetDouble();
mDelta = delta;
}
else
KRATOS_ERROR << "Specified gradient_mode '" << gradient_mode << "' not recognized. The only option is: finite_differencing" << std::endl;                    
}
}
}

virtual ~MassOptResponse()
{
}




void Initialize() override {
for(long unsigned int i=0;i<mrResponseSettings["evaluated_objects"].size();i++){
auto eval_obj = mrResponseSettings["evaluated_objects"][i].GetString();
ModelPart& eval_model_part = mrModel.GetModelPart(eval_obj);
auto controlled_obj = mrResponseSettings["controlled_objects"][i].GetString();
ModelPart& controlled_model_part = mrModel.GetModelPart(controlled_obj);
auto control_type = mrResponseSettings["control_types"][i].GetString();

KRATOS_ERROR_IF_NOT(eval_model_part.Elements().size()>0)
<<"MassOptResponse::Initialize: evaluated object "<<eval_obj<<" must have elements !"<<std::endl;

KRATOS_ERROR_IF_NOT(controlled_model_part.Elements().size()>0)
<<"MassOptResponse::Initialize: controlled object "<<controlled_obj<<" for "<<control_type<<" sensitivity must have elements !"<<std::endl;

}
};
double CalculateValue() override {
double total_mass = 0.0;
for(auto& eval_obj : mrResponseSettings["evaluated_objects"]){
ModelPart& r_eval_object = mrModel.GetModelPart(eval_obj.GetString());
const std::size_t domain_size = r_eval_object.GetProcessInfo()[DOMAIN_SIZE];
#pragma omp parallel for reduction(+:total_mass)
for (auto& elem_i : r_eval_object.Elements()){
const bool element_is_active = elem_i.IsDefined(ACTIVE) ? elem_i.Is(ACTIVE) : true;
if(element_is_active)
total_mass += CalculateElementMass(elem_i,domain_size);
}
}
return total_mass;
};    

double CalculateElementMass(Element& elem_i, const std::size_t DomainSize){
auto& r_this_geometry = elem_i.GetGeometry();
const std::size_t local_space_dimension = r_this_geometry.LocalSpaceDimension();

double volume_area = 0.0;
const IntegrationMethod this_integration_method = r_this_geometry.GetDefaultIntegrationMethod();    
const GeometryType::IntegrationPointsArrayType& integration_points = r_this_geometry.IntegrationPoints(this_integration_method);
for(std::size_t i_point = 0; i_point<integration_points.size(); ++i_point)
{
Matrix J0;
GeometryUtils::JacobianOnInitialConfiguration(r_this_geometry, integration_points[i_point], J0); 
volume_area += integration_points[i_point].Weight() * MathUtils<double>::GeneralizedDet(J0);
}           


double element_mass = 0;
if (local_space_dimension == 2 && DomainSize == 3 && elem_i.GetProperties().Has(THICKNESS) && elem_i.GetProperties().Has(PT))
element_mass = volume_area * elem_i.GetProperties().GetValue(PT) * elem_i.GetProperties().GetValue(DENSITY);
else if (local_space_dimension == 2 && DomainSize == 3 && elem_i.GetProperties().Has(THICKNESS))
element_mass = volume_area * elem_i.GetProperties().GetValue(THICKNESS) * elem_i.GetProperties().GetValue(DENSITY);            
else if (local_space_dimension == 3 && DomainSize == 3)
element_mass = volume_area * elem_i.GetProperties().GetValue(DENSITY);
else
KRATOS_ERROR << "MassOptResponse::CalculateElementMass: the element is neither shell nor solid element !" << std::endl;

return element_mass;
}

void CalculateGradient() override {

KRATOS_TRY;

for(long unsigned int i=0;i<mrResponseSettings["controlled_objects"].size();i++){
auto controlled_obj = mrResponseSettings["controlled_objects"][i].GetString();
ModelPart& controlled_model_part = mrModel.GetModelPart(controlled_obj);
const std::size_t domain_size = controlled_model_part.GetProcessInfo()[DOMAIN_SIZE];
auto control_type = mrResponseSettings["control_types"][i].GetString();
if(control_type=="shape"){
VariableUtils().SetHistoricalVariableToZero(D_MASS_D_X, controlled_model_part.Nodes());
for (auto& elem_i : controlled_model_part.Elements())
if(elem_i.IsDefined(ACTIVE) ? elem_i.Is(ACTIVE) : true)
CalculateElementShapeGradients(elem_i,domain_size);
}
else if(control_type=="material"){
VariableUtils().SetHistoricalVariableToZero(D_MASS_D_FD, controlled_model_part.Nodes());
#pragma omp parallel for
for (auto& elem_i : controlled_model_part.Elements())
if(elem_i.IsDefined(ACTIVE) ? elem_i.Is(ACTIVE) : true)  
CalculateElementMaterialGradients(elem_i,domain_size);              
}
else if(control_type=="thickness"){
VariableUtils().SetHistoricalVariableToZero(D_MASS_D_FT, controlled_model_part.Nodes());
#pragma omp parallel for
for (auto& elem_i : controlled_model_part.Elements())
if(elem_i.IsDefined(ACTIVE) ? elem_i.Is(ACTIVE) : true)  
CalculateElementThicknessGradients(elem_i,domain_size);              
}            

}

KRATOS_CATCH("");

};  

void CalculateElementShapeGradients(Element& elem_i, const std::size_t DomainSize){

auto& r_this_geometry = elem_i.GetGeometry();
const std::size_t number_of_nodes = r_this_geometry.size();

std::vector<array_1d<double, 3>> current_coordinates(number_of_nodes);
for (std::size_t i_node = 0; i_node < number_of_nodes; ++i_node) {
noalias(current_coordinates[i_node]) = r_this_geometry[i_node].Coordinates();
noalias(r_this_geometry[i_node].Coordinates()) = r_this_geometry[i_node].GetInitialPosition().Coordinates();
}
double mass_before_fd = CalculateElementMass(elem_i,DomainSize);

for (SizeType i_node = 0; i_node < number_of_nodes; ++i_node){
auto& node_grads = r_this_geometry[i_node].FastGetSolutionStepValue(D_MASS_D_X);	

r_this_geometry[i_node].X() += mDelta;
r_this_geometry[i_node].X0() += mDelta;
double mass_after_fd = CalculateElementMass(elem_i,DomainSize);
node_grads[0] += (mass_after_fd - mass_before_fd) / mDelta;
r_this_geometry[i_node].X() -= mDelta;
r_this_geometry[i_node].X0() -= mDelta;

r_this_geometry[i_node].Y() += mDelta;
r_this_geometry[i_node].Y0() += mDelta;
mass_after_fd = CalculateElementMass(elem_i,DomainSize);
node_grads[1] += (mass_after_fd - mass_before_fd) / mDelta;
r_this_geometry[i_node].Y() -= mDelta;
r_this_geometry[i_node].Y0() -= mDelta;   

r_this_geometry[i_node].Z() += mDelta;
r_this_geometry[i_node].Z0() += mDelta;
mass_after_fd = CalculateElementMass(elem_i,DomainSize);
node_grads[2] += (mass_after_fd - mass_before_fd) / mDelta;
r_this_geometry[i_node].Z() -= mDelta;
r_this_geometry[i_node].Z0() -= mDelta;                                  
}

for (std::size_t i_node = 0; i_node < number_of_nodes; ++i_node) {
noalias(r_this_geometry[i_node].Coordinates()) = current_coordinates[i_node];
}

};

void CalculateElementMaterialGradients(Element& elem_i, const std::size_t DomainSize){

auto& r_this_geometry = elem_i.GetGeometry();
const std::size_t number_of_nodes = r_this_geometry.size();

double curr_density = elem_i.GetProperties().GetValue(DENSITY);
elem_i.GetProperties().SetValue(DENSITY,1.0);
double elem_dens_grad = CalculateElementMass(elem_i,DomainSize);
elem_i.GetProperties().SetValue(DENSITY,curr_density);

for (SizeType i_node = 0; i_node < number_of_nodes; ++i_node){
const auto& d_pd_d_fd = r_this_geometry[i_node].FastGetSolutionStepValue(D_PD_D_FD);
#pragma omp atomic
r_this_geometry[i_node].FastGetSolutionStepValue(D_MASS_D_FD) += d_pd_d_fd * elem_dens_grad / number_of_nodes;
}
};        

void CalculateElementThicknessGradients(Element& elem_i, const std::size_t DomainSize){

auto& r_this_geometry = elem_i.GetGeometry();
const std::size_t number_of_nodes = r_this_geometry.size();

double curr_thickness = elem_i.GetProperties().GetValue(PT);
elem_i.GetProperties().SetValue(PT,1.0);
double elem_thick_grad = CalculateElementMass(elem_i,DomainSize);
elem_i.GetProperties().SetValue(PT,curr_thickness);

for (SizeType i_node = 0; i_node < number_of_nodes; ++i_node){
const auto& d_pt_d_ft = r_this_geometry[i_node].FastGetSolutionStepValue(D_PT_D_FT);
#pragma omp atomic
r_this_geometry[i_node].FastGetSolutionStepValue(D_MASS_D_FT) += d_pt_d_ft * elem_thick_grad / number_of_nodes;
}

};








virtual std::string Info() const override
{
return "MassOptResponse";
}

virtual void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MassOptResponse";
}

virtual void PrintData(std::ostream& rOStream) const override
{
}





protected:
















private:
double mDelta;
















}; 







}  

#endif 
