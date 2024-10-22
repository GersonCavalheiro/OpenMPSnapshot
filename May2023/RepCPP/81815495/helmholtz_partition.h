
#ifndef HELMHOLTZ_PARTITION_H
#define HELMHOLTZ_PARTITION_H

#include <iostream>
#include <string>


#include "custom_controls/material_controls/material_control.h"
#include "custom_elements/helmholtz_bulk_element.h"
#include "custom_strategies/strategies/helmholtz_strategy.h"



namespace Kratos
{








class KRATOS_API(OPTIMIZATION_APPLICATION) HelmholtzPartition : public MaterialControl
{
public:

typedef HelmholtzStrategy<SparseSpaceType, LocalSpaceType,LinearSolverType> StrategyType;

KRATOS_CLASS_POINTER_DEFINITION(HelmholtzPartition);


HelmholtzPartition( std::string ControlName, Model& rModel, std::vector<LinearSolverType::Pointer>& rLinearSolvers, Parameters ControlSettings )
:  MaterialControl(ControlName,rModel,ControlSettings){
for(long unsigned int lin_i=0;lin_i<rLinearSolvers.size();lin_i++)
rLinearSystemSolvers.push_back(rLinearSolvers[lin_i]);
mTechniqueSettings = ControlSettings["technique_settings"];
}

virtual ~HelmholtzPartition()
{
}




void Initialize() override {

BuiltinTimer timer;
KRATOS_INFO("HelmholtzPartition:Initialize ") << "Starting initialization of material control "<<mControlName<<" ..." << std::endl;

CreateModelParts();

CalculateNodeNeighbourCount();

for(long unsigned int model_i=0;model_i<mpVMModelParts.size();model_i++){
StrategyType* mpStrategy = new StrategyType (*mpVMModelParts[model_i],rLinearSystemSolvers[model_i]);            
mpStrategy->Initialize();
mpStrategies.push_back(mpStrategy);
}


initial_density = mTechniqueSettings["initial_density"].GetDouble();
beta = mTechniqueSettings["beta"].GetDouble();        

double initial_filtered_density = ProjectBackward(initial_density,beta);
double initial_control_density = initial_filtered_density;

for(long unsigned int model_i=0;model_i<mpVMModelParts.size();model_i++){    
SetVariable(mpVMModelParts[model_i],CD,initial_control_density); 
SetVariable(mpVMModelParts[model_i],FD,initial_filtered_density); 
SetVariable(mpVMModelParts[model_i],PD,initial_density);
}  


for(long unsigned int model_i =0;model_i<mpVMModelParts.size();model_i++)
{
ModelPart* mpVMModePart = mpVMModelParts[model_i];
ProcessInfo &rCurrentProcessInfo = (mpVMModePart)->GetProcessInfo();
rCurrentProcessInfo[COMPUTE_CONTROL_DENSITIES] = false;
}        

KRATOS_INFO("HelmholtzPartition:Initialize") << "Finished initialization of material control "<<mControlName<<" in " << timer.ElapsedSeconds() << " s." << std::endl;

};
void Update() override {

opt_itr++;


std::cout<<"++++++++++++++++++++++ beta : "<<beta<<" ++++++++++++++++++++++"<<std::endl;

ComputeFilteredDensity();
ComputePhyiscalDensity();

};  
void MapControlUpdate(const Variable<double> &rOriginVariable, const Variable<double> &rDestinationVariable) override{};
void MapFirstDerivative(const Variable<double> &rDerivativeVariable, const Variable<double> &rMappedDerivativeVariable) override{


BuiltinTimer timer;
KRATOS_INFO("") << std::endl;
KRATOS_INFO("HelmholtzPartition:MapFirstDerivative") << "Starting mapping of " << rDerivativeVariable.Name() << "..." << std::endl;

for(long unsigned int model_i =0;model_i<mpVMModelParts.size();model_i++)
{
ModelPart* mpVMModePart = mpVMModelParts[model_i];
SetVariable1ToVarible2(mpVMModePart,rDerivativeVariable,HELMHOLTZ_SOURCE_DENSITY);
SetVariable(mpVMModePart,HELMHOLTZ_VAR_DENSITY,0.0);

mpStrategies[model_i]->Solve();
SetVariable1ToVarible2(mpVMModePart,HELMHOLTZ_VAR_DENSITY,rMappedDerivativeVariable);
}
KRATOS_INFO("HelmholtzPartition:MapFirstDerivative") << "Finished mapping in " << timer.ElapsedSeconds() << " s." << std::endl;
};  






virtual std::string Info() const override
{
return "HelmholtzPartition";
}

virtual void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "HelmholtzPartition";
}

virtual void PrintData(std::ostream& rOStream) const override
{
}





protected:



std::vector<LinearSolverType::Pointer> rLinearSystemSolvers;
std::vector<StrategyType*>mpStrategies;
std::vector<ModelPart*> mpVMModelParts;
std::vector<Properties::Pointer> mpVMModelPartsProperties;
Parameters mTechniqueSettings;
double beta;
int opt_itr = 0;
double initial_density;












private:









void CalculateNodeNeighbourCount()
{
for(long unsigned int model_i =0;model_i<mpVMModelParts.size();model_i++)
{
ModelPart* mpVMModePart = mpVMModelParts[model_i];
auto& r_nodes = mpVMModePart->Nodes();
int mNumNodes = r_nodes.size();

VariableUtils variable_utils;
variable_utils.SetFlag(STRUCTURE,true,r_nodes);

for (auto& r_node : r_nodes)
{
r_node.SetValue(NUMBER_OF_NEIGHBOUR_ELEMENTS,0);
}

mNumNodes = mpVMModePart->GetCommunicator().GetDataCommunicator().SumAll(mNumNodes);

auto& r_elements = mpVMModePart->Elements();
const int num_elements = r_elements.size();

#pragma omp parallel for
for (int i = 0; i < num_elements; i++)
{
auto i_elem = r_elements.begin() + i;
auto& r_geom = i_elem->GetGeometry();
for (unsigned int i = 0; i < r_geom.PointsNumber(); i++)
{
auto& r_node = r_geom[i];
if (r_node.Is(STRUCTURE))
{
r_node.SetLock();
r_node.GetValue(NUMBER_OF_NEIGHBOUR_ELEMENTS) += 1;
r_node.UnSetLock();
}
}
}

mpVMModePart->GetCommunicator().AssembleNonHistoricalData(NUMBER_OF_NEIGHBOUR_ELEMENTS);

}
} 

void CreateModelParts()
{
for(auto& control_obj : mControlSettings["controlling_objects"]){
ModelPart& r_controlling_object = mrModel.GetModelPart(control_obj.GetString());
ModelPart& root_model_part = r_controlling_object.GetRootModelPart();
std::string vm_model_part_name =  root_model_part.Name()+"_HELMHOLTZ_PARTITION_Part";
ModelPart* p_vm_model_part;
Properties::Pointer p_vm_model_part_property;

if (root_model_part.HasSubModelPart(vm_model_part_name)){
p_vm_model_part = &(root_model_part.GetSubModelPart(vm_model_part_name));
for(long unsigned int i =0; i<mpVMModelParts.size(); i++)
if(mpVMModelParts[i]->Name()==p_vm_model_part->Name())
p_vm_model_part_property = mpVMModelPartsProperties[i];
}
else{
p_vm_model_part = &(root_model_part.CreateSubModelPart(vm_model_part_name));
p_vm_model_part_property = p_vm_model_part->CreateNewProperties(root_model_part.NumberOfProperties()+1);
mpVMModelPartsProperties.push_back(p_vm_model_part_property);
mpVMModelParts.push_back(p_vm_model_part);
}

p_vm_model_part_property->SetValue(HELMHOLTZ_RADIUS_DENSITY,mTechniqueSettings["filter_radius"].GetDouble());

for(auto& node : r_controlling_object.Nodes())
p_vm_model_part->AddNode(&node);

ModelPart::ElementsContainerType &rmesh_elements = p_vm_model_part->Elements();   

if(!(r_controlling_object.Elements().size()>0))
KRATOS_ERROR << "HelmholtzPartition:CreateModelParts : controlling model part " <<control_obj.GetString()<<" does not have elements"<<std::endl;

for (int i = 0; i < (int)r_controlling_object.Elements().size(); i++) {
ModelPart::ElementsContainerType::iterator it = r_controlling_object.ElementsBegin() + i;
const Properties& elem_i_prop = it->GetProperties();
Properties::Pointer elem_i_new_prop = r_controlling_object.CreateNewProperties(r_controlling_object.NumberOfProperties()+1);
*elem_i_new_prop = elem_i_prop;
it->SetProperties(elem_i_new_prop);
Element::Pointer p_element = new HelmholtzBulkElement(it->Id(), it->pGetGeometry(), p_vm_model_part_property);
rmesh_elements.push_back(p_element);
}   
}

for(long unsigned int model_i =0;model_i<mpVMModelParts.size();model_i++)
{
ModelPart* mpVMModePart = mpVMModelParts[model_i];
for(auto& node_i : mpVMModePart->Nodes())
{
node_i.AddDof(HELMHOLTZ_VAR_DENSITY);
}
}     

}

void ComputeFilteredDensity(){   

for(long unsigned int model_i=0;model_i<mpVMModelParts.size();model_i++){

AddVariable1ToVarible2(mpVMModelParts[model_i],D_CD,CD);

SetVariable(mpVMModelParts[model_i],HELMHOLTZ_VAR_DENSITY,0.0);
SetVariable(mpVMModelParts[model_i],HELMHOLTZ_SOURCE_DENSITY,0.0);
for(auto& elem_i : mpVMModelParts[model_i]->Elements())
{
VectorType origin_values;
GetElementVariableValuesVector(elem_i,CD,origin_values);
MatrixType mass_matrix;
elem_i.Calculate(HELMHOLTZ_MASS_MATRIX,mass_matrix,mpVMModelParts[model_i]->GetProcessInfo());           
VectorType int_vals = prod(mass_matrix,origin_values);
AddElementVariableValuesVector(elem_i,HELMHOLTZ_SOURCE_DENSITY,int_vals);
}

mpStrategies[model_i]->Solve();
SetVariable1ToVarible2(mpVMModelParts[model_i],HELMHOLTZ_VAR_DENSITY,FD);
}        
} 

void ComputePhyiscalDensity(){

for(long unsigned int model_i=0;model_i<mpVMModelParts.size();model_i++){
for(auto& node_i : mpVMModelParts[model_i]->Nodes()){
const auto& filtered_density = node_i.FastGetSolutionStepValue(FD);
auto& physical_density = node_i.FastGetSolutionStepValue(PD);
auto& physical_density_der = node_i.FastGetSolutionStepValue(D_PD_D_FD);
physical_density = ProjectForward(filtered_density,beta);
physical_density_der = FirstFilterDerivative(filtered_density,beta);
}
}

for(auto& control_obj : mControlSettings["controlling_objects"]){
ModelPart& r_controlling_object = mrModel.GetModelPart(control_obj.GetString());
for (int i = 0; i < (int)r_controlling_object.Elements().size(); i++) {
ModelPart::ElementsContainerType::iterator it = r_controlling_object.ElementsBegin() + i;
double elem_i_density = 0.0;
for(unsigned int node_element = 0; node_element<it->GetGeometry().size(); node_element++)
elem_i_density += it->GetGeometry()[node_element].FastGetSolutionStepValue(PD);
elem_i_density /= it->GetGeometry().size();
it->GetProperties().SetValue(DENSITY,elem_i_density);
}
}
}

double ProjectForward(double x,double beta){

double x_c = 1.0;
double x_f = 0.0;

double pow_val = -2.0*beta*(x-(x_c+x_f)/2);
double value = (x_c-x_f)/(1+std::exp(pow_val)) + x_f;
if(value>1.0)
value = 1.0;
if(value<0.0)
value = 0.0;         
return value;
}

double ProjectBackward(double y,double beta){

double y_c = 1.0;
double y_f = 0.0; 

if((y>y_f) && (y<y_c))
return ((y_f+y_c)/2.0) + (1.0/(-2.0*beta)) * std::log(((y_c-y_f)/(y-y_f))-1);
else if(y==y_c)
return y_c;
else if(y==y_f)
return y_f;
else if(y>y_c)
return y_c;
else if(y<y_f)
return y_f;
else
return 0.0;            
}

double FirstFilterDerivative(double x, double beta){

double x_c = 1.0;
double x_f = 0.0;

double pow_val = -2.0*beta*(x-(x_c+x_f)/2);
return (1.0/(1+std::exp(pow_val))) * (1.0/(1+std::exp(pow_val))) * 2.0 * beta * std::exp(pow_val);
}    

void GetElementVariableValuesVector(const Element& rElement,
const Variable<double> &rVariable,
VectorType &rValues) const
{
const GeometryType &rgeom = rElement.GetGeometry();
const SizeType num_nodes = rgeom.PointsNumber();

const unsigned int local_size = num_nodes;

if (rValues.size() != local_size)
rValues.resize(local_size, false);

for (SizeType i_node = 0; i_node < num_nodes; ++i_node) {
const auto& r_nodal_variable = rgeom[i_node].FastGetSolutionStepValue(rVariable);    
rValues[i_node] = r_nodal_variable; 
}
}
void GetConditionVariableValuesVector(const Condition& rCondition,
const Variable<double> &rVariable,
VectorType &rValues) const
{
const GeometryType &rgeom = rCondition.GetGeometry();
const SizeType num_nodes = rgeom.PointsNumber();
const unsigned int local_size = num_nodes;

if (rValues.size() != local_size)
rValues.resize(local_size, false);

SizeType index = 0;
for (SizeType i_node = 0; i_node < num_nodes; ++i_node) {
const auto& r_nodal_variable = rgeom[i_node].FastGetSolutionStepValue(rVariable);    
rValues[index++] = r_nodal_variable;
}
}    
void AddElementVariableValuesVector(Element& rElement,
const Variable<double> &rVariable,
const VectorType &rValues,
const double& rWeight = 1.0
) 
{
GeometryType &rgeom = rElement.GetGeometry();
const SizeType num_nodes = rgeom.PointsNumber();

for (SizeType i_node = 0; i_node < num_nodes; ++i_node) {
auto& r_nodal_variable = rgeom[i_node].FastGetSolutionStepValue(rVariable);
r_nodal_variable += rWeight * rValues[i_node];
}
}

void AddConditionVariableValuesVector(Condition& rCondition,
const Variable<double> &rVariable,
const VectorType &rValues,
const double& rWeight = 1.0
) 
{
GeometryType &rgeom = rCondition.GetGeometry();
const SizeType num_nodes = rgeom.PointsNumber();

SizeType index = 0;
for (SizeType i_node = 0; i_node < num_nodes; ++i_node) {
auto& r_nodal_variable = rgeom[i_node].FastGetSolutionStepValue(rVariable);
r_nodal_variable += rWeight * rValues[index++];
}
}    

void SetVariable(ModelPart* mpVMModePart, const Variable<double> &rVariable, const double value) 
{
for(auto& node_i : mpVMModePart->Nodes())
{
auto& r_nodal_variable = node_i.FastGetSolutionStepValue(rVariable);
r_nodal_variable = value;                   
}
}

void SetVariable1ToVarible2(ModelPart* mpVMModePart,const Variable<double> &rVariable1,const Variable<double> &rVariable2) 
{
for(auto& node_i : mpVMModePart->Nodes())
{
auto& r_nodal_variable1 = node_i.FastGetSolutionStepValue(rVariable1);
auto& r_nodal_variable2 = node_i.FastGetSolutionStepValue(rVariable2);
r_nodal_variable2 = r_nodal_variable1;                  
}
}    

void AddVariable1ToVarible2(ModelPart* mpVMModePart,const Variable<double> &rVariable1,const Variable<double> &rVariable2) 
{
for(auto& node_i : mpVMModePart->Nodes())
{
auto& r_nodal_variable1 = node_i.FastGetSolutionStepValue(rVariable1);
auto& r_nodal_variable2 = node_i.FastGetSolutionStepValue(rVariable2);
r_nodal_variable2 += r_nodal_variable1;                  
}
}    








}; 







}  

#endif 
