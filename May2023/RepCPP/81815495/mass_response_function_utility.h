
#pragma once

#include <iostream>
#include <string>


#include "includes/define.h"
#include "includes/kratos_parameters.h"
#include "includes/model_part.h"
#include "utilities/variable_utils.h"
#include "processes/find_nodal_neighbours_process.h"
#include "structural_mechanics_application_variables.h"
#include "custom_processes/total_structural_mass_process.h"


namespace Kratos
{








class MassResponseFunctionUtility
{
public:

typedef array_1d<double, 3> array_3d;

KRATOS_CLASS_POINTER_DEFINITION(MassResponseFunctionUtility);


MassResponseFunctionUtility(ModelPart& model_part, Parameters responseSettings)
: mrModelPart(model_part)
{
std::string gradient_mode = responseSettings["gradient_mode"].GetString();
if (gradient_mode.compare("finite_differencing") == 0)
{
double delta = responseSettings["step_size"].GetDouble();
mDelta = delta;
}
else
KRATOS_ERROR << "Specified gradient_mode '" << gradient_mode << "' not recognized. The only option is: finite_differencing" << std::endl;
}

virtual ~MassResponseFunctionUtility()
{
}



void Initialize()
{}

double CalculateValue()
{
KRATOS_TRY;

double total_mass = 0.0;
const std::size_t domain_size = mrModelPart.GetProcessInfo()[DOMAIN_SIZE];

for (auto& elem_i : mrModelPart.Elements()){
const bool element_is_active = elem_i.IsDefined(ACTIVE) ? elem_i.Is(ACTIVE) : true;
if(element_is_active)
total_mass += TotalStructuralMassProcess::CalculateElementMass(elem_i, domain_size);
}

return total_mass;

KRATOS_CATCH("");
}

void CalculateGradient()
{
KRATOS_TRY;


VariableUtils().SetHistoricalVariableToZero(SHAPE_SENSITIVITY, mrModelPart.Nodes());

const std::size_t domain_size = mrModelPart.GetProcessInfo()[DOMAIN_SIZE];

FindNodalNeighboursProcess neighorFinder(mrModelPart);
neighorFinder.Execute();

for(auto& node_i : mrModelPart.Nodes())
{
GlobalPointersVector<Element >& ng_elem = node_i.GetValue(NEIGHBOUR_ELEMENTS);

double mass_before_fd = 0.0;
for(std::size_t i = 0; i < ng_elem.size(); i++)
{
Element& ng_elem_i = ng_elem[i];
const bool element_is_active = ng_elem_i.IsDefined(ACTIVE) ? ng_elem_i.Is(ACTIVE) : true;

if(element_is_active)
mass_before_fd += TotalStructuralMassProcess::CalculateElementMass(ng_elem_i, domain_size);
}

array_3d gradient(3, 0.0);

double mass_after_fd = 0.0;
node_i.X() += mDelta;
node_i.X0() += mDelta;
for(std::size_t i = 0; i < ng_elem.size(); i++)
{
Element& ng_elem_i = ng_elem[i];
const bool element_is_active = ng_elem_i.IsDefined(ACTIVE) ? ng_elem_i.Is(ACTIVE) : true;

if(element_is_active)
mass_after_fd += TotalStructuralMassProcess::CalculateElementMass(ng_elem_i, domain_size);
}
gradient[0] = (mass_after_fd - mass_before_fd) / mDelta;
node_i.X() -= mDelta;
node_i.X0() -= mDelta;

mass_after_fd = 0.0;
node_i.Y() += mDelta;
node_i.Y0() += mDelta;
for(std::size_t i = 0; i < ng_elem.size(); i++)
{
Element& ng_elem_i = ng_elem[i];
const bool element_is_active = ng_elem_i.IsDefined(ACTIVE) ? ng_elem_i.Is(ACTIVE) : true;

if(element_is_active)
mass_after_fd += TotalStructuralMassProcess::CalculateElementMass(ng_elem_i, domain_size);
}
gradient[1] = (mass_after_fd - mass_before_fd) / mDelta;
node_i.Y() -= mDelta;
node_i.Y0() -= mDelta;

mass_after_fd = 0.0;
node_i.Z() += mDelta;
node_i.Z0() += mDelta;
for(std::size_t i = 0; i < ng_elem.size(); i++)
{
Element& ng_elem_i = ng_elem[i];
const bool element_is_active = ng_elem_i.IsDefined(ACTIVE) ? ng_elem_i.Is(ACTIVE) : true;

if(element_is_active)
mass_after_fd += TotalStructuralMassProcess::CalculateElementMass(ng_elem_i, domain_size);
}
gradient[2] = (mass_after_fd - mass_before_fd) / mDelta;
node_i.Z() -= mDelta;
node_i.Z0() -= mDelta;

noalias(node_i.FastGetSolutionStepValue(SHAPE_SENSITIVITY)) = gradient;

}

KRATOS_CATCH("");
}





std::string Info() const
{
return "MassResponseFunctionUtility";
}

virtual void PrintInfo(std::ostream &rOStream) const
{
rOStream << "MassResponseFunctionUtility";
}

virtual void PrintData(std::ostream &rOStream) const
{
}



protected:









private:


ModelPart &mrModelPart;
double mDelta;









}; 





} 
