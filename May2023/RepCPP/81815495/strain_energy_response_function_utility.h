
#pragma once

#include <iostream>
#include <string>


#include "includes/define.h"
#include "includes/kratos_parameters.h"
#include "includes/model_part.h"
#include "utilities/variable_utils.h"
#include "processes/find_nodal_neighbours_process.h"
#include "finite_difference_utility.h"


namespace Kratos
{








class StrainEnergyResponseFunctionUtility
{
public:

typedef array_1d<double, 3> array_3d;

KRATOS_CLASS_POINTER_DEFINITION(StrainEnergyResponseFunctionUtility);


StrainEnergyResponseFunctionUtility(ModelPart& model_part, Parameters responseSettings)
: mrModelPart(model_part)
{
std::string gradient_mode = responseSettings["gradient_mode"].GetString();

if (gradient_mode.compare("semi_analytic") == 0)
{
double delta = responseSettings["step_size"].GetDouble();
mDelta = delta;
}
else
KRATOS_ERROR << "Specified gradient_mode '" << gradient_mode << "' not recognized. The only option is: semi_analytic" << std::endl;

}

virtual ~StrainEnergyResponseFunctionUtility()
{
}



void Initialize()
{
}

double CalculateValue()
{
KRATOS_TRY;

const ProcessInfo &CurrentProcessInfo = mrModelPart.GetProcessInfo();
double strain_energy = 0.0;

for (auto& elem_i : mrModelPart.Elements())
{
const bool element_is_active = elem_i.IsDefined(ACTIVE) ? elem_i.Is(ACTIVE) : true;
if(element_is_active)
{
Matrix LHS;
Vector RHS;
Vector u;

const auto& rConstElemRef = elem_i;
rConstElemRef.GetValuesVector(u,0);

elem_i.CalculateLocalSystem(LHS,RHS,CurrentProcessInfo);

strain_energy += 0.5 * inner_prod(u,prod(LHS,u));
}
}

return strain_energy;

KRATOS_CATCH("");
}

void CalculateGradient()
{
KRATOS_TRY;




VariableUtils().SetHistoricalVariableToZero(SHAPE_SENSITIVITY, mrModelPart.Nodes());



CalculateResponseDerivativePartByFiniteDifferencing();
CalculateAdjointField();
CalculateStateDerivativePartByFiniteDifferencing();

KRATOS_CATCH("");
}





std::string Info() const
{
return "StrainEnergyResponseFunctionUtility";
}

virtual void PrintInfo(std::ostream &rOStream) const
{
rOStream << "StrainEnergyResponseFunctionUtility";
}

virtual void PrintData(std::ostream &rOStream) const
{
}



protected:




void CalculateAdjointField()
{
KRATOS_TRY;


KRATOS_CATCH("");
}

void CalculateStateDerivativePartByFiniteDifferencing()
{
KRATOS_TRY;

const ProcessInfo &CurrentProcessInfo = mrModelPart.GetProcessInfo();

for (auto& elem_i : mrModelPart.Elements())
{
const bool element_is_active = elem_i.IsDefined(ACTIVE) ? elem_i.Is(ACTIVE) : true;
if(element_is_active)
{
Vector u;
Vector lambda;
Vector RHS;

const auto& rConstElemRef = elem_i;
rConstElemRef.GetValuesVector(u,0);

lambda = 0.5*u;

elem_i.CalculateRightHandSide(RHS, CurrentProcessInfo);
for (auto& node_i : elem_i.GetGeometry())
{
array_3d gradient_contribution(3, 0.0);
Vector derived_RHS = Vector(0);

FiniteDifferenceUtility::CalculateRightHandSideDerivative(elem_i, RHS, SHAPE_SENSITIVITY_X, node_i, mDelta, derived_RHS, CurrentProcessInfo);
gradient_contribution[0] = inner_prod(lambda, derived_RHS);

FiniteDifferenceUtility::CalculateRightHandSideDerivative(elem_i, RHS, SHAPE_SENSITIVITY_Y, node_i, mDelta, derived_RHS, CurrentProcessInfo);
gradient_contribution[1] = inner_prod(lambda, derived_RHS);

FiniteDifferenceUtility::CalculateRightHandSideDerivative(elem_i, RHS, SHAPE_SENSITIVITY_Z, node_i, mDelta, derived_RHS, CurrentProcessInfo);
gradient_contribution[2] = inner_prod(lambda, derived_RHS);

noalias(node_i.FastGetSolutionStepValue(SHAPE_SENSITIVITY)) += gradient_contribution;
}
}
}

for (auto& cond_i : mrModelPart.Conditions())
{
const bool condition_is_active = cond_i.IsDefined(ACTIVE) ? cond_i.Is(ACTIVE) : true;
if (condition_is_active)
{
Vector u;
Vector lambda;
Vector RHS;

const auto& rConstCondRef = cond_i;
rConstCondRef.GetValuesVector(u,0);
lambda = 0.5*u;

cond_i.CalculateRightHandSide(RHS, CurrentProcessInfo);
for (auto& node_i : cond_i.GetGeometry())
{
array_3d gradient_contribution(3, 0.0);
Vector perturbed_RHS = Vector(0);

node_i.X0() += mDelta;
cond_i.CalculateRightHandSide(perturbed_RHS, CurrentProcessInfo);
gradient_contribution[0] = inner_prod(lambda, (perturbed_RHS - RHS) / mDelta);
node_i.X0() -= mDelta;

perturbed_RHS = Vector(0);

node_i.Y0() += mDelta;
cond_i.CalculateRightHandSide(perturbed_RHS, CurrentProcessInfo);
gradient_contribution[1] = inner_prod(lambda, (perturbed_RHS - RHS) / mDelta);
node_i.Y0() -= mDelta;

perturbed_RHS = Vector(0);

node_i.Z0() += mDelta;
cond_i.CalculateRightHandSide(perturbed_RHS, CurrentProcessInfo);
gradient_contribution[2] = inner_prod(lambda, (perturbed_RHS - RHS) / mDelta);
node_i.Z0() -= mDelta;

noalias(node_i.FastGetSolutionStepValue(SHAPE_SENSITIVITY)) += gradient_contribution;
}
}
}

KRATOS_CATCH("");
}

void CalculateResponseDerivativePartByFiniteDifferencing()
{
KRATOS_TRY;

const ProcessInfo &CurrentProcessInfo = mrModelPart.GetProcessInfo();

for (auto& cond_i : mrModelPart.Conditions())
{
const bool condition_is_active = cond_i.IsDefined(ACTIVE) ? cond_i.Is(ACTIVE) : true;
if (condition_is_active)
{
Vector u;
Vector RHS;

const auto& rConstCondRef = cond_i;
rConstCondRef.GetValuesVector(u,0);

cond_i.CalculateRightHandSide(RHS, CurrentProcessInfo);
for (auto& node_i : cond_i.GetGeometry())
{
array_3d gradient_contribution(3, 0.0);
Vector perturbed_RHS = Vector(0);

node_i.X0() += mDelta;
cond_i.CalculateRightHandSide(perturbed_RHS, CurrentProcessInfo);
gradient_contribution[0] = inner_prod(0.5*u, (perturbed_RHS - RHS) / mDelta);
node_i.X0() -= mDelta;

perturbed_RHS = Vector(0);

node_i.Y0() += mDelta;
cond_i.CalculateRightHandSide(perturbed_RHS, CurrentProcessInfo);
gradient_contribution[1] = inner_prod(0.5*u, (perturbed_RHS - RHS) / mDelta);
node_i.Y0() -= mDelta;

perturbed_RHS = Vector(0);

node_i.Z0() += mDelta;
cond_i.CalculateRightHandSide(perturbed_RHS, CurrentProcessInfo);
gradient_contribution[2] = inner_prod(0.5*u, (perturbed_RHS - RHS) / mDelta);
node_i.Z0() -= mDelta;

noalias(node_i.FastGetSolutionStepValue(SHAPE_SENSITIVITY)) += gradient_contribution;
}
}
}
KRATOS_CATCH("");
}






private:


ModelPart &mrModelPart;
double mDelta;









}; 





} 