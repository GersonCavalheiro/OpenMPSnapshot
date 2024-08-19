
#if !defined(KRATOS_ASSIGN_TORQUE_ABOUT_AN_AXIS_TO_CONDITIONS_PROCESS_H_INCLUDED)
#define  KRATOS_ASSIGN_TORQUE_ABOUT_AN_AXIS_TO_CONDITIONS_PROCESS_H_INCLUDED




#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"
#include "utilities/beam_math_utilities.hpp"

namespace Kratos
{



class AssignTorqueAboutAnAxisToConditionsProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AssignTorqueAboutAnAxisToConditionsProcess);



AssignTorqueAboutAnAxisToConditionsProcess(ModelPart& model_part) : Process(Flags()) , mrModelPart(model_part) {}



AssignTorqueAboutAnAxisToConditionsProcess(ModelPart& model_part,
Parameters rParameters
) : Process(Flags()) , mrModelPart(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"MODEL_PART_NAME",
"variable_name": "VARIABLE_NAME",
"modulus" : 1.0,
"direction" : [],
"center" : []
}  )" );


rParameters.ValidateAndAssignDefaults(default_parameters);

mvariable_name = rParameters["variable_name"].GetString();


if( KratosComponents< Variable<array_1d<double, 3> > >::Has( mvariable_name ) ) 
{

mvalue = rParameters["modulus"].GetDouble();

for( unsigned int i=0; i<3; i++)
{
mdirection[i] = rParameters["direction"][i].GetDouble();
mcenter[i] = rParameters["center"][i].GetDouble();
}

double norm = norm_2(mdirection);
if(norm!=0)
mdirection/=norm;

}
else 
{
KRATOS_ERROR << "trying to set a variable that is not in the model_part - variable name is " << mvariable_name <<std::endl;
}

KRATOS_CATCH("");
}


~AssignTorqueAboutAnAxisToConditionsProcess() override {}



void operator()()
{
Execute();
}




void Execute() override
{

KRATOS_TRY;

this->AssignTorqueAboutAnAxis(KratosComponents< Variable<array_1d<double,3> > >::Get(mvariable_name));

KRATOS_CATCH("");

}

void ExecuteInitialize() override
{
}

void ExecuteBeforeSolutionLoop() override
{
}


void ExecuteInitializeSolutionStep() override
{
}

void ExecuteFinalizeSolutionStep() override
{
}


void ExecuteBeforeOutputStep() override
{
}


void ExecuteAfterOutputStep() override
{
}


void ExecuteFinalize() override
{
array_1d<double,3> vector_value;
vector_value.clear();
InternalAssignValue(KratosComponents< Variable<array_1d<double,3> > >::Get(mvariable_name), vector_value);
}







std::string Info() const override
{
return "AssignTorqueAboutAnAxisToConditionsProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AssignTorqueAboutAnAxisToConditionsProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:


ModelPart& mrModelPart;
std::string mvariable_name;
double mvalue;
array_1d<double,3> mdirection;
array_1d<double,3> mcenter;


AssignTorqueAboutAnAxisToConditionsProcess(AssignTorqueAboutAnAxisToConditionsProcess const& rOther);


private:



void InternalAssignValue(const Variable<array_1d<double,3> >& rVariable,
const array_1d<double,3>& rvector_value)
{
const int nconditions = mrModelPart.GetMesh().Conditions().size();

if(nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin = mrModelPart.GetMesh().ConditionsBegin();

#pragma omp parallel for
for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

it->SetValue(rVariable, rvector_value);
}
}
}

void AssignTorqueAboutAnAxis(const Variable<array_1d<double,3> >& rVariable)
{
KRATOS_TRY

const int nconditions = mrModelPart.GetMesh().Conditions().size();

if(nconditions != 0)
{
ModelPart::ConditionsContainerType::iterator it_begin = mrModelPart.GetMesh().ConditionsBegin();

std::vector<array_1d<double,3> > Couples(nconditions);
std::vector<array_1d<double,3> > Forces(nconditions);

Matrix rotation_matrix;
array_1d<double,3> radius;
array_1d<double,3> distance;
array_1d<double,3> force;

#pragma omp parallel for private(rotation_matrix,radius,distance,force)
for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;

Geometry< Node >& rGeometry = it->GetGeometry();

unsigned int size  = rGeometry.size();
array_1d<double,3> couple;
couple.clear();
array_1d<double,3> moment;
moment.clear();
for ( unsigned int j = 0; j < size; j++ )
{

noalias(distance) = rGeometry[j].GetInitialPosition() - mcenter;

noalias(radius)  = distance-inner_prod(distance,mdirection) * mdirection,

BeamMathUtils<double>::VectorToSkewSymmetricTensor(mdirection, rotation_matrix);

double norm_radius = norm_2(radius);
if(norm_radius!=0)
radius/=norm_radius;

noalias(force) = prod(rotation_matrix, radius);

noalias(couple) += force*norm_radius;

double norm = norm_2(force);
if(norm!=0)
force/=norm;

BeamMathUtils<double>::VectorToSkewSymmetricTensor(radius, rotation_matrix);

noalias(moment) += norm_radius * prod(rotation_matrix, force);
}

const unsigned int dimension = rGeometry.WorkingSpaceDimension();
double domain_size = 1.0;
if(dimension==3)
domain_size = rGeometry.Area();
if(dimension==2)
domain_size = rGeometry.Length();

Couples[i] = moment * domain_size * (1.0/double(size));
Forces[i]  = couple * (1.0/double(size));

}

double total_size = 1.0;
array_1d<double,3> torque;
for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;
Geometry< Node >& rGeometry = it->GetGeometry();
const unsigned int dimension = rGeometry.WorkingSpaceDimension();
double domain_size = 0.0;
if(dimension==3)
domain_size = rGeometry.Area();
if(dimension==2)
domain_size = rGeometry.Length();

total_size += domain_size;
torque += Couples[i];
}

torque /=total_size;

double value = 0;
for(int i = 0; i<3; i++)
{
if( torque[i] != 0 )
value = mdirection[i]*mvalue/torque[i];
if( value != 0 )
break;
}

array_1d<double,3> load;
#pragma omp parallel for private(torque)
for(int i = 0; i<nconditions; i++)
{
ModelPart::ConditionsContainerType::iterator it = it_begin + i;
load = value * Forces[i];
it->SetValue(rVariable, load);
}

}

KRATOS_CATCH( "" )

}



AssignTorqueAboutAnAxisToConditionsProcess& operator=(AssignTorqueAboutAnAxisToConditionsProcess const& rOther);



}; 







inline std::istream& operator >> (std::istream& rIStream,
AssignTorqueAboutAnAxisToConditionsProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const AssignTorqueAboutAnAxisToConditionsProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
