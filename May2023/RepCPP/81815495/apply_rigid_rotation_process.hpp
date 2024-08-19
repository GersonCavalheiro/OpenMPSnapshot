
#if !defined(KRATOS_APPLY_RIGID_ROTATION_PROCESS )
#define  KRATOS_APPLY_RIGID_ROTATION_PROCESS

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"
#include "utilities/math_utils.h"

#include "swimming_dem_application_variables.h"

namespace Kratos
{


class ApplyRigidRotationProcess : public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(ApplyRigidRotationProcess);


ApplyRigidRotationProcess(ModelPart& model_part,
Parameters& rParameters
) : Process(Flags()) , mr_model_part(model_part)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"angular_velocity": 0.0,
"rotation_axis_initial_point": [0.0,0.0,0.0],
"rotation_axis_final_point": [0.0,0.0,1.0],
"initial_time": 0.0
}  )" );

rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mangular_velocity = rParameters["angular_velocity"].GetDouble();
maxis_initial_point = rParameters["rotation_axis_initial_point"].GetVector();
maxis_final_point = rParameters["rotation_axis_final_point"].GetVector();
minitial_time = rParameters["initial_time"].GetDouble();

KRATOS_CATCH("");
}


~ApplyRigidRotationProcess() override {}


void Execute() override
{
}

void ExecuteInitialize() override
{
KRATOS_TRY;

array_1d<double,3> rotation_axis;
noalias(rotation_axis) = maxis_final_point - maxis_initial_point;
const double axis_norm = norm_2(rotation_axis);
if (axis_norm > 1.0e-15){
rotation_axis[0] *= 1.0/axis_norm;
rotation_axis[1] *= 1.0/axis_norm;
rotation_axis[2] *= 1.0/axis_norm;
}

midentity_matrix = ZeroMatrix(3,3);
midentity_matrix(0,0) = 1.0;
midentity_matrix(1,1) = 1.0;
midentity_matrix(2,2) = 1.0;

for (int i = 0; i < 3; ++i) {
for (int j = 0; j < 3; ++j) {
maxis_matrix(i,j) = rotation_axis[i] * rotation_axis[j];
}
}

mantisym_axis_matrix = ZeroMatrix(3, 3);
mantisym_axis_matrix(0, 1) = - rotation_axis[2];
mantisym_axis_matrix(0, 2) =   rotation_axis[1];
mantisym_axis_matrix(1, 0) =   rotation_axis[2];
mantisym_axis_matrix(1, 2) = - rotation_axis[0];
mantisym_axis_matrix(2, 0) = - rotation_axis[1];
mantisym_axis_matrix(2, 1) =   rotation_axis[0];

KRATOS_CATCH("");
}

void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY;

const int nnodes = static_cast<int>(mr_model_part.Nodes().size());

if(nnodes != 0)
{
const double current_time = mr_model_part.GetProcessInfo()[TIME];

if(current_time >= minitial_time)
{
this->CalculateRodriguesMatrices(current_time);

ModelPart::NodesContainerType::iterator it_begin = mr_model_part.NodesBegin();

#pragma omp parallel for
for(int i = 0; i<nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

noalias(it->Coordinates()) = maxis_initial_point + prod(mrotation_matrix, it->GetInitialPosition().Coordinates() - maxis_initial_point);


array_1d<double, 3>& mesh_velocity = it->FastGetSolutionStepValue(MESH_VELOCITY);
noalias(mesh_velocity) = prod(mrotation_dt_matrix, it->GetInitialPosition().Coordinates() - maxis_initial_point);

}
}
}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "ApplyRigidRotationProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ApplyRigidRotationProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mr_model_part;

double mangular_velocity;
double minitial_time;
array_1d<double, 3> maxis_initial_point;
array_1d<double, 3> maxis_final_point;
BoundedMatrix<double, 3, 3> midentity_matrix;
BoundedMatrix<double, 3, 3> maxis_matrix;
BoundedMatrix<double, 3, 3> mantisym_axis_matrix;
BoundedMatrix<double, 3, 3> mrotation_matrix;
BoundedMatrix<double, 3, 3> mrotation_dt_matrix;


private:

void CalculateRodriguesMatrices(const double current_time)
{
const double delta_time = current_time - minitial_time;
const double sin_theta = std::sin(delta_time * mangular_velocity);
const double cos_theta = std::cos(delta_time * mangular_velocity);

noalias(mrotation_matrix) = cos_theta * midentity_matrix
+ sin_theta * mantisym_axis_matrix
+ (1.0 - cos_theta) * maxis_matrix;

noalias(mrotation_dt_matrix) = - mangular_velocity * sin_theta * midentity_matrix
+ mangular_velocity * cos_theta * mantisym_axis_matrix
+ mangular_velocity * sin_theta * maxis_matrix;
}

ApplyRigidRotationProcess& operator=(ApplyRigidRotationProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
ApplyRigidRotationProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ApplyRigidRotationProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
