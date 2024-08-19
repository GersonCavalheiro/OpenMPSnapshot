
#if !defined(KRATOS_LAGRANGIAN_ROTATION_PROCESS)
#define KRATOS_LAGRANGIAN_ROTATION_PROCESS

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"
#include "utilities/math_utils.h"

#include "pfem_fluid_dynamics_application_variables.h"

namespace Kratos
{


class LagrangianRotationProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(LagrangianRotationProcess);


LagrangianRotationProcess(ModelPart &model_part,
Parameters rParameters) : Process(Flags()), mr_model_part(model_part)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"angular_velocity": 0.0,
"rotation_axis_initial_point": [0.0,0.0,0.0],
"rotation_axis_final_point": [0.0,0.0,1.0],
"initial_time": 0.0,
"final_time": 10.0
}  )");

rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mangular_velocity = rParameters["angular_velocity"].GetDouble();
maxis_initial_point = rParameters["rotation_axis_initial_point"].GetVector();
maxis_final_point = rParameters["rotation_axis_final_point"].GetVector();
minitial_time = rParameters["initial_time"].GetDouble();
mfinal_time = rParameters["final_time"].GetDouble();

KRATOS_CATCH("");
}


~LagrangianRotationProcess() override {}


void Execute() override
{
}

void ExecuteInitialize() override
{
KRATOS_TRY;

array_1d<double, 3> rotation_axis;
noalias(rotation_axis) = maxis_final_point - maxis_initial_point;
const double axis_norm = norm_2(rotation_axis);
if (axis_norm > 1.0e-15)
{
rotation_axis[0] *= 1.0 / axis_norm;
rotation_axis[1] *= 1.0 / axis_norm;
rotation_axis[2] *= 1.0 / axis_norm;
}

noalias(midentity_matrix) = ZeroMatrix(3, 3);
midentity_matrix(0, 0) = 1.0;
midentity_matrix(1, 1) = 1.0;
midentity_matrix(2, 2) = 1.0;

for (int i = 0; i < 3; ++i)
{
for (int j = 0; j < 3; ++j)
{
maxis_matrix(i, j) = rotation_axis[i] * rotation_axis[j];
}
}

noalias(mantisym_axis_matrix) = ZeroMatrix(3, 3);
mantisym_axis_matrix(0, 1) = -rotation_axis[2];
mantisym_axis_matrix(0, 2) = rotation_axis[1];
mantisym_axis_matrix(1, 0) = rotation_axis[2];
mantisym_axis_matrix(1, 2) = -rotation_axis[0];
mantisym_axis_matrix(2, 0) = -rotation_axis[1];
mantisym_axis_matrix(2, 1) = rotation_axis[0];

KRATOS_CATCH("");
}













void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY;

const int nnodes = static_cast<int>(mr_model_part.Nodes().size());

if (nnodes != 0)
{
const double current_time = mr_model_part.GetProcessInfo()[TIME];

this->CalculateRodriguesMatrices(current_time);

ModelPart::NodesContainerType::iterator it_begin = mr_model_part.NodesBegin();
array_1d<double, 3> point_initial_position;

#pragma omp parallel for private(point_initial_position)
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

noalias(point_initial_position) = it->Coordinates();

it->Fix(VELOCITY_X);
it->Fix(VELOCITY_Y);
it->Fix(VELOCITY_Z);

array_1d<double, 3> &velocity = it->FastGetSolutionStepValue(VELOCITY);
noalias(velocity) = prod(mrotation_dt_matrix, point_initial_position - maxis_initial_point);
}



}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "LagrangianRotationProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "LagrangianRotationProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

ModelPart &mr_model_part;

double mangular_velocity;
double minitial_time;
double mfinal_time;
array_1d<double, 3> maxis_initial_point;
array_1d<double, 3> maxis_final_point;
BoundedMatrix<double, 3, 3> midentity_matrix;
BoundedMatrix<double, 3, 3> maxis_matrix;
BoundedMatrix<double, 3, 3> mantisym_axis_matrix;
BoundedMatrix<double, 3, 3> mrotation_dt_matrix;


private:
void CalculateRodriguesMatrices(const double current_time)
{
const double delta_time = mr_model_part.GetProcessInfo()[DELTA_TIME];
double current_angular_velocity = mangular_velocity * current_time / minitial_time; 

if (current_time >= minitial_time)
{
current_angular_velocity = mangular_velocity;
}

if (current_time > mfinal_time)
{
current_angular_velocity = 0;
}

double sin_theta = std::sin(delta_time * current_angular_velocity);
double cos_theta = std::cos(delta_time * current_angular_velocity);

noalias(mrotation_dt_matrix) = -current_angular_velocity * sin_theta * midentity_matrix + current_angular_velocity * cos_theta * mantisym_axis_matrix + current_angular_velocity * sin_theta * maxis_matrix;
}

LagrangianRotationProcess &operator=(LagrangianRotationProcess const &rOther);


}; 

inline std::istream &operator>>(std::istream &rIStream,
LagrangianRotationProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const LagrangianRotationProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
