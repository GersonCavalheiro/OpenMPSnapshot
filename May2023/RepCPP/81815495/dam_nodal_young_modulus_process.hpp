
#if !defined(KRATOS_DAM_NODAL_YOUNG_MODULUS_PROCESS)
#define KRATOS_DAM_NODAL_YOUNG_MODULUS_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamNodalYoungModulusProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamNodalYoungModulusProcess);


DamNodalYoungModulusProcess(ModelPart &rModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"is_fixed"                                         : false,
"Young_Modulus_1"                                  : 10.0,
"Young_Modulus_2"                                  : 60.0,
"Young_Modulus_3"                                  : 50.0,
"Young_Modulus_4"                                  : 70.0,
"interval":[
0.0,
0.0
]
}  )");

rParameters["Young_Modulus_1"];
rParameters["variable_name"];
rParameters["model_part_name"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mIsFixed = rParameters["is_fixed"].GetBool();
mYoung1 = rParameters["Young_Modulus_1"].GetDouble();
mYoung2 = rParameters["Young_Modulus_2"].GetDouble();
mYoung3 = rParameters["Young_Modulus_3"].GetDouble();
mYoung4 = rParameters["Young_Modulus_4"].GetDouble();

KRATOS_CATCH("");
}


virtual ~DamNodalYoungModulusProcess() {}


void Execute() override
{
}


void ExecuteInitialize() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if (mIsFixed)
{
it->Fix(var);
}

double Young = mYoung1 + (mYoung2 * it->X()) + (mYoung3 * it->Y()) + (mYoung4 * it->Z());

if (Young <= 0.0)
{
it->FastGetSolutionStepValue(var) = 0.0;
}
else
it->FastGetSolutionStepValue(var) = Young;
}
}

KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{

KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;

if (mIsFixed)
{
it->Fix(var);
}

double Young = mYoung1 + (mYoung2 * it->X()) + (mYoung3 * it->Y()) + (mYoung4 * it->Z());

if (Young <= 0.0)
{
it->FastGetSolutionStepValue(var) = 0.0;
}
else
it->FastGetSolutionStepValue(var) = Young;
}
}

KRATOS_CATCH("");
}

std::string Info() const override
{
return "DamNodalYoungModulusProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamNodalYoungModulusProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:

ModelPart &mrModelPart;
std::string mVariableName;
bool mIsFixed;
double mYoung1;
double mYoung2;
double mYoung3;
double mYoung4;


private:
DamNodalYoungModulusProcess &operator=(DamNodalYoungModulusProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamNodalYoungModulusProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamNodalYoungModulusProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
