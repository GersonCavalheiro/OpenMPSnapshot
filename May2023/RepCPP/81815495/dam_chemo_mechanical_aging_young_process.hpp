
#if !defined(KRATOS_DAM_CHEMO_MECHANICAL_AGING_YOUNG_PROCESS)
#define KRATOS_DAM_CHEMO_MECHANICAL_AGING_YOUNG_PROCESS

#include <cmath>

#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

#include "dam_application_variables.h"

namespace Kratos
{

class DamChemoMechanicalAgingYoungProcess : public Process
{

public:
KRATOS_CLASS_POINTER_DEFINITION(DamChemoMechanicalAgingYoungProcess);


DamChemoMechanicalAgingYoungProcess(ModelPart &rModelPart,
Parameters &rParameters) : Process(Flags()), mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters(R"(
{
"model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
"variable_name": "PLEASE_PRESCRIBE_VARIABLE_NAME",
"initial_elastic_modulus"                          : 30.0e9,
"initial_porosity"                                 : 0.2,
"max_chemical_porosity"                            : 0.32,
"chemical_characteristic_aging_time"               : 100.0,
"max_mechanical_damage"                            : 0.32,
"damage_characteristic_aging_time"                 : 100.0,
"interval":[
0.0,
0.0
]
}  )");

rParameters["initial_elastic_modulus"];
rParameters["initial_porosity"];
rParameters["max_chemical_porosity"];

rParameters.ValidateAndAssignDefaults(default_parameters);

mVariableName = rParameters["variable_name"].GetString();
mInitialElasticModulus = rParameters["initial_elastic_modulus"].GetDouble();
mInitialPorosity = rParameters["initial_porosity"].GetDouble();
mMaxChemicalPorosity = rParameters["max_chemical_porosity"].GetDouble();
mChemicalTime = rParameters["chemical_characteristic_aging_time"].GetDouble();
mMaxMechaDamage = rParameters["max_mechanical_damage"].GetDouble();
mDamageTime = rParameters["damage_characteristic_aging_time"].GetDouble();

KRATOS_CATCH("");
}


virtual ~DamChemoMechanicalAgingYoungProcess() {}



void Execute() override
{
}



void ExecuteInitialize() override
{
KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

double time = mrModelPart.GetProcessInfo()[TIME] / 31536000.0;

double sound_concrete = mInitialElasticModulus * sqrt(1.0 + 0.0805 * log(time));
double chemical_porosity = mMaxChemicalPorosity * (1.0 - exp(-time / mChemicalTime));
double damage_mechanical = mMaxMechaDamage * (1.0 - exp(-time / mDamageTime));
double young = ((1.0 - mInitialPorosity - chemical_porosity) * (1.0 - damage_mechanical) * sound_concrete) / (1.0 - mInitialPorosity);

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->FastGetSolutionStepValue(var) = young;
}
}

KRATOS_CATCH("");
}


void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY;

const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

double time = mrModelPart.GetProcessInfo()[TIME] / 31536000.0;

double sound_concrete = mInitialElasticModulus * sqrt(1.0 + 0.0805 * log(time));
double chemical_porosity = mMaxChemicalPorosity * (1.0 - exp(-time / mChemicalTime));
double damage_mechanical = mMaxMechaDamage * (1.0 - exp(-time / mDamageTime));
double young = ((1.0 - mInitialPorosity - chemical_porosity) * (1.0 - damage_mechanical) * sound_concrete) / (1.0 - mInitialPorosity);

if (nnodes != 0)
{
ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

#pragma omp parallel for
for (int i = 0; i < nnodes; i++)
{
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->FastGetSolutionStepValue(var) = young;
}
}

KRATOS_CATCH("");
}


std::string Info() const override
{
return "DamChemoMechanicalAgingYoungProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "DamChemoMechanicalAgingYoungProcess";
}

void PrintData(std::ostream &rOStream) const override
{
}


protected:
ModelPart &mrModelPart;
std::string mVariableName;
double mInitialElasticModulus;
double mInitialPorosity;
double mMaxChemicalPorosity;
double mChemicalTime;
double mMaxMechaDamage;
double mDamageTime;


private:
DamChemoMechanicalAgingYoungProcess &operator=(DamChemoMechanicalAgingYoungProcess const &rOther);

}; 

inline std::istream &operator>>(std::istream &rIStream,
DamChemoMechanicalAgingYoungProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const DamChemoMechanicalAgingYoungProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
