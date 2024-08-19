


#include "thermal_explicit_solver_strategy.h"

namespace Kratos {


ThermalExplicitSolverStrategy::ThermalExplicitSolverStrategy():ExplicitSolverStrategy() {}

ThermalExplicitSolverStrategy::ThermalExplicitSolverStrategy(ExplicitSolverSettings&            settings,
const double                       max_delta_time,
const int                          n_step_search,
const double                       safety_factor,
const int                          delta_option,
ParticleCreatorDestructor::Pointer p_creator_destructor,
DEM_FEM_Search::Pointer            p_dem_fem_search,
SpatialSearch::Pointer             pSpSearch,
Parameters                         strategy_parameters):
ExplicitSolverStrategy(settings,
max_delta_time,
n_step_search,
safety_factor,
delta_option,
p_creator_destructor,
p_dem_fem_search,
pSpSearch,
strategy_parameters) {}

ThermalExplicitSolverStrategy::~ThermalExplicitSolverStrategy() {}


void ThermalExplicitSolverStrategy::Initialize(void) {
KRATOS_TRY

ExplicitSolverStrategy::Initialize();

if (GetModelPart().GetProcessInfo()[AUTO_SOLVE_FREQUENCY_OPTION])
SetSolveFrequency();

KRATOS_CATCH("")
}

void ThermalExplicitSolverStrategy::SetSearchRadiiOnAllParticles(ModelPart& r_model_part, double added_search_distance, double amplification) {
SetSearchRadii(r_model_part, added_search_distance, amplification);
}

void ThermalExplicitSolverStrategy::SetSearchRadiiWithFemOnAllParticles(ModelPart& r_model_part, double added_search_distance, double amplification) {
SetSearchRadii(r_model_part, added_search_distance, amplification);
}


double ThermalExplicitSolverStrategy::SolveSolutionStepStatic(void) {
KRATOS_TRY

ExplicitSolverStrategy::GetForce();
PerformThermalTimeIntegration();
return 0.0;

KRATOS_CATCH("")
}

void ThermalExplicitSolverStrategy::SetSolveFrequency(void) {
KRATOS_TRY

ModelPart&         r_model_part   = GetModelPart();
ProcessInfo&       r_process_info = r_model_part.GetProcessInfo();
ElementsArrayType& r_elements     = r_model_part.GetCommunicator().LocalMesh().Elements();

const double safety_factor = 0.1;
const int    max_freq      = 10000;

double min_diff_time = DBL_MAX;

block_for_each(r_elements, [&](ModelPart::ElementType& r_element) {
Element* p_element = &(r_element);
ThermalSphericParticle* particle = dynamic_cast<ThermalSphericParticle*>(p_element);

const double radius    = particle->GetParticleRadius();
const double diff_time = radius * radius / particle->GetParticleDiffusivity();

#pragma omp critical
{
if (diff_time < min_diff_time) min_diff_time = diff_time;
}

});

min_diff_time *= safety_factor;

const double time_step = r_process_info[DELTA_TIME];
if (min_diff_time < time_step)
KRATOS_ERROR << "Time step is too large for the thermal analysis!" << std::endl;

int solve_freq = floor(min_diff_time / time_step);

if (solve_freq > max_freq)
solve_freq = max_freq;

r_process_info[THERMAL_FREQUENCY] = solve_freq;

KRATOS_CATCH("")
}

void ThermalExplicitSolverStrategy::PerformThermalTimeIntegration(void) {
KRATOS_TRY

ModelPart&         r_model_part   = GetModelPart();
ProcessInfo&       r_process_info = r_model_part.GetProcessInfo();
ElementsArrayType& r_elements     = r_model_part.GetCommunicator().LocalMesh().Elements();

block_for_each(r_elements, [&](ModelPart::ElementType& r_element) {
Element* p_element = &(r_element);
ThermalSphericParticle* particle = dynamic_cast<ThermalSphericParticle*>(p_element);

particle->Move(r_process_info[DELTA_TIME], false, 0.0, 0);
});

KRATOS_CATCH("")
}

void ThermalExplicitSolverStrategy::SetSearchRadii(ModelPart& r_model_part, double added_search_distance, double amplification) {
KRATOS_TRY

int number_of_elements = r_model_part.GetCommunicator().LocalMesh().ElementsArray().end() - r_model_part.GetCommunicator().LocalMesh().ElementsArray().begin();

IndexPartition<unsigned int>(number_of_elements).for_each([&](unsigned int i) {
ThermalSphericParticle* particle = dynamic_cast<ThermalSphericParticle*>(mListOfSphericParticles[i]);

particle->ComputeAddedSearchDistance(r_model_part.GetProcessInfo(), added_search_distance);
particle->SetSearchRadius(amplification * (added_search_distance + mListOfSphericParticles[i]->GetRadius()));
});

KRATOS_CATCH("")
}

} 
