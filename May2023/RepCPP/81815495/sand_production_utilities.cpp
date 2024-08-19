
#include "custom_elements/spheric_continuum_particle.h"
#include "sand_production_utilities.hpp"

#include "dem_structures_coupling_application_variables.h"


namespace Kratos
{


SandProductionUtilities::SandProductionUtilities(){}


SandProductionUtilities::~SandProductionUtilities(){}


void SandProductionUtilities::MarkSandProductionParticlesForErasing(ModelPart& r_model_part) {

KRATOS_TRY

Configure::ElementsContainerType& rElements = r_model_part.GetCommunicator().LocalMesh().Elements();
ProcessInfo& r_process_info = r_model_part.GetProcessInfo();
const double sigma_3_average = fabs(r_process_info[SIGMA_3_AVERAGE]);
const double percentage_of_stress = 0.1; 

#pragma omp parallel for
for (int k = 0; k < (int)rElements.size(); k++){
Configure::ElementsContainerType::ptr_iterator particle_pointer_it = rElements.ptr_begin() + k;

if ((*particle_pointer_it)->Is(DEMFlags::BELONGS_TO_A_CLUSTER)) continue;
if ((*particle_pointer_it)->Is(BLOCKED)) continue;
if ((*particle_pointer_it)->Is(DEMFlags::STICKY)) continue;

Element* p_element = particle_pointer_it->get();
SphericContinuumParticle* pDemElem = dynamic_cast<SphericContinuumParticle*>(p_element);
BoundedMatrix<double, 3, 3> stress_tensor = (*(pDemElem->mSymmStressTensor));
Vector principal_stresses(3);
noalias(principal_stresses) = AuxiliaryFunctions::EigenValuesDirectMethod(stress_tensor);
const double particle_sigma_3 = fabs(*std::min_element(principal_stresses.begin(), principal_stresses.end()));

bool the_particle_plays_a_structural_role = (particle_sigma_3 > percentage_of_stress * sigma_3_average);

the_particle_plays_a_structural_role = false;

if ((*particle_pointer_it)->Is(DEMFlags::IS_SAND_PRODUCTION) && !the_particle_plays_a_structural_role) {
(*particle_pointer_it)->GetGeometry()[0].Set(TO_ERASE);
(*particle_pointer_it)->Set(TO_ERASE);
}
}

KRATOS_CATCH("")
}

std::string SandProductionUtilities::Info() const
{
return "";
}


void SandProductionUtilities::PrintInfo(std::ostream& rOStream) const
{
}


void SandProductionUtilities::PrintData(std::ostream& rOStream) const
{
}


}  
