

#ifndef KRATOS_STRESS_FAILURE_CHECK_UTILITIES
#define KRATOS_STRESS_FAILURE_CHECK_UTILITIES

#include <fstream>
#include <iostream>
#include <cmath>

#include "geometries/geometry.h"
#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "utilities/openmp_utils.h"
#include "utilities/parallel_utilities.h"

#include "custom_utilities/AuxiliaryFunctions.h"
#include "custom_elements/spheric_continuum_particle.h"

#include "dem_structures_coupling_application_variables.h"


namespace Kratos
{


class StressFailureCheckUtilities
{

public:

KRATOS_CLASS_POINTER_DEFINITION(StressFailureCheckUtilities);


StressFailureCheckUtilities(ModelPart& rModelPart,
Parameters& rParameters
) :
mrModelPart(rModelPart)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"cylinder_center": [0.0,0.0,0.0],
"min_radius": 0.00381,
"max_radius": 0.00481
}  )" );

rParameters.ValidateAndAssignDefaults(default_parameters);

mCylinderCenter[0] = rParameters["cylinder_center"][0].GetDouble();
mCylinderCenter[1] = rParameters["cylinder_center"][1].GetDouble();
mCylinderCenter[2] = rParameters["cylinder_center"][2].GetDouble();
mMinRadius = rParameters["min_radius"].GetDouble();
mMaxRadius = rParameters["max_radius"].GetDouble();

KRATOS_CATCH("");
}


virtual ~StressFailureCheckUtilities(){}


void ExecuteFinalizeSolutionStep()
{
std::vector<double> ThreadSigma1(ParallelUtilities::GetNumThreads(), 0.0);
std::vector<double> ThreadSigma3(ParallelUtilities::GetNumThreads(), 0.0);
std::vector<int> ThreadNParticles(ParallelUtilities::GetNumThreads(), 0);

ModelPart::ElementsContainerType& rElements = mrModelPart.GetCommunicator().LocalMesh().Elements();

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();

#pragma omp for
for (int i = 0; i < (int)rElements.size(); i++) {

ModelPart::ElementsContainerType::ptr_iterator ptr_itElem = rElements.ptr_begin() + i;

const array_1d<double,3> DemPosition = (*ptr_itElem)->GetGeometry()[0].Coordinates();
const double Distance2 = std::pow(DemPosition[0] - mCylinderCenter[0], 2) + std::pow(DemPosition[1] - mCylinderCenter[1], 2);

Element* p_element = ptr_itElem->get();
SphericContinuumParticle* pDemElem = dynamic_cast<SphericContinuumParticle*>(p_element);

if ((pDemElem->IsNot(DEMFlags::STICKY)) && (Distance2 >= std::pow(mMinRadius,2)) && (Distance2 <= std::pow(mMaxRadius,2))) {

BoundedMatrix<double, 3, 3> stress_tensor = (*(pDemElem->mSymmStressTensor));
Vector principal_stresses(3);
noalias(principal_stresses) = AuxiliaryFunctions::EigenValuesDirectMethod(stress_tensor);
const double max_stress = *std::max_element(principal_stresses.begin(), principal_stresses.end());
const double min_stress = *std::min_element(principal_stresses.begin(), principal_stresses.end());
ThreadSigma1[k] += max_stress;
ThreadSigma3[k] += min_stress;
ThreadNParticles[k] += 1;
}
}
}

double Sigma1Average = 0.0;
double Sigma3Average = 0.0;
int NParticles = 0;
for (int k = 0; k < ParallelUtilities::GetNumThreads(); k++) {
Sigma1Average += ThreadSigma1[k];
Sigma3Average += ThreadSigma3[k];
NParticles += ThreadNParticles[k];
}

if (NParticles > 0) {
Sigma1Average = Sigma1Average / NParticles;
Sigma3Average = Sigma3Average / NParticles;
}

double CurrentTime = mrModelPart.GetProcessInfo()[TIME];
mrModelPart.GetProcessInfo()[SIGMA_3_AVERAGE] = Sigma3Average;

std::fstream Sigma1File;
std::fstream Sigma3File;

Sigma1File.open("sigma1average_t.txt", std::fstream::out | std::fstream::app);
Sigma1File.precision(12);
Sigma1File << CurrentTime << " " << Sigma1Average << std::endl;
Sigma1File.close();

Sigma3File.open("sigma3average_t.txt", std::fstream::out | std::fstream::app);
Sigma3File.precision(12);
Sigma3File << CurrentTime << " " << Sigma3Average << std::endl;
Sigma3File.close();
}








virtual std::string Info() const
{
return "";
}


virtual void PrintInfo(std::ostream& rOStream) const
{
}


virtual void PrintData(std::ostream& rOStream) const
{
}




protected:

ModelPart& mrModelPart;
array_1d<double,3> mCylinderCenter;
double mMinRadius;
double mMaxRadius;













private:











StressFailureCheckUtilities & operator=(StressFailureCheckUtilities const& rOther);



}; 

}  

#endif 
