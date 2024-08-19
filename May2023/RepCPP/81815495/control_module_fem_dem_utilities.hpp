

#ifndef KRATOS_CONTROL_MODULE_FEM_DEM_UTILITIES
#define KRATOS_CONTROL_MODULE_FEM_DEM_UTILITIES



#include "includes/variables.h"


#include <limits>
#include <iostream>
#include <iomanip>


#ifdef _OPENMP
#include <omp.h>
#endif


#include "geometries/geometry.h"
#include "includes/define.h"
#include "includes/model_part.h"

#include "includes/table.h"
#include "includes/kratos_parameters.h"

#include "custom_elements/spheric_continuum_particle.h"

#include "dem_structures_coupling_application_variables.h"


namespace Kratos
{
class ControlModuleFemDemUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ControlModuleFemDemUtilities);

typedef Table<double,double> TableType;


ControlModuleFemDemUtilities(ModelPart& rFemModelPart,
ModelPart& rDemModelPart,
Parameters& rParameters
) :
mrFemModelPart(rFemModelPart),
mrDemModelPart(rDemModelPart)
{
KRATOS_TRY

Parameters default_parameters( R"(
{
"imposed_direction" : 2,
"alternate_axis_loading": false,
"target_stress_table_id" : 0,
"initial_velocity" : 0.0,
"limit_velocity" : 1.0,
"velocity_factor" : 1.0,
"compression_length" : 1.0,
"young_modulus" : 1.0e7,
"stress_increment_tolerance": 100.0,
"update_stiffness": true,
"start_time" : 0.0,
"stress_averaging_time": 1.0e-5
}  )" );

rParameters.ValidateAndAssignDefaults(default_parameters);

mImposedDirection = rParameters["imposed_direction"].GetInt();
mTargetStressTableId = rParameters["target_stress_table_id"].GetInt();
mVelocity = rParameters["initial_velocity"].GetDouble();
if (mImposedDirection == 0) {
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_X_VALUE");
mrDemModelPart[DemVelocityVar] = mVelocity;
} else if (mImposedDirection == 1) {
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_Y_VALUE");
mrDemModelPart[DemVelocityVar] = mVelocity;
} else {
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_Z_VALUE");
mrDemModelPart[DemVelocityVar] = mVelocity;
}
mLimitVelocity = rParameters["limit_velocity"].GetDouble();
mVelocityFactor = rParameters["velocity_factor"].GetDouble();
mStartTime = rParameters["start_time"].GetDouble();
mStressIncrementTolerance = rParameters["stress_increment_tolerance"].GetDouble();
mUpdateStiffness = rParameters["update_stiffness"].GetBool();
mReactionStressOld = 0.0;
mStiffness = rParameters["young_modulus"].GetDouble()/rParameters["compression_length"].GetDouble(); 
mStressAveragingTime = rParameters["stress_averaging_time"].GetDouble();
mVectorOfLastStresses.resize(0);

mAlternateAxisLoading = rParameters["alternate_axis_loading"].GetBool();
mXCounter = 1;
mYCounter = 2;
mZCounter = 3;

mrDemModelPart.GetProcessInfo().SetValue(TARGET_STRESS_Z,0.0);
int NNodes = static_cast<int>(mrFemModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator it_begin = mrFemModelPart.NodesBegin();
array_1d<double,3> zero_vector = ZeroVector(3);
#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->SetValue(TARGET_STRESS,zero_vector);
it->SetValue(REACTION_STRESS,zero_vector);
it->SetValue(LOADING_VELOCITY,zero_vector);
}
NNodes = static_cast<int>(mrDemModelPart.Nodes().size());
it_begin = mrDemModelPart.NodesBegin();
#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->SetValue(TARGET_STRESS,zero_vector);
it->SetValue(REACTION_STRESS,zero_vector);
it->SetValue(LOADING_VELOCITY,zero_vector);
}

mApplyCM = false;

KRATOS_CATCH("");
}


virtual ~ControlModuleFemDemUtilities(){}


void ExecuteInitialize()
{
KRATOS_TRY;

const int NNodes = static_cast<int>(mrFemModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator it_begin = mrFemModelPart.NodesBegin();

if (mImposedDirection == 0) {
#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->Fix(DISPLACEMENT_X);
it->FastGetSolutionStepValue(DISPLACEMENT_X) = 0.0;
}
} else if (mImposedDirection == 1) {
#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->Fix(DISPLACEMENT_Y);
it->FastGetSolutionStepValue(DISPLACEMENT_Y) = 0.0;
}
} else if (mImposedDirection == 2) {
#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->Fix(DISPLACEMENT_Z);
it->FastGetSolutionStepValue(DISPLACEMENT_Z) = 0.0;
}
}

KRATOS_CATCH("");
}

void ExecuteInitializeSolutionStep()
{
const double CurrentTime = mrFemModelPart.GetProcessInfo()[TIME];
const double DeltaTime = mrFemModelPart.GetProcessInfo()[DELTA_TIME];
const int NNodes = static_cast<int>(mrFemModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator it_begin = mrFemModelPart.NodesBegin();
TableType::Pointer pTargetStressTable = mrFemModelPart.pGetTable(mTargetStressTableId);

double ReactionStress = CalculateReactionStress();
ReactionStress = UpdateVectorOfHistoricalStressesAndComputeNewAverage(ReactionStress);

IsTimeToApplyCM();

if (mApplyCM == true) {

if (mAlternateAxisLoading == false) {
if(mUpdateStiffness == true) {
mStiffness = EstimateStiffness(ReactionStress,DeltaTime);
}
}
mReactionStressOld = ReactionStress;

const double NextTargetStress = pTargetStressTable->GetValue(CurrentTime+DeltaTime);
const double df_target = NextTargetStress - ReactionStress;
double delta_velocity = df_target/(mStiffness * DeltaTime) - mVelocity;

if(std::abs(df_target) < mStressIncrementTolerance) {
delta_velocity = -mVelocity;
}

mVelocity += mVelocityFactor * delta_velocity;

if(std::abs(mVelocity) > std::abs(mLimitVelocity)) {
if(mVelocity >= 0.0) {
mVelocity = std::abs(mLimitVelocity);
} else {
mVelocity = - std::abs(mLimitVelocity);
}
}

if (mImposedDirection == 0) { 
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_X_VALUE");
mrDemModelPart[DemVelocityVar] = mVelocity;

#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->FastGetSolutionStepValue(DISPLACEMENT_X) += mVelocity * DeltaTime;
it->GetValue(TARGET_STRESS_X) = pTargetStressTable->GetValue(CurrentTime);
it->GetValue(REACTION_STRESS_X) = ReactionStress;
it->GetValue(LOADING_VELOCITY_X) = mVelocity;
}
} else if (mImposedDirection == 1) { 
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_Y_VALUE");
mrDemModelPart[DemVelocityVar] = mVelocity;

#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->FastGetSolutionStepValue(DISPLACEMENT_Y) += mVelocity * DeltaTime;
it->GetValue(TARGET_STRESS_Y) = pTargetStressTable->GetValue(CurrentTime);
it->GetValue(REACTION_STRESS_Y) = ReactionStress;
it->GetValue(LOADING_VELOCITY_Y) = mVelocity;
}
} else if (mImposedDirection == 2) { 
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_Z_VALUE");
mrDemModelPart[DemVelocityVar] = mVelocity;

#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->FastGetSolutionStepValue(DISPLACEMENT_Z) += mVelocity * DeltaTime;
it->GetValue(TARGET_STRESS_Z) = pTargetStressTable->GetValue(CurrentTime);
it->GetValue(REACTION_STRESS_Z) = ReactionStress;
it->GetValue(LOADING_VELOCITY_Z) = mVelocity;
}
mrDemModelPart.GetProcessInfo()[TARGET_STRESS_Z] = std::abs(pTargetStressTable->GetValue(CurrentTime));
}
} else {
if (mImposedDirection == 0) { 
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_X_VALUE");
mrDemModelPart[DemVelocityVar] = 0.0;

#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->GetValue(TARGET_STRESS_X) = pTargetStressTable->GetValue(CurrentTime);
it->GetValue(REACTION_STRESS_X) = ReactionStress;
it->GetValue(LOADING_VELOCITY_X) = 0.0;
}
} else if (mImposedDirection == 1) { 
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_Y_VALUE");
mrDemModelPart[DemVelocityVar] = 0.0;

#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->GetValue(TARGET_STRESS_Y) = pTargetStressTable->GetValue(CurrentTime);
it->GetValue(REACTION_STRESS_Y) = ReactionStress;
it->GetValue(LOADING_VELOCITY_Y) = 0.0;
}
} else if (mImposedDirection == 2) { 
const Variable<double>& DemVelocityVar = KratosComponents< Variable<double> >::Get("IMPOSED_VELOCITY_Z_VALUE");
mrDemModelPart[DemVelocityVar] = 0.0;

#pragma omp parallel for
for(int i = 0; i<NNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_begin + i;
it->GetValue(TARGET_STRESS_Z) = pTargetStressTable->GetValue(CurrentTime);
it->GetValue(REACTION_STRESS_Z) = ReactionStress;
it->GetValue(LOADING_VELOCITY_Z) = 0.0;
}
mrDemModelPart.GetProcessInfo()[TARGET_STRESS_Z] = std::abs(pTargetStressTable->GetValue(CurrentTime));
}
}
}

void ExecuteFinalizeSolutionStep()
{
if (mApplyCM == true) {
if (mAlternateAxisLoading == true) {
const double delta_time = mrFemModelPart.GetProcessInfo()[DELTA_TIME];
double ReactionStress = CalculateReactionStress();
if(mUpdateStiffness == true) {
mStiffness = EstimateStiffness(ReactionStress,delta_time);
}
}
}
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

ModelPart& mrFemModelPart;
ModelPart& mrDemModelPart;
unsigned int mImposedDirection;
unsigned int mTargetStressTableId;
double mVelocity;
double mLimitVelocity;
double mVelocityFactor;
double mStartTime;
double mReactionStressOld;
double mStressIncrementTolerance;
double mStiffness;
bool mUpdateStiffness;
std::vector<double> mVectorOfLastStresses;
double mStressAveragingTime;
bool mAlternateAxisLoading;
unsigned int mXCounter;
unsigned int mYCounter;
unsigned int mZCounter;
bool mApplyCM;













private:





double UpdateVectorOfHistoricalStressesAndComputeNewAverage(const double& last_reaction) {
KRATOS_TRY;
int length_of_vector = mVectorOfLastStresses.size();
if (length_of_vector == 0) { 
int number_of_steps_for_stress_averaging = (int) (mStressAveragingTime / mrFemModelPart.GetProcessInfo()[DELTA_TIME]);
if(number_of_steps_for_stress_averaging < 1) number_of_steps_for_stress_averaging = 1;
mVectorOfLastStresses.resize(number_of_steps_for_stress_averaging);
KRATOS_INFO("DEM") << " 'number_of_steps_for_stress_averaging' is "<< number_of_steps_for_stress_averaging << std::endl;
}

length_of_vector = mVectorOfLastStresses.size();

if(length_of_vector > 1) {
for(int i=1; i<length_of_vector; i++) {
mVectorOfLastStresses[i-1] = mVectorOfLastStresses[i];
}
}
mVectorOfLastStresses[length_of_vector-1] = last_reaction;

double average = 0.0;
for(int i=0; i<length_of_vector; i++) {
average += mVectorOfLastStresses[i];
}
average /= (double) length_of_vector;
return average;

KRATOS_CATCH("");
}

void IsTimeToApplyCM(){
const double current_time = mrFemModelPart.GetProcessInfo()[TIME];
mApplyCM = false;

if(current_time >= mStartTime) {
if (mAlternateAxisLoading == true) {
const unsigned int step = mrFemModelPart.GetProcessInfo()[STEP];
if (mImposedDirection == 0) {
if(step == mXCounter){
mApplyCM = true;
mXCounter += 3;
}
} else if (mImposedDirection == 1) {
if(step == mYCounter){
mApplyCM = true;
mYCounter += 3;
}
} else if (mImposedDirection == 2) {
if(step == mZCounter){
mApplyCM = true;
mZCounter += 3;
}
}
} else {
mApplyCM = true;
}
}
}

double CalculateReactionStress() {

ModelPart::ElementsContainerType& rElements = mrDemModelPart.GetCommunicator().LocalMesh().Elements();
const int NNodes = static_cast<int>(mrFemModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator it_begin = mrFemModelPart.NodesBegin();

double face_area = 0.0;
#pragma omp parallel for reduction(+:face_area)
for (int i = 0; i < (int)rElements.size(); i++) {
ModelPart::ElementsContainerType::ptr_iterator ptr_itElem = rElements.ptr_begin() + i;
Element* p_element = ptr_itElem->get();
SphericContinuumParticle* pDemElem = dynamic_cast<SphericContinuumParticle*>(p_element);
const double radius = pDemElem->GetRadius();
face_area += Globals::Pi*radius*radius;
}
const int NCons = static_cast<int>(mrFemModelPart.Conditions().size());
ModelPart::ConditionsContainerType::iterator con_begin = mrFemModelPart.ConditionsBegin();
#pragma omp parallel for reduction(+:face_area)
for(int i = 0; i < NCons; i++)
{
ModelPart::ConditionsContainerType::iterator itCond = con_begin + i;
face_area += itCond->GetGeometry().Area();
}

double FaceReaction = 0.0;

if (mImposedDirection == 0) { 
#pragma omp parallel for reduction(+:FaceReaction)
for(int i = 0; i<NNodes; i++){
ModelPart::NodesContainerType::iterator it = it_begin + i;
FaceReaction += it->FastGetSolutionStepValue(REACTION_X);
}
} else if (mImposedDirection == 1) { 
#pragma omp parallel for reduction(+:FaceReaction)
for(int i = 0; i<NNodes; i++){
ModelPart::NodesContainerType::iterator it = it_begin + i;
FaceReaction += it->FastGetSolutionStepValue(REACTION_Y);
}
} else if (mImposedDirection == 2) { 
#pragma omp parallel for reduction(+:FaceReaction)
for(int i = 0; i<NNodes; i++){
ModelPart::NodesContainerType::iterator it = it_begin + i;
FaceReaction += it->FastGetSolutionStepValue(REACTION_Z);
}
}

const int NDemNodes = static_cast<int>(mrDemModelPart.Nodes().size());
ModelPart::NodesContainerType::iterator it_dem_begin = mrDemModelPart.NodesBegin();

if (mImposedDirection == 0) { 
#pragma omp parallel for reduction(-:FaceReaction)
for(int i = 0; i<NDemNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_dem_begin + i;
FaceReaction -= it->FastGetSolutionStepValue(TOTAL_FORCES_X);
}
} else if (mImposedDirection == 1) { 
#pragma omp parallel for reduction(-:FaceReaction)
for(int i = 0; i<NDemNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_dem_begin + i;
FaceReaction -= it->FastGetSolutionStepValue(TOTAL_FORCES_Y);
}
} else if (mImposedDirection == 2) { 
#pragma omp parallel for reduction(-:FaceReaction)
for(int i = 0; i<NDemNodes; i++) {
ModelPart::NodesContainerType::iterator it = it_dem_begin + i;
FaceReaction -= it->FastGetSolutionStepValue(TOTAL_FORCES_Z);
}
}

double reaction_stress;
if (std::abs(face_area) > 1.0e-12) {
reaction_stress = FaceReaction / face_area;
} else {
reaction_stress = 0.0;
}

return reaction_stress;
}

double EstimateStiffness(const double& rReactionStress, const double& rDeltaTime) {
double K_estimated = mStiffness;
if(std::abs(mVelocity) > 1.0e-12 && std::abs(rReactionStress-mReactionStressOld) > mStressIncrementTolerance) {
K_estimated = std::abs((rReactionStress-mReactionStressOld)/(mVelocity * rDeltaTime));
}
return K_estimated;
}






ControlModuleFemDemUtilities & operator=(ControlModuleFemDemUtilities const& rOther);



}; 

} 
#endif 
