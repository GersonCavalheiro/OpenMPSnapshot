
#if !defined(KRATOS_AITKEN_RELAXATION_UTILITY)
#define  KRATOS_AITKEN_RELAXATION_UTILITY


#include "utilities/math_utils.h"
#include "spaces/ublas_space.h"
#include "fem_to_dem_application_variables.h"

namespace Kratos
{






class AitkenRelaxationFEMDEMUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AitkenRelaxationFEMDEMUtility);

typedef UblasSpace<double, Matrix, Vector> TSpace;

typedef typename TSpace::VectorType VectorType;

typedef typename TSpace::VectorPointerType VectorPointerType;

static constexpr unsigned int Dimension = 3;



AitkenRelaxationFEMDEMUtility(const double OmegaOld = 0.825, const double MaximumOmega = 0.825, const double MinimumOmega = 0.825)
{
mOmegaOld = OmegaOld;
mOmegaMax = MaximumOmega;
mOmegaMin = MinimumOmega;
}


AitkenRelaxationFEMDEMUtility(const AitkenRelaxationFEMDEMUtility& rOther)
{
mOmegaOld = rOther.mOmegaOld;
}


virtual ~AitkenRelaxationFEMDEMUtility() {}





void InitializeSolutionStep()
{
KRATOS_TRY
mConvergenceAcceleratorIteration = 1;
mOmegaNew = 0.825;
KRATOS_CATCH( "" )
}


void UpdateSolution(
const Vector& rResidualVector,
Vector& rIterationGuess
)
{
KRATOS_TRY
VectorPointerType pAux(new VectorType(rResidualVector));
std::swap(mpResidualVectorNew, pAux);

if (mConvergenceAcceleratorIteration == 1) {
TSpace::UnaliasedAdd(rIterationGuess, mOmegaOld, *mpResidualVectorNew);
} else {
VectorType Aux1minus0(*mpResidualVectorNew);                  
TSpace::UnaliasedAdd(Aux1minus0, -1.0, *mpResidualVectorOld); 

const double denominator = TSpace::Dot(Aux1minus0, Aux1minus0);
const double numerator   = TSpace::Dot(*mpResidualVectorOld, Aux1minus0);

mOmegaNew = -mOmegaOld * (numerator / denominator);

mOmegaNew = (mOmegaNew > mOmegaMax) ? mOmegaMax : mOmegaNew;
mOmegaNew = (mOmegaNew < mOmegaMin) ? mOmegaMin : mOmegaNew;

TSpace::UnaliasedAdd(rIterationGuess, mOmegaNew, *mpResidualVectorNew);
mOmegaOld = mOmegaNew;
}
KRATOS_CATCH("")
}


void FinalizeNonLinearIteration()
{
KRATOS_TRY
std::swap(mpResidualVectorOld, mpResidualVectorNew);
mConvergenceAcceleratorIteration += 1;
KRATOS_CATCH("")
}


void FinalizeSolutionStep()
{
KRATOS_TRY
mConvergenceAcceleratorIteration = 1;
mVectorSize = 0;
KRATOS_CATCH("")
}


double ComputeNorm(const Vector& rVector)
{
return MathUtils<double>::Norm(rVector);
}


void InitializeInterfaceSubModelPart(ModelPart &rSolidModelPart)
{
mVectorSize = 0;
if (rSolidModelPart.HasSubModelPart("fsi_interface_model_part")) {
auto &r_interface_sub_model = rSolidModelPart.GetSubModelPart("fsi_interface_model_part");
r_interface_sub_model.Nodes().clear();
} else {
auto &r_interface_sub_model = rSolidModelPart.CreateSubModelPart("fsi_interface_model_part");
}

auto &r_interface_sub_model  = rSolidModelPart.GetSubModelPart("fsi_interface_model_part");
auto &r_solid_skin_sub_model = rSolidModelPart.GetSubModelPart("SkinDEMModelPart");

const auto it_node_begin = r_solid_skin_sub_model.NodesBegin();
for (int i = 0; i < static_cast<int>(r_solid_skin_sub_model.Nodes().size()); i++) {
auto it_node = it_node_begin + i;
if (it_node->FastGetSolutionStepValue(PRESSURE) != 0.0) {
r_interface_sub_model.AddNode(*(it_node.base()));
mVectorSize++;
}
}
mVectorSize *= Dimension;
}


void ResetNodalValues(ModelPart &rSolidModelPart)
{
auto &r_interface_sub_model = rSolidModelPart.GetSubModelPart("fsi_interface_model_part");
const Vector& r_zero_vector = ZeroVector(Dimension);

const auto it_node_begin = r_interface_sub_model.NodesBegin();
#pragma omp parallel for
for (int i = 0; i < static_cast<int>(r_interface_sub_model.Nodes().size()); i++) {
auto it_node = it_node_begin + i;
auto &r_var_1 = it_node->FastGetSolutionStepValue(RELAXED_VELOCITY);
auto &r_var_2 = it_node->FastGetSolutionStepValue(OLD_RELAXED_VELOCITY);
auto &r_var_3 = it_node->FastGetSolutionStepValue(FSI_INTERFACE_RESIDUAL);
r_var_1 = r_zero_vector;
r_var_2 = r_zero_vector;
r_var_3 = r_zero_vector;
}

mpResidualVectorOld.reset();
mpResidualVectorNew.reset();
}


void SavePreviousRelaxedValues(ModelPart &rSolidModelPart)
{
auto &r_interface_sub_model = rSolidModelPart.GetSubModelPart("fsi_interface_model_part");
const auto it_node_begin = r_interface_sub_model.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(r_interface_sub_model.Nodes().size()); i++) {
auto it_node = it_node_begin + i;
const auto &r_relaxed_velocity  = it_node->FastGetSolutionStepValue(RELAXED_VELOCITY);
auto &r_old_relaxed_velocity    = it_node->GetSolutionStepValue(OLD_RELAXED_VELOCITY);
noalias(r_old_relaxed_velocity) = r_relaxed_velocity;
}
}


unsigned int GetVectorSize()
{
return mVectorSize;
}


void FillOldRelaxedValuesVector(
ModelPart &rSolidModelPart,
Vector& rIterationValueVector
)
{
auto &r_interface_sub_model = rSolidModelPart.GetSubModelPart("fsi_interface_model_part");
const auto it_node_begin = r_interface_sub_model.NodesBegin();

if (rIterationValueVector.size() != mVectorSize) {
rIterationValueVector.resize(mVectorSize);
noalias(rIterationValueVector) = ZeroVector(mVectorSize);
}

#pragma omp parallel for firstprivate(it_node_begin)
for (int i = 0; i < static_cast<int>(r_interface_sub_model.Nodes().size()); i++) {
auto it_node = it_node_begin + i;
const auto &r_value = it_node->FastGetSolutionStepValue(OLD_RELAXED_VELOCITY);

const unsigned int base_i = i * Dimension;
for (unsigned int jj = 0; jj < Dimension; ++jj) {
rIterationValueVector[base_i + jj] = r_value[jj];
}
}
}


double ComputeInterfaceResidualVector(
ModelPart &rSolidModelPart,
Vector& rInterfaceResidualVector
)
{
if (rInterfaceResidualVector.size() != mVectorSize) {
rInterfaceResidualVector.resize(mVectorSize);
noalias(rInterfaceResidualVector) = ZeroVector(mVectorSize);
}

auto &r_interface_sub_model = rSolidModelPart.GetSubModelPart("fsi_interface_model_part");
const auto it_node_begin = r_interface_sub_model.NodesBegin();

TSpace::SetToZero(rInterfaceResidualVector);

#pragma omp parallel for firstprivate(it_node_begin)
for (int i = 0; i < static_cast<int>(r_interface_sub_model.Nodes().size()); i++) {
auto it_node = it_node_begin + i;

const auto &r_origin_value    = it_node->FastGetSolutionStepValue(VELOCITY);
const auto &r_modified_value  = it_node->FastGetSolutionStepValue(OLD_RELAXED_VELOCITY);
auto& r_interface_residual    = it_node->FastGetSolutionStepValue(FSI_INTERFACE_RESIDUAL);
noalias(r_interface_residual) = r_origin_value - r_modified_value;

const unsigned int base_i = i * Dimension;
for (unsigned int jj = 0; jj < Dimension; ++jj)
rInterfaceResidualVector[base_i + jj] = r_interface_residual[jj];
}
return MathUtils<double>::Norm(rInterfaceResidualVector);
}


void UpdateInterfaceValues(
ModelPart &rSolidModelPart,
const Vector& rRelaxedValuesVector
)
{
auto &r_interface_sub_model = rSolidModelPart.GetSubModelPart("fsi_interface_model_part");
const auto it_node_begin = r_interface_sub_model.NodesBegin();

#pragma omp parallel for firstprivate(it_node_begin)
for (int i = 0; i < static_cast<int>(r_interface_sub_model.Nodes().size()); i++) {
auto it_node = it_node_begin + i;
auto &r_value         = it_node->FastGetSolutionStepValue(VELOCITY);
auto &r_value_relaxed = it_node->FastGetSolutionStepValue(RELAXED_VELOCITY);

const int base_i = i * Dimension;
for (unsigned int jj = 0; jj < Dimension; ++jj) {
r_value[jj]         = rRelaxedValuesVector[base_i + jj];
r_value_relaxed[jj] = rRelaxedValuesVector[base_i + jj];
}
}
}


void ResetPFEMkinematicValues(
ModelPart &rFluidModelPart
)
{
const auto it_node_begin = rFluidModelPart.NodesBegin();

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rFluidModelPart.Nodes().size()); ++i) {
auto it_node = it_node_begin + i;

if (it_node->IsNot(SOLID)) { 
auto &r_current_displ = it_node->FastGetSolutionStepValue(DISPLACEMENT, 0);
auto &r_current_vel   = it_node->FastGetSolutionStepValue(VELOCITY, 0);
auto &r_current_acc   = it_node->FastGetSolutionStepValue(ACCELERATION, 0);

auto &r_old_displ     = it_node->FastGetSolutionStepValue(DISPLACEMENT, 1);
auto &r_old_vel       = it_node->FastGetSolutionStepValue(VELOCITY, 1);
auto &r_old_acc       = it_node->FastGetSolutionStepValue(ACCELERATION, 1);

auto copy_old_displ   = r_old_displ;
auto copy_old_vel     = r_old_vel;
auto copy_old_acc     = r_old_acc;

auto& r_coordinates = it_node->Coordinates();
const auto& r_initial_coordinates = it_node->GetInitialPosition();
noalias(r_coordinates) = r_initial_coordinates + copy_old_displ;

noalias(r_current_displ) = copy_old_displ;
noalias(r_current_vel)   = copy_old_vel;
noalias(r_current_acc)   = copy_old_acc;
}
}
}






protected:



unsigned int mConvergenceAcceleratorIteration;

double mOmegaOld;
double mOmegaNew;

double mOmegaMax;
double mOmegaMin;

VectorPointerType mpResidualVectorOld;
VectorPointerType mpResidualVectorNew;

unsigned int mVectorSize;







private:









}; 




} 

#endif 