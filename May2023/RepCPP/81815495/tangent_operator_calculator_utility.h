
#pragma once



#include "includes/constitutive_law.h"
#include "constitutive_laws_application_variables.h"
#include "structural_mechanics_application_variables.h"

namespace Kratos
{







class TangentOperatorCalculatorUtility
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TangentOperatorCalculatorUtility);

typedef std::size_t SizeType;

typedef std::size_t IndexType;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();

static constexpr double PerturbationCoefficient1 = 1.0e-5;
static constexpr double PerturbationCoefficient2 = 1.0e-10;

static constexpr double PerturbationThreshold = 1.0e-8;


TangentOperatorCalculatorUtility()
{
}

virtual ~TangentOperatorCalculatorUtility() {}




static void CalculateTangentTensor(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw* pConstitutiveLaw,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_Cauchy,
const bool ConsiderPertubationThreshold = true,
const IndexType ApproximationOrder = 2
)
{
Flags& cl_options = rValues.GetOptions();
const bool use_element_provided_strain = cl_options.Is(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN);

if (use_element_provided_strain) {
CalculateTangentTensorSmallDeformationProvidedStrain(rValues, pConstitutiveLaw, rStressMeasure, ConsiderPertubationThreshold, ApproximationOrder);
} else {
CalculateTangentTensorSmallDeformationNotProvidedStrain(rValues, pConstitutiveLaw, rStressMeasure, ConsiderPertubationThreshold, ApproximationOrder);
}
}


static void CalculateTangentTensorSmallDeformationProvidedStrain(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw *pConstitutiveLaw,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_Cauchy,
const bool ConsiderPertubationThreshold = true,
const IndexType ApproximationOrder = 2
)
{
const Vector unperturbed_strain_vector_gp = Vector(rValues.GetStrainVector());
const Vector unperturbed_stress_vector_gp = Vector(rValues.GetStressVector());
const auto &r_properties = rValues.GetMaterialProperties();
const bool symmetrize_operator = (r_properties.Has(SYMMETRIZE_TANGENT_OPERATOR)) ? r_properties[SYMMETRIZE_TANGENT_OPERATOR] : false;

const SizeType num_components = unperturbed_stress_vector_gp.size();

Matrix& r_tangent_tensor = rValues.GetConstitutiveMatrix();
r_tangent_tensor.clear();
Matrix auxiliary_tensor = ZeroMatrix(num_components,num_components);

double pertubation = PerturbationThreshold;
if (r_properties.Has(PERTURBATION_SIZE)) {
pertubation = r_properties[PERTURBATION_SIZE];
if (pertubation == -1.0)
pertubation = std::sqrt(tolerance);
} else {
for (IndexType i_component = 0; i_component < num_components; ++i_component) {
double component_perturbation;
CalculatePerturbation(unperturbed_strain_vector_gp, i_component, component_perturbation);
pertubation = std::max(component_perturbation, pertubation);
}
if (ConsiderPertubationThreshold && pertubation < PerturbationThreshold) pertubation = PerturbationThreshold;
}

Vector& r_perturbed_strain = rValues.GetStrainVector();
Vector& r_perturbed_integrated_stress = rValues.GetStressVector();
if (ApproximationOrder == 1) {
for (IndexType i_component = 0; i_component < num_components; ++i_component) {

PerturbateStrainVector(r_perturbed_strain, unperturbed_strain_vector_gp, pertubation, i_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector delta_stress = r_perturbed_integrated_stress - unperturbed_stress_vector_gp;
CalculateComponentsToTangentTensorFirstOrder(auxiliary_tensor, r_perturbed_strain-unperturbed_strain_vector_gp, delta_stress, i_component);

noalias(r_perturbed_strain) = unperturbed_strain_vector_gp;
noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;
}
} else if (ApproximationOrder == 2) {
for (IndexType i_component = 0; i_component < num_components; ++i_component) {
PerturbateStrainVector(r_perturbed_strain, unperturbed_strain_vector_gp, pertubation, i_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector strain_plus = r_perturbed_strain;
const Vector stress_plus = r_perturbed_integrated_stress;

noalias(r_perturbed_strain) = unperturbed_strain_vector_gp;
noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;

PerturbateStrainVector(r_perturbed_strain, unperturbed_strain_vector_gp, - pertubation, i_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector strain_minus = r_perturbed_strain;
const Vector stress_minus = r_perturbed_integrated_stress;

CalculateComponentsToTangentTensorSecondOrder(auxiliary_tensor, strain_plus, strain_minus, stress_plus, stress_minus, i_component);

noalias(r_perturbed_strain) = unperturbed_strain_vector_gp;
noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;
}
} else if (ApproximationOrder == 4) { 
for (IndexType i_component = 0; i_component < num_components; ++i_component) {
PerturbateStrainVector(r_perturbed_strain, unperturbed_strain_vector_gp, pertubation, i_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector strain_plus = r_perturbed_strain;
const Vector stress_plus = r_perturbed_integrated_stress;

noalias(r_perturbed_strain) = unperturbed_strain_vector_gp;
noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;

PerturbateStrainVector(r_perturbed_strain, unperturbed_strain_vector_gp, 2.0*pertubation, i_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector strain_2_plus = r_perturbed_strain;
const Vector stress_2_plus = r_perturbed_integrated_stress;

const SizeType voigt_size = stress_plus.size();
for (IndexType row = 0; row < voigt_size; ++row) {
auxiliary_tensor(row, i_component) = (stress_plus[row] - unperturbed_stress_vector_gp[row]) / pertubation - (stress_2_plus[row] - 2.0 * stress_plus[row] + unperturbed_stress_vector_gp[row]) / (2.0 * pertubation);
}

noalias(r_perturbed_strain) = unperturbed_strain_vector_gp;
noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;
}
}
if (symmetrize_operator)
noalias(r_tangent_tensor) = 0.5*(auxiliary_tensor + trans(auxiliary_tensor));
else
noalias(r_tangent_tensor) = auxiliary_tensor;
}


static void CalculateTangentTensorSmallDeformationNotProvidedStrain(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw *pConstitutiveLaw,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_Cauchy,
const bool ConsiderPertubationThreshold = true,
const IndexType ApproximationOrder = 2
)
{
const Vector unperturbed_strain_vector_gp = Vector(rValues.GetStrainVector());
const Vector unperturbed_stress_vector_gp = Vector(rValues.GetStressVector());

const auto &r_properties = rValues.GetMaterialProperties();
const bool symmetrize_operator = (r_properties.Has(SYMMETRIZE_TANGENT_OPERATOR)) ? r_properties[SYMMETRIZE_TANGENT_OPERATOR] : false;

const SizeType num_components = unperturbed_stress_vector_gp.size();

const Matrix unperturbed_deformation_gradient_gp = Matrix(rValues.GetDeformationGradientF());
const double det_unperturbed_deformation_gradient_gp = double(rValues.GetDeterminantF());

Matrix& r_tangent_tensor = rValues.GetConstitutiveMatrix();
r_tangent_tensor.clear();
Matrix auxiliary_tensor = ZeroMatrix(num_components,num_components);

const std::size_t size1 = unperturbed_deformation_gradient_gp.size1();
const std::size_t size2 = unperturbed_deformation_gradient_gp.size2();

KRATOS_ERROR_IF_NOT(ApproximationOrder == 1 || ApproximationOrder == 2) << "The approximation order for the perturbation is " << ApproximationOrder << ". Options are 1 and 2" << std::endl;

double pertubation = PerturbationThreshold;
if (r_properties.Has(PERTURBATION_SIZE)) {
pertubation = r_properties[PERTURBATION_SIZE];
} else {
for (IndexType i_component = 0; i_component < num_components; ++i_component) {
double component_perturbation;
CalculatePerturbation(unperturbed_strain_vector_gp, i_component, component_perturbation);
pertubation = std::max(component_perturbation, pertubation);
}
if (ConsiderPertubationThreshold && pertubation < PerturbationThreshold) pertubation = PerturbationThreshold;
}

Vector& r_perturbed_strain = rValues.GetStrainVector();
Vector& r_perturbed_integrated_stress = rValues.GetStressVector();
Matrix& r_perturbed_deformation_gradient = const_cast<Matrix&>(rValues.GetDeformationGradientF());
double& r_perturbed_det_deformation_gradient = const_cast<double&>(rValues.GetDeterminantF());

if (ApproximationOrder == 1) {
for (IndexType i_component = 0; i_component < size1; ++i_component) {
for (IndexType j_component = i_component; j_component < size2; ++j_component) {
PerturbateDeformationGradient(r_perturbed_deformation_gradient, unperturbed_deformation_gradient_gp, pertubation, i_component, j_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector delta_stress = r_perturbed_integrated_stress - unperturbed_stress_vector_gp;

const IndexType voigt_index = CalculateVoigtIndex(delta_stress.size(), i_component, j_component);
CalculateComponentsToTangentTensorFirstOrder(auxiliary_tensor, r_perturbed_strain-unperturbed_strain_vector_gp, delta_stress, voigt_index);

noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;
noalias(r_perturbed_deformation_gradient) = unperturbed_deformation_gradient_gp;
r_perturbed_det_deformation_gradient = det_unperturbed_deformation_gradient_gp;
}
}
} else if (ApproximationOrder == 2) {
for (IndexType i_component = 0; i_component < size1; ++i_component) {
for (IndexType j_component = i_component; j_component < size2; ++j_component) {
PerturbateDeformationGradient(r_perturbed_deformation_gradient, unperturbed_deformation_gradient_gp, pertubation, i_component, j_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector strain_plus = r_perturbed_strain;
const Vector stress_plus = r_perturbed_integrated_stress;

noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;
noalias(r_perturbed_deformation_gradient) = unperturbed_deformation_gradient_gp;
r_perturbed_det_deformation_gradient = det_unperturbed_deformation_gradient_gp;

PerturbateDeformationGradient(r_perturbed_deformation_gradient, unperturbed_deformation_gradient_gp, - pertubation, i_component, j_component);

IntegratePerturbedStrain(rValues, pConstitutiveLaw, rStressMeasure);

const Vector strain_minus = r_perturbed_strain;
const Vector stress_minus = r_perturbed_integrated_stress;

const IndexType voigt_index = CalculateVoigtIndex(stress_plus.size(), i_component, j_component);
CalculateComponentsToTangentTensorSecondOrder(auxiliary_tensor, strain_plus, strain_minus, stress_plus, stress_minus, voigt_index);

noalias(r_perturbed_integrated_stress) = unperturbed_stress_vector_gp;
noalias(r_perturbed_deformation_gradient) = unperturbed_deformation_gradient_gp;
r_perturbed_det_deformation_gradient = det_unperturbed_deformation_gradient_gp;
}
}
}
if (symmetrize_operator)
noalias(r_tangent_tensor) = 0.5*(auxiliary_tensor + trans(auxiliary_tensor));
else
noalias(r_tangent_tensor) = auxiliary_tensor;
}


static void CalculateTangentTensorFiniteDeformation(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw* pConstitutiveLaw,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_PK2,
const bool ConsiderPertubationThreshold = true,
const IndexType ApproximationOrder = 2
)
{
CalculateTangentTensorSmallDeformationNotProvidedStrain(rValues, pConstitutiveLaw, rStressMeasure, ConsiderPertubationThreshold, ApproximationOrder);
}

protected:







private:





static void CalculatePerturbation(
const Vector& rStrainVector,
const IndexType Component,
double& rPerturbation
)
{
double perturbation_1, perturbation_2;
if (std::abs(rStrainVector[Component]) > tolerance) {
perturbation_1 = PerturbationCoefficient1 * rStrainVector[Component];
} else {
double min_strain_component;
GetMinAbsValue(rStrainVector, min_strain_component);
perturbation_1 = PerturbationCoefficient1 * min_strain_component;
}
double max_strain_component;
GetMaxAbsValue(rStrainVector, max_strain_component);
perturbation_2 = PerturbationCoefficient2 * max_strain_component;
rPerturbation = std::max(perturbation_1, perturbation_2);
}


static void CalculatePerturbationFiniteDeformation(
const Matrix& rDeformationGradient,
const IndexType ComponentI,
const IndexType ComponentJ,
double& rPerturbation
)
{
double perturbation_1, perturbation_2;
if (std::abs(rDeformationGradient(ComponentI, ComponentJ)) > tolerance) {
perturbation_1 = PerturbationCoefficient1 * rDeformationGradient(ComponentI, ComponentJ);
} else {
double min_strain_component;
GetMinAbsValue(rDeformationGradient, min_strain_component);
perturbation_1 = PerturbationCoefficient1 * min_strain_component;
}
double max_strain_component;
GetMaxAbsValue(rDeformationGradient, max_strain_component);
perturbation_2 = PerturbationCoefficient2 * max_strain_component;
rPerturbation = std::max(perturbation_1, perturbation_2);
}


static void PerturbateStrainVector(
Vector& rPerturbedStrainVector,
const Vector& rStrainVectorGP,
const double Perturbation,
const IndexType Component
)
{
noalias(rPerturbedStrainVector) = rStrainVectorGP;
rPerturbedStrainVector[Component] += Perturbation;
}


static void PerturbateDeformationGradient(
Matrix& rPerturbedDeformationGradient,
const Matrix& rDeformationGradientGP,
const double Perturbation,
const IndexType ComponentI,
const IndexType ComponentJ
)
{
Matrix aux_perturbation_matrix = IdentityMatrix(rDeformationGradientGP.size1());
if (ComponentI == ComponentJ) {
aux_perturbation_matrix(ComponentI, ComponentJ) += Perturbation;
} else {
aux_perturbation_matrix(ComponentI, ComponentJ) += 0.5 * Perturbation;
aux_perturbation_matrix(ComponentJ, ComponentI) += 0.5 * Perturbation;
}
noalias(rPerturbedDeformationGradient) = prod(aux_perturbation_matrix, rDeformationGradientGP);
}


static void IntegratePerturbedStrain(
ConstitutiveLaw::Parameters& rValues,
ConstitutiveLaw* pConstitutiveLaw,
const ConstitutiveLaw::StressMeasure& rStressMeasure = ConstitutiveLaw::StressMeasure_Cauchy
)
{
Flags& cl_options = rValues.GetOptions();

const bool flag_back_up_1 = cl_options.Is(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR);
const bool flag_back_up_2 = cl_options.Is(ConstitutiveLaw::COMPUTE_STRESS);

cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);

pConstitutiveLaw->CalculateMaterialResponse(rValues, rStressMeasure);

cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, flag_back_up_1);
cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, flag_back_up_2);
}


static void GetMaxAbsValue(
const Vector& rArrayValues,
double& rMaxValue
)
{
const SizeType dimension = rArrayValues.size();

IndexType counter = 0;
double aux = 0.0;
for (IndexType i = 0; i < dimension; ++i) {
if (std::abs(rArrayValues[i]) > aux) {
aux = std::abs(rArrayValues[i]);
++counter;
}
}

rMaxValue = aux;
}


static void GetMaxAbsValue(
const Matrix& rMatrixValues,
double& rMaxValue
)
{
const SizeType size1 = rMatrixValues.size1();
const SizeType size2 = rMatrixValues.size2();

const Matrix working_matrix = rMatrixValues - IdentityMatrix(size1);

IndexType counter = 0;
double aux = 0.0;
for (IndexType i = 0; i < size1; ++i) {
for (IndexType j = 0; j < size2; ++j) {
if (std::abs(working_matrix(i, j)) > aux) {
aux = std::abs(working_matrix(i, j));
++counter;
}
}
}

rMaxValue = aux;
}


static void GetMinAbsValue(
const Vector& rArrayValues,
double& rMinValue
)
{
const SizeType dimension = rArrayValues.size();

IndexType counter = 0;
double aux = std::numeric_limits<double>::max();
for (IndexType i = 0; i < dimension; ++i) {
if (std::abs(rArrayValues[i]) < aux) {
aux = std::abs(rArrayValues[i]);
++counter;
}
}

rMinValue = aux;
}


static void GetMinAbsValue(
const Matrix& rMatrixValues,
double& rMinValue
)
{
const SizeType size1 = rMatrixValues.size1();
const SizeType size2 = rMatrixValues.size2();

const Matrix working_matrix = rMatrixValues - IdentityMatrix(size1);

IndexType counter = 0;
double aux = std::numeric_limits<double>::max();
for (IndexType i = 0; i < size1; ++i) {
for (IndexType j = 0; j < size2; ++j) {
if (std::abs(working_matrix(i, j)) < aux) {
aux = std::abs(working_matrix(i, j));
++counter;
}
}
}

rMinValue = aux;
}


static void CalculateComponentsToTangentTensorFirstOrder(
Matrix& rTangentTensor,
const Vector& rVectorStrain,
const Vector& rDeltaStress,
const IndexType Component
)
{
const double perturbation = rVectorStrain[Component];
const SizeType voigt_size = rDeltaStress.size();
for (IndexType row = 0; row < voigt_size; ++row) {
rTangentTensor(row, Component) = rDeltaStress[row] / perturbation;
}
}


static void CalculateComponentsToTangentTensorSecondOrder(
Matrix& rTangentTensor,
const Vector& rVectorStrainPlus,
const Vector& rVectorStrainMinus,
const Vector& rStressPlus,
const Vector& rStressMinus,
const IndexType Component
)
{
const double perturbation = (rVectorStrainPlus[Component] - rVectorStrainMinus[Component]);
const SizeType voigt_size = rStressPlus.size();
for (IndexType row = 0; row < voigt_size; ++row) {
rTangentTensor(row, Component) = (rStressPlus[row] - rStressMinus[row]) / perturbation;
}
}


static IndexType CalculateVoigtIndex(
const SizeType VoigtSize,
const IndexType ComponentI,
const IndexType ComponentJ
)
{
if (VoigtSize == 6) {
switch(ComponentI) {
case 0:
switch(ComponentJ) {
case 0:
return 0;
case 1:
return 3;
case 2:
return 5;
default:
return 0;
}
case 1:
switch(ComponentJ) {
case 0:
return 3;
case 1:
return 1;
case 2:
return 4;
default:
return 0;
}
case 2:
switch(ComponentJ) {
case 0:
return 5;
case 1:
return 4;
case 2:
return 2;
default:
return 0;
}
default:
return 0;
}
} else {
switch(ComponentI) {
case 0:
switch(ComponentJ) {
case 0:
return 0;
case 1:
return 2;
default:
return 0;
}
case 1:
switch(ComponentJ) {
case 0:
return 2;
case 1:
return 1;
default:
return 0;
}
default:
return 0;
}
}
}

protected:







private:







TangentOperatorCalculatorUtility &operator=(TangentOperatorCalculatorUtility const &rOther);
};
} 
