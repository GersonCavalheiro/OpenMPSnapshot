


#include "includes/checks.h"
#include "custom_elements/base_solid_element.h"
#include "utilities/math_utils.h"
#include "utilities/geometry_utilities.h"

namespace Kratos
{
void BaseSolidElement::Initialize(const ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY


mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(this->GetIntegrationMethod());

if ( mConstitutiveLawVector.size() != integration_points.size() )
mConstitutiveLawVector.resize( integration_points.size() );

InitializeMaterial();

KRATOS_CATCH( "" )
}




void BaseSolidElement::InitializeSolutionStep( const ProcessInfo& rCurrentProcessInfo )
{
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

Values.SetStrainVector(this_constitutive_variables.StrainVector);
Values.SetStressVector(this_constitutive_variables.StressVector);
Values.SetConstitutiveMatrix(this_constitutive_variables.D);

const GeometryType& r_geometry = GetGeometry();
const Properties& r_properties = GetProperties();
const auto& N_values = r_geometry.ShapeFunctionsValues(mThisIntegrationMethod);

const GeometryType::IntegrationPointsArrayType& integration_points = r_geometry.IntegrationPoints(mThisIntegrationMethod);

for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
CalculateKinematicVariables(this_kinematic_variables, point_number, mThisIntegrationMethod);

SetConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points);

mConstitutiveLawVector[point_number]->InitializeMaterialResponse(Values, GetStressMeasure());

mConstitutiveLawVector[point_number]->InitializeSolutionStep( r_properties, r_geometry, row( N_values, point_number ), rCurrentProcessInfo);
}
}




void BaseSolidElement::InitializeNonLinearIteration( const ProcessInfo& rCurrentProcessInfo )
{
const GeometryType& r_geometry = GetGeometry();
const Properties& r_properties = GetProperties();
const auto& N_values = r_geometry.ShapeFunctionsValues(mThisIntegrationMethod);
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->InitializeNonLinearIteration( r_properties, r_geometry, row( N_values, point_number ), rCurrentProcessInfo);
}
}




void BaseSolidElement::FinalizeNonLinearIteration( const ProcessInfo& rCurrentProcessInfo )
{
const GeometryType& r_geometry = GetGeometry();
const Properties& r_properties = GetProperties();
const auto& N_values = r_geometry.ShapeFunctionsValues(mThisIntegrationMethod);
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->FinalizeNonLinearIteration( r_properties, r_geometry, row( N_values, point_number ), rCurrentProcessInfo);
}
}




void BaseSolidElement::FinalizeSolutionStep( const ProcessInfo& rCurrentProcessInfo )
{
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

Values.SetStrainVector(this_constitutive_variables.StrainVector);
Values.SetStressVector(this_constitutive_variables.StressVector);
Values.SetConstitutiveMatrix(this_constitutive_variables.D);

const GeometryType& r_geometry = GetGeometry();
const Properties& r_properties = GetProperties();
const auto& N_values = r_geometry.ShapeFunctionsValues(mThisIntegrationMethod);

const GeometryType::IntegrationPointsArrayType& integration_points = r_geometry.IntegrationPoints(mThisIntegrationMethod);

for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
CalculateKinematicVariables(this_kinematic_variables, point_number, mThisIntegrationMethod);

SetConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points);

mConstitutiveLawVector[point_number]->FinalizeMaterialResponse(Values, GetStressMeasure());

mConstitutiveLawVector[point_number]->FinalizeSolutionStep( r_properties, r_geometry, row( N_values, point_number ), rCurrentProcessInfo);
}
}




void BaseSolidElement::InitializeMaterial()
{
KRATOS_TRY

if ( GetProperties()[CONSTITUTIVE_LAW] != nullptr ) {
const GeometryType& r_geometry = GetGeometry();
const Properties& r_properties = GetProperties();
const auto& N_values = r_geometry.ShapeFunctionsValues(mThisIntegrationMethod);
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number] = GetProperties()[CONSTITUTIVE_LAW]->Clone();
mConstitutiveLawVector[point_number]->InitializeMaterial( r_properties, r_geometry, row(N_values , point_number ));
}
} else
KRATOS_ERROR << "A constitutive law needs to be specified for the element with ID " << this->Id() << std::endl;

KRATOS_CATCH( "" );
}




ConstitutiveLaw::StressMeasure BaseSolidElement::GetStressMeasure() const
{
return ConstitutiveLaw::StressMeasure_PK2;
}




bool BaseSolidElement::UseElementProvidedStrain() const
{
return false;
}




void BaseSolidElement::ResetConstitutiveLaw()
{
KRATOS_TRY

if ( GetProperties()[CONSTITUTIVE_LAW] != nullptr ) {
const GeometryType& r_geometry = GetGeometry();
const Properties& r_properties = GetProperties();
const auto& N_values = r_geometry.ShapeFunctionsValues(mThisIntegrationMethod);
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number )
mConstitutiveLawVector[point_number]->ResetMaterial( r_properties,  r_geometry, row( N_values, point_number ) );
}

KRATOS_CATCH( "" )
}




Element::Pointer BaseSolidElement::Clone (
IndexType NewId,
NodesArrayType const& rThisNodes
) const
{
KRATOS_TRY

KRATOS_WARNING("BaseSolidElement") << " Call BaseSolidElement (base class) Clone " << std::endl;

BaseSolidElement::Pointer p_new_elem = Kratos::make_intrusive<BaseSolidElement>(NewId, GetGeometry().Create(rThisNodes), pGetProperties());
p_new_elem->SetData(this->GetData());
p_new_elem->Set(Flags(*this));

p_new_elem->SetIntegrationMethod(mThisIntegrationMethod);

p_new_elem->SetConstitutiveLawVector(mConstitutiveLawVector);

return p_new_elem;

KRATOS_CATCH("");
}




void BaseSolidElement::EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const
{
KRATOS_TRY;

const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();

if (rResult.size() != dimension * number_of_nodes)
rResult.resize(dimension * number_of_nodes,false);

const SizeType pos = this->GetGeometry()[0].GetDofPosition(DISPLACEMENT_X);

if(dimension == 2) {
for (IndexType i = 0; i < number_of_nodes; ++i) {
const SizeType index = i * 2;
rResult[index] = GetGeometry()[i].GetDof(DISPLACEMENT_X,pos).EquationId();
rResult[index + 1] = GetGeometry()[i].GetDof(DISPLACEMENT_Y,pos+1).EquationId();
}
} else {
for (IndexType i = 0; i < number_of_nodes; ++i) {
const SizeType index = i * 3;
rResult[index] = GetGeometry()[i].GetDof(DISPLACEMENT_X,pos).EquationId();
rResult[index + 1] = GetGeometry()[i].GetDof(DISPLACEMENT_Y,pos+1).EquationId();
rResult[index + 2] = GetGeometry()[i].GetDof(DISPLACEMENT_Z,pos+2).EquationId();
}
}

KRATOS_CATCH("")
};




void BaseSolidElement::GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo
) const
{
KRATOS_TRY;

const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
rElementalDofList.resize(0);
rElementalDofList.reserve(dimension*number_of_nodes);

if(dimension == 2) {
for (IndexType i = 0; i < number_of_nodes; ++i) {
rElementalDofList.push_back(GetGeometry()[i].pGetDof(DISPLACEMENT_X));
rElementalDofList.push_back( GetGeometry()[i].pGetDof(DISPLACEMENT_Y));
}
} else {
for (IndexType i = 0; i < number_of_nodes; ++i) {
rElementalDofList.push_back(GetGeometry()[i].pGetDof(DISPLACEMENT_X));
rElementalDofList.push_back( GetGeometry()[i].pGetDof(DISPLACEMENT_Y));
rElementalDofList.push_back( GetGeometry()[i].pGetDof(DISPLACEMENT_Z));
}
}

KRATOS_CATCH("")
};




void BaseSolidElement::GetValuesVector(
Vector& rValues,
int Step
) const
{
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType mat_size = number_of_nodes * dimension;
if (rValues.size() != mat_size)
rValues.resize(mat_size, false);
for (IndexType i = 0; i < number_of_nodes; ++i)
{
const array_1d<double, 3 >& displacement = GetGeometry()[i].FastGetSolutionStepValue(DISPLACEMENT, Step);
const SizeType index = i * dimension;
for(unsigned int k = 0; k < dimension; ++k)
{
rValues[index + k] = displacement[k];
}
}
}




void BaseSolidElement::GetFirstDerivativesVector(
Vector& rValues,
int Step
) const
{
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType mat_size = number_of_nodes * dimension;
if (rValues.size() != mat_size)
rValues.resize(mat_size, false);
for (IndexType i = 0; i < number_of_nodes; ++i) {
const array_1d<double, 3 >& velocity = GetGeometry()[i].FastGetSolutionStepValue(VELOCITY, Step);
const SizeType index = i * dimension;
for(unsigned int k = 0; k < dimension; ++k)
rValues[index + k] = velocity[k];
}
}




void BaseSolidElement::GetSecondDerivativesVector(
Vector& rValues,
int Step
) const
{
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType mat_size = number_of_nodes * dimension;
if (rValues.size() != mat_size)
rValues.resize(mat_size, false);
for (IndexType i = 0; i < number_of_nodes; ++i) {
const array_1d<double, 3 >& acceleration = GetGeometry()[i].FastGetSolutionStepValue(ACCELERATION, Step);
const SizeType index = i * dimension;
for(unsigned int k = 0; k < dimension; ++k)
rValues[index + k] = acceleration[k];
}
}




void BaseSolidElement::AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double>& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
)
{
KRATOS_TRY;

auto& r_geom = this->GetGeometry();
const SizeType dimension = r_geom.WorkingSpaceDimension();
const SizeType number_of_nodes = r_geom.size();
const SizeType mat_size = number_of_nodes * dimension;

if (rDestinationVariable == NODAL_MASS ) {
VectorType element_mass_vector(mat_size);
this->CalculateLumpedMassVector(element_mass_vector, rCurrentProcessInfo);

for (IndexType i = 0; i < number_of_nodes; ++i) {
const IndexType index = i * dimension;

#pragma omp atomic
r_geom[i].GetValue(NODAL_MASS) += element_mass_vector[index];
}
}

KRATOS_CATCH("")
}




void BaseSolidElement::AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double, 3>>& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
)
{
KRATOS_TRY;

auto& r_geom = this->GetGeometry();
const auto& r_prop = this->GetProperties();
const SizeType dimension = r_geom.WorkingSpaceDimension();
const SizeType number_of_nodes = r_geom.size();
const SizeType element_size = dimension * number_of_nodes;

Vector damping_residual_contribution = ZeroVector(element_size);

if (r_prop.Has(RAYLEIGH_ALPHA) || r_prop.Has(RAYLEIGH_BETA)) {
Vector current_nodal_velocities = ZeroVector(element_size);
this->GetFirstDerivativesVector(current_nodal_velocities);

Matrix damping_matrix(element_size, element_size);
this->CalculateDampingMatrixWithLumpedMass(damping_matrix, rCurrentProcessInfo);

noalias(damping_residual_contribution) = prod(damping_matrix, current_nodal_velocities);
}

if (rRHSVariable == RESIDUAL_VECTOR && rDestinationVariable == FORCE_RESIDUAL) {
for (IndexType i = 0; i < number_of_nodes; ++i) {
const IndexType index = dimension * i;

array_1d<double, 3>& r_force_residual = r_geom[i].FastGetSolutionStepValue(FORCE_RESIDUAL);

for (IndexType j = 0; j < dimension; ++j) {
#pragma omp atomic
r_force_residual[j] += rRHSVector[index + j] - damping_residual_contribution[index + j];
}
}
}

KRATOS_CATCH("")
}




void BaseSolidElement::CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
)
{
const bool CalculateStiffnessMatrixFlag = true;
const bool CalculateResidualVectorFlag = true;

CalculateAll( rLeftHandSideMatrix, rRightHandSideVector, rCurrentProcessInfo, CalculateStiffnessMatrixFlag, CalculateResidualVectorFlag );
}




void BaseSolidElement::CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo
)
{
KRATOS_TRY;
VectorType RHS;
CalculateLocalSystem(rLeftHandSideMatrix, RHS, rCurrentProcessInfo);
KRATOS_CATCH("");
}




void BaseSolidElement::CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
)
{
const bool CalculateStiffnessMatrixFlag = false;
const bool CalculateResidualVectorFlag = true;
MatrixType temp = Matrix();

CalculateAll( temp, rRightHandSideVector, rCurrentProcessInfo, CalculateStiffnessMatrixFlag, CalculateResidualVectorFlag );
}




void BaseSolidElement::CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo
)
{
KRATOS_TRY;

const auto& r_prop = GetProperties();

const auto& r_geom = GetGeometry();
SizeType dimension = r_geom.WorkingSpaceDimension();
SizeType number_of_nodes = r_geom.size();
SizeType mat_size = dimension * number_of_nodes;

if (rMassMatrix.size1() != mat_size || rMassMatrix.size2() != mat_size)
rMassMatrix.resize( mat_size, mat_size, false );
rMassMatrix = ZeroMatrix( mat_size, mat_size );

KRATOS_ERROR_IF_NOT(r_prop.Has(DENSITY)) << "DENSITY has to be provided for the calculation of the MassMatrix!" << std::endl;

const bool compute_lumped_mass_matrix =  r_prop.Has(COMPUTE_LUMPED_MASS_MATRIX) ? r_prop[COMPUTE_LUMPED_MASS_MATRIX] : false;

if (compute_lumped_mass_matrix) {
VectorType temp_vector(mat_size);
CalculateLumpedMassVector(temp_vector, rCurrentProcessInfo);
for (IndexType i = 0; i < mat_size; ++i)
rMassMatrix(i, i) = temp_vector[i];
} else { 
const double density = r_prop[DENSITY];
const double thickness = (dimension == 2 && r_prop.Has(THICKNESS)) ? r_prop[THICKNESS] : 1.0;

Matrix J0(dimension, dimension);

IntegrationMethod integration_method = IntegrationUtilities::GetIntegrationMethodForExactMassMatrixEvaluation(r_geom);
const GeometryType::IntegrationPointsArrayType& integration_points = r_geom.IntegrationPoints( integration_method );
const Matrix& Ncontainer = r_geom.ShapeFunctionsValues(integration_method);

for ( IndexType point_number = 0; point_number < integration_points.size(); ++point_number ) {
GeometryUtils::JacobianOnInitialConfiguration(
r_geom, integration_points[point_number], J0);
const double detJ0 = MathUtils<double>::Det(J0);
const double integration_weight =
GetIntegrationWeight(integration_points, point_number, detJ0) * thickness;
const Vector& rN = row(Ncontainer,point_number);

for ( IndexType i = 0; i < number_of_nodes; ++i ) {
const SizeType index_i = i * dimension;

for ( IndexType j = 0; j < number_of_nodes; ++j ) {
const SizeType index_j = j * dimension;
const double NiNj_weight = rN[i] * rN[j] * integration_weight * density;

for ( IndexType k = 0; k < dimension; ++k )
rMassMatrix( index_i + k, index_j + k ) += NiNj_weight;
}
}
}
}

KRATOS_CATCH("");
}




void BaseSolidElement::CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
)
{
const unsigned int mat_size = GetGeometry().PointsNumber() * GetGeometry().WorkingSpaceDimension();
this->CalculateRayleighDampingMatrix(*this, rDampingMatrix, rCurrentProcessInfo, mat_size);
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<bool>& rVariable,
std::vector<bool>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const GeometryType::IntegrationPointsArrayType &integration_points = GetGeometry().IntegrationPoints(this->GetIntegrationMethod());

const SizeType number_of_integration_points = integration_points.size();
if (rOutput.size() != number_of_integration_points)
rOutput.resize(number_of_integration_points);

if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number <number_of_integration_points; ++point_number ) {
bool value;
mConstitutiveLawVector[point_number]->GetValue( rVariable, value);
rOutput[point_number] = value;
}
} else {
ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

for ( IndexType ii = 0; ii < mConstitutiveLawVector.size(); ++ii ) {
bool solution;
solution = mConstitutiveLawVector[ii]->CalculateValue( Values, rVariable, solution);
rOutput[ii] = solution;
}
}
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<int>& rVariable,
std::vector<int>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const GeometryType::IntegrationPointsArrayType &integration_points = GetGeometry().IntegrationPoints(this->GetIntegrationMethod());

const SizeType number_of_integration_points = integration_points.size();
if (rOutput.size() != number_of_integration_points)
rOutput.resize(number_of_integration_points);

if (mConstitutiveLawVector[0]->Has( rVariable)) {
GetValueOnConstitutiveLaw(rVariable, rOutput);
} else {
CalculateOnConstitutiveLaw(rVariable, rOutput, rCurrentProcessInfo);
}
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const GeometryType::IntegrationPointsArrayType &integration_points = GetGeometry().IntegrationPoints(this->GetIntegrationMethod());

const std::size_t number_of_integration_points = integration_points.size();
const auto& r_geometry = GetGeometry();

if ( rOutput.size() != number_of_integration_points )
rOutput.resize( number_of_integration_points );

if (mConstitutiveLawVector[0]->Has( rVariable)) {
GetValueOnConstitutiveLaw(rVariable, rOutput);
} else {
if (rVariable == INTEGRATION_WEIGHT) {
const SizeType number_of_nodes = r_geometry.size();
const SizeType dimension = r_geometry.WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);

for (IndexType point_number = 0; point_number < number_of_integration_points; ++point_number) {
this_kinematic_variables.detJ0 = CalculateDerivativesOnReferenceConfiguration(this_kinematic_variables.J0,
this_kinematic_variables.InvJ0,
this_kinematic_variables.DN_DX,
point_number,
this->GetIntegrationMethod());

double integration_weight = GetIntegrationWeight(integration_points,
point_number,
this_kinematic_variables.detJ0);

if (dimension == 2 && this->GetProperties().Has(THICKNESS))
integration_weight *= this->GetProperties()[THICKNESS];

rOutput[point_number] = integration_weight;
}
} else if ( rVariable == STRAIN_ENERGY ) {
const SizeType number_of_nodes = r_geometry.size();
const SizeType dimension = r_geometry.WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(r_geometry,GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, false);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

const GeometryType::IntegrationPointsArrayType& integration_points = r_geometry.IntegrationPoints(this->GetIntegrationMethod());

Values.SetStrainVector(this_constitutive_variables.StrainVector); 

for (IndexType point_number = 0; point_number < number_of_integration_points; ++point_number) {
CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

SetConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points);

double StrainEnergy = 0.0;

mConstitutiveLawVector[point_number]->CalculateValue(Values, STRAIN_ENERGY, StrainEnergy);

rOutput[point_number] = StrainEnergy;
}
} else if ( rVariable == ERROR_INTEGRATION_POINT ) {
const SizeType number_of_nodes = r_geometry.size();
const SizeType dimension = r_geometry.WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(r_geometry,GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, false);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);

const GeometryType::IntegrationPointsArrayType& integration_points = r_geometry.IntegrationPoints(  );

std::vector<Vector> sigma_FE_solution(number_of_nodes);
const Variable<Vector>& r_variable_stress = CAUCHY_STRESS_VECTOR;
CalculateOnIntegrationPoints(r_variable_stress, sigma_FE_solution, rCurrentProcessInfo);

Vector detJ(number_of_integration_points);
detJ = r_geometry.DeterminantOfJacobian(detJ);

Values.SetStrainVector(this_constitutive_variables.StrainVector); 

if (r_geometry[0].Has(RECOVERED_STRESS)) {
for (IndexType point_number = 0; point_number < number_of_integration_points; point_number++) {
CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

CalculateConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points, GetStressMeasure());

double integration_weight = GetIntegrationWeight(integration_points, point_number, detJ[point_number]);

if (dimension == 2 && this->GetProperties().Has(THICKNESS))
integration_weight *= this->GetProperties()[THICKNESS];

Vector sigma_recovered = ZeroVector(strain_size);

for (IndexType node_number=0; node_number<number_of_nodes; node_number++) {
const auto& r_sigma_recovered_node = r_geometry[node_number].GetValue(RECOVERED_STRESS);
for (IndexType stress_component = 0; stress_component<strain_size; stress_component++) {
sigma_recovered[stress_component] += this_kinematic_variables.N[node_number] * r_sigma_recovered_node[stress_component];
}
}

Vector error_sigma(strain_size);
error_sigma = sigma_recovered - sigma_FE_solution[point_number];

KRATOS_TRACE("ERROR_INTEGRATION_POINT")
<<"sigma recovered: " << sigma_recovered << std::endl
<<"sigma FE: " << sigma_FE_solution[point_number] << std::endl;

Matrix invD(strain_size,strain_size);
double detD;
MathUtils<double>::InvertMatrix(this_constitutive_variables.D, invD,detD);

rOutput[point_number] = integration_weight * inner_prod(error_sigma, prod(invD, error_sigma));
}
} else {
for (IndexType point_number = 0; point_number < number_of_integration_points; point_number++) {
rOutput[point_number] = 0.0;
}
}
} else if (rVariable == VON_MISES_STRESS) {
const SizeType number_of_nodes = r_geometry.size();
const SizeType dimension = r_geometry.WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(r_geometry,GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

Values.SetStrainVector(this_constitutive_variables.StrainVector);

for (IndexType point_number = 0; point_number < number_of_integration_points; ++point_number) {
CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

CalculateConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points, GetStressMeasure());

const Matrix stress_tensor = MathUtils<double>::StressVectorToTensor( this_constitutive_variables.StressVector );

double sigma_equivalent = 0.0;

if (dimension == 2) {
sigma_equivalent = std::pow((stress_tensor(0,0) - stress_tensor(1,1)), 2.0) +
3*(stress_tensor(0,1) * stress_tensor(1,0));
} else {
sigma_equivalent = 0.5*(std::pow((stress_tensor(0,0) - stress_tensor(1,1)), 2.0) +
std::pow((stress_tensor(1,1) - stress_tensor(2,2)), 2.0) +
std::pow((stress_tensor(2,2) - stress_tensor(0,0)), 2.0) +
6*(stress_tensor(0,1) * stress_tensor(1,0) +
stress_tensor(1,2) * stress_tensor(2,1) +
stress_tensor(2,0) * stress_tensor(0,2)));
}

if( sigma_equivalent < 0.0 )
rOutput[point_number] = 0.0;
else
rOutput[point_number] = std::sqrt(sigma_equivalent);
}
} else {
CalculateOnConstitutiveLaw(rVariable, rOutput, rCurrentProcessInfo);
}
}
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3>>& rVariable,
std::vector<array_1d<double, 3>>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const GeometryType::IntegrationPointsArrayType &integration_points = GetGeometry().IntegrationPoints(this->GetIntegrationMethod());

const SizeType number_of_integration_points = integration_points.size();
if ( rOutput.size() != number_of_integration_points )
rOutput.resize( number_of_integration_points );

if (mConstitutiveLawVector[0]->Has( rVariable)) {
GetValueOnConstitutiveLaw(rVariable, rOutput);
} else {
if (rVariable == INTEGRATION_COORDINATES) {
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);

for (IndexType point_number = 0; point_number < number_of_integration_points; ++point_number) {
Point global_point;
GetGeometry().GlobalCoordinates(global_point, integration_points[point_number]);

rOutput[point_number] = global_point.Coordinates();
}
} else {
CalculateOnConstitutiveLaw(rVariable, rOutput, rCurrentProcessInfo);
}
}
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<array_1d<double, 6>>& rVariable,
std::vector<array_1d<double, 6>>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const GeometryType::IntegrationPointsArrayType &integration_points = GetGeometry().IntegrationPoints(this->GetIntegrationMethod());

const SizeType number_of_integration_points = integration_points.size();
if (rOutput.size() != number_of_integration_points)
rOutput.resize(number_of_integration_points);

if (mConstitutiveLawVector[0]->Has( rVariable)) {
GetValueOnConstitutiveLaw(rVariable, rOutput);
}  else {
CalculateOnConstitutiveLaw(rVariable, rOutput, rCurrentProcessInfo);
}
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<Vector>& rVariable,
std::vector<Vector>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints( this->GetIntegrationMethod() );

const SizeType number_of_integration_points = integration_points.size();
if ( rOutput.size() != number_of_integration_points )
rOutput.resize( number_of_integration_points );

if (mConstitutiveLawVector[0]->Has( rVariable)) {
GetValueOnConstitutiveLaw(rVariable, rOutput);
} else {
if ( rVariable == INSITU_STRESS ) {
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();
Vector strain_vector( strain_size );

for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
if ( rOutput[point_number].size() != strain_vector.size() )
rOutput[point_number].resize( strain_vector.size(), false );

rOutput[point_number] = mConstitutiveLawVector[point_number]->GetValue( INSITU_STRESS, rOutput[point_number] );
}
} else if ( rVariable == CAUCHY_STRESS_VECTOR || rVariable == PK2_STRESS_VECTOR ) {
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

Values.SetStrainVector(this_constitutive_variables.StrainVector);

for ( IndexType point_number = 0; point_number < number_of_integration_points; ++point_number ) {
CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

if( rVariable == CAUCHY_STRESS_VECTOR) {
CalculateConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points, ConstitutiveLaw::StressMeasure_Cauchy);
} else {
CalculateConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points,ConstitutiveLaw::StressMeasure_PK2);
}

if ( rOutput[point_number].size() != strain_size )
rOutput[point_number].resize( strain_size, false );

rOutput[point_number] = this_constitutive_variables.StressVector;
}
} else if( rVariable == GREEN_LAGRANGE_STRAIN_VECTOR  || rVariable == ALMANSI_STRAIN_VECTOR ) {
const SizeType number_of_nodes = GetGeometry().size();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

Flags &ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, false);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

Values.SetStrainVector(this_constitutive_variables.StrainVector);

const ConstitutiveLaw::StressMeasure this_stress_measure = rVariable == GREEN_LAGRANGE_STRAIN_VECTOR ? ConstitutiveLaw::StressMeasure_PK2 : ConstitutiveLaw::StressMeasure_Kirchhoff;

for ( IndexType point_number = 0; point_number < number_of_integration_points; ++point_number ) {
CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

CalculateConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points, this_stress_measure);

if ( rOutput[point_number].size() != strain_size)
rOutput[point_number].resize( strain_size, false );

rOutput[point_number] = this_constitutive_variables.StrainVector;
}
} else {
CalculateOnConstitutiveLaw(rVariable, rOutput, rCurrentProcessInfo);
}
}
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<Matrix>& rVariable,
std::vector<Matrix>& rOutput,
const ProcessInfo& rCurrentProcessInfo
)
{
const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints( this->GetIntegrationMethod() );
const SizeType dimension = GetGeometry().WorkingSpaceDimension();

if ( rOutput.size() != integration_points.size() )
rOutput.resize( integration_points.size() );

if (mConstitutiveLawVector[0]->Has( rVariable)) {
GetValueOnConstitutiveLaw(rVariable, rOutput);
} else {
if ( rVariable == CAUCHY_STRESS_TENSOR || rVariable == PK2_STRESS_TENSOR ) {
std::vector<Vector> stress_vector;

if( rVariable == CAUCHY_STRESS_TENSOR )
this->CalculateOnIntegrationPoints( CAUCHY_STRESS_VECTOR, stress_vector, rCurrentProcessInfo );
else
this->CalculateOnIntegrationPoints( PK2_STRESS_VECTOR, stress_vector, rCurrentProcessInfo );

for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
if ( rOutput[point_number].size2() != dimension )
rOutput[point_number].resize( dimension, dimension, false );

rOutput[point_number] = MathUtils<double>::StressVectorToTensor(stress_vector[point_number]);
}
}
else if ( rVariable == GREEN_LAGRANGE_STRAIN_TENSOR  || rVariable == ALMANSI_STRAIN_TENSOR) {
std::vector<Vector> strain_vector;
if( rVariable == GREEN_LAGRANGE_STRAIN_TENSOR )
CalculateOnIntegrationPoints( GREEN_LAGRANGE_STRAIN_VECTOR, strain_vector, rCurrentProcessInfo );
else
CalculateOnIntegrationPoints( ALMANSI_STRAIN_VECTOR, strain_vector, rCurrentProcessInfo );

for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
if ( rOutput[point_number].size2() != dimension )
rOutput[point_number].resize( dimension, dimension, false );

rOutput[point_number] = MathUtils<double>::StrainVectorToTensor(strain_vector[point_number]);
}
} else if ( rVariable == CONSTITUTIVE_MATRIX ) {
const SizeType number_of_nodes = GetGeometry().size();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

Flags& ConstitutiveLawOptions=Values.GetOptions();
ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, false);
ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);

Values.SetStrainVector(this_constitutive_variables.StrainVector);
Values.SetConstitutiveMatrix(this_constitutive_variables.D); 

for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

CalculateConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points, GetStressMeasure());

if( rOutput[point_number].size2() != this_constitutive_variables.D.size2() )
rOutput[point_number].resize( this_constitutive_variables.D.size1() , this_constitutive_variables.D.size2() , false );

rOutput[point_number] = this_constitutive_variables.D;
}
} else if ( rVariable == DEFORMATION_GRADIENT ) { 
const SizeType number_of_nodes = GetGeometry().size();
const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
ConstitutiveVariables this_constitutive_variables(strain_size);

ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
CalculateKinematicVariables(this_kinematic_variables, point_number, this->GetIntegrationMethod());

if( rOutput[point_number].size2() != this_kinematic_variables.F.size2() )
rOutput[point_number].resize( this_kinematic_variables.F.size1() , this_kinematic_variables.F.size2() , false );

rOutput[point_number] = this_kinematic_variables.F;
}
}  else {
CalculateOnConstitutiveLaw(rVariable, rOutput, rCurrentProcessInfo);
}
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<bool>& rVariable,
const std::vector<bool>& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->SetValue( rVariable,rValues[point_number], rCurrentProcessInfo);
}
} else {
KRATOS_WARNING("BaseSolidElement") << "The variable " << rVariable << " is not implemented in the current ConstitutiveLaw" << std::endl;
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<int>& rVariable,
const std::vector<int>& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->SetValue( rVariable,rValues[point_number], rCurrentProcessInfo);
}
} else {
KRATOS_WARNING("BaseSolidElement") << "The variable " << rVariable << " is not implemented in the current ConstitutiveLaw" << std::endl;
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<double>& rVariable,
const std::vector<double>& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->SetValue( rVariable,rValues[point_number], rCurrentProcessInfo);
}
} else {
KRATOS_WARNING("BaseSolidElement") << "The variable " << rVariable << " is not implemented in the current ConstitutiveLaw" << std::endl;
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<Vector>& rVariable,
const std::vector<Vector>& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->SetValue( rVariable,rValues[point_number], rCurrentProcessInfo);
}
} else {
KRATOS_WARNING("BaseSolidElement") << "The variable " << rVariable << " is not implemented in the current ConstitutiveLaw" << std::endl;
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<ConstitutiveLaw::Pointer>& rVariable,
const std::vector<ConstitutiveLaw::Pointer>& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (rVariable == CONSTITUTIVE_LAW) {
const SizeType integration_points_number = mConstitutiveLawVector.size();
for ( IndexType point_number = 0; point_number < integration_points_number; ++point_number ) {
mConstitutiveLawVector[point_number] = rValues[point_number];
}
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
const std::vector<array_1d<double, 3 > >& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->SetValue( rVariable,rValues[point_number], rCurrentProcessInfo);
}
} else {
KRATOS_WARNING("BaseSolidElement") << "The variable " << rVariable << " is not implemented in the current ConstitutiveLaw" << std::endl;
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<array_1d<double, 6 > >& rVariable,
const std::vector<array_1d<double, 6 > >& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->SetValue( rVariable,rValues[point_number], rCurrentProcessInfo);
}
} else {
KRATOS_WARNING("BaseSolidElement") << "The variable " << rVariable << " is not implemented in the current ConstitutiveLaw" << std::endl;
}
}




void BaseSolidElement::SetValuesOnIntegrationPoints(
const Variable<Matrix>& rVariable,
const std::vector<Matrix>& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (mConstitutiveLawVector[0]->Has( rVariable)) {
for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
mConstitutiveLawVector[point_number]->SetValue( rVariable,rValues[point_number], rCurrentProcessInfo);
}
} else {
KRATOS_WARNING("BaseSolidElement") << "The variable " << rVariable << " is not implemented in the current ConstitutiveLaw" << std::endl;
}
}




void BaseSolidElement::CalculateOnIntegrationPoints(
const Variable<ConstitutiveLaw::Pointer>& rVariable,
std::vector<ConstitutiveLaw::Pointer>& rValues,
const ProcessInfo& rCurrentProcessInfo
)
{
if (rVariable == CONSTITUTIVE_LAW) {
const SizeType integration_points_number = mConstitutiveLawVector.size();
if (rValues.size() != integration_points_number) {
rValues.resize(integration_points_number);
}
for (IndexType point_number = 0; point_number < integration_points_number; ++point_number) {
rValues[point_number] = mConstitutiveLawVector[point_number];
}
}
}




int  BaseSolidElement::Check( const ProcessInfo& rCurrentProcessInfo ) const
{
KRATOS_TRY;

Element::Check(rCurrentProcessInfo);

const SizeType number_of_nodes = this->GetGeometry().size();
const SizeType dimension = this->GetGeometry().WorkingSpaceDimension();

for ( IndexType i = 0; i < number_of_nodes; i++ ) {
const NodeType &rnode = this->GetGeometry()[i];
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(DISPLACEMENT,rnode)

KRATOS_CHECK_DOF_IN_NODE(DISPLACEMENT_X, rnode)
KRATOS_CHECK_DOF_IN_NODE(DISPLACEMENT_Y, rnode)
KRATOS_CHECK_DOF_IN_NODE(DISPLACEMENT_Z, rnode)
}

KRATOS_ERROR_IF_NOT(this->GetProperties().Has( CONSTITUTIVE_LAW )) << "Constitutive law not provided for property " << this->GetProperties().Id() << std::endl;

const SizeType strain_size = this->GetProperties().GetValue( CONSTITUTIVE_LAW )->GetStrainSize();
if ( dimension == 2 ) {
KRATOS_ERROR_IF( strain_size < 3 || strain_size > 4) << "Wrong constitutive law used. This is a 2D element! expected strain size is 3 or 4 (el id = ) " << this->Id() << std::endl;
} else {
KRATOS_ERROR_IF_NOT(strain_size == 6) << "Wrong constitutive law used. This is a 3D element! expected strain size is 6 (el id = ) "<<  this->Id() << std::endl;
}

if ( mConstitutiveLawVector.size() > 0 ) {
return mConstitutiveLawVector[0]->Check( GetProperties(), GetGeometry(), rCurrentProcessInfo );
}

return 0;

KRATOS_CATCH( "" );
}




void BaseSolidElement::CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
)
{
KRATOS_ERROR << "You have called to the CalculateAll from the base class for solid elements" << std::endl;
}

/


void BaseSolidElement::CalculateKinematicVariables(
KinematicVariables& rThisKinematicVariables,
const IndexType PointNumber,
const GeometryType::IntegrationMethod& rIntegrationMethod
)
{
KRATOS_ERROR << "You have called to the CalculateKinematicVariables from the base class for solid elements" << std::endl;
}




void BaseSolidElement::CalculateConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const ConstitutiveLaw::StressMeasure ThisStressMeasure
)
{
SetConstitutiveVariables(rThisKinematicVariables, rThisConstitutiveVariables, rValues, PointNumber, IntegrationPoints);

mConstitutiveLawVector[PointNumber]->CalculateMaterialResponse(rValues, ThisStressMeasure); 
}




void BaseSolidElement::SetConstitutiveVariables(
KinematicVariables& rThisKinematicVariables,
ConstitutiveVariables& rThisConstitutiveVariables,
ConstitutiveLaw::Parameters& rValues,
const IndexType PointNumber,
const GeometryType::IntegrationPointsArrayType& IntegrationPoints
)
{
rValues.SetShapeFunctionsValues(rThisKinematicVariables.N); 
rValues.SetDeterminantF(rThisKinematicVariables.detF); 
rValues.SetDeformationGradientF(rThisKinematicVariables.F); 

rValues.SetConstitutiveMatrix(rThisConstitutiveVariables.D); 
rValues.SetStressVector(rThisConstitutiveVariables.StressVector); 
}




Matrix& BaseSolidElement::CalculateDeltaDisplacement(Matrix& DeltaDisplacement) const
{
KRATOS_TRY

const SizeType number_of_nodes = GetGeometry().PointsNumber();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();

DeltaDisplacement.resize(number_of_nodes , dimension, false);

for ( IndexType i_node = 0; i_node < number_of_nodes; i_node++ ) {
const array_1d<double, 3 >& current_displacement  = GetGeometry()[i_node].FastGetSolutionStepValue(DISPLACEMENT);
const array_1d<double, 3 >& previous_displacement = GetGeometry()[i_node].FastGetSolutionStepValue(DISPLACEMENT,1);

for ( IndexType j_dim = 0; j_dim < dimension; ++j_dim )
DeltaDisplacement(i_node, j_dim) = current_displacement[j_dim] - previous_displacement[j_dim];
}

return DeltaDisplacement;

KRATOS_CATCH( "" )
}




double BaseSolidElement::CalculateDerivativesOnReferenceConfiguration(
Matrix& rJ0,
Matrix& rInvJ0,
Matrix& rDN_DX,
const IndexType PointNumber,
IntegrationMethod ThisIntegrationMethod
) const
{
const GeometryType& r_geom = GetGeometry();
GeometryUtils::JacobianOnInitialConfiguration(
r_geom,
r_geom.IntegrationPoints(ThisIntegrationMethod)[PointNumber], rJ0);
double detJ0;
MathUtils<double>::InvertMatrix(rJ0, rInvJ0, detJ0);
const Matrix& rDN_De =
GetGeometry().ShapeFunctionsLocalGradients(ThisIntegrationMethod)[PointNumber];
GeometryUtils::ShapeFunctionsGradients(rDN_De, rInvJ0, rDN_DX);
return detJ0;
}




double BaseSolidElement::CalculateDerivativesOnCurrentConfiguration(
Matrix& rJ,
Matrix& rInvJ,
Matrix& rDN_DX,
const IndexType PointNumber,
IntegrationMethod ThisIntegrationMethod
) const
{
double detJ;
rJ = GetGeometry().Jacobian( rJ, PointNumber, ThisIntegrationMethod );
const Matrix& DN_De = GetGeometry().ShapeFunctionsLocalGradients(ThisIntegrationMethod)[PointNumber];
MathUtils<double>::InvertMatrix( rJ, rInvJ, detJ );
GeometryUtils::ShapeFunctionsGradients(DN_De, rInvJ, rDN_DX);
return detJ;
}




array_1d<double, 3> BaseSolidElement::GetBodyForce(
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const IndexType PointNumber
) const
{
array_1d<double, 3> body_force;
for (IndexType i = 0; i < 3; ++i)
body_force[i] = 0.0;

const auto& r_properties = GetProperties();
double density = 0.0;
if (r_properties.Has( DENSITY ))
density = r_properties[DENSITY];

if (r_properties.Has( VOLUME_ACCELERATION ))
noalias(body_force) += density * r_properties[VOLUME_ACCELERATION];

const auto& r_geometry = this->GetGeometry();
if( r_geometry[0].SolutionStepsDataHas(VOLUME_ACCELERATION) ) {
Vector N;
N = r_geometry.ShapeFunctionsValues(N, IntegrationPoints[PointNumber].Coordinates());
for (IndexType i_node = 0; i_node < r_geometry.size(); ++i_node)
noalias(body_force) += N[i_node] * density * r_geometry[i_node].FastGetSolutionStepValue(VOLUME_ACCELERATION);
}

return body_force;
}




void BaseSolidElement::CalculateAndAddKm(
MatrixType& rLeftHandSideMatrix,
const Matrix& B,
const Matrix& D,
const double IntegrationWeight
) const
{
KRATOS_TRY

noalias( rLeftHandSideMatrix ) += IntegrationWeight * prod( trans( B ), Matrix(prod(D, B)));

KRATOS_CATCH( "" )
}




void BaseSolidElement::CalculateAndAddKg(
MatrixType& rLeftHandSideMatrix,
const Matrix& DN_DX,
const Vector& StressVector,
const double IntegrationWeight
) const
{
KRATOS_TRY

const SizeType dimension = GetGeometry().WorkingSpaceDimension();
Matrix stress_tensor = MathUtils<double>::StressVectorToTensor( StressVector );
Matrix reduced_Kg = prod( DN_DX, IntegrationWeight * Matrix( prod( stress_tensor, trans( DN_DX ) ) ) ); 
MathUtils<double>::ExpandAndAddReducedMatrix( rLeftHandSideMatrix, reduced_Kg, dimension );

KRATOS_CATCH( "" )
}




void BaseSolidElement::CalculateAndAddResidualVector(
VectorType& rRightHandSideVector,
const KinematicVariables& rThisKinematicVariables,
const ProcessInfo& rCurrentProcessInfo,
const array_1d<double, 3>& rBodyForce,
const Vector& rStressVector,
const double IntegrationWeight
) const
{
KRATOS_TRY

this->CalculateAndAddExtForceContribution( rThisKinematicVariables.N, rCurrentProcessInfo, rBodyForce, rRightHandSideVector, IntegrationWeight );

noalias( rRightHandSideVector ) -= IntegrationWeight * prod( trans( rThisKinematicVariables.B ), rStressVector );

KRATOS_CATCH( "" )
}




void BaseSolidElement::CalculateAndAddExtForceContribution(
const Vector& rN,
const ProcessInfo& rCurrentProcessInfo,
const array_1d<double, 3>& rBodyForce,
VectorType& rRightHandSideVector,
const double Weight
) const
{
KRATOS_TRY;

const SizeType number_of_nodes = GetGeometry().PointsNumber();
const SizeType dimension = GetGeometry().WorkingSpaceDimension();

for ( IndexType i = 0; i < number_of_nodes; ++i ) {
const SizeType index = dimension * i;

for ( IndexType j = 0; j < dimension; ++j )
rRightHandSideVector[index + j] += Weight * rN[i] * rBodyForce[j];
}

KRATOS_CATCH( "" )
}




void BaseSolidElement::CalculateLumpedMassVector(
VectorType &rLumpedMassVector,
const ProcessInfo &rCurrentProcessInfo) const
{
KRATOS_TRY;

const auto& r_geom = GetGeometry();
const auto& r_prop = GetProperties();
const SizeType dimension = r_geom.WorkingSpaceDimension();
const SizeType number_of_nodes = r_geom.size();
const SizeType mat_size = dimension * number_of_nodes;

if (rLumpedMassVector.size() != mat_size)
rLumpedMassVector.resize( mat_size, false );

const double density = r_prop[DENSITY];
const double thickness = (dimension == 2 && r_prop.Has(THICKNESS)) ? r_prop[THICKNESS] : 1.0;

const double total_mass = GetGeometry().DomainSize() * density * thickness;

Vector lumping_factors;
lumping_factors = GetGeometry().LumpingFactors( lumping_factors );

for ( IndexType i = 0; i < number_of_nodes; ++i ) {
const double temp = lumping_factors[i] * total_mass;
for ( IndexType j = 0; j < dimension; ++j ) {
IndexType index = i * dimension + j;
rLumpedMassVector[index] = temp;
}
}

KRATOS_CATCH("");
}




void BaseSolidElement::CalculateDampingMatrixWithLumpedMass(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
)
{
KRATOS_TRY;

unsigned int number_of_nodes = GetGeometry().size();
unsigned int dimension = GetGeometry().WorkingSpaceDimension();

unsigned int mat_size = number_of_nodes * dimension;

if ( rDampingMatrix.size1() != mat_size )
rDampingMatrix.resize( mat_size, mat_size, false );

noalias( rDampingMatrix ) = ZeroMatrix( mat_size, mat_size );

double alpha = 0.0;
if( GetProperties().Has(RAYLEIGH_ALPHA) )
alpha = GetProperties()[RAYLEIGH_ALPHA];
else if( rCurrentProcessInfo.Has(RAYLEIGH_ALPHA) )
alpha = rCurrentProcessInfo[RAYLEIGH_ALPHA];

double beta  = 0.0;
if( GetProperties().Has(RAYLEIGH_BETA) )
beta = GetProperties()[RAYLEIGH_BETA];
else if( rCurrentProcessInfo.Has(RAYLEIGH_BETA) )
beta = rCurrentProcessInfo[RAYLEIGH_BETA];


if (alpha > std::numeric_limits<double>::epsilon()) {
VectorType temp_vector(mat_size);
CalculateLumpedMassVector(temp_vector, rCurrentProcessInfo);
for (IndexType i = 0; i < mat_size; ++i)
rDampingMatrix(i, i) += alpha * temp_vector[i];
}

if (beta > std::numeric_limits<double>::epsilon()) {
MatrixType stiffness_matrix( mat_size, mat_size );
VectorType residual_vector( mat_size );

this->CalculateAll(stiffness_matrix, residual_vector, rCurrentProcessInfo, true, false);

noalias( rDampingMatrix ) += beta  * stiffness_matrix;
}

KRATOS_CATCH( "" )
}




void BaseSolidElement::CalculateRayleighDampingMatrix(
Element& rElement,
Element::MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo,
const std::size_t MatrixSize)
{
KRATOS_TRY;

if (rDampingMatrix.size1() != MatrixSize || rDampingMatrix.size2() != MatrixSize) {
rDampingMatrix.resize(MatrixSize, MatrixSize, false);
}
noalias(rDampingMatrix) = ZeroMatrix(MatrixSize, MatrixSize);

const double beta = GetRayleighBeta(rElement.GetProperties(), rCurrentProcessInfo);
if (std::abs(beta) > 0.0) {
Element::MatrixType stiffness_matrix;
rElement.CalculateLeftHandSide(stiffness_matrix, rCurrentProcessInfo);
noalias(rDampingMatrix) += beta  * stiffness_matrix;
}

const double alpha = GetRayleighAlpha(rElement.GetProperties(), rCurrentProcessInfo);
if (std::abs(alpha) > 0.0) {
Element::MatrixType mass_matrix;
rElement.CalculateMassMatrix(mass_matrix, rCurrentProcessInfo);
noalias(rDampingMatrix) += alpha * mass_matrix;
}

KRATOS_CATCH("CalculateRayleighDampingMatrix")
}




double BaseSolidElement::GetRayleighAlpha(
const Properties& rProperties,
const ProcessInfo& rCurrentProcessInfo)
{
if (rProperties.Has(RAYLEIGH_ALPHA)) {
return rProperties[RAYLEIGH_ALPHA];
} else if (rCurrentProcessInfo.Has(RAYLEIGH_ALPHA)) {
return rCurrentProcessInfo[RAYLEIGH_ALPHA];
}

return 0.0;
}




double BaseSolidElement::GetRayleighBeta(
const Properties& rProperties,
const ProcessInfo& rCurrentProcessInfo)
{
if (rProperties.Has(RAYLEIGH_BETA)) {
return rProperties[RAYLEIGH_BETA];
} else if (rCurrentProcessInfo.Has(RAYLEIGH_BETA)) {
return rCurrentProcessInfo[RAYLEIGH_BETA];
}

return 0.0;
}

void BaseSolidElement::save( Serializer& rSerializer ) const
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, Element );
int IntMethod = int(this->GetIntegrationMethod());
rSerializer.save("IntegrationMethod",IntMethod);
rSerializer.save("mConstitutiveLawVector", mConstitutiveLawVector);
}




void BaseSolidElement::load( Serializer& rSerializer )
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, Element );
int IntMethod;
rSerializer.load("IntegrationMethod",IntMethod);
mThisIntegrationMethod = IntegrationMethod(IntMethod);
rSerializer.load("mConstitutiveLawVector", mConstitutiveLawVector);
}
} 
