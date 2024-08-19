
#pragma once

#include "solving_strategies/schemes/residual_based_implicit_time_scheme.h"
#include "includes/variables.h"
#include "includes/checks.h"

namespace Kratos
{



template<class TSparseSpace,  class TDenseSpace >
class ResidualBasedBossakDisplacementScheme
: public ResidualBasedImplicitTimeScheme<TSparseSpace, TDenseSpace>
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedBossakDisplacementScheme );

typedef Scheme<TSparseSpace,TDenseSpace>                                  BaseType;

typedef ResidualBasedImplicitTimeScheme<TSparseSpace,TDenseSpace> ImplicitBaseType;

typedef ResidualBasedBossakDisplacementScheme<TSparseSpace, TDenseSpace> ClassType;

typedef typename ImplicitBaseType::TDataType                             TDataType;

typedef typename ImplicitBaseType::DofsArrayType                     DofsArrayType;

typedef typename Element::DofsVectorType                            DofsVectorType;

typedef typename ImplicitBaseType::TSystemMatrixType             TSystemMatrixType;

typedef typename ImplicitBaseType::TSystemVectorType             TSystemVectorType;

typedef typename ImplicitBaseType::LocalSystemVectorType     LocalSystemVectorType;

typedef typename ImplicitBaseType::LocalSystemMatrixType     LocalSystemMatrixType;

typedef ModelPart::NodeIterator                                       NodeIterator;

typedef ModelPart::NodesContainerType                               NodesArrayType;

typedef ModelPart::ElementsContainerType                         ElementsArrayType;

typedef ModelPart::ConditionsContainerType                     ConditionsArrayType;

typedef typename BaseType::Pointer                                 BaseTypePointer;

typedef double              ComponentType;



explicit ResidualBasedBossakDisplacementScheme(Parameters ThisParameters)
: ImplicitBaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);

mNewmark.gamma = 0.5;

AuxiliarInitializeBossak();
}


explicit ResidualBasedBossakDisplacementScheme(const double Alpha = 0.0)
: ResidualBasedBossakDisplacementScheme(Alpha, 0.25)
{
}


explicit ResidualBasedBossakDisplacementScheme(const double Alpha, const double NewmarkBeta)
:ImplicitBaseType()
{
mBossak.alpha = Alpha;
mNewmark.beta = NewmarkBeta;
mNewmark.gamma = 0.5;

AuxiliarInitializeBossak();
}

explicit ResidualBasedBossakDisplacementScheme(ResidualBasedBossakDisplacementScheme& rOther)
:ImplicitBaseType(rOther)
,mBossak(rOther.mBossak)
,mNewmark(rOther.mNewmark)
,mVector(rOther.mVector)
{
}

BaseTypePointer Clone() override
{
return BaseTypePointer( new ResidualBasedBossakDisplacementScheme(*this) );
}

~ResidualBasedBossakDisplacementScheme
() override {}



typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
}

void CalculateBossakCoefficients()
{
mBossak.beta  = (1.0 - mBossak.alpha) * (1.0 - mBossak.alpha) * mNewmark.beta;
mBossak.gamma = mNewmark.gamma  - mBossak.alpha;
}


void Update(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY;

mpDofUpdater->UpdateDofs(rDofSet, rDx);

block_for_each(rModelPart.Nodes(), array_1d<double,3>(), [&](Node& rNode, array_1d<double,3>& rDeltaDisplacementTLS){
noalias(rDeltaDisplacementTLS) = rNode.FastGetSolutionStepValue(DISPLACEMENT) - rNode.FastGetSolutionStepValue(DISPLACEMENT, 1);

array_1d<double, 3>& r_current_velocity = rNode.FastGetSolutionStepValue(VELOCITY);
const array_1d<double, 3>& r_previous_velocity = rNode.FastGetSolutionStepValue(VELOCITY, 1);

array_1d<double, 3>& r_current_acceleration = rNode.FastGetSolutionStepValue(ACCELERATION);
const array_1d<double, 3>& r_previous_acceleration = rNode.FastGetSolutionStepValue(ACCELERATION, 1);

UpdateVelocity(r_current_velocity, rDeltaDisplacementTLS, r_previous_velocity, r_previous_acceleration);
UpdateAcceleration(r_current_acceleration, rDeltaDisplacementTLS, r_previous_velocity, r_previous_acceleration);
});

KRATOS_CATCH( "" );
}


void Predict(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& rA,
TSystemVectorType& rDx,
TSystemVectorType& rb
) override
{
KRATOS_TRY;

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();
const double delta_time = r_current_process_info[DELTA_TIME];

const auto it_node_begin = rModelPart.Nodes().begin();

KRATOS_ERROR_IF_NOT(it_node_begin->HasDofFor(DISPLACEMENT_X)) << "ResidualBasedBossakDisplacementScheme:: DISPLACEMENT is not added" << std::endl;
const int disppos = it_node_begin->GetDofPosition(DISPLACEMENT_X);
const int velpos = it_node_begin->HasDofFor(VELOCITY_X) ? static_cast<int>(it_node_begin->GetDofPosition(VELOCITY_X)) : -1;
const int accelpos = it_node_begin->HasDofFor(ACCELERATION_X) ? static_cast<int>(it_node_begin->GetDofPosition(ACCELERATION_X)) : -1;

KRATOS_WARNING_IF("ResidualBasedBossakDisplacementScheme", !r_current_process_info.Has(DOMAIN_SIZE)) << "DOMAIN_SIZE not defined. Please define DOMAIN_SIZE. 3D case will be assumed" << std::endl;
const std::size_t dimension = r_current_process_info.Has(DOMAIN_SIZE) ? r_current_process_info.GetValue(DOMAIN_SIZE) : 3;

array_1d<double, 3 > delta_displacement;
std::array<bool, 3> predicted = {false, false, false};
const std::array<const Variable<ComponentType>*, 3> disp_components = {&DISPLACEMENT_X, &DISPLACEMENT_Y, &DISPLACEMENT_Z};
const std::array<const Variable<ComponentType>*, 3> vel_components = {&VELOCITY_X, &VELOCITY_Y, &VELOCITY_Z};
const std::array<const Variable<ComponentType>*, 3> accel_components = {&ACCELERATION_X, &ACCELERATION_Y, &ACCELERATION_Z};

typedef std::tuple<array_1d<double,3>, std::array<bool,3>> TLSContainerType;
TLSContainerType aux_TLS = std::make_tuple(delta_displacement, predicted);

block_for_each(rModelPart.Nodes(), aux_TLS, [&](Node& rNode, TLSContainerType& rAuxTLS){
auto& r_delta_displacement = std::get<0>(rAuxTLS);
auto& r_predicted = std::get<1>(rAuxTLS);

for (std::size_t i_dim = 0; i_dim < dimension; ++i_dim) {
r_predicted[i_dim] = false;
}

const array_1d<double, 3>& r_previous_acceleration = rNode.FastGetSolutionStepValue(ACCELERATION, 1);
const array_1d<double, 3>& r_previous_velocity     = rNode.FastGetSolutionStepValue(VELOCITY,     1);
const array_1d<double, 3>& r_previous_displacement = rNode.FastGetSolutionStepValue(DISPLACEMENT, 1);
array_1d<double, 3>& r_current_acceleration        = rNode.FastGetSolutionStepValue(ACCELERATION);
array_1d<double, 3>& r_current_velocity            = rNode.FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3>& r_current_displacement        = rNode.FastGetSolutionStepValue(DISPLACEMENT);

if (accelpos > -1) {
for (std::size_t i_dim = 0; i_dim < dimension; ++i_dim) {
if (rNode.GetDof(*accel_components[i_dim], accelpos + i_dim).IsFixed()) {
r_delta_displacement[i_dim] = (r_current_acceleration[i_dim] + mBossak.c3 * r_previous_acceleration[i_dim] +  mBossak.c2 * r_previous_velocity[i_dim])/mBossak.c0;
r_current_displacement[i_dim] =  r_previous_displacement[i_dim] + r_delta_displacement[i_dim];
r_predicted[i_dim] = true;
}
}
}
if (velpos > -1) {
for (std::size_t i_dim = 0; i_dim < dimension; ++i_dim) {
if (rNode.GetDof(*vel_components[i_dim], velpos + i_dim).IsFixed() && !r_predicted[i_dim]) {
r_delta_displacement[i_dim] = (r_current_velocity[i_dim] + mBossak.c4 * r_previous_velocity[i_dim] + mBossak.c5 * r_previous_acceleration[i_dim])/mBossak.c1;
r_current_displacement[i_dim] =  r_previous_displacement[i_dim] + r_delta_displacement[i_dim];
r_predicted[i_dim] = true;
}
}
}
for (std::size_t i_dim = 0; i_dim < dimension; ++i_dim) {
if (!rNode.GetDof(*disp_components[i_dim], disppos + i_dim).IsFixed() && !r_predicted[i_dim]) {
r_current_displacement[i_dim] = r_previous_displacement[i_dim] + delta_time * r_previous_velocity[i_dim] + 0.5 * std::pow(delta_time, 2) * r_previous_acceleration[i_dim];
}
}

noalias(r_delta_displacement) = r_current_displacement - r_previous_displacement;
UpdateVelocity(r_current_velocity, r_delta_displacement, r_previous_velocity, r_previous_acceleration);
UpdateAcceleration(r_current_acceleration, r_delta_displacement, r_previous_velocity, r_previous_acceleration);
});

KRATOS_CATCH( "" );
}


void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
{
KRATOS_TRY;

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

ImplicitBaseType::InitializeSolutionStep(rModelPart, A, Dx, b);

const double delta_time = r_current_process_info[DELTA_TIME];

mBossak.c0 = ( 1.0 / (mBossak.beta * delta_time * delta_time) );
mBossak.c1 = ( mBossak.gamma / (mBossak.beta * delta_time) );
mBossak.c2 = ( 1.0 / (mBossak.beta * delta_time) );
mBossak.c3 = ( 0.5 / (mBossak.beta) - 1.0 );
mBossak.c4 = ( (mBossak.gamma / mBossak.beta) - 1.0  );
mBossak.c5 = ( delta_time * 0.5 * ( ( mBossak.gamma / mBossak.beta ) - 2.0 ) );

const auto it_node_begin = rModelPart.Nodes().begin();

KRATOS_WARNING_IF("ResidualBasedBossakDisplacementScheme", !r_current_process_info.Has(DOMAIN_SIZE)) << "DOMAIN_SIZE not defined. Please define DOMAIN_SIZE. 3D case will be assumed" << std::endl;
const std::size_t dimension = r_current_process_info.Has(DOMAIN_SIZE) ? r_current_process_info.GetValue(DOMAIN_SIZE) : 3;

const int velpos = it_node_begin->HasDofFor(VELOCITY_X) ? static_cast<int>(it_node_begin->GetDofPosition(VELOCITY_X)) : -1;
const int accelpos = it_node_begin->HasDofFor(ACCELERATION_X) ? static_cast<int>(it_node_begin->GetDofPosition(ACCELERATION_X)) : -1;

std::array<bool, 3> fixed = {false, false, false};
const std::array<const Variable<ComponentType>*, 3> disp_components = {&DISPLACEMENT_X, &DISPLACEMENT_Y, &DISPLACEMENT_Z};
const std::array<const Variable<ComponentType>*, 3> vel_components = {&VELOCITY_X, &VELOCITY_Y, &VELOCITY_Z};
const std::array<const Variable<ComponentType>*, 3> accel_components = {&ACCELERATION_X, &ACCELERATION_Y, &ACCELERATION_Z};

block_for_each(rModelPart.Nodes(), fixed, [&](Node& rNode, std::array<bool,3>& rFixedTLS){
for (std::size_t i_dim = 0; i_dim < dimension; ++i_dim) {
rFixedTLS[i_dim] = false;
}

if (accelpos > -1) {
for (std::size_t i_dim = 0; i_dim < dimension; ++i_dim) {
if (rNode.GetDof(*accel_components[i_dim], accelpos + i_dim).IsFixed()) {
rNode.Fix(*disp_components[i_dim]);
rFixedTLS[i_dim] = true;
}
}
}
if (velpos > -1) {
for (std::size_t i_dim = 0; i_dim < dimension; ++i_dim) {
if (rNode.GetDof(*vel_components[i_dim], velpos + i_dim).IsFixed() && !rFixedTLS[i_dim]) {
rNode.Fix(*disp_components[i_dim]);
}
}
}
});

KRATOS_CATCH( "" );
}


int Check(const ModelPart& rModelPart) const override
{
KRATOS_TRY;

const int err = ImplicitBaseType::Check(rModelPart);
if(err != 0) return err;

for (const auto& rnode : rModelPart.Nodes()) {
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(DISPLACEMENT,rnode)
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(VELOCITY,rnode)
KRATOS_CHECK_VARIABLE_IN_NODAL_DATA(ACCELERATION,rnode)

KRATOS_CHECK_DOF_IN_NODE(DISPLACEMENT_X, rnode)
KRATOS_CHECK_DOF_IN_NODE(DISPLACEMENT_Y, rnode)
KRATOS_CHECK_DOF_IN_NODE(DISPLACEMENT_Z, rnode)
}

KRATOS_ERROR_IF(rModelPart.GetBufferSize() < 2)
<< "Insufficient buffer size. Buffer size should be greater than 2. Current size is: "
<< rModelPart.GetBufferSize() << std::endl;

KRATOS_ERROR_IF(mBossak.alpha > 0.0 || mBossak.alpha < -0.5) << "Value not admissible for "
<< "AlphaBossak. Admissible values are between 0.0 and -0.5\nCurrent value is: "
<< mBossak.alpha << std::endl;

static const double epsilon = 1e-12;
KRATOS_ERROR_IF_NOT(std::abs(mNewmark.beta - 0.0)   < epsilon ||
std::abs(mNewmark.beta - 0.167) < epsilon ||
std::abs(mNewmark.beta - 0.25)  < epsilon)
<< "Value not admissible for NewmarkBeta. Admissible values are:\n"
<< "0.0 for central-differencing\n"
<< "0.25 for mean-constant-acceleration\n"
<< "0.167 for linear-acceleration\n"
<< "Current value is: " << mNewmark.beta << std::endl;

return 0;
KRATOS_CATCH( "" );
}

void Clear() override
{
this->mpDofUpdater->Clear();
}

Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"          : "bossak_scheme",
"damp_factor_m" : -0.3,
"newmark_beta"  : 0.25
})");

const Parameters base_default_parameters = ImplicitBaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}

static std::string Name()
{
return "bossak_scheme";
}


std::string Info() const override
{
return "ResidualBasedBossakDisplacementScheme";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info();
}


protected:
typename TSparseSpace::DofUpdaterPointerType mpDofUpdater = TSparseSpace::CreateDofUpdater();

struct BossakAlphaMethod
{
double alpha;

double beta;

double gamma;

double c0, c1, c2, c3, c4, c5;
};

struct NewmarkMethod
{
double beta;

double gamma;
};

struct GeneralVectors
{
std::vector< Vector > v;

std::vector< Vector > a;

std::vector< Vector > ap;
};

BossakAlphaMethod mBossak;

NewmarkMethod mNewmark;

GeneralVectors mVector;



inline void UpdateVelocity(
array_1d<double, 3>& rCurrentVelocity,
const array_1d<double, 3>& rDeltaDisplacement,
const array_1d<double, 3>& rPreviousVelocity,
const array_1d<double, 3>& rPreviousAcceleration
)
{
noalias(rCurrentVelocity) = (mBossak.c1 * rDeltaDisplacement - mBossak.c4 * rPreviousVelocity - mBossak.c5 * rPreviousAcceleration);
}


inline void UpdateAcceleration(
array_1d<double, 3>& rCurrentAcceleration,
const array_1d<double, 3>& rDeltaDisplacement,
const array_1d<double, 3>& rPreviousVelocity,
const array_1d<double, 3>& rPreviousAcceleration
)
{
noalias(rCurrentAcceleration) = (mBossak.c0 * rDeltaDisplacement - mBossak.c2 * rPreviousVelocity -  mBossak.c3 * rPreviousAcceleration);
}


void AddDynamicsToLHS(
LocalSystemMatrixType& LHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
const ProcessInfo& rCurrentProcessInfo
) override
{
if (M.size1() != 0) 
noalias(LHS_Contribution) += M * (1.0 - mBossak.alpha) * mBossak.c0;

if (D.size1() != 0) 
noalias(LHS_Contribution) += D * mBossak.c1;
}


void AddDynamicsToRHS(
Element& rElement,
LocalSystemVectorType& RHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
const ProcessInfo& rCurrentProcessInfo
) override
{
const std::size_t this_thread = OpenMPUtils::ThisThread();

const auto& r_const_elem_ref = rElement;
if (M.size1() != 0) {

r_const_elem_ref.GetSecondDerivativesVector(mVector.a[this_thread], 0);
mVector.a[this_thread] *= (1.00 - mBossak.alpha);

r_const_elem_ref.GetSecondDerivativesVector(mVector.ap[this_thread], 1);
noalias(mVector.a[this_thread]) += mBossak.alpha * mVector.ap[this_thread];

noalias(RHS_Contribution) -= prod(M, mVector.a[this_thread]);
}

if (D.size1() != 0) {
r_const_elem_ref.GetFirstDerivativesVector(mVector.v[this_thread], 0);
noalias(RHS_Contribution) -= prod(D, mVector.v[this_thread]);
}
}


void AddDynamicsToRHS(
Condition& rCondition,
LocalSystemVectorType& RHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
const ProcessInfo& rCurrentProcessInfo
) override
{
const std::size_t this_thread = OpenMPUtils::ThisThread();
const auto& r_const_cond_ref = rCondition;

if (M.size1() != 0) {
r_const_cond_ref.GetSecondDerivativesVector(mVector.a[this_thread], 0);
mVector.a[this_thread] *= (1.00 - mBossak.alpha);

r_const_cond_ref.GetSecondDerivativesVector(mVector.ap[this_thread], 1);
noalias(mVector.a[this_thread]) += mBossak.alpha * mVector.ap[this_thread];

noalias(RHS_Contribution) -= prod(M, mVector.a[this_thread]);
}

if (D.size1() != 0) {
r_const_cond_ref.GetFirstDerivativesVector(mVector.v[this_thread], 0);

noalias(RHS_Contribution) -= prod(D, mVector.v[this_thread]);
}
}

void AssignSettings(const Parameters ThisParameters) override
{
ImplicitBaseType::AssignSettings(ThisParameters);
mBossak.alpha = ThisParameters["damp_factor_m"].GetDouble();
mNewmark.beta = ThisParameters["newmark_beta"].GetDouble();
}


private:

void AuxiliarInitializeBossak()
{
CalculateBossakCoefficients();

const std::size_t num_threads = ParallelUtilities::GetNumThreads();

mVector.v.resize(num_threads);
mVector.a.resize(num_threads);
mVector.ap.resize(num_threads);

KRATOS_DETAIL("MECHANICAL SCHEME: The Bossak Time Integration Scheme ") << "[alpha_m= " << mBossak.alpha << " beta= " << mNewmark.beta << " gamma= " << mNewmark.gamma << "]" <<std::endl;
}

}; 


} 

