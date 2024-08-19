
#pragma once



#include "solving_strategies/schemes/residual_based_bossak_displacement_scheme.hpp"

namespace Kratos
{


template<class TSparseSpace,  class TDenseSpace >
class ResidualBasedPseudoStaticDisplacementScheme
: public ResidualBasedBossakDisplacementScheme<TSparseSpace,TDenseSpace>
{
public:
KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedPseudoStaticDisplacementScheme );

typedef Scheme<TSparseSpace,TDenseSpace>                                        BaseType;

typedef ResidualBasedPseudoStaticDisplacementScheme<TSparseSpace, TDenseSpace> ClassType;

typedef typename BaseType::TDataType                                           TDataType;

typedef typename BaseType::DofsArrayType                                   DofsArrayType;

typedef typename Element::DofsVectorType                                  DofsVectorType;

typedef typename BaseType::TSystemMatrixType                           TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                           TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType                   LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType                   LocalSystemMatrixType;

typedef ModelPart::ElementsContainerType                               ElementsArrayType;

typedef ModelPart::ConditionsContainerType                           ConditionsArrayType;

typedef typename BaseType::Pointer                                       BaseTypePointer;

typedef ResidualBasedBossakDisplacementScheme<TSparseSpace,TDenseSpace>  DerivedBaseType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();



explicit ResidualBasedPseudoStaticDisplacementScheme()
: DerivedBaseType(0.0),
mpRayleighBeta(&NODAL_MAUX)
{
}


explicit ResidualBasedPseudoStaticDisplacementScheme(Parameters ThisParameters)
: DerivedBaseType()
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ResidualBasedPseudoStaticDisplacementScheme(const Variable<double>& RayleighBetaVariable)
:DerivedBaseType(0.0),
mpRayleighBeta(&RayleighBetaVariable)
{
}


explicit ResidualBasedPseudoStaticDisplacementScheme(ResidualBasedPseudoStaticDisplacementScheme& rOther)
:DerivedBaseType(rOther),
mpRayleighBeta(rOther.mpRayleighBeta)
{
}


BaseTypePointer Clone() override
{
return BaseTypePointer( new ResidualBasedPseudoStaticDisplacementScheme(*this) );
}


~ResidualBasedPseudoStaticDisplacementScheme() override
{
}




typename BaseType::Pointer Create(Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(ThisParameters);
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

DerivedBaseType::mpDofUpdater->UpdateDofs(rDofSet, rDx);

array_1d<double, 3 > delta_displacement;
block_for_each(rModelPart.Nodes(), delta_displacement, [&](Node& rNode, array_1d<double,3>& rDeltaDisplacementTLS){

noalias(rDeltaDisplacementTLS) = rNode.FastGetSolutionStepValue(DISPLACEMENT) - rNode.FastGetSolutionStepValue(DISPLACEMENT, 1);

array_1d<double, 3>& r_current_velocity = rNode.FastGetSolutionStepValue(VELOCITY);
const array_1d<double, 3>& r_previous_velocity = rNode.FastGetSolutionStepValue(VELOCITY, 1);

noalias(r_current_velocity) = (this->mBossak.c1 * rDeltaDisplacementTLS - this->mBossak.c4 * r_previous_velocity);
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

const array_1d<double, 3> zero_array = ZeroVector(3);
array_1d<double, 3 > delta_displacement = zero_array;

const int disppos_x = it_node_begin->HasDofFor(DISPLACEMENT_X) ? static_cast<int>(it_node_begin->GetDofPosition(DISPLACEMENT_X)) : -1;
const int velpos_x = it_node_begin->HasDofFor(VELOCITY_X) ? static_cast<int>(it_node_begin->GetDofPosition(VELOCITY_X)) : -1;
const int disppos_y = it_node_begin->HasDofFor(DISPLACEMENT_Y) ? static_cast<int>(it_node_begin->GetDofPosition(DISPLACEMENT_Y)) : -1;
const int velpos_y = it_node_begin->HasDofFor(VELOCITY_Y) ? static_cast<int>(it_node_begin->GetDofPosition(VELOCITY_Y)) : -1;
const int disppos_z = it_node_begin->HasDofFor(DISPLACEMENT_Z) ? static_cast<int>(it_node_begin->GetDofPosition(DISPLACEMENT_Z)) : -1;
const int velpos_z = it_node_begin->HasDofFor(VELOCITY_Z) ? static_cast<int>(it_node_begin->GetDofPosition(VELOCITY_Z)) : -1;

block_for_each(rModelPart.Nodes(), delta_displacement, [&](Node& rNode, array_1d<double,3>& rDeltaDisplacementTLS){

bool predicted_x = false;
bool predicted_y = false;
bool predicted_z = false;


const array_1d<double, 3>& r_previous_velocity     = rNode.FastGetSolutionStepValue(VELOCITY,     1);
const array_1d<double, 3>& r_previous_displacement = rNode.FastGetSolutionStepValue(DISPLACEMENT, 1);
array_1d<double, 3>& r_current_acceleration        = rNode.FastGetSolutionStepValue(ACCELERATION);
array_1d<double, 3>& r_current_velocity            = rNode.FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3>& r_current_displacement        = rNode.FastGetSolutionStepValue(DISPLACEMENT);

if (velpos_x > -1) {
if (rNode.GetDof(VELOCITY_X, velpos_x).IsFixed()) {
rDeltaDisplacementTLS[0] = (r_current_velocity[0] + this->mBossak.c4 * r_previous_velocity[0])/this->mBossak.c1;
r_current_displacement[0] =  r_previous_displacement[0] + rDeltaDisplacementTLS[0];
predicted_x = true;
}
}
if (disppos_x > -1 && !predicted_x) {
if (!rNode.GetDof(DISPLACEMENT_X, disppos_x).IsFixed() && !predicted_x) {
r_current_displacement[0] = r_previous_displacement[0] + delta_time * r_previous_velocity[0];
}
}

if (velpos_y > -1) {
if (rNode.GetDof(VELOCITY_Y, velpos_y).IsFixed()) {
rDeltaDisplacementTLS[1] = (r_current_velocity[1] + this->mBossak.c4 * r_previous_velocity[1])/this->mBossak.c1;
r_current_displacement[1] =  r_previous_displacement[1] + rDeltaDisplacementTLS[1];
predicted_y = true;
}
}
if (disppos_y > -1 && !predicted_y) {
if (!rNode.GetDof(DISPLACEMENT_Y, disppos_y).IsFixed() && !predicted_y) {
r_current_displacement[1] = r_previous_displacement[1] + delta_time * r_previous_velocity[1];
}
}

if (velpos_z > -1) {
if (rNode.GetDof(VELOCITY_Z, velpos_z).IsFixed()) {
rDeltaDisplacementTLS[2] = (r_current_velocity[2] + this->mBossak.c4 * r_previous_velocity[2])/this->mBossak.c1;
r_current_displacement[2] =  r_previous_displacement[2] + rDeltaDisplacementTLS[2];
predicted_z = true;
}
}
if (disppos_z > -1 && !predicted_z) {
if (!rNode.GetDof(DISPLACEMENT_Z, disppos_z).IsFixed() && !predicted_z) {
r_current_displacement[2] = r_previous_displacement[2] + delta_time * r_previous_velocity[2];
}
}

noalias(r_current_acceleration) = zero_array;
noalias(r_current_velocity) = r_previous_velocity;
});

KRATOS_CATCH( "" );
}


Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                   : "pseudo_static_scheme",
"rayleigh_beta_variable" : "RAYLEIGH_BETA"
})");

const Parameters base_default_parameters = DerivedBaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
return default_parameters;
}


static std::string Name()
{
return "pseudo_static_scheme";
}




std::string Info() const override
{
return "ResidualBasedPseudoStaticDisplacementScheme";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info() << ". Considering the following damping variable " << *mpRayleighBeta;
}


protected:





void AddDynamicsToLHS(
LocalSystemMatrixType& rLHSContribution,
LocalSystemMatrixType& rD,
LocalSystemMatrixType& rM,
const ProcessInfo& rCurrentProcessInfo
) override
{
if (rD.size1() != 0 && TDenseSpace::TwoNorm(rD) > ZeroTolerance) 
noalias(rLHSContribution) += rD * this->mBossak.c1;
else if (rM.size1() != 0) {
const double beta = rCurrentProcessInfo[*mpRayleighBeta];
noalias(rLHSContribution) += rM * beta * this->mBossak.c1;
}
}


void AddDynamicsToRHS(
Element& rElement,
LocalSystemVectorType& rRHSContribution,
LocalSystemMatrixType& rD,
LocalSystemMatrixType& rM,
const ProcessInfo& rCurrentProcessInfo
) override
{
const std::size_t this_thread = OpenMPUtils::ThisThread();
const auto& r_const_elem_ref = rElement;
if (rD.size1() != 0 && TDenseSpace::TwoNorm(rD) > ZeroTolerance) {
r_const_elem_ref.GetFirstDerivativesVector(this->mVector.v[this_thread], 0);
noalias(rRHSContribution) -= prod(rD, this->mVector.v[this_thread]);
} else if (rM.size1() != 0) {
const double beta = rCurrentProcessInfo[*mpRayleighBeta];
r_const_elem_ref.GetFirstDerivativesVector(this->mVector.v[this_thread], 0);
noalias(rRHSContribution) -= beta * prod(rM, this->mVector.v[this_thread]);
}
}


void AddDynamicsToRHS(
Condition& rCondition,
LocalSystemVectorType& rRHSContribution,
LocalSystemMatrixType& rD,
LocalSystemMatrixType& rM,
const ProcessInfo& rCurrentProcessInfo
) override
{
const std::size_t this_thread = OpenMPUtils::ThisThread();
const auto& r_const_cond_ref = rCondition;
if (rD.size1() != 0 && TDenseSpace::TwoNorm(rD) > ZeroTolerance) {
r_const_cond_ref.GetFirstDerivativesVector(this->mVector.v[this_thread], 0);
noalias(rRHSContribution) -= prod(rD, this->mVector.v[this_thread]);
} else if (rM.size1() != 0) {
const double beta = rCurrentProcessInfo[*mpRayleighBeta];
r_const_cond_ref.GetFirstDerivativesVector(this->mVector.v[this_thread], 0);
noalias(rRHSContribution) -= beta * prod(rM, this->mVector.v[this_thread]);
}
}


void AssignSettings(const Parameters ThisParameters) override
{
DerivedBaseType::AssignSettings(ThisParameters);
mpRayleighBeta = &KratosComponents<Variable<double>>::Get(ThisParameters["rayleigh_beta_variable"].GetString());
}



private:

const Variable<double>* mpRayleighBeta = nullptr; 





}; 
}  