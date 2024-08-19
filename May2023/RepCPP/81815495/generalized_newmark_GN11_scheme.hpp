
#if !defined(KRATOS_GENERALIZED_NEWMARK_GN11_SCHEME )
#define  KRATOS_GENERALIZED_NEWMARK_GN11_SCHEME

#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/convection_diffusion_settings.h"

#include "fluid_transport_application_variables.h"

namespace Kratos
{

template<class TSparseSpace, class TDenseSpace>

class GeneralizedNewmarkGN11Scheme : public Scheme<TSparseSpace,TDenseSpace>
{

public:

KRATOS_CLASS_POINTER_DEFINITION( GeneralizedNewmarkGN11Scheme );

typedef Scheme<TSparseSpace,TDenseSpace>                      BaseType;
typedef typename BaseType::DofsArrayType                 DofsArrayType;
typedef typename BaseType::TSystemMatrixType         TSystemMatrixType;
typedef typename BaseType::TSystemVectorType         TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;


GeneralizedNewmarkGN11Scheme(double theta) : Scheme<TSparseSpace,TDenseSpace>()
{
mTheta = theta;

int NumThreads = ParallelUtilities::GetNumThreads();
mMMatrix.resize(NumThreads);
mMVector.resize(NumThreads);
}


~GeneralizedNewmarkGN11Scheme() override {}


int Check(ModelPart& r_model_part) override
{
KRATOS_TRY

ConvectionDiffusionSettings::Pointer my_settings = r_model_part.GetProcessInfo().GetValue(CONVECTION_DIFFUSION_SETTINGS);
const Variable<double>& rUnknownVar = my_settings->GetUnknownVariable();


if(DELTA_TIME.Key() == 0)
KRATOS_THROW_ERROR( std::invalid_argument,"DELTA_TIME Key is 0. Check if all applications were correctly registered.", "")
if(rUnknownVar.Key() == 0)
KRATOS_THROW_ERROR( std::invalid_argument, "UnknownVar Key is 0. Check if all applications were correctly registered.", "" )

if(TEMPERATURE.Key() == 0)
KRATOS_THROW_ERROR( std::invalid_argument, "TEMPERATURE Key is 0. Check if all applications were correctly registered.", "" )

for(ModelPart::NodesContainerType::iterator it=r_model_part.NodesBegin(); it!=r_model_part.NodesEnd(); it++)
{
if(it->SolutionStepsDataHas(rUnknownVar) == false)
KRATOS_THROW_ERROR( std::logic_error, "UnknownVar variable is not allocated for node ", it->Id() )
if(it->SolutionStepsDataHas(TEMPERATURE) == false)
KRATOS_THROW_ERROR( std::logic_error, "TEMPERATURE variable is not allocated for node ", it->Id() )

if(it->HasDofFor(rUnknownVar) == false)
KRATOS_THROW_ERROR( std::invalid_argument,"missing UnknownVar dof on node ",it->Id() )
}

if (r_model_part.GetBufferSize() < 2)
KRATOS_THROW_ERROR( std::logic_error, "insufficient buffer size. Buffer size should be greater than 2. Current size is", r_model_part.GetBufferSize() )

if(mTheta <= 0.0)
KRATOS_THROW_ERROR( std::invalid_argument,"Some of the scheme variables: theta has an invalid value ", "" )

return 0;

KRATOS_CATCH( "" )
}


void Initialize(ModelPart& r_model_part) override
{
KRATOS_TRY

mDeltaTime = r_model_part.GetProcessInfo()[DELTA_TIME];
r_model_part.GetProcessInfo().SetValue(THETA,mTheta);

BaseType::mSchemeIsInitialized = true;

KRATOS_CATCH("")
}


void InitializeSolutionStep(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

mDeltaTime = r_model_part.GetProcessInfo()[DELTA_TIME];
r_model_part.GetProcessInfo().SetValue(THETA,mTheta);

const ProcessInfo& CurrentProcessInfo = r_model_part.GetProcessInfo();

int NElems = static_cast<int>(r_model_part.Elements().size());
ModelPart::ElementsContainerType::iterator el_begin = r_model_part.ElementsBegin();

#pragma omp parallel for
for(int i = 0; i < NElems; i++)
{
ModelPart::ElementsContainerType::iterator itElem = el_begin + i;
itElem -> InitializeSolutionStep(CurrentProcessInfo);
}

int NCons = static_cast<int>(r_model_part.Conditions().size());
ModelPart::ConditionsContainerType::iterator con_begin = r_model_part.ConditionsBegin();

#pragma omp parallel for
for(int i = 0; i < NCons; i++)
{
ModelPart::ConditionsContainerType::iterator itCond = con_begin + i;
itCond -> InitializeSolutionStep(CurrentProcessInfo);
}

KRATOS_CATCH("")
}


void FinalizeSolutionStep(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY


const int NNodes = static_cast<int>(r_model_part.Nodes().size());
ModelPart::NodesContainerType::iterator node_begin = r_model_part.NodesBegin();


#pragma omp parallel for
for(int i = 0; i < NNodes; i++)
{
ModelPart::NodesContainerType::iterator itNode = node_begin + i;

itNode->FastGetSolutionStepValue(NODAL_AREA) = 0.0;

array_1d<double,3>& rNodalPhiGradient = itNode->FastGetSolutionStepValue(NODAL_PHI_GRADIENT);
noalias(rNodalPhiGradient) = ZeroVector(3);
}

BaseType::FinalizeSolutionStep(r_model_part,A,Dx,b);


#pragma omp parallel for
for(int i = 0; i < NNodes; i++)
{
ModelPart::NodesContainerType::iterator itNode = node_begin + i;

const double& NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
if (NodalArea>1.0e-20)
{
const double InvNodalArea = 1.0/NodalArea;
array_1d<double,3>& rNodalPhiGradient = itNode->FastGetSolutionStepValue(NODAL_PHI_GRADIENT);
for(unsigned int i = 0; i<3; i++)
{
rNodalPhiGradient[i] *= InvNodalArea;
}
}
}







KRATOS_CATCH("")
}



void Predict(
ModelPart& r_model_part,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
this->UpdateVariablesDerivatives(r_model_part);
}


void InitializeNonLinIteration(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY



const int NNodes = static_cast<int>(r_model_part.Nodes().size());
ModelPart::NodesContainerType::iterator node_begin = r_model_part.NodesBegin();


#pragma omp parallel for
for(int i = 0; i < NNodes; i++)
{
ModelPart::NodesContainerType::iterator itNode = node_begin + i;

itNode->FastGetSolutionStepValue(NODAL_AREA) = 0.0;

array_1d<double,3>& rNodalPhiGradient = itNode->FastGetSolutionStepValue(NODAL_PHI_GRADIENT);
noalias(rNodalPhiGradient) = ZeroVector(3);
}



const ProcessInfo& CurrentProcessInfo = r_model_part.GetProcessInfo();

int NElems = static_cast<int>(r_model_part.Elements().size());
ModelPart::ElementsContainerType::iterator el_begin = r_model_part.ElementsBegin();

#pragma omp parallel for
for(int i = 0; i < NElems; i++)
{
ModelPart::ElementsContainerType::iterator itElem = el_begin + i;
itElem -> InitializeNonLinearIteration(CurrentProcessInfo);
}

int NCons = static_cast<int>(r_model_part.Conditions().size());
ModelPart::ConditionsContainerType::iterator con_begin = r_model_part.ConditionsBegin();

#pragma omp parallel for
for(int i = 0; i < NCons; i++)
{
ModelPart::ConditionsContainerType::iterator itCond = con_begin + i;
itCond -> InitializeNonLinearIteration(CurrentProcessInfo);
}



#pragma omp parallel for
for(int i = 0; i < NNodes; i++)
{
ModelPart::NodesContainerType::iterator itNode = node_begin + i;

const double& NodalArea = itNode->FastGetSolutionStepValue(NODAL_AREA);
if (NodalArea>1.0e-20)
{
const double InvNodalArea = 1.0/NodalArea;
array_1d<double,3>& rNodalPhiGradient = itNode->FastGetSolutionStepValue(NODAL_PHI_GRADIENT);
for(unsigned int i = 0; i<3; i++)
{
rNodalPhiGradient[i] *= InvNodalArea;
}
}
}


KRATOS_CATCH("")
}


void FinalizeNonLinIteration(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

const ProcessInfo& CurrentProcessInfo = r_model_part.GetProcessInfo();

int NElems = static_cast<int>(r_model_part.Elements().size());
ModelPart::ElementsContainerType::iterator el_begin = r_model_part.ElementsBegin();

#pragma omp parallel for
for(int i = 0; i < NElems; i++)
{
ModelPart::ElementsContainerType::iterator itElem = el_begin + i;
itElem -> FinalizeNonLinearIteration(CurrentProcessInfo);
}

int NCons = static_cast<int>(r_model_part.Conditions().size());
ModelPart::ConditionsContainerType::iterator con_begin = r_model_part.ConditionsBegin();

#pragma omp parallel for
for(int i = 0; i < NCons; i++)
{
ModelPart::ConditionsContainerType::iterator itCond = con_begin + i;
itCond -> FinalizeNonLinearIteration(CurrentProcessInfo);
}

KRATOS_CATCH("")
}




void CalculateSystemContributions(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

int thread = OpenMPUtils::ThisThread();

rCurrentElement.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,CurrentProcessInfo);

rCurrentElement.CalculateFirstDerivativesContributions(mMMatrix[thread], mMVector[thread],CurrentProcessInfo);

if (mMMatrix[thread].size1() != 0)
noalias(LHS_Contribution) += 1.0 / (mTheta*mDeltaTime) * mMMatrix[thread];

if (mMVector[thread].size() != 0)
noalias(RHS_Contribution) += mMVector[thread];

rCurrentElement.EquationIdVector(EquationId,CurrentProcessInfo);


KRATOS_CATCH( "" )
}



void CalculateSystemContributions(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

rCurrentCondition.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,CurrentProcessInfo);

rCurrentCondition.EquationIdVector(EquationId,CurrentProcessInfo);

KRATOS_CATCH( "" )
}



void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

int thread = OpenMPUtils::ThisThread();

rCurrentElement.CalculateRightHandSide(RHS_Contribution,CurrentProcessInfo);

rCurrentElement.CalculateFirstDerivativesRHS(mMVector[thread],CurrentProcessInfo);


if (mMVector[thread].size() != 0)
noalias(RHS_Contribution) += mMVector[thread];

rCurrentElement.EquationIdVector(EquationId,CurrentProcessInfo);

KRATOS_CATCH( "" )
}



void CalculateRHSContribution(
Condition& rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

rCurrentCondition.CalculateRightHandSide(RHS_Contribution, CurrentProcessInfo);

rCurrentCondition.EquationIdVector(EquationId, CurrentProcessInfo);

KRATOS_CATCH( "" )
}



void CalculateLHSContribution(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

int thread = OpenMPUtils::ThisThread();

rCurrentElement.CalculateLeftHandSide(LHS_Contribution,CurrentProcessInfo);

rCurrentElement.CalculateFirstDerivativesLHS(mMMatrix[thread], CurrentProcessInfo);

if (mMMatrix[thread].size1() != 0)
noalias(LHS_Contribution) += 1.0 / (mTheta*mDeltaTime) * mMMatrix[thread];

rCurrentElement.EquationIdVector(EquationId,CurrentProcessInfo);

KRATOS_CATCH( "" )
}



void CalculateLHSContribution(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo) override
{
KRATOS_TRY

rCurrentCondition.CalculateLeftHandSide(LHS_Contribution, CurrentProcessInfo);

rCurrentCondition.EquationIdVector(EquationId, CurrentProcessInfo);

KRATOS_CATCH( "" )
}


void Update(
ModelPart& r_model_part,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector DofSetPartition;
OpenMPUtils::DivideInPartitions(rDofSet.size(), NumThreads, DofSetPartition);

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();

typename DofsArrayType::iterator DofsBegin = rDofSet.begin() + DofSetPartition[k];
typename DofsArrayType::iterator DofsEnd = rDofSet.begin() + DofSetPartition[k+1];

for (typename DofsArrayType::iterator itDof = DofsBegin; itDof != DofsEnd; ++itDof)
{
if (itDof->IsFree())
itDof->GetSolutionStepValue() += TSparseSpace::GetValue(Dx, itDof->EquationId());
}
}

this->UpdateVariablesDerivatives(r_model_part);

KRATOS_CATCH( "" )
}


protected:


double mTheta;
double mDeltaTime;

std::vector< Matrix > mMMatrix;
std::vector< Vector > mMVector;


inline void UpdateVariablesDerivatives(ModelPart& r_model_part)
{
KRATOS_TRY

ConvectionDiffusionSettings::Pointer my_settings = r_model_part.GetProcessInfo().GetValue(CONVECTION_DIFFUSION_SETTINGS);
const Variable<double>& rUnknownVar = my_settings->GetUnknownVariable();

const int NNodes = static_cast<int>(r_model_part.Nodes().size());
ModelPart::NodesContainerType::iterator node_begin = r_model_part.NodesBegin();

#pragma omp parallel for
for(int i = 0; i < NNodes; i++)
{
ModelPart::NodesContainerType::iterator itNode = node_begin + i;

double& CurrentPhi = itNode->FastGetSolutionStepValue(TEMPERATURE);
const double& PreviousPhi = itNode->FastGetSolutionStepValue(TEMPERATURE, 1);
const double& CurrentPhiTheta = itNode->FastGetSolutionStepValue(rUnknownVar);

CurrentPhi = 1.0 / mTheta * CurrentPhiTheta + (1.0 - 1.0 / mTheta) * PreviousPhi;
}

KRATOS_CATCH( "" )
}

}; 
}  

#endif 
