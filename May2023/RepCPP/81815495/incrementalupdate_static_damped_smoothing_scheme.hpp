
#if !defined(KRATOS_INCREMENTAL_UPDATE_STATIC_DAMPED_SMOOTHING_SCHEME )
#define  KRATOS_INCREMENTAL_UPDATE_STATIC_DAMPED_SMOOTHING_SCHEME

#include "custom_strategies/schemes/incrementalupdate_static_smoothing_scheme.hpp"
#include "dam_application_variables.h"

namespace Kratos
{

template<class TSparseSpace, class TDenseSpace>

class IncrementalUpdateStaticDampedSmoothingScheme : public IncrementalUpdateStaticSmoothingScheme<TSparseSpace,TDenseSpace>
{

public:

KRATOS_CLASS_POINTER_DEFINITION( IncrementalUpdateStaticDampedSmoothingScheme );

typedef Scheme<TSparseSpace,TDenseSpace>     BaseType;
typedef typename BaseType::TSystemMatrixType TSystemMatrixType;
typedef typename BaseType::TSystemVectorType TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;
typedef typename BaseType::DofsArrayType     DofsArrayType;
typedef ModelPart::NodesContainerType        NodesArrayType;
using Scheme<TSparseSpace,TDenseSpace>::mSchemeIsInitialized;


IncrementalUpdateStaticDampedSmoothingScheme(double rayleigh_m, double rayleigh_k)
: IncrementalUpdateStaticSmoothingScheme<TSparseSpace,TDenseSpace>()

{
mRayleighAlpha = rayleigh_m;
mRayleighBeta = rayleigh_k;

int NumThreads = ParallelUtilities::GetNumThreads();
mDampingMatrix.resize(NumThreads);
mVelocityVector.resize(NumThreads);

}


virtual ~IncrementalUpdateStaticDampedSmoothingScheme() {}


int Check(ModelPart& r_model_part) override
{
KRATOS_TRY

int ierr = Scheme<TSparseSpace,TDenseSpace>::Check(r_model_part);
if(ierr != 0) return ierr;

if ( RAYLEIGH_ALPHA.Key() == 0 )
KRATOS_THROW_ERROR( std::invalid_argument, "RAYLEIGH_ALPHA has Key zero! (check if the application is correctly registered", "" )
if ( RAYLEIGH_BETA.Key() == 0 )
KRATOS_THROW_ERROR( std::invalid_argument, "RAYLEIGH_BETA has Key zero! (check if the application is correctly registered", "" )

if( mRayleighAlpha < 0.0 || mRayleighBeta < 0.0 )
KRATOS_THROW_ERROR( std::invalid_argument,"Some of the rayleigh coefficients has an invalid value ", "" )

return 0;

KRATOS_CATCH( "" )
}



void Initialize(ModelPart& r_model_part) override
{
KRATOS_TRY

mDeltaTime = r_model_part.GetProcessInfo()[DELTA_TIME];

mbeta = 0.25;
mgamma = 0.5;

mNewmark0 = ( 1.0 / (mbeta * mDeltaTime * mDeltaTime) );
mNewmark1 = ( mgamma / (mbeta * mDeltaTime) );
mNewmark2 = ( 1.0 / (mbeta * mDeltaTime) );
mNewmark3 = ( 0.5 / (mbeta) - 1.0 );
mNewmark4 = ( (mgamma / mbeta) - 1.0  );
mNewmark5 = ( mDeltaTime * 0.5 * ( ( mgamma / mbeta ) - 2.0 ) );

r_model_part.GetProcessInfo()[RAYLEIGH_ALPHA] = mRayleighAlpha;
r_model_part.GetProcessInfo()[RAYLEIGH_BETA] = mRayleighBeta;

mSchemeIsInitialized = true;

KRATOS_CATCH("")
}



void Predict(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b
) override
{
KRATOS_TRY;


const unsigned int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector NodePartition;
OpenMPUtils::DivideInPartitions(rModelPart.Nodes().size(), NumThreads, NodePartition);

const int nnodes = static_cast<int>( rModelPart.Nodes().size() );
NodesArrayType::iterator NodeBegin = rModelPart.Nodes().begin();

#pragma omp parallel for firstprivate(NodeBegin)
for(int i = 0;  i< nnodes; i++)
{
array_1d<double, 3 > DeltaDisplacement;

NodesArrayType::iterator itNode = NodeBegin + i;

const array_1d<double, 3 > & PreviousAcceleration = (itNode)->FastGetSolutionStepValue(ACCELERATION, 1);
const array_1d<double, 3 > & PreviousVelocity     = (itNode)->FastGetSolutionStepValue(VELOCITY,     1);
const array_1d<double, 3 > & PreviousDisplacement = (itNode)->FastGetSolutionStepValue(DISPLACEMENT, 1);
array_1d<double, 3 > & CurrentAcceleration        = (itNode)->FastGetSolutionStepValue(ACCELERATION, 0);
array_1d<double, 3 > & CurrentVelocity            = (itNode)->FastGetSolutionStepValue(VELOCITY,     0);
array_1d<double, 3 > & CurrentDisplacement        = (itNode)->FastGetSolutionStepValue(DISPLACEMENT, 0);

noalias(DeltaDisplacement) = CurrentDisplacement - PreviousDisplacement;

this->UpdateVelocity     (CurrentVelocity,     DeltaDisplacement, PreviousVelocity, PreviousAcceleration);

this->UpdateAcceleration (CurrentAcceleration, DeltaDisplacement, PreviousVelocity, PreviousAcceleration);
}

KRATOS_CATCH( "" );
}


void Update(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b ) override
{
KRATOS_TRY;

int NumThreads = ParallelUtilities::GetNumThreads();

block_for_each(rDofSet, [&Dx](auto& dof)
{
if (dof.IsFree())
{
dof.GetSolutionStepValue() += TSparseSpace::GetValue(Dx, dof.EquationId());
}

}
);

OpenMPUtils::PartitionVector NodePartition;
OpenMPUtils::DivideInPartitions(rModelPart.Nodes().size(), NumThreads, NodePartition);

const int nnodes = static_cast<int>(rModelPart.Nodes().size());
NodesArrayType::iterator NodeBegin = rModelPart.Nodes().begin();

#pragma omp parallel for firstprivate(NodeBegin)
for(int i = 0;  i < nnodes; i++)
{
array_1d<double, 3 > DeltaDisplacement;

NodesArrayType::iterator itNode = NodeBegin + i;

noalias(DeltaDisplacement) = (itNode)->FastGetSolutionStepValue(DISPLACEMENT) - (itNode)->FastGetSolutionStepValue(DISPLACEMENT, 1);

array_1d<double, 3 > & CurrentVelocity            = (itNode)->FastGetSolutionStepValue(VELOCITY, 0);
const array_1d<double, 3 > & PreviousVelocity     = (itNode)->FastGetSolutionStepValue(VELOCITY, 1);

array_1d<double, 3 > & CurrentAcceleration        = (itNode)->FastGetSolutionStepValue(ACCELERATION, 0);
const array_1d<double, 3 > & PreviousAcceleration = (itNode)->FastGetSolutionStepValue(ACCELERATION, 1);

this->UpdateVelocity     (CurrentVelocity,     DeltaDisplacement, PreviousVelocity, PreviousAcceleration);

this->UpdateAcceleration (CurrentAcceleration, DeltaDisplacement, PreviousVelocity, PreviousAcceleration);
}

KRATOS_CATCH( "" );
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

rCurrentElement.CalculateLocalSystem(LHS_Contribution, RHS_Contribution, CurrentProcessInfo);

rCurrentElement.CalculateDampingMatrix(mDampingMatrix[thread], CurrentProcessInfo);

this->AddDampingToLHS (LHS_Contribution, mDampingMatrix[thread], CurrentProcessInfo);

this->AddDampingToRHS (rCurrentElement, RHS_Contribution, mDampingMatrix[thread], CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);

KRATOS_CATCH( "" )
}



void Calculate_RHS_Contribution(
Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo)
{
KRATOS_TRY

int thread = OpenMPUtils::ThisThread();

rCurrentElement.CalculateRightHandSide(RHS_Contribution, CurrentProcessInfo);

rCurrentElement.CalculateDampingMatrix(mDampingMatrix[thread], CurrentProcessInfo);

this->AddDampingToRHS (rCurrentElement, RHS_Contribution, mDampingMatrix[thread], CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);

KRATOS_CATCH( "" )
}




void Calculate_LHS_Contribution(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
Element::EquationIdVectorType& EquationId,
const ProcessInfo& CurrentProcessInfo)
{
KRATOS_TRY

int thread = OpenMPUtils::ThisThread();

rCurrentElement.CalculateLeftHandSide(LHS_Contribution, CurrentProcessInfo);

rCurrentElement.CalculateDampingMatrix(mDampingMatrix[thread], CurrentProcessInfo);

this->AddDampingToLHS (LHS_Contribution, mDampingMatrix[thread], CurrentProcessInfo);

rCurrentElement.EquationIdVector(EquationId, CurrentProcessInfo);

KRATOS_CATCH( "" )
}



protected:

double mNewmark0,mNewmark1,mNewmark2,mNewmark3,mNewmark4,mNewmark5;

double mDeltaTime;
double mbeta;
double mgamma;

double mRayleighAlpha;
double mRayleighBeta;

std::vector< Matrix > mDampingMatrix;
std::vector< Vector > mVelocityVector;


void AddDampingToLHS(LocalSystemMatrixType& LHS_Contribution, LocalSystemMatrixType& C, const ProcessInfo& CurrentProcessInfo)
{
if (C.size1() != 0)
{
noalias(LHS_Contribution) += mgamma/(mbeta*mDeltaTime)*C;
}
}


void AddDampingToRHS(Element& rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
LocalSystemMatrixType& C,
const ProcessInfo& CurrentProcessInfo)
{
int thread = OpenMPUtils::ThisThread();

if (C.size1() != 0)
{
rCurrentElement.GetFirstDerivativesVector(mVelocityVector[thread], 0);

noalias(RHS_Contribution) -= prod(C, mVelocityVector[thread]);

}
}


inline void UpdateVelocity(
array_1d<double, 3 > & CurrentVelocity,
const array_1d<double, 3 > & DeltaDisplacement,
const array_1d<double, 3 > & PreviousVelocity,
const array_1d<double, 3 > & PreviousAcceleration
)
{
noalias(CurrentVelocity) =  (mNewmark1 * DeltaDisplacement - mNewmark4 * PreviousVelocity
- mNewmark5 * PreviousAcceleration);
}


inline void UpdateAcceleration(
array_1d<double, 3 > & CurrentAcceleration,
const array_1d<double, 3 > & DeltaDisplacement,
const array_1d<double, 3 > & PreviousVelocity,
const array_1d<double, 3 > & PreviousAcceleration
)
{
noalias(CurrentAcceleration) =  (mNewmark0 * DeltaDisplacement - mNewmark2 * PreviousVelocity
-  mNewmark3 * PreviousAcceleration);
}

}; 
}  

#endif 
