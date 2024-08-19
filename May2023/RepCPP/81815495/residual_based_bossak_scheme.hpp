
#if !defined(KRATOS_RESIDUAL_BASED_BOSSAK_SCHEME )
#define  KRATOS_RESIDUAL_BASED_BOSSAK_SCHEME


#include "boost/smart_ptr.hpp"

#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "containers/array_1d.h"
#include "includes/element.h"
#include "pfem_solid_mechanics_application_variables.h"

namespace Kratos
{


template<class TSparseSpace,  class TDenseSpace >
class ResidualBasedBossakScheme: public Scheme<TSparseSpace,TDenseSpace>
{
protected:

struct GeneralAlphaMethod
{

double f;  
double m;  

};

struct NewmarkMethod
{

double beta;
double gamma;

double c0;
double c1;
double c2;
double c3;
double c4;
double c5;
double c6;

double static_dynamic;

};


struct  GeneralMatrices
{

std::vector< Matrix > M;     
std::vector< Matrix > D;     

};

struct GeneralVectors
{

std::vector< Vector > v;    
std::vector< Vector > a;    
std::vector< Vector > ap;   

};



public:


KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedBossakScheme );

typedef Scheme<TSparseSpace,TDenseSpace>                      BaseType;

typedef typename BaseType::TDataType                         TDataType;

typedef typename BaseType::DofsArrayType                 DofsArrayType;

typedef typename Element::DofsVectorType                DofsVectorType;

typedef typename BaseType::TSystemMatrixType         TSystemMatrixType;

typedef typename BaseType::TSystemVectorType         TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef ModelPart::ElementsContainerType             ElementsArrayType;

typedef ModelPart::ConditionsContainerType         ConditionsArrayType;

typedef typename BaseType::Pointer                     BaseTypePointer;


ResidualBasedBossakScheme(double rAlpham=0,double rDynamic=1)
:Scheme<TSparseSpace,TDenseSpace>()
{
mAlpha.f= 0;
mAlpha.m= rAlpham;

mNewmark.beta= (1.0+mAlpha.f-mAlpha.m)*(1.0+mAlpha.f-mAlpha.m)*0.25;
mNewmark.gamma= 0.5+mAlpha.f-mAlpha.m;

mNewmark.static_dynamic= rDynamic;



int NumThreads = OpenMPUtils::GetNumThreads();

mMatrix.M.resize(NumThreads);
mMatrix.D.resize(NumThreads);

mVector.v.resize(NumThreads);
mVector.a.resize(NumThreads);
mVector.ap.resize(NumThreads);


}


ResidualBasedBossakScheme(ResidualBasedBossakScheme& rOther)
:BaseType(rOther)
,mAlpha(rOther.mAlpha)
,mNewmark(rOther.mNewmark)
,mMatrix(rOther.mMatrix)
,mVector(rOther.mVector)
{
}


virtual ~ResidualBasedBossakScheme
() {}



BaseTypePointer Clone() override
{
return BaseTypePointer( new ResidualBasedBossakScheme(*this) );
}



/
void Update(
ModelPart& r_model_part,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b ) override
{
KRATOS_TRY

for (typename DofsArrayType::iterator i_dof = rDofSet.begin(); i_dof != rDofSet.end(); ++i_dof)
{
if (i_dof->IsFree() )
{
i_dof->GetSolutionStepValue() += Dx[i_dof->EquationId()];
}
}

array_1d<double, 3 > DeltaDisplacement;
for (ModelPart::NodeIterator i = r_model_part.NodesBegin();
i != r_model_part.NodesEnd(); ++i)
{

noalias(DeltaDisplacement) = (i)->FastGetSolutionStepValue(DISPLACEMENT) - (i)->FastGetSolutionStepValue(DISPLACEMENT, 1);


array_1d<double, 3 > & CurrentVelocity      = (i)->FastGetSolutionStepValue(VELOCITY, 0);
array_1d<double, 3 > & PreviousVelocity     = (i)->FastGetSolutionStepValue(VELOCITY, 1);

array_1d<double, 3 > & CurrentAcceleration  = (i)->FastGetSolutionStepValue(ACCELERATION, 0);
array_1d<double, 3 > & PreviousAcceleration = (i)->FastGetSolutionStepValue(ACCELERATION, 1);

UpdateVelocity     (CurrentVelocity, DeltaDisplacement, PreviousVelocity, PreviousAcceleration);

UpdateAcceleration (CurrentAcceleration, DeltaDisplacement, PreviousVelocity, PreviousAcceleration);

}

KRATOS_CATCH( "" )
}


/
void InitializeElements(ModelPart& rModelPart) override
{
KRATOS_TRY

int NumThreads = OpenMPUtils::GetNumThreads();
OpenMPUtils::PartitionVector ElementPartition;
OpenMPUtils::DivideInPartitions(rModelPart.Elements().size(), NumThreads, ElementPartition);

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();
ElementsArrayType::iterator ElemBegin = rModelPart.Elements().begin() + ElementPartition[k];
ElementsArrayType::iterator ElemEnd = rModelPart.Elements().begin() + ElementPartition[k + 1];

for (ElementsArrayType::iterator itElem = ElemBegin; itElem != ElemEnd; itElem++)
{
itElem->Initialize(); 
}

}

this->mElementsAreInitialized = true;


KRATOS_CATCH( "" )
}

/
void InitializeConditions(ModelPart& rModelPart) override
{
KRATOS_TRY

if(this->mElementsAreInitialized==false)
KRATOS_THROW_ERROR( std::logic_error, "Before initilizing Conditions, initialize Elements FIRST", "" )

int NumThreads = OpenMPUtils::GetNumThreads();
OpenMPUtils::PartitionVector ConditionPartition;
OpenMPUtils::DivideInPartitions(rModelPart.Conditions().size(), NumThreads, ConditionPartition);

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();
ConditionsArrayType::iterator CondBegin = rModelPart.Conditions().begin() + ConditionPartition[k];
ConditionsArrayType::iterator CondEnd = rModelPart.Conditions().begin() + ConditionPartition[k + 1];

for (ConditionsArrayType::iterator itCond = CondBegin; itCond != CondEnd; itCond++)
{
itCond->Initialize(); 
}

}

this->mConditionsAreInitialized = true;
KRATOS_CATCH( "" )
}

/
void InitializeSolutionStep(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

ProcessInfo CurrentProcessInfo= r_model_part.GetProcessInfo();


Scheme<TSparseSpace,TDenseSpace>::InitializeSolutionStep(r_model_part,A,Dx,b);


double DeltaTime = CurrentProcessInfo[DELTA_TIME];


if (DeltaTime == 0){
std::cout<<" WARNING: detected delta_time = 0 in the Solution Scheme "<<std::endl;
std::cout<<" DELTA_TIME set to 1 considering a Quasistatic step with one step only "<<std::endl;
std::cout<<" PLEASE : check if the time step is created correctly for the current model part "<<std::endl;

CurrentProcessInfo[DELTA_TIME] = 1;
DeltaTime = CurrentProcessInfo[DELTA_TIME];
}


mNewmark.c0 = ( 1.0 / (mNewmark.beta * DeltaTime * DeltaTime) );
mNewmark.c1 = ( mNewmark.gamma / (mNewmark.beta * DeltaTime) );
mNewmark.c2 = ( 1.0 / (mNewmark.beta * DeltaTime) );
mNewmark.c3 = ( 0.5 / (mNewmark.beta) - 1.0 );
mNewmark.c4 = ( (mNewmark.gamma / mNewmark.beta) - 1.0  );
mNewmark.c5 = ( DeltaTime * 0.5 * ( ( mNewmark.gamma / mNewmark.beta ) - 2 ) );



KRATOS_CATCH( "" )
}

/
int Check(ModelPart& r_model_part) override
{
KRATOS_TRY

int err = Scheme<TSparseSpace, TDenseSpace>::Check(r_model_part);
if(err!=0) return err;

if(DISPLACEMENT.Key() == 0)
KRATOS_THROW_ERROR( std::invalid_argument,"DISPLACEMENT has Key zero! (check if the application is correctly registered", "" )
if(VELOCITY.Key() == 0)
KRATOS_THROW_ERROR( std::invalid_argument,"VELOCITY has Key zero! (check if the application is correctly registered", "" )
if(ACCELERATION.Key() == 0)
KRATOS_THROW_ERROR( std::invalid_argument,"ACCELERATION has Key zero! (check if the application is correctly registered", "" )

for(ModelPart::NodesContainerType::iterator it=r_model_part.NodesBegin();
it!=r_model_part.NodesEnd(); it++)
{
if (it->SolutionStepsDataHas(DISPLACEMENT) == false)
KRATOS_THROW_ERROR( std::logic_error, "DISPLACEMENT variable is not allocated for node ", it->Id() )
if (it->SolutionStepsDataHas(VELOCITY) == false)
KRATOS_THROW_ERROR( std::logic_error, "DISPLACEMENT variable is not allocated for node ", it->Id() )
if (it->SolutionStepsDataHas(ACCELERATION) == false)
KRATOS_THROW_ERROR( std::logic_error, "DISPLACEMENT variable is not allocated for node ", it->Id() )
}

for(ModelPart::NodesContainerType::iterator it=r_model_part.NodesBegin();
it!=r_model_part.NodesEnd(); it++)
{
if(it->HasDofFor(DISPLACEMENT_X) == false)
KRATOS_THROW_ERROR( std::invalid_argument,"missing DISPLACEMENT_X dof on node ",it->Id() )
if(it->HasDofFor(DISPLACEMENT_Y) == false)
KRATOS_THROW_ERROR( std::invalid_argument,"missing DISPLACEMENT_Y dof on node ",it->Id() )
if(it->HasDofFor(DISPLACEMENT_Z) == false)
KRATOS_THROW_ERROR( std::invalid_argument,"missing DISPLACEMENT_Z dof on node ",it->Id() )
}


if(mAlpha.m > 0.0 || mAlpha.m < -0.3)
KRATOS_THROW_ERROR( std::logic_error,"Value not admissible for AlphaBossak. Admissible values should be between 0.0 and -0.3. Current value is ", mAlpha.m )

if (r_model_part.GetBufferSize() < 2)
KRATOS_THROW_ERROR( std::logic_error, "insufficient buffer size. Buffer size should be greater than 2. Current size is", r_model_part.GetBufferSize() )


return 0;
KRATOS_CATCH( "" )
}


protected:

GeneralAlphaMethod  mAlpha;
NewmarkMethod       mNewmark;

GeneralMatrices     mMatrix;
GeneralVectors      mVector;



inline void UpdateVelocity(array_1d<double, 3 > & CurrentVelocity,
const array_1d<double, 3 > & DeltaDisplacement,
const array_1d<double, 3 > & PreviousVelocity,
const array_1d<double, 3 > & PreviousAcceleration)
{

noalias(CurrentVelocity) =  (mNewmark.c1 * DeltaDisplacement - mNewmark.c4 * PreviousVelocity
- mNewmark.c5 * PreviousAcceleration) * mNewmark.static_dynamic;

}



inline void UpdateAcceleration(array_1d<double, 3 > & CurrentAcceleration,
const array_1d<double, 3 > & DeltaDisplacement,
const array_1d<double, 3 > & PreviousVelocity,
const array_1d<double, 3 > & PreviousAcceleration)
{

noalias(CurrentAcceleration) =  (mNewmark.c0 * DeltaDisplacement - mNewmark.c2 * PreviousVelocity
-  mNewmark.c3 * PreviousAcceleration) * mNewmark.static_dynamic;


}



void AddDynamicsToLHS(
LocalSystemMatrixType& LHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
ProcessInfo& CurrentProcessInfo)
{

if (M.size1() != 0) 
{
noalias(LHS_Contribution) += M * (1-mAlpha.m) * mNewmark.c0 * mNewmark.static_dynamic;

}

if (D.size1() != 0) 
{
noalias(LHS_Contribution) += D * (1-mAlpha.f) * mNewmark.c1 * mNewmark.static_dynamic;

}

}


void AddDynamicsToRHS(
Element::Pointer rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
ProcessInfo& CurrentProcessInfo)
{
int thread = OpenMPUtils::ThisThread();

if (M.size1() != 0)
{
rCurrentElement->GetSecondDerivativesVector(mVector.a[thread], 0);

(mVector.a[thread]) *= (1.00 - mAlpha.m) * mNewmark.static_dynamic ;

rCurrentElement->GetSecondDerivativesVector(mVector.ap[thread], 1);

noalias(mVector.a[thread]) += mAlpha.m * mVector.ap[thread] * mNewmark.static_dynamic;

noalias(RHS_Contribution)  -= prod(M, mVector.a[thread]);

}

if (D.size1() != 0)
{
rCurrentElement->GetFirstDerivativesVector(mVector.v[thread], 0);

(mVector.v[thread]) *= mNewmark.static_dynamic ;

noalias(RHS_Contribution) -= prod(D, mVector.v[thread]);
}


}



void AddDynamicsToRHS(
Condition::Pointer rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
ProcessInfo& CurrentProcessInfo)
{
int thread = OpenMPUtils::ThisThread();

if (M.size1() != 0)
{
rCurrentCondition->GetSecondDerivativesVector(mVector.a[thread], 0);

(mVector.a[thread]) *= (1.00 - mAlpha.m) * mNewmark.static_dynamic;

rCurrentCondition->GetSecondDerivativesVector(mVector.ap[thread], 1);

noalias(mVector.a[thread]) += mAlpha.m * mVector.ap[thread] * mNewmark.static_dynamic;

noalias(RHS_Contribution)  -= prod(M, mVector.a[thread]);
}

if (D.size1() != 0)
{
rCurrentCondition->GetFirstDerivativesVector(mVector.v[thread], 0);

(mVector.v[thread]) *= mNewmark.static_dynamic ;

noalias(RHS_Contribution) -= prod(D, mVector.v [thread]);
}

}

}; 


}  

#endif 


