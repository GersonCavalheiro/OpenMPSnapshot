
#if !defined(KRATOS_RESIDUAL_BASED_BOSSAK_DISPLACEMENT_ROTATION_SCHEME)
#define KRATOS_RESIDUAL_BASED_BOSSAK_DISPLACEMENT_ROTATION_SCHEME




#include "boost/smart_ptr.hpp"


#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "containers/array_1d.h"
#include "includes/element.h"
#include "utilities/beam_math_utilities.hpp"

#include "solid_mechanics_application_variables.h"

namespace Kratos
{

















template<class TSparseSpace,  class TDenseSpace >
class ResidualBasedBossakDisplacementRotationScheme: public Scheme<TSparseSpace,TDenseSpace>
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
double deltatime;

double c0;
double c1;
double c2;
double c3;
double c4;
double c5;
double c6;
double c7;

double static_dynamic;

};


struct  GeneralMatrices
{
std::vector< Matrix > M;     
std::vector< Matrix > D;     

};

struct GeneralVectors
{
std::vector< Vector > fm;  
std::vector< Vector > fd;  
};



public:





KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedBossakDisplacementRotationScheme );

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

typedef BeamMathUtils<double>                        BeamMathUtilsType;

typedef Quaternion<double>                              QuaternionType;





ResidualBasedBossakDisplacementRotationScheme(double rDynamic = 1, double mAlphaM = 0.0)
:Scheme<TSparseSpace,TDenseSpace>()
{

mAlpha.f= 0;
mAlpha.m= mAlphaM; 

mNewmark.beta= (1.0+mAlpha.f-mAlpha.m)*(1.0+mAlpha.f-mAlpha.m)*0.25;
mNewmark.gamma= 0.5+mAlpha.f-mAlpha.m;


mNewmark.static_dynamic= rDynamic;




int NumThreads = OpenMPUtils::GetNumThreads();

mMatrix.M.resize(NumThreads);
mMatrix.D.resize(NumThreads);

mVector.fm.resize(NumThreads);
mVector.fd.resize(NumThreads);

}


virtual ~ResidualBasedBossakDisplacementRotationScheme
() {}










/
void Update(
ModelPart& r_model_part,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b )
{
KRATOS_TRY


for (ModelPart::NodeIterator i = r_model_part.NodesBegin();
i != r_model_part.NodesEnd(); ++i)
{
if( i->IsNot(SLAVE) && i->IsNot(RIGID) ){


array_1d<double, 3 > & CurrentRotation      = (i)->FastGetSolutionStepValue(ROTATION);
array_1d<double, 3 > & ReferenceRotation    = (i)->FastGetSolutionStepValue(DELTA_ROTATION, 1);

ReferenceRotation    = CurrentRotation;

array_1d<double, 3 > & CurrentDisplacement  = (i)->FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3 > & ReferenceDisplacement= (i)->FastGetSolutionStepValue(STEP_DISPLACEMENT, 1);

ReferenceDisplacement = CurrentDisplacement;

}
}


for (typename DofsArrayType::iterator i_dof = rDofSet.begin(); i_dof != rDofSet.end(); ++i_dof)
{
if (i_dof->IsFree() )
{
i_dof->GetSolutionStepValue() += Dx[i_dof->EquationId()];
}
}


array_1d<double, 3 >  DeltaDisplacement;

for (ModelPart::NodeIterator i = r_model_part.NodesBegin();
i != r_model_part.NodesEnd(); ++i)
{

if( i->IsNot(SLAVE) && i->IsNot(RIGID) ){

array_1d<double, 3 > & CurrentDisplacement      = (i)->FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double, 3 > & ReferenceDisplacement    = (i)->FastGetSolutionStepValue(STEP_DISPLACEMENT, 1); 

noalias(DeltaDisplacement) = CurrentDisplacement-ReferenceDisplacement;

array_1d<double, 3 > & CurrentStepDisplacement  = (i)->FastGetSolutionStepValue(STEP_DISPLACEMENT);

noalias(CurrentStepDisplacement)  = CurrentDisplacement - (i)->FastGetSolutionStepValue(DISPLACEMENT,1);


array_1d<double, 3 > & CurrentVelocity      = (i)->FastGetSolutionStepValue(VELOCITY);
array_1d<double, 3 > & CurrentAcceleration  = (i)->FastGetSolutionStepValue(ACCELERATION);

array_1d<double, 3 > & PreviousVelocity      = (i)->FastGetSolutionStepValue(VELOCITY,1);
array_1d<double, 3 > & PreviousAcceleration  = (i)->FastGetSolutionStepValue(ACCELERATION,1);

UpdateAcceleration ((*i), CurrentAcceleration, DeltaDisplacement, CurrentStepDisplacement, PreviousAcceleration, PreviousVelocity);
UpdateVelocity     ((*i), CurrentVelocity, DeltaDisplacement, CurrentAcceleration, PreviousAcceleration, PreviousVelocity);

}
}


Vector TotalRotationVector = ZeroVector(3);
Vector StepRotationVector  = ZeroVector(3);
Vector DeltaRotationVector = ZeroVector(3);

Vector LinearDeltaRotationVector = ZeroVector(3);

QuaternionType DeltaRotationQuaternion;
QuaternionType CurrentRotationQuaternion;
QuaternionType ReferenceRotationQuaternion;


for (ModelPart::NodeIterator i = r_model_part.NodesBegin();
i != r_model_part.NodesEnd(); ++i)
{
if( i->IsNot(SLAVE) && i->IsNot(RIGID) ){

array_1d<double, 3 > & CurrentRotation     = (i)->FastGetSolutionStepValue(ROTATION);
array_1d<double, 3 > & ReferenceRotation   = (i)->FastGetSolutionStepValue(DELTA_ROTATION, 1);

array_1d<double, 3 > & CurrentStepRotation = (i)->FastGetSolutionStepValue(STEP_ROTATION);

array_1d<double, 3 > & CurrentDeltaRotation= (i)->FastGetSolutionStepValue(DELTA_ROTATION);

CurrentDeltaRotation = CurrentRotation-ReferenceRotation;



for( unsigned int j=0; j<3; j++)
{
DeltaRotationVector[j]  = CurrentDeltaRotation[j]; 

TotalRotationVector[j]  = ReferenceRotation[j];    

StepRotationVector[j]   = CurrentStepRotation[j];  
}


DeltaRotationQuaternion = QuaternionType::FromRotationVector( DeltaRotationVector );

ReferenceRotationQuaternion = QuaternionType::FromRotationVector( StepRotationVector );

CurrentRotationQuaternion = DeltaRotationQuaternion * ReferenceRotationQuaternion;

CurrentRotationQuaternion.ToRotationVector( StepRotationVector );



ReferenceRotationQuaternion = QuaternionType::FromRotationVector( TotalRotationVector );

CurrentRotationQuaternion = DeltaRotationQuaternion * ReferenceRotationQuaternion;

CurrentRotationQuaternion.ToRotationVector( TotalRotationVector );


for( unsigned int j=0; j<3; j++)
{
LinearDeltaRotationVector[j] = StepRotationVector[j] - CurrentStepRotation[j];
}

Matrix RotationMatrix;
(ReferenceRotationQuaternion.conjugate()).ToRotationMatrix(RotationMatrix);
LinearDeltaRotationVector = prod( RotationMatrix, LinearDeltaRotationVector );


array_1d<double, 3 > LinearDeltaRotation;

for( unsigned int j=0; j<3; j++)
{
CurrentRotation[j]      = TotalRotationVector[j];
CurrentStepRotation[j]  = StepRotationVector[j];

LinearDeltaRotation[j]  = LinearDeltaRotationVector[j];
}


array_1d<double, 3 > & CurrentVelocity      = (i)->FastGetSolutionStepValue(ANGULAR_VELOCITY);
array_1d<double, 3 > & CurrentAcceleration  = (i)->FastGetSolutionStepValue(ANGULAR_ACCELERATION);


array_1d<double, 3 > & PreviousAngularAcceleration  = (i)->FastGetSolutionStepValue(ANGULAR_ACCELERATION,1);
array_1d<double, 3 > & PreviousAngularVelocity      = (i)->FastGetSolutionStepValue(ANGULAR_VELOCITY,1);


UpdateAngularAcceleration ((*i), CurrentAcceleration, LinearDeltaRotation, CurrentStepRotation, PreviousAngularAcceleration, PreviousAngularVelocity);
UpdateAngularVelocity     ((*i), CurrentVelocity, LinearDeltaRotation, CurrentAcceleration, PreviousAngularAcceleration, PreviousAngularVelocity);


}

}


SlaveNodesUpdate(r_model_part);


KRATOS_CATCH( "" )
}




/
void InitializeElements(ModelPart& rModelPart)
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



void InitializeConditions(ModelPart& rModelPart)
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


void InitializeSolutionStep(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b)
{
KRATOS_TRY

ProcessInfo& CurrentProcessInfo= r_model_part.GetProcessInfo();


Scheme<TSparseSpace,TDenseSpace>::InitializeSolutionStep(r_model_part,A,Dx,b);


double DeltaTime = CurrentProcessInfo[DELTA_TIME];

if (DeltaTime == 0)
KRATOS_THROW_ERROR(std::logic_error, "detected delta_time = 0 in the Solution Scheme ... check if the time step is created correctly for the current model part", "" )


if( mNewmark.static_dynamic != 0 ){
CurrentProcessInfo[NEWMARK_BETA]  = mNewmark.beta;
CurrentProcessInfo[NEWMARK_GAMMA] = mNewmark.gamma;
CurrentProcessInfo[BOSSAK_ALPHA]  = mAlpha.m;
CurrentProcessInfo[COMPUTE_DYNAMIC_TANGENT] = true;
}

mNewmark.deltatime = DeltaTime;

mNewmark.c0 = ( mNewmark.gamma / ( DeltaTime * mNewmark.beta ) );
mNewmark.c1 = ( 1.0 / ( DeltaTime * DeltaTime * mNewmark.beta ) );

mNewmark.c2 = ( DeltaTime * ( 1.0 - mNewmark.gamma ) );
mNewmark.c3 = ( DeltaTime * mNewmark.gamma );
mNewmark.c4 = ( DeltaTime / mNewmark.beta );
mNewmark.c5 = ( DeltaTime * DeltaTime * ( 0.5 - mNewmark.beta ) / mNewmark.beta );

mNewmark.c6 = ( 1.0 / (mNewmark.beta * DeltaTime) );
mNewmark.c7 = ( 0.5 / (mNewmark.beta) - 1.0 );



KRATOS_CATCH( "" )
}
/
void FinalizeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b)
{
KRATOS_TRY
ElementsArrayType& rElements = rModelPart.Elements();
ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();

int NumThreads = OpenMPUtils::GetNumThreads();
OpenMPUtils::PartitionVector ElementPartition;
OpenMPUtils::DivideInPartitions(rElements.size(), NumThreads, ElementPartition);

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();

ElementsArrayType::iterator ElementsBegin = rElements.begin() + ElementPartition[k];
ElementsArrayType::iterator ElementsEnd = rElements.begin() + ElementPartition[k + 1];

for (ElementsArrayType::iterator itElem = ElementsBegin; itElem != ElementsEnd; itElem++)
{
itElem->FinalizeSolutionStep(CurrentProcessInfo);
}
}

ConditionsArrayType& rConditions = rModelPart.Conditions();

OpenMPUtils::PartitionVector ConditionPartition;
OpenMPUtils::DivideInPartitions(rConditions.size(), NumThreads, ConditionPartition);

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();

ConditionsArrayType::iterator ConditionsBegin = rConditions.begin() + ConditionPartition[k];
ConditionsArrayType::iterator ConditionsEnd = rConditions.begin() + ConditionPartition[k + 1];

for (ConditionsArrayType::iterator itCond = ConditionsBegin; itCond != ConditionsEnd; itCond++)
{
itCond->FinalizeSolutionStep(CurrentProcessInfo);
}
}
KRATOS_CATCH( "" )
}

/

void CalculateSystemContributions(
Element::Pointer rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
ProcessInfo& CurrentProcessInfo)
{
KRATOS_TRY

int thread = OpenMPUtils::ThisThread();


(rCurrentElement) -> CalculateLocalSystem(LHS_Contribution,RHS_Contribution,CurrentProcessInfo);


if(mNewmark.static_dynamic !=0)
{

(rCurrentElement) -> CalculateSecondDerivativesContributions(mMatrix.M[thread],mVector.fm[thread],CurrentProcessInfo);

(rCurrentElement) -> CalculateFirstDerivativesContributions(mMatrix.D[thread],mVector.fd[thread],CurrentProcessInfo);

}


(rCurrentElement) -> EquationIdVector(EquationId,CurrentProcessInfo);


if(mNewmark.static_dynamic !=0)
{

AddDynamicsToLHS(LHS_Contribution, mMatrix.D[thread], mMatrix.M[thread], CurrentProcessInfo);

AddDynamicsToRHS(RHS_Contribution, mVector.fd[thread], mVector.fm[thread], CurrentProcessInfo);

}


KRATOS_CATCH( "" )
}

void Calculate_RHS_Contribution(
Element::Pointer rCurrentElement,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
ProcessInfo& CurrentProcessInfo)
{

KRATOS_TRY

int thread = OpenMPUtils::ThisThread();


(rCurrentElement) -> CalculateRightHandSide(RHS_Contribution,CurrentProcessInfo);

if(mNewmark.static_dynamic !=0)
{

(rCurrentElement) -> CalculateSecondDerivativesRHS(mVector.fm[thread],CurrentProcessInfo);

(rCurrentElement) -> CalculateFirstDerivativesRHS(mVector.fd[thread],CurrentProcessInfo);

}

(rCurrentElement) -> EquationIdVector(EquationId,CurrentProcessInfo);

if(mNewmark.static_dynamic !=0)
{

AddDynamicsToRHS(RHS_Contribution, mVector.fd[thread], mVector.fm[thread], CurrentProcessInfo);

}

KRATOS_CATCH( "" )

}

/
void Condition_CalculateSystemContributions(
Condition::Pointer rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
ProcessInfo& CurrentProcessInfo)
{


KRATOS_TRY

int thread = OpenMPUtils::ThisThread();


(rCurrentCondition) -> CalculateLocalSystem(LHS_Contribution,RHS_Contribution,CurrentProcessInfo);

if(mNewmark.static_dynamic !=0)
{

(rCurrentCondition) -> CalculateSecondDerivativesContributions(mMatrix.M[thread],mVector.fm[thread],CurrentProcessInfo);

(rCurrentCondition) -> CalculateFirstDerivativesContributions(mMatrix.D[thread],mVector.fd[thread],CurrentProcessInfo);

}

(rCurrentCondition) -> EquationIdVector(EquationId,CurrentProcessInfo);

if(mNewmark.static_dynamic !=0)
{

AddDynamicsToLHS(LHS_Contribution, mMatrix.D[thread], mMatrix.M[thread], CurrentProcessInfo);

AddDynamicsToRHS(RHS_Contribution, mVector.fd[thread], mVector.fm[thread], CurrentProcessInfo);

}


KRATOS_CATCH( "" )
}


/
void GetElementalDofList(
Element::Pointer rCurrentElement,
Element::DofsVectorType& ElementalDofList,
ProcessInfo& CurrentProcessInfo)
{
rCurrentElement->GetDofList(ElementalDofList, CurrentProcessInfo);
}

/
void GetConditionDofList(
Condition::Pointer rCurrentCondition,
Element::DofsVectorType& ConditionDofList,
ProcessInfo& CurrentProcessInfo)
{
rCurrentCondition->GetDofList(ConditionDofList, CurrentProcessInfo);
}

/
virtual int Check(ModelPart& r_model_part)
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






/
void AddDynamicsToLHS(
LocalSystemMatrixType& LHS_Contribution,
LocalSystemMatrixType& D,
LocalSystemMatrixType& M,
ProcessInfo& CurrentProcessInfo)
{

if (M.size1() != 0) 
{
noalias(LHS_Contribution) += M * mNewmark.static_dynamic;

}

if (D.size1() != 0) 
{
noalias(LHS_Contribution) += D * mNewmark.static_dynamic;

}

}


void AddDynamicsToRHS(
LocalSystemVectorType& RHS_Contribution,
LocalSystemVectorType& fd,
LocalSystemVectorType& fm,
ProcessInfo& CurrentProcessInfo)
{

if (fm.size() != 0)
{
noalias(RHS_Contribution) -=  mNewmark.static_dynamic * fm;
}

if (fd.size() != 0)
{
noalias(RHS_Contribution) -=  mNewmark.static_dynamic * fd;
}


}


/
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

(mVector.a[thread]) *= mNewmark.static_dynamic ;

rCurrentElement->GetSecondDerivativesVector(mVector.ap[thread], 1);

noalias(mVector.a[thread]) += mVector.ap[thread] * mNewmark.static_dynamic;

noalias(RHS_Contribution)  -= prod(M, mVector.a[thread]);

}

if (D.size1() != 0)
{
rCurrentElement->GetFirstDerivativesVector(mVector.v[thread], 0);

(mVector.v[thread]) *= mNewmark.static_dynamic ;

noalias(RHS_Contribution) -= prod(D, mVector.v[thread]);
}


}



/

















private:




















}; 
}  

#endif 
