
#if !defined(KRATOS_EXPLICIT_HAMILTON_SCHEME_H_INCLUDED)
#define  KRATOS_EXPLICIT_HAMILTON_SCHEME_H_INCLUDED


#ifdef _OPENMP
#include <omp.h>
#endif


#include "boost/smart_ptr.hpp"



#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "utilities/quaternion.h"

#include "solid_mechanics_application_variables.h"

namespace Kratos
{


























template<class TSparseSpace,
class TDenseSpace 
>
class ExplicitHamiltonScheme : public Scheme<TSparseSpace,TDenseSpace>
{

public:



KRATOS_CLASS_POINTER_DEFINITION( ExplicitHamiltonScheme );

typedef Scheme<TSparseSpace,TDenseSpace> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef ModelPart::ElementsContainerType ElementsArrayType;

typedef ModelPart::ConditionsContainerType ConditionsArrayType;

typedef ModelPart::NodesContainerType NodesArrayType;

typedef Quaternion<double> QuaternionType;






ExplicitHamiltonScheme(const double  rMaximumDeltaTime,
const double  rDeltaTimeFraction,
const double  rDeltaTimePredictionLevel,
const bool    rRayleighDamping)
: Scheme<TSparseSpace,TDenseSpace>()
{

mDeltaTime.PredictionLevel  = rDeltaTimePredictionLevel;

mDeltaTime.Maximum          = rMaximumDeltaTime;
mDeltaTime.Fraction         = rDeltaTimeFraction;

mRayleighDamping            = rRayleighDamping;

int NumThreads = ParallelUtilities::GetNumThreads();


mMatrix.D.resize(NumThreads);

mVector.v.resize(NumThreads);


}



virtual ~ExplicitHamiltonScheme() {}






virtual int Check(ModelPart& r_model_part)
{
KRATOS_TRY

BaseType::Check(r_model_part);

if(r_model_part.GetBufferSize() < 2)
{
KRATOS_THROW_ERROR(std::logic_error, "Insufficient buffer size for Central Difference Scheme. It has to be 2", "")
}

KRATOS_CATCH("")

return 0;
}


virtual void Initialize(ModelPart& r_model_part)
{
KRATOS_TRY

if(mDeltaTime.PredictionLevel>0)
{
this->CalculateDeltaTime(r_model_part);
}

this->InitializeExplicitScheme(r_model_part);

ProcessInfo& rCurrentProcessInfo = r_model_part.GetProcessInfo();
rCurrentProcessInfo[COMPUTE_DYNAMIC_TANGENT] = true;

mTime.Previous       = 0.0;
mTime.PreviousMiddle = 0.0;

mSchemeIsInitialized = true;

KRATOS_CATCH("")
}

/
/

void CalculateSystemContributions(
Element::Pointer rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY

(rCurrentElement) -> CalculateSecondDerivativesContributions(LHS_Contribution,RHS_Contribution,rCurrentProcessInfo);

(rCurrentElement) -> AddExplicitContribution(LHS_Contribution, TANGENT_MATRIX, TANGENT_LYAPUNOV, rCurrentProcessInfo);

(rCurrentElement) -> AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, RESIDUAL_LYAPUNOV, rCurrentProcessInfo);







KRATOS_CATCH( "" )

}

/
void Condition_CalculateSystemContributions(
Condition::Pointer rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationId,
ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY










KRATOS_CATCH( "" )
}


/
virtual void Condition_Calculate_RHS_Contribution(Condition::Pointer rCurrentCondition,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& EquationId,
ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY

(rCurrentCondition) -> CalculateRightHandSide(RHS_Contribution, rCurrentProcessInfo);


(rCurrentCondition) -> AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, FORCE_RESIDUAL, rCurrentProcessInfo);

(rCurrentCondition) -> AddExplicitContribution(RHS_Contribution, RESIDUAL_VECTOR, MOMENT_RESIDUAL, rCurrentProcessInfo);


KRATOS_CATCH( "" )
}
























protected:




struct GeneralMatrices
{
std::vector< Matrix > D;     
};

struct GeneralVectors
{
std::vector< Vector > v;    
};


struct DeltaTimeParameters
{
double PredictionLevel; 

double Maximum;  
double Fraction; 
};


struct TimeVariables
{
double PreviousMiddle; 
double Previous;       
double Middle;         
double Current;        

double Delta;          
};


GeneralMatrices     mMatrix;
GeneralVectors      mVector;

bool                mSchemeIsInitialized;


TimeVariables       mTime;
DeltaTimeParameters mDeltaTime;
bool                mRayleighDamping;

bool                mUpdatePositionFlag;
bool                mUpdateRotationFlag;
bool                mUpdateMomentumFlag;














/













private:





























}; 









}  

#endif 
