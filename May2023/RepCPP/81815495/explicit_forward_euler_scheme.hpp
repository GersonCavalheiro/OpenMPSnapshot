
#if !defined(KRATOS_EXPLICIT_FORWARD_EULER_SCHEME )
#define  KRATOS_EXPLICIT_FORWARD_EULER_SCHEME

#include "includes/define.h"
#include "includes/model_part.h"
#include "custom_strategies/schemes/generalized_newmark_GN11_scheme.hpp"
#include "includes/convection_diffusion_settings.h"

#include "fluid_transport_application_variables.h"

namespace Kratos
{

template<class TSparseSpace, class TDenseSpace>

class ExplicitForwardEulerScheme : public GeneralizedNewmarkGN11Scheme<TSparseSpace,TDenseSpace>
{

public:

KRATOS_CLASS_POINTER_DEFINITION( ExplicitForwardEulerScheme );

typedef GeneralizedNewmarkGN11Scheme<TSparseSpace,TDenseSpace>                      BaseType;
typedef typename BaseType::DofsArrayType                 DofsArrayType;
typedef typename BaseType::TSystemMatrixType         TSystemMatrixType;
typedef typename BaseType::TSystemVectorType         TSystemVectorType;
typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;


ExplicitForwardEulerScheme(double theta) : GeneralizedNewmarkGN11Scheme<TSparseSpace,TDenseSpace>(theta)
{
}


~ExplicitForwardEulerScheme() override {}


void Predict(
ModelPart& r_model_part,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
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
itDof->GetSolutionStepValue() = TSparseSpace::GetValue(Dx, itDof->EquationId());
}
}


KRATOS_CATCH( "" )
}


protected:










}; 
}  

#endif 
