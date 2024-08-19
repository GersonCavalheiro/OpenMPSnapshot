

#if !defined(KRATOS_RESIDUALBASED_PREDICTOR_CORRECTOR_VELOCITY_BOSSAK_ALE_SCHEME )
#define  KRATOS_RESIDUALBASED_PREDICTOR_CORRECTOR_VELOCITY_BOSSAK_ALE_SCHEME






#include "boost/smart_ptr.hpp"


#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/deprecated_variables.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "includes/cfd_variables.h"
#include "containers/array_1d.h"
#include "utilities/openmp_utils.h"
#include "utilities/dof_updater.h"
#include "utilities/coordinate_transformation_utilities.h"
#include "processes/process.h"

#include "../../applications/FluidDynamicsApplication/custom_strategies/schemes/residualbased_predictorcorrector_velocity_bossak_scheme_turbulent.h"

namespace Kratos {



























template<class TSparseSpace, class TDenseSpace >
class ResidualBasedPredictorCorrectorVelocityBossakAleScheme : public ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace> {

public:



KRATOS_CLASS_POINTER_DEFINITION(ResidualBasedPredictorCorrectorVelocityBossakAleScheme);

typedef Scheme<TSparseSpace, TDenseSpace> BaseType;

typedef ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace> SchemeType;

typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename Element::DofsVectorType DofsVectorType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef Element::GeometryType  GeometryType;








ResidualBasedPredictorCorrectorVelocityBossakAleScheme(
double NewAlphaBossak,
double MoveMeshStrategy,
unsigned int DomainSize)
: ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace>(NewAlphaBossak,MoveMeshStrategy,DomainSize)
{
}



ResidualBasedPredictorCorrectorVelocityBossakAleScheme(
double NewAlphaBossak,
unsigned int DomainSize,
const Variable<int>& rPeriodicIdVar)
: ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace>(NewAlphaBossak,DomainSize,rPeriodicIdVar)
{
}



ResidualBasedPredictorCorrectorVelocityBossakAleScheme(
double NewAlphaBossak,
double MoveMeshStrategy,
unsigned int DomainSize,
Variable<double>& rSlipVar)
: ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace>(NewAlphaBossak,MoveMeshStrategy,rSlipVar)
{
}


ResidualBasedPredictorCorrectorVelocityBossakAleScheme(
double NewAlphaBossak,
double MoveMeshStrategy,
unsigned int DomainSize,
Process::Pointer pTurbulenceModel)
: ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, TDenseSpace>(NewAlphaBossak,MoveMeshStrategy,DomainSize,pTurbulenceModel)
{
}


~ResidualBasedPredictorCorrectorVelocityBossakAleScheme() override {
}







/





















protected:





































private:


































}; 









} 

#endif 
