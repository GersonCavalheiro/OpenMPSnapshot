




#if !defined(PFEM2_EXPLICIT_STRATEGY)
#define  KRATOS_PFEM2_EXPLICIT_STRATEGY



#include <string>
#include <iostream>
#include <algorithm>



#ifdef _OPENMP
#include <omp.h>
#endif

#include "boost/smart_ptr.hpp"



#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/element.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "solving_strategies/schemes/scheme.h"
#include "includes/variables.h"
#include "includes/cfd_variables.h"
#include "containers/array_1d.h"
#include "pfem_2_application_variables.h"

namespace Kratos
{

template<
class TSparseSpace,
class TDenseSpace,
class TLinearSolver>
class PFEM2_Explicit_Strategy : public ImplicitSolvingStrategy<TSparseSpace,TDenseSpace,TLinearSolver>
{

public:

KRATOS_CLASS_POINTER_DEFINITION(PFEM2_Explicit_Strategy);

typedef ImplicitSolvingStrategy<TSparseSpace,TDenseSpace,TLinearSolver> BaseType;

typedef typename BaseType::TDataType TDataType;

typedef TSparseSpace SparseSpaceType;

typedef typename BaseType::TBuilderAndSolverType TBuilderAndSolverType;

typedef typename BaseType::TSchemeType TSchemeType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename Element::DofsVectorType DofsVectorType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;

typedef ModelPart::NodesContainerType NodesArrayType;

typedef ModelPart::ElementsContainerType ElementsArrayType;

typedef ModelPart::ConditionsContainerType ConditionsArrayType;

typedef ModelPart::ConditionsContainerType::ContainerType ConditionsContainerType;

typedef ConditionsContainerType::iterator                 ConditionsContainerIterator;

typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;

typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;

typedef ModelPart::PropertiesType PropertiesType;







PFEM2_Explicit_Strategy(
ModelPart& model_part,
const int        dimension,
const bool       MoveMeshFlag
)

: ImplicitSolvingStrategy<TSparseSpace,TDenseSpace,TLinearSolver>(model_part, MoveMeshFlag)
{

}

virtual ~PFEM2_Explicit_Strategy () {}



/
#endif 
