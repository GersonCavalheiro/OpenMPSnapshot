
#pragma once

@{KRATOS_SYSTEM_INCLUDES}
#include <string>
#include <iostream>
#include <algorithm>

@{KRATOS_EXTERNA_INCLUDES}
#include "includes/kratos_flags.h"

@{KRATOS_PROJECT_INCLUDES}
#include "includes/define.h"
#include "processes/process.h"
#include "includes/kratos_flags.h"
#include "includes/element.h"
#include "includes/model_part.h"
#include "geometries/geometry_data.h"

#include "spaces/ublas_space.h"
#include "linear_solvers/linear_solver.h"
#include "solving_strategies/schemes/residualbased_incrementalupdate_static_scheme.h"
#include "solving_strategies/builder_and_solvers/residualbased_block_builder_and_solver.h"
#include "solving_strategies/strategies/residualbased_linear_strategy.h"
#include "elements/distance_calculation_element_simplex.h"

namespace Kratos {






@{KRATOS_CLASS_TEMPLATE}
class @{KRATOS_NAME_CAMEL} : public @{KRATOS_CLASS_BASE} {
public:


KRATOS_CLASS_POINTER_DEFINITION(@{KRATOS_NAME_CAMEL});


@{KRATOS_NAME_CAMEL}() {
}

virtual ~@{KRATOS_NAME_CAMEL}() {
}


void operator()() {
Execute();
}


virtual void Execute() {
}

virtual void Clear() {
}




virtual std::string Info() const {
return "@{KRATOS_NAME_CAMEL}";
}

virtual void PrintInfo(std::ostream& rOStream) const {
rOStream << "@{KRATOS_NAME_CAMEL}";
}

virtual void PrintData(std::ostream& rOStream) const {
}



protected:









private:








@{KRATOS_NAME_CAMEL}& operator=(@{KRATOS_NAME_CAMEL} const& rOther);



}; 

template< unsigned int TDim,class TSparseSpace, class TDenseSpace, class TLinearSolver >
const Kratos::Flags @{KRATOS_NAME_CAMEL}<TDim,TSparseSpace,TDenseSpace,TLinearSolver>::PERFORM_STEP1(Kratos::Flags::Create(0));

template< unsigned int TDim,class TSparseSpace, class TDenseSpace, class TLinearSolver >
const Kratos::Flags @{KRATOS_NAME_CAMEL}<TDim,TSparseSpace,TDenseSpace,TLinearSolver>::DO_EXPENSIVE_CHECKS(Kratos::Flags::Create(1));




template< unsigned int TDim, class TSparseSpace, class TDenseSpace, class TLinearSolver>
inline std::istream& operator >> (std::istream& rIStream,
@{KRATOS_NAME_CAMEL}<TDim,TSparseSpace,TDenseSpace,TLinearSolver>& rThis);

template< unsigned int TDim, class TSparseSpace, class TDenseSpace, class TLinearSolver>
inline std::ostream& operator << (std::ostream& rOStream,
const @{KRATOS_NAME_CAMEL}<TDim,TSparseSpace,TDenseSpace,TLinearSolver>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
