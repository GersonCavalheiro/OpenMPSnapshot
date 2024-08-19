
#pragma once

#include <pybind11/pybind11.h>


#include "includes/define.h"

namespace Kratos::Python
{
void  AddCustomLinearSolversToPython(pybind11::module& m);
}  
