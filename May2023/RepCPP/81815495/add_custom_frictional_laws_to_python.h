
#pragma once

#include <pybind11/pybind11.h>


#include "includes/define.h"

namespace Kratos::Python
{
void  AddCustomFrictionalLawsToPython(pybind11::module& m);
}  
