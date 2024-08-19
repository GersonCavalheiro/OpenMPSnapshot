
#pragma once

#include <pybind11/pybind11.h>



namespace Kratos::Python {

void AddTrilinosConvergenceAcceleratorsToPython(pybind11::module& m);

}  
