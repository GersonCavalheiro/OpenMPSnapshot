
#pragma once


#include <pybind11/pybind11.h>


namespace Kratos::Python {

void AddMPICommunicatorToPython(pybind11::module& m);

} 