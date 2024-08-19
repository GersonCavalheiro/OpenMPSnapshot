
#pragma once


#include <pybind11/pybind11.h>


namespace Kratos::Python {

void AddMPIDataCommunicatorToPython(pybind11::module &m);

} 