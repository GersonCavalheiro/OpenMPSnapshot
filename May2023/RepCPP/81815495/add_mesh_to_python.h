
#pragma once

#include <pybind11/pybind11.h>



namespace Kratos::Python
{

void  AddNodeToPython(pybind11::module& m);
void  AddPropertiesToPython(pybind11::module& m);
void  AddMeshToPython(pybind11::module& m);

}  
