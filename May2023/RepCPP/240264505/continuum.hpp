

#ifndef LBT_CONTINUUM
#define LBT_CONTINUUM
#pragma once

#include "../general/use_vtk.hpp"

#ifdef LBT_USE_VTK
#include "vtk_continuum.hpp"

namespace lbt {
template <typename T>
using Continuum = VtkContinuum<T>;
}
#else
#include "simple_continuum.hpp"

namespace lbt {
template <typename T>
using Continuum = SimpleContinuum<T>;
}
#endif 

#endif 
