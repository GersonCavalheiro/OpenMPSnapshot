
#pragma once

#include "decs.hpp"
#include "emhd.hpp"
#include "gr_coordinates.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "prob_common.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

TaskStatus InitializeAtmosphere(MeshBlockData<Real> *rc, ParameterInput *pin);

TaskStatus dirichlet_bc(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse);