
#pragma once

#include "decs.hpp"

#include "b_flux_ct.hpp"
#include "b_cd.hpp"

namespace KHARMA {


void SeedAndNormalizeB(ParameterInput *pin, std::shared_ptr<MeshData<Real>> md);


void PostInitialize(ParameterInput *pin, Mesh *pmesh, bool is_restart, bool is_resize);

}
