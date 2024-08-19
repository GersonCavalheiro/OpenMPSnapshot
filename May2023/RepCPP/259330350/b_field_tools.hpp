
#pragma once

#include "decs.hpp"
#include "types.hpp"

enum BSeedType{constant, monopole, sane, ryan, ryan_quadrupole, r3s3, steep, gaussian, bz_monopole, vertical};


inline BSeedType ParseBSeedType(std::string b_field_type)
{
if (b_field_type == "constant") {
return BSeedType::constant;
} else if (b_field_type == "monopole") {
return BSeedType::monopole;
} else if (b_field_type == "sane") {
return BSeedType::sane;
} else if (b_field_type == "mad" || b_field_type == "ryan") {
return BSeedType::ryan;
} else if (b_field_type == "mad_quadrupole" || b_field_type == "ryan_quadrupole") {
return BSeedType::ryan_quadrupole;
} else if (b_field_type == "r3s3") {
return BSeedType::r3s3;
} else if (b_field_type == "mad_steep" || b_field_type == "steep") {
return BSeedType::steep;
} else if (b_field_type == "gaussian") {
return BSeedType::gaussian;
} else if (b_field_type == "bz_monopole") {
return BSeedType::bz_monopole;
} else if (b_field_type == "vertical") {
return BSeedType::vertical;
} else {
throw std::invalid_argument("Magnetic field seed type not supported: " + b_field_type);
}
}


Real GetLocalBetaMin(parthenon::MeshBlockData<Real> *rc);


Real GetLocalBsqMax(parthenon::MeshBlockData<Real> *rc);
Real GetLocalBsqMin(parthenon::MeshBlockData<Real> *rc);


Real GetLocalPMax(parthenon::MeshBlockData<Real> *rc);


TaskStatus NormalizeBField(parthenon::MeshBlockData<Real> *rc, Real factor);
