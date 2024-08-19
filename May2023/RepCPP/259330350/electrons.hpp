
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

#include "grmhd_functions.hpp"

using namespace parthenon;


namespace Electrons {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);


TaskStatus InitElectrons(MeshBlockData<Real> *rc, ParameterInput *pin);


void UtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerivedBlock(MeshBlockData<Real> *rc) { UtoP(rc); }


TaskStatus ApplyElectronHeating(MeshBlockData<Real> *rc_old, MeshBlockData<Real> *rc);


TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);


void FillOutput(MeshBlock *pmb, ParameterInput *pin);


KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
const int& k, const int& j, const int& i,
const VariablePack<Real>& flux, const VarMap m_u, const Loci loc=Loci::center)
{
const Real ut = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc) * m::sqrt(-G.gcon(loc, j, i, 0, 0));
const Real rho_ut = P(m_p.RHO, k, j, i) * ut * G.gdet(loc, j, i);

flux(m_u.KTOT, k, j, i) = rho_ut * P(m_p.KTOT, k, j, i);
if (m_p.K_CONSTANT >= 0)
flux(m_u.K_CONSTANT, k, j, i) = rho_ut * P(m_p.K_CONSTANT, k, j, i);
if (m_p.K_HOWES >= 0)
flux(m_u.K_HOWES, k, j, i) = rho_ut * P(m_p.K_HOWES, k, j, i);
if (m_p.K_KAWAZURA >= 0)
flux(m_u.K_KAWAZURA, k, j, i) = rho_ut * P(m_p.K_KAWAZURA, k, j, i);
if (m_p.K_WERNER >= 0)
flux(m_u.K_WERNER, k, j, i) = rho_ut * P(m_p.K_WERNER, k, j, i);
if (m_p.K_ROWAN >= 0)
flux(m_u.K_ROWAN, k, j, i) = rho_ut * P(m_p.K_ROWAN, k, j, i);
if (m_p.K_SHARMA >= 0)
flux(m_u.K_SHARMA, k, j, i) = rho_ut * P(m_p.K_SHARMA, k, j, i);
}

}
