
#pragma once

#include "decs.hpp"
#include "types.hpp"

#include "emhd_sources.hpp"
#include "emhd.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"


using namespace EMHD;

#define FLOOP for(int ip=0; ip < nfvar; ++ip)

namespace Implicit
{


std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);


TaskStatus Step(MeshData<Real> *mdi, MeshData<Real> *md0, MeshData<Real> *dudt,
MeshData<Real> *mc_solver, const Real& dt);


std::vector<std::string> get_ordered_names(MeshBlockData<Real> *rc, const MetadataFlag& flag, bool only_implicit=false);


template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_residual(const GRCoordinates& G, const Local& P_test,
const Local& Pi, const Local& Ui, const Local& Ps,
const Local& dudt_explicit, const Local& dUi, const Local& tmp, 
const VarMap& m_p, const VarMap& m_u, const EMHD_parameters& emhd_params,
const EMHD_parameters& emhd_params_tau,const int& nfvar, 
const int& k, const int& j, const int& i, 
const Real& gam, const double& dt, Local& residual)
{
Flux::p_to_u(G, P_test, m_p, emhd_params, gam, j, i, tmp, m_u); 
FLOOP residual(ip) = (tmp(ip) - Ui(ip)) / dt - dudt_explicit(ip);

if (m_p.Q >= 0) {
Real dUq, dUdP; 
EMHD::implicit_sources(G, P_test, Ps, m_p, gam, k, j, i, emhd_params_tau, dUq, dUdP); 
residual(m_u.Q)  -= 0.5*(dUq + dUi(m_u.Q));
residual(m_u.DP) -= 0.5*(dUdP + dUi(m_u.DP));
EMHD::time_derivative_sources(G, P_test, Pi, Ps, m_p, emhd_params, gam, dt, k, j, i, dUq, dUdP); 
residual(m_u.Q)  -= dUq;
residual(m_u.DP) -= dUdP;

Real tau, chi_e, nu_e;
EMHD::set_parameters(G, P_test, m_p, emhd_params, gam, k, j, i, tau, chi_e, nu_e);
residual(m_u.Q)  *= tau;
residual(m_u.DP) *= tau;
if (emhd_params.higher_order_terms){
Real rho   = P_test(m_p.RHO);
Real u     = P_test(m_p.UU);
Real Theta = (gam - 1.) * u / rho;

residual(m_u.Q)  *= (chi_e != 0) ? sqrt(rho * chi_e * tau * pow(Theta, 2)) / tau : 1.;
residual(m_u.DP) *= (nu_e != 0)  ? sqrt(rho * nu_e * tau * Theta) / tau : 1.;
}
}

}


template<typename Local, typename Local2>
KOKKOS_INLINE_FUNCTION void calc_jacobian(const GRCoordinates& G, const Local& P_solver,
const Local& P_full_step_init, const Local& U_full_step_init, const Local& P_sub_step_init,
const Local& flux_src, const Local& dU_implicit, Local& tmp1, Local& tmp2, Local& tmp3,
const VarMap& m_p, const VarMap& m_u, const EMHD_parameters& emhd_params_full_step_init,
const EMHD_parameters& emhd_params_sub_step_init, const int& nvar, const int& nfvar,
const int& k, const int& j, const int& i,
const Real& jac_delta, const Real& gam, const double& dt,
Local2& jacobian, Local& residual)
{
calc_residual(G, P_solver, P_full_step_init, U_full_step_init, P_sub_step_init, flux_src, dU_implicit, tmp3,
m_p, m_u, emhd_params_full_step_init, emhd_params_sub_step_init, nfvar, k, j, i, gam, dt, residual);

auto& P_delta        = tmp1;
auto& residual_delta = tmp2;
PLOOP P_delta(ip)    = P_solver(ip);

for (int col = 0; col < nfvar; col++) {
if (m::abs(P_solver(col)) < (0.5 * jac_delta)) {
P_delta(col) = P_solver(col) + jac_delta;
} else {
P_delta(col) = (1 + jac_delta) * P_solver(col);
}

calc_residual(G, P_delta, P_full_step_init, U_full_step_init, P_sub_step_init, flux_src, dU_implicit, tmp3, 
m_p, m_u, emhd_params_full_step_init, emhd_params_sub_step_init, nfvar, k, j, i, gam, dt, residual_delta);

for (int row = 0; row < nfvar; row++) {
jacobian(row, col) = (residual_delta(row) - residual(row)) / (P_delta(col) - P_solver(col) + SMALL);
}

P_delta(col) = P_solver(col);

}
}   

} 
