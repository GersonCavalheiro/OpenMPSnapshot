
#pragma once

#include "decs.hpp"

#include "emhd.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"



namespace EMHD {


template<typename Local>
KOKKOS_INLINE_FUNCTION void implicit_sources(const GRCoordinates& G, const Local& P, const Local& P_tau, const VarMap& m_p,
const Real& gam, const int& k, const int& j, const int& i,
const EMHD_parameters& emhd_params_tau,
Real& dUq, Real& dUdP)
{
Real tau, chi_e, nu_e;
EMHD::set_parameters(G, P_tau, m_p, emhd_params_tau, gam, k, j, i, tau, chi_e, nu_e);
dUq  = -G.gdet(Loci::center, j, i) * (P(m_p.Q) / tau);
dUdP = -G.gdet(Loci::center, j, i) * (P(m_p.DP) / tau);
}


template<typename Local>
KOKKOS_INLINE_FUNCTION void time_derivative_sources(const GRCoordinates& G, const Local& P_new,
const Local& P_old, const Local& P,
const VarMap& m_p, const EMHD_parameters& emhd_params,
const Real& gam, const Real& dt, 
const int & k, const int& j, const int& i,
Real& dUq, Real& dUdP)
{
Real tau, chi_e, nu_e;
EMHD::set_parameters(G, P, m_p, emhd_params, gam, k, j, i, tau, chi_e, nu_e);

FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, j, i, Loci::center, Dtmp);
double bsq = m::max(dot(Dtmp.bcon, Dtmp.bcov), SMALL);

Real ucon[GR_DIM], ucov_new[GR_DIM], ucov_old[GR_DIM];
GRMHD::calc_ucon(G, P_old, m_p, j, i, Loci::center, ucon);
G.lower(ucon, ucov_old, 0, j, i, Loci::center);
GRMHD::calc_ucon(G, P_new, m_p, j, i, Loci::center, ucon);
G.lower(ucon, ucov_new, 0, j, i, Loci::center);
Real dt_ucov[GR_DIM];
DLOOP1 dt_ucov[mu] = (ucov_new[mu] - ucov_old[mu]) / dt;

Real div_ucon = 0;
DLOOP1 div_ucon += G.gcon(Loci::center, j, i, 0, mu) * dt_ucov[mu];
const Real Theta_new = m::max((gam-1) * P_new(m_p.UU) / P_new(m_p.RHO), SMALL);
const Real Theta_old = m::max((gam-1) * P_old(m_p.UU) / P_old(m_p.RHO), SMALL);
const Real dt_Theta = (Theta_new - Theta_old) / dt;

const Real& rho     = P(m_p.RHO);
const Real& qtilde  = P(m_p.Q);
const Real& dPtilde = P(m_p.DP);
const Real& Theta   = (gam-1) * P(m_p.UU) / P(m_p.RHO);

Real q0 = -rho * chi_e * (Dtmp.bcon[0] / m::sqrt(bsq)) * dt_Theta;
DLOOP1 q0 -= rho * chi_e * (Dtmp.bcon[mu] / m::sqrt(bsq)) * Theta * Dtmp.ucon[0] * dt_ucov[mu];

Real dP0 = -rho * nu_e * div_ucon;
DLOOP1 dP0 += 3. * rho * nu_e * (Dtmp.bcon[0] * Dtmp.bcon[mu] / bsq) * dt_ucov[mu];

Real q0_tilde  = q0; 
Real dP0_tilde = dP0;
if (emhd_params.higher_order_terms) {
q0_tilde  *= (chi_e != 0) ? sqrt(tau / (chi_e * rho * pow(Theta, 2)) ) : 0.;
dP0_tilde *= (nu_e  != 0) ? sqrt(tau / (nu_e * rho * Theta) ) : 0.;
}

dUq  = G.gdet(Loci::center, j, i) * (q0_tilde / tau);
dUdP = G.gdet(Loci::center, j, i) * (dP0_tilde / tau);

if (emhd_params.higher_order_terms) {
dUq  += G.gdet(Loci::center, j, i) * (qtilde / 2.) * div_ucon;
dUdP += G.gdet(Loci::center, j, i) * (dPtilde / 2.) * div_ucon;
}
}

} 
