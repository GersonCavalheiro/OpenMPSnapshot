
#pragma once

#include <parthenon/parthenon.hpp>

#include "grmhd_functions.hpp"

using namespace parthenon;


namespace EMHD {

enum ClosureType{constant=0, soundspeed, kappa_eta, torus};

class EMHD_parameters {
public:

bool higher_order_terms;
ClosureType type;
Real tau;
Real conduction_alpha;
Real viscosity_alpha;

Real kappa;
Real eta;

};


std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);


TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);


template<typename Local>
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Local& P, const VarMap& m_p,
const EMHD_parameters& emhd_params, const Real& gam,
const int& k, const int& j, const int& i,
Real& tau, Real& chi_e, Real& nu_e)
{
if (emhd_params.type == ClosureType::constant) {

tau   = emhd_params.tau;
chi_e = emhd_params.conduction_alpha;
nu_e  = emhd_params.viscosity_alpha;

} else if (emhd_params.type == ClosureType::soundspeed) {
Real cs2 = (gam * (gam - 1.) * P(m_p.UU)) / (P(m_p.RHO) + (gam * P(m_p.UU)));

tau   = emhd_params.tau;
chi_e = emhd_params.conduction_alpha * cs2 * tau;
nu_e  = emhd_params.viscosity_alpha * cs2 * tau;

} else if (emhd_params.type == ClosureType::kappa_eta){

tau   = emhd_params.tau;
chi_e = emhd_params.kappa / m::max(P(m_p.RHO), SMALL);
nu_e  = emhd_params.eta / m::max(P(m_p.RHO), SMALL);

} else if (emhd_params.type == ClosureType::torus) {
FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, j, i, Loci::center, Dtmp);
double bsq = m::max(dot(Dtmp.bcon, Dtmp.bcov), SMALL);

GReal Xembed[GR_DIM];
G.coord_embed(k, j, i, Loci::center, Xembed);
GReal r = Xembed[1];

Real tau_dyn = m::pow(r, 1.5);
tau          = tau_dyn;

Real pg    = (gam - 1.) * P(m_p.UU);
Real Theta = pg / P(m_p.RHO);
Real cs    = m::sqrt(gam * pg / (P(m_p.RHO) + (gam * P(m_p.UU)))); 

Real lambda    = 0.01;
Real inv_exp_g = 0.;
Real f_fmin    = 0.;

Real q = P(m_p.Q);
if (emhd_params.higher_order_terms)
q *= sqrt(P(m_p.RHO) * emhd_params.conduction_alpha * m::pow(cs, 2.) * m::pow(Theta, 2.));
Real q_max   = emhd_params.conduction_alpha * P(m_p.RHO) * m::pow(cs, 3.);
Real q_ratio = fabs(q) / q_max;
inv_exp_g    = exp(-(q_ratio - 1.) / lambda);
f_fmin       = inv_exp_g / (inv_exp_g + 1.) + 1.e-5;

tau = m::min(tau, f_fmin * tau_dyn);

Real dP = P(m_p.DP);
if (emhd_params.higher_order_terms)
dP *= sqrt(P(m_p.RHO) * emhd_params.viscosity_alpha * m::pow(cs, 2.) * Theta);
Real dP_comp_ratio = m::max(pg - 2./3. * dP, SMALL) / m::max(pg  + 1./3. * dP, SMALL);
Real dP_plus       = m::min(0.5 * bsq * dP_comp_ratio, 1.49 * pg / 1.07);
Real dP_minus      = m::max(-bsq, -2.99 * pg / 1.07);

Real dP_max = 0.;
if (dP > 0.)
dP_max = dP_plus;
else
dP_max = dP_minus;

Real dP_ratio = m::abs(dP) / (m::abs(dP_max) + SMALL);
inv_exp_g     = m::exp(-(dP_comp_ratio - 1.) / lambda);
f_fmin        = inv_exp_g / (inv_exp_g + 1.) + 1.e-5;

tau = m::min(tau, f_fmin * tau_dyn);

Real max_alpha = (1 - m::pow(cs, 2.)) / (2*m::pow(cs, 2.) + 1.e-12);
chi_e = m::min(max_alpha, emhd_params.conduction_alpha) * m::pow(cs, 2.) * tau;
nu_e  = m::min(max_alpha, emhd_params.viscosity_alpha) * m::pow(cs, 2.) * tau;
} 
}

template<typename Global>
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Global& P, const VarMap& m_p,
const EMHD_parameters& emhd_params, const Real& gam,
const int& k, const int& j, const int& i,
Real& tau, Real& chi_e, Real& nu_e, const char* global_flag)
{
if (emhd_params.type == ClosureType::constant) {
tau   = emhd_params.tau;
chi_e = emhd_params.conduction_alpha;
nu_e  = emhd_params.viscosity_alpha;
} else if (emhd_params.type == ClosureType::soundspeed) {
const Real cs2 = (gam * (gam - 1.) * P(m_p.UU, k, j, i)) /
(P(m_p.RHO, k, j, i) + (gam * P(m_p.UU, k, j, i)));

tau   = emhd_params.tau;
chi_e = emhd_params.conduction_alpha * cs2 * tau;
nu_e  = emhd_params.viscosity_alpha * cs2 * tau;
} else if (emhd_params.type == ClosureType::kappa_eta){

tau   = emhd_params.tau;
chi_e = emhd_params.kappa / m::max(P(m_p.RHO, k, j, i), SMALL);
nu_e  = emhd_params.eta / m::max(P(m_p.RHO, k, j, i), SMALL);

} else if (emhd_params.type == ClosureType::torus) {
Real rho     = P(m_p.RHO, k, j, i);
Real uu      = P(m_p.UU, k, j, i);
Real qtilde  = P(m_p.Q, k, j, i);
Real dPtilde = P(m_p.DP, k, j, i);

FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
double bsq = m::max(dot(Dtmp.bcon, Dtmp.bcov), SMALL);

GReal Xembed[GR_DIM];
G.coord_embed(k, j, i, Loci::center, Xembed);
GReal r = Xembed[1];

Real tau_dyn = pow(r, 1.5);
tau          = tau_dyn;

Real pg    = (gam - 1.) * uu;
Real Theta = pg / rho;
Real cs    = sqrt(gam * pg / (rho + (gam * uu))); 

Real lambda    = 0.01;
Real inv_exp_g = 0.;
Real f_fmin    = 0.;

Real q = qtilde;
if (emhd_params.higher_order_terms)
q *= (rho * emhd_params.conduction_alpha * pow(cs, 2.) * pow(Theta, 2.));
Real q_max   = emhd_params.conduction_alpha * rho * pow(cs, 3.);
Real q_ratio = fabs(q) / q_max;
inv_exp_g    = exp(-(q_ratio - 1.) / lambda);
f_fmin       = inv_exp_g / (inv_exp_g + 1.) + 1.e-5;

tau = m::min(tau, f_fmin * tau_dyn);

Real dP = dPtilde;
if (emhd_params.higher_order_terms)
dP *= sqrt(rho * emhd_params.viscosity_alpha * pow(cs, 2.) * Theta);
Real dP_comp_ratio = m::max(pg - 2./3. * dP, SMALL) / m::max(pg  + 1./3. * dP, SMALL);
Real dP_plus       = m::min(0.5 * bsq * dP_comp_ratio, 1.49 * pg / 1.07);
Real dP_minus      = m::max(-bsq, -2.99 * pg / 1.07);

Real dP_max = 0.;
if (dP > 0.)
dP_max = dP_plus;
else
dP_max = dP_minus;

Real dP_ratio = m::abs(dP) / (m::abs(dP_max) + SMALL);
inv_exp_g     = m::exp(-(dP_comp_ratio - 1.) / lambda);
f_fmin        = inv_exp_g / (inv_exp_g + 1.) + 1.e-5;

tau = m::min(tau, f_fmin * tau_dyn);

Real max_alpha = (1 - m::pow(cs, 2.)) / (2*m::pow(cs, 2.) + 1.e-12);
chi_e = m::min(max_alpha, emhd_params.conduction_alpha) * m::pow(cs, 2.) * tau;
nu_e  = m::min(max_alpha, emhd_params.viscosity_alpha) * m::pow(cs, 2.) * tau;
} 
}

KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Real& rho, const Real& u,
const EMHD_parameters& emhd_params, const Real& gam,
const int& k, const int& j, const int& i,
Real& tau, Real& chi_e, Real& nu_e)
{
if (emhd_params.type == ClosureType::constant) {
tau   = emhd_params.tau;
chi_e = emhd_params.conduction_alpha;
nu_e  = emhd_params.viscosity_alpha;

} else if (emhd_params.type == ClosureType::soundspeed) {
const Real cs2 = (gam * (gam - 1.) * u) / (rho + (gam * u));
tau   = emhd_params.tau;
chi_e = emhd_params.conduction_alpha * cs2 * tau;
nu_e  = emhd_params.viscosity_alpha * cs2 * tau;

} else if (emhd_params.type == ClosureType::kappa_eta){
tau   = emhd_params.tau;
chi_e = emhd_params.kappa / m::max(rho, SMALL);
nu_e  = emhd_params.eta / m::max(rho, SMALL);

} 
}


KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
const Real& q, const Real& dP,
const FourVectors& D, const int& dir,
Real emhd[GR_DIM])
{
const Real bsq = m::max(dot(D.bcon, D.bcov), SMALL);
const Real eta = pgas + rho + u + bsq;
const Real ptot = pgas + 0.5 * bsq;

DLOOP1 {
emhd[mu] = eta * D.ucon[dir] * D.ucov[mu]
+ ptot * (dir == mu)
- D.bcon[dir] * D.bcov[mu]
+ (q / m::sqrt(bsq)) * ((D.ucon[dir] * D.bcov[mu]) +
(D.bcon[dir] * D.ucov[mu]))
- dP * ((D.bcon[dir] * D.bcov[mu] / bsq)
- (1./3) * ((dir == mu) + D.ucon[dir] * D.ucov[mu]));
}
}

KOKKOS_INLINE_FUNCTION void convert_prims_to_q_dP(const Real& q_tilde, const Real& dP_tilde,
const Real& rho, const Real& Theta, const Real& cs2, 
const EMHD_parameters& emhd_params, Real& q, Real& dP)
{
q  = q_tilde;
dP = dP_tilde;

if (emhd_params.higher_order_terms) {
q  *= m::sqrt(rho * emhd_params.conduction_alpha * cs2 * m::pow(Theta, 2));
dP *= m::sqrt(rho * emhd_params.viscosity_alpha * cs2 * Theta);
}
}

} 
