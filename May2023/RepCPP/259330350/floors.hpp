
#pragma once

#include "decs.hpp"


#include "b_flux_ct.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "U_to_P.hpp"
#include "emhd.hpp"

#include <parthenon/parthenon.hpp>

#define RECORD_POST_RECON 0

#define HIT_FLOOR_GEOM_RHO 32
#define HIT_FLOOR_GEOM_U 64
#define HIT_FLOOR_B_RHO 128
#define HIT_FLOOR_B_U 256
#define HIT_FLOOR_TEMP 512
#define HIT_FLOOR_GAMMA 1024
#define HIT_FLOOR_KTOT 2048
#define HIT_FLOOR_GEOM_RHO_FLUX 4096
#define HIT_FLOOR_GEOM_U_FLUX 8192

#define HIT_Q_LIMIT  1
#define HIT_DP_LIMIT 2

namespace Floors
{


std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);


TaskStatus ApplyFloors(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire);


TaskStatus PostFillDerivedBlock(MeshBlockData<Real> *rc);


class Prescription {
public:
double rho_min_geom, u_min_geom, r_char, frame_switch;
double bsq_over_rho_max, bsq_over_u_max, u_over_rho_max;
double ktot_max;
double gamma_max;
bool fluid_frame, mixed_frame, drift_frame;
bool use_r_char, temp_adjust_u, adjust_k;

bool enable_emhd_limits;

Prescription(const parthenon::Params& params)
{
rho_min_geom = params.Get<Real>("rho_min_geom");
u_min_geom   = params.Get<Real>("u_min_geom");
r_char       = params.Get<GReal>("r_char");
frame_switch = params.Get<GReal>("frame_switch");

bsq_over_rho_max = params.Get<Real>("bsq_over_rho_max");
bsq_over_u_max   = params.Get<Real>("bsq_over_u_max");
u_over_rho_max   = params.Get<Real>("u_over_rho_max");
ktot_max         = params.Get<Real>("ktot_max");
gamma_max        = params.Get<Real>("gamma_max");

use_r_char    = params.Get<bool>("use_r_char");
temp_adjust_u = params.Get<bool>("temp_adjust_u");
adjust_k      = params.Get<bool>("adjust_k");

fluid_frame   = params.Get<bool>("fluid_frame");
mixed_frame   = params.Get<bool>("mixed_frame");
drift_frame   = params.Get<bool>("drift_frame");

enable_emhd_limits = params.Get<bool>("enable_emhd_limits");
}
};


KOKKOS_INLINE_FUNCTION int apply_ceilings(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
const Real& gam, const int& k, const int& j, const int& i, const Floors::Prescription& floors,
const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
int fflag = 0;
Real gamma = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc);

if (gamma > floors.gamma_max) {
fflag |= HIT_FLOOR_GAMMA;

Real f = m::sqrt((m::pow(floors.gamma_max, 2) - 1.)/(m::pow(gamma, 2) - 1.));
VLOOP P(m_p.U1+v, k, j, i) *= f;
}

Real ktot = (gam - 1.) * P(m_p.UU, k, j, i) / m::pow(P(m_p.RHO, k, j, i), gam);
if (ktot > floors.ktot_max) {
fflag |= HIT_FLOOR_KTOT;

P(m_p.UU, k, j, i) = floors.ktot_max / ktot * P(m_p.UU, k, j, i);
}
if (m_p.KTOT >= 0 && (P(m_p.KTOT, k, j, i) > floors.ktot_max)) {
fflag |= HIT_FLOOR_KTOT;
P(m_p.KTOT, k, j, i) = floors.ktot_max;
}

if (floors.temp_adjust_u && P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i) > floors.u_over_rho_max) {
fflag |= HIT_FLOOR_TEMP;

P(m_p.UU, k, j, i) = floors.u_over_rho_max * P(m_p.RHO, k, j, i);
}

if (fflag) {
GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u, loc);
}

return fflag;
}


KOKKOS_INLINE_FUNCTION int apply_floors(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
const Real& gam, const int& k, const int& j, const int& i, const Floors::Prescription& floors,
const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
int fflag = 0;
InversionStatus pflag = InversionStatus::success;
Real rhoflr_geom, uflr_geom;
bool use_ff, use_df;
if(G.coords.spherical()) {
GReal Xembed[GR_DIM];
G.coord_embed(k, j, i, loc, Xembed);
GReal r = Xembed[1];

use_ff = floors.fluid_frame || (floors.mixed_frame && r > floors.frame_switch);
use_df = floors.drift_frame;

if (floors.use_r_char) {
Real rhoscal = m::pow(r, -2.) * 1 / (1 + r / floors.r_char);
rhoflr_geom  = floors.rho_min_geom * rhoscal;
uflr_geom    = floors.u_min_geom * m::pow(rhoscal, gam);
} else {
rhoflr_geom = floors.rho_min_geom * m::pow(r, -1.5);
uflr_geom   = floors.u_min_geom * m::pow(r, -2.5); 
}
} else {
rhoflr_geom = floors.rho_min_geom;
uflr_geom   = floors.u_min_geom;
use_ff      = floors.fluid_frame;
use_df      = floors.drift_frame;
}
Real rho = P(m_p.RHO, k, j, i);
Real u   = P(m_p.UU, k, j, i);

FourVectors Dtmp;
GRMHD::calc_4vecs(G, P, m_p, k, j, i, loc, Dtmp);
double bsq      = dot(Dtmp.bcon, Dtmp.bcov);
double rhoflr_b = bsq / floors.bsq_over_rho_max;
double uflr_b   = bsq / floors.bsq_over_u_max;

double uflr_max = m::max(uflr_geom, uflr_b);

double rhoflr_max;
if (!floors.temp_adjust_u) {
double rhoflr_temp = m::max(u, uflr_max) / floors.u_over_rho_max;
fflag |= (rhoflr_temp > rho) * HIT_FLOOR_TEMP; 

rhoflr_max = m::max(m::max(rhoflr_geom, rhoflr_b), rhoflr_temp);
} else {
rhoflr_max = m::max(rhoflr_geom, rhoflr_b);
}

if (rhoflr_max > rho || uflr_max > u) {

fflag |= (rhoflr_geom > rho) * HIT_FLOOR_GEOM_RHO;
fflag |= (uflr_geom > u) * HIT_FLOOR_GEOM_U;
fflag |= (rhoflr_b > rho) * HIT_FLOOR_B_RHO;
fflag |= (uflr_b > u) * HIT_FLOOR_B_U;

if (use_ff) {
P(m_p.RHO, k, j, i) += m::max(0., rhoflr_max - rho);
P(m_p.UU, k, j, i)  += m::max(0., uflr_max - u);
GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u, loc);

} else if (use_df) {
const Real gdet     = G.gdet(Loci::center, j, i);
const Real lapse    = 1./m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
double beta[GR_DIM] = {0};

beta[1] = lapse * lapse * G.gcon(Loci::center, j, i, 0, 1);
beta[2] = lapse * lapse * G.gcon(Loci::center, j, i, 0, 2);
beta[3] = lapse * lapse * G.gcon(Loci::center, j, i, 0, 3);

const Real rho   = P(m_p.RHO, k, j, i);
const Real uu    = P(m_p.UU, k, j, i);
const Real pg    = (gam - 1.) * uu;
const Real w_old = m::max(rho + uu + pg, SMALL);

Real Bcon[GR_DIM] = {0};
Real Bcov[GR_DIM] = {0};
Bcon[0] = 0;
Bcon[1] = P(m_p.B1, k, j, i);
Bcon[2] = P(m_p.B2, k, j, i);
Bcon[3] = P(m_p.B3, k, j, i);
DLOOP2 Bcov[mu] += G.gcov(Loci::center, j, i, mu, nu) * Bcon[nu];
const Real Bsq   = dot(Bcon, Bcov);

Real Qcov[GR_DIM] = {0};
Qcov[0] = w_old * Dtmp.ucon[0] * Dtmp.ucov[0] + pg;
Qcov[1] = w_old * Dtmp.ucon[0] * Dtmp.ucov[1];
Qcov[2] = w_old * Dtmp.ucon[0] * Dtmp.ucov[2];
Qcov[3] = w_old * Dtmp.ucon[0] * Dtmp.ucov[3];

double QdotB = dot(Bcon, Qcov);

Real vpar = QdotB / (sqrt(Bsq) * w_old * pow(Dtmp.ucon[0], 2.));

Real ucon_dr[GR_DIM] = {0};
ucon_dr[0] = 1. / sqrt(pow(Dtmp.ucon[0], -2.) + pow(vpar, 2.));
for (int mu = 1; mu < GR_DIM; mu++) {
ucon_dr[mu] = Dtmp.ucon[mu] * (ucon_dr[0] / Dtmp.ucon[0]) - (vpar * Bcon[mu] * ucon_dr[0] / sqrt(Bsq));
}

P(m_p.RHO, k, j, i) = m::max(rho, rhoflr_max);
P(m_p.UU, k, j, i)  = m::max(uu, uflr_max);
const Real pg_new   = (gam - 1.) * P(m_p.UU, k, j, i);
const Real w_new    = P(m_p.RHO, k, j, i) + P(m_p.UU, k, j, i) + pg_new;

const Real x = (2. * QdotB) / (sqrt(Bsq) * w_new * ucon_dr[0]);
vpar = x / (1 + sqrt(1 + x*x)) * (1. / ucon_dr[0]);

Dtmp.ucon[0] = 1. / sqrt(pow(ucon_dr[0], -2.) - pow(vpar, 2.));
for (int mu = 1; mu < GR_DIM; mu++) {
Dtmp.ucon[mu] = ucon_dr[mu] * (Dtmp.ucon[0] / ucon_dr[0]) + (vpar * Bcon[mu] * Dtmp.ucon[0] / sqrt(Bsq));
}
G.lower(Dtmp.ucon, Dtmp.ucov, k, j, i, Loci::center);

const Real gamma = Dtmp.ucon[0] * lapse;

P(m_p.U1, k, j, i) = Dtmp.ucon[1] + (beta[1] * gamma/lapse);
P(m_p.U2, k, j, i) = Dtmp.ucon[2] + (beta[2] * gamma/lapse);
P(m_p.U3, k, j, i) = Dtmp.ucon[3] + (beta[3] * gamma/lapse);

} else {
const Real rho_add = m::max(0., rhoflr_max - rho);
const Real u_add   = m::max(0., uflr_max - u);
const Real uvec[NVEC] = {0}, B[NVEC] = {0};

Real rho_ut, T[GR_DIM];
GRMHD::p_to_u_mhd(G, rho_add, u_add, uvec, B, gam, k, j, i, rho_ut, T, loc);

P(m_p.RHO, k, j, i) += rho_add;
P(m_p.UU, k, j, i)  += u_add;
U(m_u.RHO, k, j, i) += rho_ut;
U(m_u.UU, k, j, i)  += T[0]; 
U(m_u.U1, k, j, i)  += T[1];
U(m_u.U2, k, j, i)  += T[2];
U(m_u.U3, k, j, i)  += T[3];

pflag = GRMHD::u_to_p(G, U, m_u, gam, k, j, i, loc, P, m_p);
if (pflag) {
GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u, loc);
}
}
}

if (floors.adjust_k && (fflag & HIT_FLOOR_GEOM_RHO || fflag & HIT_FLOOR_B_RHO)) {
const Real reduce   = m::pow(rho / P(m_p.RHO, k, j, i), gam);
const Real reduce_e = m::pow(rho / P(m_p.RHO, k, j, i), 4./3); 
if (m_p.KTOT >= 0) P(m_p.KTOT, k, j, i) *= reduce;
if (m_p.K_CONSTANT >= 0) P(m_p.K_CONSTANT, k, j, i) *= reduce_e;
if (m_p.K_HOWES >= 0)    P(m_p.K_HOWES, k, j, i)    *= reduce_e;
if (m_p.K_KAWAZURA >= 0) P(m_p.K_KAWAZURA, k, j, i) *= reduce_e;
if (m_p.K_WERNER >= 0)   P(m_p.K_WERNER, k, j, i)   *= reduce_e;
if (m_p.K_ROWAN >= 0)    P(m_p.K_ROWAN, k, j, i)    *= reduce_e;
if (m_p.K_SHARMA >= 0)   P(m_p.K_SHARMA, k, j, i)   *= reduce_e;
}

return fflag + pflag;
}


template<typename Local>
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Local& P, const VarMap& m,
const Real& gam, const int& j, const int& i,
const Floors::Prescription& floors, const Loci loc=Loci::center)
{
Real rhoflr_geom, uflr_geom;
if(G.coords.spherical()) {
GReal Xembed[GR_DIM];
G.coord_embed(0, j, i, loc, Xembed);
GReal r = Xembed[1];

if (floors.use_r_char) {
Real rhoscal = m::pow(r, -2.) * 1 / (1 + r / floors.r_char);
rhoflr_geom = floors.rho_min_geom * rhoscal;
uflr_geom = floors.u_min_geom * m::pow(rhoscal, gam);
} else {
rhoflr_geom = floors.rho_min_geom * m::pow(r, -1.5);
uflr_geom = floors.u_min_geom * m::pow(r, -2.5); 
}
} else {
rhoflr_geom = floors.rho_min_geom;
uflr_geom = floors.u_min_geom;
}

int fflag = 0;
#if RECORD_POST_RECON
fflag |= (rhoflr_geom > P(m.RHO)) * HIT_FLOOR_GEOM_RHO_FLUX;
fflag |= (uflr_geom > P(m.UU)) * HIT_FLOOR_GEOM_U_FLUX;
#endif

P(m.RHO) += m::max(0., rhoflr_geom - P(m.RHO));
P(m.UU) += m::max(0., uflr_geom - P(m.UU));

return fflag;
}

template<typename Global>
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Global& P, const VarMap& m,
const Real& gam, const int& k, const int& j, const int& i,
const Floors::Prescription& floors, const Loci loc=Loci::center)
{
Real rhoflr_geom, uflr_geom;
if(G.coords.spherical()) {
GReal Xembed[GR_DIM];
G.coord_embed(k, j, i, loc, Xembed);
GReal r = Xembed[1];

if (floors.use_r_char) {
Real rhoscal = m::pow(r, -2.) * 1 / (1 + r / floors.r_char);
rhoflr_geom = floors.rho_min_geom * rhoscal;
uflr_geom = floors.u_min_geom * m::pow(rhoscal, gam);
} else {
rhoflr_geom = floors.rho_min_geom * m::pow(r, -1.5);
uflr_geom = floors.u_min_geom * m::pow(r, -2.5); 
}
} else {
rhoflr_geom = floors.rho_min_geom;
uflr_geom = floors.u_min_geom;
}

int fflag = 0;
#if RECORD_POST_RECON
fflag |= (rhoflr_geom > P(m.RHO, k, j, i)) * HIT_FLOOR_GEOM_RHO_FLUX;
fflag |= (uflr_geom > P(m.UU, k, j, i)) * HIT_FLOOR_GEOM_U_FLUX;
#endif

P(m.RHO, k, j, i) += m::max(0., rhoflr_geom - P(m.RHO, k, j, i));
P(m.UU, k, j, i) += m::max(0., uflr_geom - P(m.UU, k, j, i));

return fflag;
}


KOKKOS_INLINE_FUNCTION int apply_instability_limits(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
const Real& gam, const EMHD::EMHD_parameters& emhd_params, 
const int& k, const int& j, const int& i, const Floors::Prescription& floors,
const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
int eflag = 0;

Real rho      = P(m_p.RHO, k, j, i);
Real uu       = P(m_p.UU, k, j, i);
Real qtilde  = P(m_p.Q, k, j, i);
Real dPtilde = P(m_p.DP, k, j, i);

Real pg    = (gam - 1.) * uu;
Real Theta = pg / rho;
Real cs    = m::sqrt(gam * pg / (rho + (gam * uu)));

FourVectors D;
GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, D);
Real bsq = m::max(dot(D.bcon, D.bcov), SMALL);

Real tau, chi_e, nu_e;
EMHD::set_parameters(G, P, m_p, emhd_params, gam, k, j, i, tau, chi_e, nu_e, "instability_limits");

Real q, dP;
EMHD::convert_prims_to_q_dP(qtilde, dPtilde, rho, Theta, cs*cs, emhd_params, q, dP);

Real qmax         = 1.07 * rho * m::pow(cs, 3.);
Real max_frac     = m::max(m::abs(q) / qmax, 1.);
if (fabs(q) / qmax > 1.)
eflag |= HIT_Q_LIMIT;

P(m_p.Q, k, j, i) = P(m_p.Q, k, j, i) / max_frac;

Real dP_comp_ratio = m::max(pg - 2./3. * dP, SMALL) / m::max(pg + 1./3. * dP, SMALL);
Real dP_plus       = m::min(1.07 * 0.5 * bsq * dP_comp_ratio, 1.49 * pg);
Real dP_minus      = m::max(-1.07 * bsq, -2.99 * pg);

if (dP > 0. && (dP / dP_plus > 1.))
eflag |= HIT_DP_LIMIT;
else if (dP < 0. && (dP / dP_minus > 1.))
eflag |= HIT_DP_LIMIT;

if (dP > 0.)
P(m_p.DP, k, j, i) = P(m_p.DP, k, j, i) * (1. / m::max(dP / dP_plus, 1.));
else
P(m_p.DP, k, j, i) = P(m_p.DP, k, j, i) * (1. / m::max(dP / dP_minus, 1.));

Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);

return eflag;

}

} 
