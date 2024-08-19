
#pragma once

#include "decs.hpp"
#include "emhd.hpp"

using namespace std::literals::complex_literals;
using namespace parthenon;

#define STRLEN 2048



TaskStatus InitializeEMHDShock(MeshBlockData<Real> *rc, ParameterInput *pin)
{
Flag(rc, "Initializing EMHD shock problem");
auto pmb = rc->GetBlockPointer();

GridScalar rho  = rc->Get("prims.rho").data;
GridScalar u    = rc->Get("prims.u").data;
GridVector uvec = rc->Get("prims.uvec").data;
GridVector B_P  = rc->Get("prims.B").data;
GridVector q    = rc->Get("prims.q").data;
GridVector dP   = rc->Get("prims.dP").data;

const auto& G = pmb->coords;

const std::string input = pin->GetOrAddString("emhdshock", "input", "BVP");

const auto& emhd_pars                    = pmb->packages.Get("EMHD")->AllParams();
const EMHD::EMHD_parameters& emhd_params = emhd_pars.Get<EMHD::EMHD_parameters>("emhd_params");
const auto& grmhd_pars                   = pmb->packages.Get("GRMHD")->AllParams();
const Real& gam                          = grmhd_pars.Get<Real>("gamma");

IndexDomain domain = IndexDomain::interior;
IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

if (input == "BVP"){

char fbvp_rho[STRLEN], fbvp_u[STRLEN], fbvp_u1[STRLEN], fbvp_q[STRLEN], fbvp_dP[STRLEN];
sprintf(fbvp_rho, "shock_soln_rho.txt");
sprintf(fbvp_u,   "shock_soln_u.txt");
sprintf(fbvp_u1,  "shock_soln_u1.txt");
sprintf(fbvp_q,   "shock_soln_q.txt");
sprintf(fbvp_dP,  "shock_soln_dP.txt");

FILE *fp_rho, *fp_u, *fp_u1, *fp_q, *fp_dP;
fp_rho = fopen(fbvp_rho, "r");
fp_u   = fopen(fbvp_u,   "r");
fp_u1  = fopen(fbvp_u1,  "r");
fp_q   = fopen(fbvp_q,   "r");
fp_dP  = fopen(fbvp_dP,  "r");

auto rho_host   = rho.GetHostMirror();
auto u_host     = u.GetHostMirror();
auto uvec_host  = uvec.GetHostMirror();
auto B_host     = B_P.GetHostMirror();
auto q_host     = q.GetHostMirror();
auto dP_host    = dP.GetHostMirror();

for (int k = kb.s; k <= kb.e; k++) {
for (int j = jb.s; j <= jb.e; j++) {
for (int i = ib.s; i <= ib.e; i++) { 

Real X[GR_DIM];
G.coord_embed(k, j, i, Loci::center, X);

fscanf(fp_rho, "%lf", &(rho_host(k, j, i)));
fscanf(fp_u,   "%lf", &(u_host(k, j, i)));
fscanf(fp_u1,  "%lf", &(uvec_host(0, k, j, i)));
fscanf(fp_q,   "%lf", &(q_host(k, j, i)));
fscanf(fp_dP,  "%lf", &(dP_host(k, j, i)));

uvec_host(1, k, j, i) = 0.;
uvec_host(2, k, j, i) = 0.;
B_host(V1, k, j, i)  = 1.e-5;
B_host(V2, k, j, i)  = 0.;
B_host(V3, k, j, i)  = 0.;

if (emhd_params.higher_order_terms) {

const Real rho_temp   = rho_host(k, j, i);
const Real u_temp     = u_host(k, j, i);
const Real Theta      = (gam - 1.) * u_temp / rho_temp;

Real tau, chi_e, nu_e;
EMHD::set_parameters(G, rho_temp, u_temp, emhd_params, gam, k, j, i, tau, chi_e, nu_e);

Real q_tilde  = q_host(k, j, i);
Real dP_tilde = dP_host(k, j, i);
if (emhd_params.higher_order_terms) {
q_tilde  *= (chi_e != 0) ? m::sqrt(tau / (chi_e * rho_temp * m::pow(Theta, 2.))) : 0.;
dP_tilde *= (nu_e  != 0) ? m::sqrt(tau / (nu_e * rho_temp * Theta)) : 0.;
}
q_host(k, j, i)  = q_tilde;
dP_host(k, j, i) = dP_tilde;
}
}
}
}

fclose(fp_rho);
fclose(fp_u);
fclose(fp_u1);
fclose(fp_q);
fclose(fp_dP);

rho.DeepCopy(rho_host);
u.DeepCopy(u_host);
uvec.DeepCopy(uvec_host);
B_P.DeepCopy(B_host);
q.DeepCopy(q_host);
dP.DeepCopy(dP_host);
Kokkos::fence();

}

else {

const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
const Real x1max = pin->GetReal("parthenon/mesh", "x1max");

double rhoL = 1.,     rhoR = 3.08312999;
double uL   = 1.,     uR   = 4.94577705;
double u1L  = 1.,     u1R  = 0.32434571;
double u2L  = 0.,     u2R  = 0.;
double u3L  = 0.,     u3R  = 0.;
double B1L  = 1.e-5,  B1R  = 1.e-5;
double B2L  = 0,      B2R  = 0.;
double B3L  = 0.,     B3R  = 0.;

pmb->par_for("emhdshock_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
KOKKOS_LAMBDA_3D {

Real X[GR_DIM];
G.coord_embed(k, j, i, Loci::center, X);
const Real x1_center = (x1min + x1max) / 2.;

bool lhs = X[1] < x1_center;

rho(k, j, i)      = (lhs) ? rhoL : rhoR;
u(k, j, i)        = (lhs) ? uL : uR;
uvec(V1, k, j, i) = (lhs) ? u1L : u1R;
uvec(V2, k, j, i) = (lhs) ? u2L : u2R;
uvec(V3, k, j, i) = (lhs) ? u3L : u3R;
B_P(V1, k, j, i)  = (lhs) ? B1L : B1R;
B_P(V2, k, j, i)  = (lhs) ? B2L : B2R;
B_P(V3, k, j, i)  = (lhs) ? B3L : B3R;
q(k ,j, i)       = 0.;   
dP(k ,j, i)      = 0.;   

}

);
}

return TaskStatus::complete;

}