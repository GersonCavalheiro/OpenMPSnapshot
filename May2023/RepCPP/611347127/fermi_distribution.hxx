#pragma once

#include <cstdio> 
#include <cassert> 
#include <cmath> 
#include <vector> 

#include "inline_math.hxx" 
#include "simple_math.hxx" 
#include "display_units.h" 
#include "recorded_warnings.hxx" 
#include "simple_stats.hxx" 

#include "status.hxx" 

namespace fermi_distribution {

template <typename real_t> inline
real_t FermiDirac(real_t const x, real_t *derivative=nullptr) { 
if (std::abs(x) > 36) {
if (derivative) *derivative = 0;
return real_t(x < 0);
} 
real_t const d = 1 + std::exp(x); 
real_t const f = 1/d;
if (derivative) *derivative = (1 - d)*f*f;
return f;
} 

template <typename real_t> inline
real_t FermiDirac_broadening(real_t const x) {
real_t der;
FermiDirac(x, &der);
return -der;
} 

template <typename real_t>
double count_electrons(
int const n
, real_t const energies[]
, double const eF
, double const kTinv
, double const weights[]=nullptr
, double *derivative=nullptr
, double occupations[]=nullptr
, double ddeF_occupations[]=nullptr
) {
double const ddeF_x = -kTinv;
double ne{0}, ddx_ne{0};
for (int i = 0; i < n; ++i) {
double ddx_f;
double const E = energies[i];
double const x = (E - eF)*kTinv;
double const f = FermiDirac(x, &ddx_f);
double const w8 = weights ? weights[i] : 1;
ne     += f     *w8;
ddx_ne += ddx_f *w8;
if (occupations) occupations[i] = f;
if (ddeF_occupations) ddeF_occupations[i] = std::abs(ddx_f * ddeF_x);
} 
if (derivative) *derivative = std::abs(ddx_ne * ddeF_x); 
return ne;
} 

template <typename real_t>
double Fermi_level(
double occupations[]
, real_t const energies[] 
, double const weights[] 
, int const nb 
, double const kT
, double const number_of_electrons
, int const spinfactor=2 
, int const echo=9
, double response_occ[]=nullptr
, double *DoS_at_eF=nullptr
, double band_energy[2]=nullptr 
) {
if (nb < 1) return 0;


double nstates{0};
double e[2] = {9e99, -9e99};
for (int ib = 0; ib < nb; ++ib) {
double const energy = energies[ib];
e[0] = std::min(e[0], energy);
e[1] = std::max(e[1], energy);
nstates += weights ? weights[ib] : 1;
} 
e[0] -= 9*kT + .5;
e[1] += 9*kT + .5;

if (echo > 6) std::printf("# %s search energy between %g and %g %s, host %g electrons in %g states\n", 
__func__, e[0]*eV, e[1]*eV, _eV, number_of_electrons, nstates);
if (number_of_electrons > nstates*spinfactor) {
warn("unable to host %g electrons in %g states (spin factor is %d)", number_of_electrons, nstates, spinfactor);
}

double const n_electrons = number_of_electrons/spinfactor;
double const kTinv = 1./std::max(1e-9, kT);

double res{1}; 
int const maxiter = 99, miniter = 5;
int iter{0}; 
while((iter < maxiter) && ((res > 2e-15) || (std::abs(e[1] - e[0]) > 1e-9) || (iter < miniter))) {
++iter;
auto const em = 0.5*(e[0] + e[1]);
auto const nem = count_electrons(nb, energies, em, kTinv, weights);
res = std::abs(nem - n_electrons); 
if (echo > 7) {
auto const ene = std::abs((e[1] - e[0])*eV); 
int const ndigits_energy = std::max(1, int(-std::log10(std::max(1e-15, ene))) - 1);
int const ndigits_charge = std::max(1, int(-std::log10(std::max(1e-15, res))) - 1);
char fmt[64]; std::snprintf(fmt, 63, "# %s with energy %%.%df %s --> %%.%df electrons\n",
__func__, ndigits_energy, _eV, ndigits_charge);
std::printf(fmt, em*eV, spinfactor*nem);
std::fflush(stdout);
} 
int const i01 = (nem > n_electrons);
e[i01] = em; 
} 

double const eF = 0.5*(e[0] + e[1]);
if (echo > 19) std::printf("# %s after convergence energy %g %s\n", __func__, eF*eV, _eV);
double DoS{0}; 
auto const nem = count_electrons(nb, energies, eF, kTinv, weights, &DoS, occupations, response_occ);
if (echo > 6) std::printf("# %s with energy %g %s --> %g electrons\n", __func__, eF*eV, _eV, spinfactor*nem);
if (res > 1e-9) {
warn("Fermi level converged only to +/- %.1e electrons in %d iterations", res*spinfactor, iter); 
} else {
if (echo > 3) std::printf("# %s at %.9f %s has a DoS of %g states\n", __func__, eF*eV, _eV, DoS*kT*spinfactor);
}
if (DoS_at_eF) *DoS_at_eF = DoS;
return eF;
} 

inline status_t density_of_states(int const n, double const energies[], double const kT, double const eF=0, double const de=3e-4, int const echo=9) {
if (n < 1) return 0;
double const kTinv = 1./std::max(1e-9, kT);
simple_stats::Stats<double> stats;
for (int i = 0; i < n; ++i) {
stats.add(energies[i] - eF);
} 
double const e_min = stats.min(), e_max = stats.max();
if (echo > 4) std::printf("# %s E_min= %g E_max= %g %s\n", __func__, e_min*eV, e_max*eV, _eV);

double const per_eV = 1./eV;
if (echo > 8) std::printf("\n## energy(%s), DoS, dDoS/dE_Fermi\n", _eV);
double integral{0};
for (int ie = (e_min - 18*kT)/de; ie <= (e_max + 18*kT)/de; ++ie) {
double const e = ie*de;
double dos{0}, ddos{0};
for (int i = 0; i < n; ++i) {
auto const ei = energies[i] - eF;
dos += FermiDirac_broadening((e - ei)*kTinv);
} 
if (echo > 8) std::printf("%g %g %g\n", e*eV, dos*kTinv*per_eV, ddos);
integral += dos*FermiDirac(e*kTinv);
} 
if (echo > 6) std::printf("# integrated DoS is %g\n", integral*kTinv*de);
return 0;
} 


class FermiLevel_t { 
public:
static double constexpr Fermi_level_not_initialized = -9e99;
static double constexpr default_temperature = 9.765625e-4; 

FermiLevel_t( 
double const n_electrons=0.0
, int const spinfactor=2
, double const kT=default_temperature
, int const echo=0 
)
: _ne(std::max(0.0, n_electrons))
, _mu(Fermi_level_not_initialized)
, _kT(kT)
, _spinfactor(std::min(std::max(1, spinfactor), 2))
{
if (echo > 0) std::printf("\n# new %s(%g electrons, kT=%g %s)\n", __func__, _ne, kT*Kelvin, _Kelvin);
set_temperature(kT, echo);
set(_band_energy, 2, 0.0);
_accu.set();
} 

bool set_temperature(double const kT=default_temperature, int const echo=0) {
_kT = std::max(minimum_temperature(), kT);
if (_kT > 0) _kTinv = 1./_kT;
if (echo > 0) std::printf("# FermiLevel %s(%g Ha) --> kT= %g %s = %g m%s\n", __func__, kT, _kT*Kelvin, _Kelvin, _kT*eV*1000, _eV);
return (kT >= 0);
} 

double get_temperature() const { return _kT; }
double get_n_electrons() const { return _ne; }
int    get_spinfactor()  const { return _spinfactor; }

inline double minimum_temperature() const { return 1e-9; }

inline bool is_initialized() const { return (Fermi_level_not_initialized != _mu); }

void set_Fermi_level(double const E_Fermi=Fermi_level_not_initialized, int const echo=0) {
_mu = E_Fermi;
if (Fermi_level_not_initialized != _mu) {
if (echo > 0) std::printf("# set FermiLevel to %.9f %s\n", _mu*eV,_eV);
} else {
if (echo > 0) std::printf("# reset FermiLevel\n");
}
} 

void add(double const E_Fermi, double const weight=1, int const echo=0) { _accu.add(E_Fermi, std::max(weight, 0.0)); }

template <typename real_t>
double get_occupations( 
double occupations[] 
, real_t const energies[] 
, int const nbands 
, double const kpoint_weight 
, int const echo=9
, double response_occ[]=nullptr 
) {
double eF{0}, DoS{0}; 
if (Fermi_level_not_initialized == _mu) {
if (echo > 0) std::printf("# FermiLevel %s for %d bands, FermiLevel has not been initialized\n", __func__, nbands);
std::vector<double> weights(nbands, kpoint_weight);
eF = Fermi_level(occupations, energies, weights.data(), nbands, _kT, _ne, _spinfactor, echo, response_occ, &DoS);
} else {
if (echo > 0) std::printf("# FermiLevel %s for %d bands at %g %s\n", __func__, nbands, _mu*eV,_eV);
count_electrons(nbands, energies, _mu, _kTinv, nullptr, &DoS, occupations, response_occ);
eF = _mu;
} 
_band_energy[0] += _spinfactor * kpoint_weight * dot_product(nbands, energies, occupations);
_band_energy[1] += _spinfactor * kpoint_weight * dot_product(nbands, energies, response_occ);
add(eF, kpoint_weight, echo); 
return DoS;
} 

double get_Fermi_level() const { return _mu; }
double get_band_sum(int const i=0) const { return _band_energy[i & 1]; }

double correct_Fermi_level( 
double const charges[] 
, int const echo=0 
) {
auto const old_mu = _mu;
_mu = _accu.mean(); 
if (echo > 0) std::printf("# %s old= %g, new= %g %s\n", __func__,
old_mu*(Fermi_level_not_initialized != old_mu)*eV, _mu*eV,_eV);
_accu.set(); 
set(_band_energy, 2, 0.0);
return 0; 
} 

private:

double _ne; 
double _mu; 
double _kT; 
double _kTinv;
double _band_energy[2]; 
simple_stats::Stats<double> _accu; 
int _spinfactor; 

}; 







#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_integration(int const echo=3, int const n=580, double const de=.0625) {
if (echo > 6) std::printf("\n## plot Fermi-Dirac distribution (%d points)\n", n);
double s0{0}, s1{0};
for (int ie = -n; ie < n; ++ie) {
double const e = (ie + .5)*de;
double dfde;
double const f = FermiDirac(e, &dfde);
if (echo > 6) std::printf("%g %g %g\n", e, f, -dfde);
s0 += f;
s1 -= dfde;
} 
double const dev[] = {s0/n - 1, s1*de - 1};
if (echo > 2) std::printf("# Fermi-Dirac distribution deviation %.1e %.1e\n", dev[0], dev[1]);
return (std::abs(dev[0]) > 2e-15) + (std::abs(dev[1]) > 1e-15);
} 

inline status_t test_bisection(int const echo=3, int const n=99) {
if (echo > 8) std::printf("\n## %s %s (%d states)\n", __FILE__, __func__, n);
std::vector<double> ene(n), wgt(n, 1);
for (int i = 0; i < n; ++i) ene[i] = simple_math::random(-10., 10.);
double const eF = Fermi_level(nullptr, ene.data(), wgt.data(), n, 3e-2, n*.75, 1, echo);
if (echo > 8) density_of_states(n, ene.data(), 3e-2, eF);
return 0;
} 

inline status_t test_FermiLevel_class(int const echo=0) {
if (echo > 6) std::printf("\n# %s %s\n", __FILE__, __func__);
auto const n_electrons = 12.;
FermiLevel_t mu(n_electrons);
mu.set_temperature(.01, echo);
for (double eF = -1.5; eF < 2; eF += 0.25) {
mu.add(eF, 1, echo - 5);
} 
mu.set_Fermi_level(mu.get_Fermi_level(), echo);
int const nb = n_electrons*1.5; 
std::vector<double> ene(nb, 0.0), occ(nb), docc(nb);
for (int ib = 0; ib < nb; ++ib) ene[ib] = ib*.01;
mu.get_occupations(occ.data(), ene.data(), nb, 1, echo, docc.data());
double sum_occ{0};
for (int ib = 0; ib < nb; ++ib) {
if (echo > 9) std::printf("# %s: band #%d \tenergy= %.6f %s \toccupation= %.6f \td/dE occ= %.3f\n",
__func__, ib, ene[ib]*eV,_eV, occ[ib], docc[ib]);
sum_occ += occ[ib];
} 
if (echo > 6) std::printf("# %s %s eF= %g %s for %g electrons, sum(occupations)= %g\n",
__FILE__, __func__, mu.get_Fermi_level()*eV,_eV, n_electrons, sum_occ);
mu.get_occupations(occ.data(), ene.data(), nb, 1, echo, docc.data());
return 0;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s %s\n", __FILE__, __func__);
status_t status(0);
status += test_integration(echo);
status += test_bisection(echo);
status += test_FermiLevel_class(echo);
return status;
} 

#endif 

} 
