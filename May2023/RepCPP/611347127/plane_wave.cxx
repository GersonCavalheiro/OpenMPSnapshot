
#include <cstdio> 
#include <cmath> 
#include <algorithm> 
#include <complex> 
#include <vector> 
#include <cassert> 
#include <set> 
#include <numeric> 
#include <cstdint> 

#include "plane_wave.hxx" 

#include "sho_potential.hxx" 
#include "geometry_analysis.hxx" 
#include "control.hxx" 
#include "display_units.h" 
#include "real_space.hxx" 
#include "sho_tools.hxx" 
#include "sho_projection.hxx" 
#include "boundary_condition.hxx" 
#include "sho_overlap.hxx" 
#include "data_view.hxx" 
#include "data_list.hxx" 
#include "inline_math.hxx" 
#include "simple_math.hxx" 
#include "vector_math.hxx" 
#include "hermite_polynomial.hxx" 
#include "simple_stats.hxx" 
#include "simple_timer.hxx" 
#include "dense_solver.hxx" 
#include "unit_system.hxx" 
#include "fourier_transform.hxx" 

#include "davidson_solver.hxx" 
#include "conjugate_gradients.hxx" 
#include "dense_operator.hxx" 
#include "brillouin_zone.hxx" 

#ifdef DEVEL
#include "print_tools.hxx" 
#endif 

namespace plane_wave {

class PlaneWave {
public:
double g2; 
int16_t x, y, z;
PlaneWave() : g2(0), x(0), y(0), z(0) {}
PlaneWave(int const ix, int const iy, int const iz, double const len2) : g2(len2), x(ix), y(iy), z(iz) {}
private:
}; 

void Hermite_Gauss_projectors(std::complex<double> pzyx[], int const numax, double const sigma, double const gv[3]) {

view2D<double> Hermite_Gauss(3, sho_tools::n1HO(numax));
for (int d = 0; d < 3; ++d) {
double const x = gv[d]*sigma; 
Gauss_Hermite_polynomials(Hermite_Gauss[d], x, numax); 
} 

{ 
double nufactorial{1};
for (int nu = 0; nu <= numax; ++nu) {
double const norm_factor = std::sqrt(sigma/(constants::sqrtpi*nufactorial));
for (int d = 0; d < 3; ++d) {
Hermite_Gauss(d,nu) *= norm_factor; 
} 
nufactorial *= (nu + 1)*0.5; 
} 
} 

std::complex<double> constexpr mi(0, -1);
std::complex<double> const imaginary[4] = {1, mi, -1, -mi}; 

int lb{0};
for (        int lz = 0; lz <= numax;           ++lz) {
for (    int ly = 0; ly <= numax - lz;      ++ly) {
for (int lx = 0; lx <= numax - lz - ly; ++lx) {
int const nu = lx + ly + lz;
pzyx[lb] = Hermite_Gauss(0,lx) *
Hermite_Gauss(1,ly) *
Hermite_Gauss(2,lz) *
imaginary[nu & 0x3]; 
++lb;
} 
} 
} 
assert( sho_tools::nSHO(numax) == lb );

} 


inline double Fourier_Gauss_factor_3D() {
return pow3(constants::sqrt2 * constants::sqrtpi);
} 


template <typename complex_t>
status_t iterative_solve(
double eigenenergies[]
, view3D<complex_t> const & HSm
, char const *x_axis=""
, int const echo=0
, int const nbands=10
, float const direct_ratio=2
) {
status_t stat(0);
int constexpr H=0, S=1;

int const nPW  = HSm.dim1();
int const nPWa = HSm.stride();

if (echo > 6) {
std::printf("# %s for the lowest %d bands of a %d x %d Hamiltonian (stride %d)\n",
__func__, nbands, nPW, nPW, nPWa);
std::fflush(stdout);
} 

view2D<complex_t> waves(nbands, nPW, complex_t(0)); 

if (nbands > nPW) {
warn("tried to find %d bands in a basis set with %d plane waves", nbands, nPW);
return -1;
} 

{ 
int const nsubspace = std::max(nbands, int(direct_ratio*nbands));
int const nsub = std::min(nsubspace, nPW);
view3D<complex_t> SHmat_b(2, nsub, align<2>(nsub));
for (int ib = 0; ib < nsub; ++ib) {
set(SHmat_b(S,ib), nsub, HSm(S,ib));
set(SHmat_b(H,ib), nsub, HSm(H,ib));
} 

if (echo > 6) { std::printf("# %s get %d start waves from diagonalization of a %d x %d Hamiltonian (stride %lu)\n",
__func__, nbands, nsub, nsub, SHmat_b.stride()); std::fflush(stdout); }
auto const stat_eig = dense_solver::solve(SHmat_b, "# start waves "); 
if (stat_eig != 0) {
warn("diagonalization of the %d x %d sub-Hamiltonian returned status= %i", nsub, nsub, int(stat_eig));
return stat_eig;
} 
stat += stat_eig;

if (echo > 6) { std::printf("# %s copy %d start waves from eigenstates of the %d x %d Hamiltonian (stride %lu)\n",
__func__, nbands, nsub, nsub, SHmat_b.stride()); std::fflush(stdout); }
for (int ib = 0; ib < nbands; ++ib) {
set(waves[ib], nsub, SHmat_b(H,ib));
} 

} 

std::vector<complex_t> precond(nPWa, complex_t(0));
{
double diag_min{9e99}; int imin{-1};
for (int iB = 0; iB < nPW; ++iB) {
double const diag = std::real(HSm(H,iB,iB));
if (diag < diag_min) imin = iB;
diag_min = std::min(diag_min, diag);
} 
if (echo > 6) { std::printf("# %s smallest diagonal element is %g %s at index #%i of %d\n",
__func__, diag_min*eV,_eV, imin, nPW); std::fflush(stdout); }
complex_t const diag_shift = diag_min - 1.0; 
for (int iB = 0; iB < nPW; ++iB) {
precond[iB] = complex_t(1)/(HSm(H,iB,iB) - diag_shift);
} 
} 

if (echo > 6) { std::printf("# %s construct a dense operator for the %d x %d Hamiltonian (stride %d)\n",
__func__, nPW, nPW, nPWa); std::fflush(stdout); }
dense_operator::dense_operator_t<complex_t> const op(nPW, nPWa, HSm(H,0), HSm(S,0));

char const method = *control::get("plane_wave.iterative.solver", "Davidson") | 32;

std::vector<double> eigvals(nbands);
status_t stat_slv(0);
if ('c' == method) {
int const nit = control::get("plane_wave.max.cg.iterations", 3.);
if (echo > 6) { std::printf("# %s envoke CG solver with max. %d iterations\n", __func__, nit); std::fflush(stdout); }
for (int it = 0; it < nit && (0 == stat_slv); ++it) {
if (echo > 6) { std::printf("# %s envoke CG solver, outer iteration #%i\n", __func__, it); std::fflush(stdout); }
stat_slv = conjugate_gradients::eigensolve(waves.data(), eigvals.data(), nbands, op, echo - 10);
stat += stat_slv;
davidson_solver::rotate(waves.data(), eigvals.data(), nbands, op, echo - 10);
} 
} else { 
int const nit = control::get("davidson_solver.max.iterations", 1.);
if (echo > 6) { std::printf("# %s envoke Davidson solver with max. %d iterations\n", __func__, nit); std::fflush(stdout); }
for (int it = 0; it < nit && (0 == stat_slv); ++it) {
stat_slv = davidson_solver::eigensolve(waves.data(), eigvals.data(), nbands, op, echo - 10, 2.0f, 2);
stat += stat_slv;
} 
} 

if (0 == stat_slv) {
if (echo > 2) {
dense_solver::display_spectrum(eigvals.data(), nbands, x_axis, eV, _eV);
if (echo > 4) std::printf("# lowest and highest eigenvalue of the Hamiltonian matrix is %g and %g %s, respectively\n",
eigvals[0]*eV, eigvals[nbands - 1]*eV, _eV);
std::fflush(stdout);
} 
} else {
warn("Davidson solver for the plane wave Hamiltonian failed with status= %i", int(stat_slv));
} 

set(eigenenergies, nbands, eigvals.data()); 

return stat;
} 



template <typename complex_t>
status_t solve_k(
double const ecut 
, double const reci[3][4] 
, view4D<double> const & Vcoeff 
, int const nG[3] 
, double const norm_factor 

, int const natoms_PAW 
, view2D<double> const & xyzZ_PAW 
, double const grid_offset[3] 
, int    const numax_PAW[] 
, double const sigma_PAW[] 
, view3D<double> const hs_PAW[] 

, double const kpoint[4] 
, char const *const x_axis 
, int & nPWs 
, int const echo=0 
, int const nbands=10 
, float const direct_ratio=2.f 
, DensityIngredients *export_rho=nullptr
, int const kpoint_id=-1 
) { 

using real_t = decltype(std::real(complex_t(1)));

int boxdim[3], max_PWs{1};
for (int d = 0; d < 3; ++d) {
boxdim[d] = std::ceil(ecut/pow2(reci[d][d]));
if (echo > 17) std::printf("# %s reci[%d] = %.6f %.6f %.6f sqRy, kpoint=%9.6f\n", __func__, d, reci[d][0], reci[d][1], reci[d][2], kpoint[d]);
max_PWs *= (boxdim[d] + 1 + boxdim[d]);
} 
if (echo > 11) std::printf("# %s boxdim = %d x %d x %d, E_cut= %g Ry\n", __func__, boxdim[0], boxdim[1], boxdim[2], 2*ecut);
std::vector<PlaneWave> pw_basis(max_PWs);
{ 
int iB{0}, outside{0};
for (int iGz = -boxdim[2]; iGz <= boxdim[2]; ++iGz) {
for (int iGy = -boxdim[1]; iGy <= boxdim[1]; ++iGy) {
for (int iGx = -boxdim[0]; iGx <= boxdim[0]; ++iGx) {
double const gv[3] = {(iGx + kpoint[0])*reci[0][0],
(iGy + kpoint[1])*reci[1][1],
(iGz + kpoint[2])*reci[2][2]}; 
double const g2 = pow2(gv[0]) + pow2(gv[1]) + pow2(gv[2]);
#ifdef FULL_DEBUG
if (echo > 91) std::printf("# %s suggest PlaneWave(%d,%d,%d) with |k+G|^2= %g Ry inside= %d\n", __func__, iGx,iGy,iGz, g2, g2 < 2*ecut);
#endif 
if (g2 < 2*ecut) {
pw_basis[iB] = PlaneWave(iGx, iGy, iGz, g2);
++iB; 
} else {
++outside;
} 
} 
} 
} 
int const number_of_PWs = iB;
assert( number_of_PWs + outside == max_PWs );
pw_basis.resize(number_of_PWs); 
} 
int const nB = pw_basis.size();
nPWs = nB; 
#ifdef DEVEL
if (echo > 6) {
auto const time = 1e-9*pow3(nPWs) + 2e-6*pow2(nPWs); 
char const *_seconds="seconds"; auto seconds=1.;
if (time > 3600) { _seconds="hours"; seconds=1/3600.; } else
if (time > 180) { _seconds="minutes"; seconds=1/60.; }
std::printf("\n# start %s<%s> Ecut= %g %s nPW=%d (est. %.1f %s)\n",
__func__, complex_name<complex_t>(), ecut*eV, _eV, nB, time*seconds, _seconds);
std::fflush(stdout);
} 
#endif 

bool constexpr sort_PWs = true;
if (sort_PWs) {
auto compare_lambda = [](PlaneWave const & lhs, PlaneWave const & rhs) { return lhs.g2 < rhs.g2; };
std::sort(pw_basis.begin(), pw_basis.end(), compare_lambda);
#ifdef DEVEL
if (echo > 8) { 
std::printf("# %s|k+G|^2 =", x_axis);
for (int i = 0; i < std::min(5, nPWs); ++i) {
std::printf(" %.4f", pw_basis[i].g2);
} 
std::printf(" ...");
for (int i = -std::min(2, nPWs); i < 0; ++i) {
std::printf(" %.3f", pw_basis[nPWs + i].g2);
} 
std::printf(" Ry\n");
} 
#endif 
} 


int constexpr H=0, S=1; 

std::vector<uint32_t> offset(natoms_PAW + 1); 
offset[0] = 0;
for (int ka = 0; ka < natoms_PAW; ++ka) {
int const nSHO = sho_tools::nSHO(numax_PAW[ka]);
assert( hs_PAW[ka].stride() >= nSHO );
assert( hs_PAW[ka].dim1()   == nSHO );
offset[ka + 1] = offset[ka] + nSHO;
} 
int const nC = offset[natoms_PAW]; 


view2D<complex_t> P_jl(nB, nC, 0.0); 
#ifdef DEVEL
std::vector<double>   P2_l(nC, 0.0); 
#endif 
view3D<complex_t> Psh_il(2, nB, nC, 0.0); 
for (int jB = 0; jB < nB; ++jB) {
auto const & pw = pw_basis[jB];
double const gv[3] = {(pw.x + kpoint[0])*reci[0][0],
(pw.y + kpoint[1])*reci[1][1],
(pw.z + kpoint[2])*reci[2][2]}; 

for (int ka = 0; ka < natoms_PAW; ++ka) {
int const nSHO = sho_tools::nSHO(numax_PAW[ka]);

double const arg = -(  (grid_offset[0] + xyzZ_PAW(ka,0))*gv[0]
+ (grid_offset[1] + xyzZ_PAW(ka,1))*gv[1]
+ (grid_offset[2] + xyzZ_PAW(ka,2))*gv[2] );
std::complex<double> const phase(std::cos(arg), std::sin(arg));

{ 
std::vector<std::complex<double>> pzyx(nSHO);
Hermite_Gauss_projectors(pzyx.data(), numax_PAW[ka], sigma_PAW[ka], gv);
auto const fgf = Fourier_Gauss_factor_3D();

for (int lb = 0; lb < nSHO; ++lb) {
int const lC = offset[ka] + lb;
P_jl(jB,lC) = complex_t(phase * pzyx[lb] * fgf * norm_factor);
#ifdef DEVEL
P2_l[lC] += std::norm(pzyx[lb]);
#endif 
} 
} 

auto const iB = jB;
for (int lb = 0; lb < nSHO; ++lb) {
int const iC = offset[ka] + lb; 
complex_t s(0), h(0);
for (int kb = 0; kb < nSHO; ++kb) { 
int const kC = offset[ka] + kb;
auto const p = P_jl(iB,kC); 
h += p * real_t(hs_PAW[ka](H,kb,lb));
s += p * real_t(hs_PAW[ka](S,kb,lb));
} 
Psh_il(H,iB,iC) = h;
Psh_il(S,iB,iC) = s;
} 
} 

} 

#ifdef DEVEL
if (echo > 29) {
for (int la = 0; la < natoms_PAW; ++la) {
std::printf("# ^2-norm of the projector of atom #%i ", la);
int const nSHO = sho_tools::nSHO(numax_PAW[la]);
printf_vector(" %g", &P2_l[offset[la]], nSHO);
std::fflush(stdout);
} 
} 
#endif 


int const nBa = align<4>(nB); 
view3D<complex_t> HSm(2, nB, nBa, complex_t(0)); 

#ifdef DEVEL
if (echo > 9) std::printf("# assume dimensions of Vcoeff(%d, %ld, %ld, %ld)\n", 2, Vcoeff.dim2(), Vcoeff.dim1(), Vcoeff.stride());
assert( nG[0] <= Vcoeff.stride() );
assert( nG[1] <= Vcoeff.dim1() );
assert( nG[2] <= Vcoeff.dim2() );

double const scale_k = control::get("hamiltonian.scale.kinetic", 1.);
double const scale_p = control::get("hamiltonian.scale.potential", 1.);
if (1 != scale_k) warn("kinetic energy is scaled by %g", scale_k);
if (1 != scale_p) warn("local potential is scaled by %g", scale_p);
real_t const scale_h = control::get("hamiltonian.scale.nonlocal.h", 1.);
real_t const scale_s = control::get("hamiltonian.scale.nonlocal.s", 1.);
if (1 != scale_h || 1 != scale_s) warn("scale PAW contributions to H and S by %g and %g, respectively", scale_h, scale_s);
#else  
real_t constexpr scale_h = 1, scale_s = 1;
double constexpr scale_k = 1, scale_p = 1;
#endif 
double const kinetic = 0.5 * scale_k; 
real_t const localpot = scale_p / (nG[0]*nG[1]*nG[2]);

for (int iB = 0; iB < nB; ++iB) {
auto const & i = pw_basis[iB];

HSm(S,iB,iB) += 1; 
HSm(H,iB,iB) += kinetic*i.g2; 

for (int jB = 0; jB < nB; ++jB) {
auto const & j = pw_basis[jB];

int const iVx = i.x - j.x, iVy = i.y - j.y, iVz = i.z - j.z;
if ((2*std::abs(iVx) < nG[0]) && (2*std::abs(iVy) < nG[1]) && (2*std::abs(iVz) < nG[2])) {
int const imz = (iVz + nG[2])%nG[2], imy = (iVy + nG[1])%nG[1], imx = (iVx + nG[0])%nG[0];
auto const V_re = Vcoeff(0, imz, imy, imx), V_im = Vcoeff(1, imz, imy, imx);
HSm(H,iB,jB) += localpot*std::complex<real_t>(V_re, V_im);
} 


{ 
complex_t h(0), s(0);
for (int lC = 0; lC < nC; ++lC) { 
auto const p = conjugate(P_jl(jB,lC)); 
h += Psh_il(H,iB,lC) * p;
s += Psh_il(S,iB,lC) * p;
} 
HSm(H,iB,jB) += h * scale_h;
HSm(S,iB,jB) += s * scale_s;
} 

} 
} 

int  const nB_auto = control::get("plane_wave.dense.solver.below", 999.);
char const solver = *control::get("plane_wave.solver", "auto") | 32; 
bool const run_solver[2] = {('i' == solver) || ('b' == solver) || (('a' == solver) && (nB >  nB_auto)),
('d' == solver) || ('b' == solver) || (('a' == solver) && (nB <= nB_auto))};
status_t solver_stat(0);
std::vector<double> eigenenergies(nbands, -9e9);
{ SimpleTimer timer("plane wave solver", __LINE__, __func__, echo);
if (run_solver[0]) solver_stat += iterative_solve(eigenenergies.data(), HSm, x_axis, echo, nbands, direct_ratio); 
if (run_solver[1]) solver_stat += dense_solver::solve(HSm, x_axis, echo, nbands, eigenenergies.data());
} 

if (export_rho && (run_solver[0] || run_solver[1])) { 
double const kpoint_weight = kpoint[3];
export_rho->constructor(nG, nbands, natoms_PAW, nC, kpoint_weight, kpoint_id, echo);
export_rho->energies.assign(eigenenergies.begin(), eigenenergies.begin() + nbands);
export_rho->offset.assign(offset.begin(), offset.end());
assert(export_rho->offset.size() == export_rho->natoms + 1);

auto const nG_all = size_t(nG[2])*size_t(nG[1])*size_t(nG[0]);

std::complex<double> const zero(0);
view3D<std::complex<double>> psi_G(nG[2], nG[1], nG[0]); 

for (int iband = 0; iband < nbands; ++iband) {
auto const atom_coeff = export_rho->coeff[iband];
set(atom_coeff, nC, zero);
set(psi_G.data(), nG_all, zero);
for (int iB = 0; iB < nB; ++iB) {
std::complex<double> const eigenvector_coeff = HSm(H,iband,iB);
auto const & i = pw_basis[iB];
int const iGx = (i.x + nG[0])%nG[0], iGy = (i.y + nG[1])%nG[1], iGz = (i.z + nG[2])%nG[2];
psi_G(iGz,iGy,iGx) = eigenvector_coeff * norm_factor;
for (int iC = 0; iC < nC; ++iC) {
atom_coeff[iC] += std::complex<double>(P_jl(iB,iC)) * eigenvector_coeff;
} 
} 

auto const fft_stat = fourier_transform::fft(export_rho->psi_r[iband], psi_G.data(), nG, false, echo);
if (int(fft_stat) != 0 && echo > 1) std::printf("# Fourier transform for band #%i returned status= %i\n", iband, int(fft_stat));

} 
} 

return solver_stat;
} 







status_t solve(
int const natoms_PAW 
, view2D<double> const & xyzZ 
, real_space::grid_t const & g 
, double const *const vtot 
, double const *const sigma_prj 
, int    const *const numax_prj 
, double *const *const atom_mat 
, int const echo 
, std::vector<DensityIngredients> *export_rho 
) {

status_t stat(0);

for (int ia = 0; ia < natoms_PAW; ++ia) {
if (echo > 0) {
std::printf("# atom#%i \tZ=%g \tposition %12.6f%12.6f%12.6f %s\n",
ia, xyzZ(ia,3), xyzZ(ia,0)*Ang, xyzZ(ia,1)*Ang, xyzZ(ia,2)*Ang, _Ang);
} 
} 

double const grid_offset[3] = {0.5*(g[0] - 1)*g.h[0], 
0.5*(g[1] - 1)*g.h[1],
0.5*(g[2] - 1)*g.h[2]};
double const cell_matrix[3][4] = {{g[0]*g.h[0], 0, 0,   0},
{0, g[1]*g.h[1], 0,   0},
{0, 0, g[2]*g.h[2],   0}};
double reci_matrix[3][4];
auto const cell_volume = simple_math::invert(3, reci_matrix[0], 4, cell_matrix[0], 4);
scale(reci_matrix[0], 3*4, 2*constants::pi); 
if (echo > 0) {
std::printf("# cell volume is %g %s^3\n", cell_volume*pow3(Ang),_Ang);
auto constexpr sqRy = 1; auto const _sqRy = "sqRy";
std::printf("# cell matrix in %s\t\tand\t\treciprocal matrix in %s:\n", _Ang, _sqRy);
for (int d = 0; d < 3; ++d) {
std::printf("# %12.6f%12.6f%12.6f    \t%12.6f%12.6f%12.6f\n",
cell_matrix[d][0]*Ang,  cell_matrix[d][1]*Ang,  cell_matrix[d][2]*Ang,
reci_matrix[d][0]*sqRy, reci_matrix[d][1]*sqRy, reci_matrix[d][2]*sqRy);
} 
std::printf("\n");
} 
stat += (std::abs(cell_volume) < 1e-9);
double const svol = 1./std::sqrt(cell_volume); 
if (echo > 1) std::printf("# normalization factor for plane waves is %g a.u.\n", svol);

char const *_ecut_u{nullptr};
auto const ecut_u = unit_system::energy_unit(control::get("plane_wave.cutoff.energy.unit", "Ha"), &_ecut_u);
auto const ecut = control::get("plane_wave.cutoff.energy", 11.)/ecut_u; 
if (echo > 1) {
auto const kcut = std::sqrt(2*ecut); 
std::printf("# plane_wave.cutoff.energy=%.3f %s corresponds to %.3f^2 Ry or %.2f %s\n",
ecut*ecut_u, _ecut_u, kcut, ecut*eV,_eV);
auto const reci_volume = pow3(2*constants::pi)/cell_volume; 
auto const pw_ball = 4*constants::pi/3*pow3(kcut); 
std::printf("# estimated number of plane waves is %.1f\n", pw_ball/reci_volume);
} 


int const nG[3] = {g[0], g[1], g[2]}; 
view4D<double> Vcoeffs(2, nG[2], nG[1], nG[0], 0.0);
view3D<double> vtot_Im(   nG[2], nG[1], nG[0], 0.0);

auto const fft_stat = fourier_transform::fft(Vcoeffs(0,0,0), Vcoeffs(1,0,0), vtot, vtot_Im(0,0), nG, true, echo);
if (0 == fft_stat) {
if (echo > 5) std::printf("# used FFT on %d x %d x %d to transform local potential into space of plane-wave differences\n", nG[0], nG[1], nG[2]);
} else
{ 
warn("FFT failed with status= %i, transform local potential manually", int(fft_stat));
SimpleTimer timer(__FILE__, __LINE__, "manual Fourier transform (not FFT)");
double const two_pi = 2*constants::pi;
double const tpi_g[] = {two_pi/g[0], two_pi/g[1], two_pi/g[2]};
for (int iGz = 0; iGz < nG[2]; ++iGz) {  auto const Gz = iGz*tpi_g[2];
for (int iGy = 0; iGy < nG[1]; ++iGy) {  auto const Gy = iGy*tpi_g[1];
for (int iGx = 0; iGx < nG[0]; ++iGx) {  auto const Gx = iGx*tpi_g[0];
double re{0}, im{0};
for (int iz = 0; iz < g[2]; ++iz) {
for (int iy = 0; iy < g[1]; ++iy) {
for (int ix = 0; ix < g[0]; ++ix) {
double const arg = -(Gz*iz + Gy*iy + Gx*ix); 
double const V = vtot[(iz*g[1] + iy)*g[0] + ix];
re += V * std::cos(arg);
im += V * std::sin(arg);
}}} 
Vcoeffs(0,iGz,iGy,iGx) = re; 
Vcoeffs(1,iGz,iGy,iGx) = im; 
}}} 

} 
if (echo > 6) {
std::printf("# 000-Fourier coefficient of the potential is %g %g %s\n",
Vcoeffs(0,0,0,0)*eV, Vcoeffs(1,0,0,0)*eV, _eV);
} 

std::vector<int32_t> numax_PAW(natoms_PAW,  3);
std::vector<double>  sigma_PAW(natoms_PAW, .5);
double maximum_sigma_PAW{0};
std::vector<view3D<double>> hs_PAW(natoms_PAW); 
for (int ka = 0; ka < natoms_PAW; ++ka) {
if (numax_prj) numax_PAW[ka] = numax_prj[ka];
if (sigma_prj) sigma_PAW[ka] = sigma_prj[ka];
maximum_sigma_PAW = std::max(maximum_sigma_PAW, sigma_PAW[ka]);
int const nSHO = sho_tools::nSHO(numax_PAW[ka]); 
if (atom_mat) {
hs_PAW[ka] = view3D<double>(atom_mat[ka], nSHO, nSHO); 
} else {
hs_PAW[ka] = view3D<double>(2, nSHO, nSHO, 0.0); 
} 
} 
if (nullptr == atom_mat) warn("atomic PAW matrices were not passed for %d atoms", natoms_PAW);

double const nbands_per_atom = control::get("bands.per.atom", 10.);
int const nbands = int(nbands_per_atom*natoms_PAW);

int const floating_point_bits = control::get("hamiltonian.floating.point.bits", 64.); 
float const iterative_direct_ratio = control::get("plane_wave.iterative.solver.ratio", 2.);

view2D<double> kmesh;
auto const nkpoints = brillouin_zone::get_kpoint_mesh(kmesh);
if (echo > 3) std::printf("# k-point mesh has %d points\n", nkpoints);

if (export_rho) export_rho->resize(nkpoints);


simple_stats::Stats<double> nPW_stats, tPW_stats;
#pragma omp parallel for
for (int ikp = 0; ikp < nkpoints; ++ikp) {
auto const *kpoint = kmesh[ikp];
char x_axis[96]; std::snprintf(x_axis, 96, "# %g %g %g spectrum ", kpoint[0], kpoint[1], kpoint[2]);
SimpleTimer timer(__FILE__, __LINE__, x_axis, 0);

bool constexpr can_be_real{false}; 
if (can_be_real) {
error("PW only implemented with complex", 0);
} else {
int nPWs{0};
if (32 == floating_point_bits) {
stat += solve_k<std::complex<float>>(
ecut, reci_matrix, Vcoeffs, nG, svol,
natoms_PAW, xyzZ, grid_offset, numax_PAW.data(), sigma_PAW.data(), hs_PAW.data(),
kpoint, x_axis, nPWs, echo,
nbands, iterative_direct_ratio,
export_rho ? &((*export_rho)[ikp]) : nullptr, ikp);
} else {
stat += solve_k<std::complex<double>>(
ecut, reci_matrix, Vcoeffs, nG, svol,
natoms_PAW, xyzZ, grid_offset, numax_PAW.data(), sigma_PAW.data(), hs_PAW.data(),
kpoint, x_axis, nPWs, echo,
nbands, iterative_direct_ratio,
export_rho ? &((*export_rho)[ikp]) : nullptr, ikp);
} 
nPW_stats.add(nPWs);
} 
if (echo > 0) std::fflush(stdout);
tPW_stats.add(timer.stop());
} 

if (echo > 3) std::printf("\n# number of plane waves is [%g, %.3f +/- %.3f, %g]\n",
nPW_stats.min(), nPW_stats.mean(), nPW_stats.dev(), nPW_stats.max());
if (echo > 3) std::printf("# time per k-point is [%.3f, %.3f +/- %.3f, %.3f] seconds\n",
tPW_stats.min(), tPW_stats.mean(), tPW_stats.dev(), tPW_stats.max());

return stat;
} 



#ifdef  NO_UNIT_TESTS
status_t all_tests(int const echo) { return STATUS_TEST_NOT_INCLUDED; }
#else 

status_t test_Hamiltonian(int const echo=5) {
status_t stat(0);

auto const vtotfile = control::get("sho_hamiltonian.test.vtot.filename", "vtot.dat"); 
int dims[] = {0, 0, 0};
std::vector<double> vtot; 
stat += sho_potential::load_local_potential(vtot, dims, vtotfile, echo);

auto const geo_file = control::get("geometry.file", "atoms.xyz");
view2D<double> xyzZ;
int natoms{0};
double cell[3] = {0, 0, 0};
int8_t bc[3] = {-7, -7, -7};
{ 
stat += geometry_analysis::read_xyz_file(xyzZ, natoms, geo_file, cell, bc, 0);
if (echo > 2) std::printf("# found %d atoms in file \"%s\" with cell=[%.3f %.3f %.3f] %s and bc=[%d %d %d]\n",
natoms, geo_file, cell[0]*Ang, cell[1]*Ang, cell[2]*Ang, _Ang, bc[0], bc[1], bc[2]);
} 

real_space::grid_t g(dims);
g.set_boundary_conditions(bc); 
g.set_grid_spacing(cell[0]/g[0], cell[1]/g[1], cell[2]/g[2]);
if (echo > 1) std::printf("# use  %g %g %g %s grid spacing\n", g.h[0]*Ang, g.h[1]*Ang, g.h[2]*Ang, _Ang);
if (echo > 1) std::printf("# cell is  %g %g %g %s\n", g.h[0]*g[0]*Ang, g.h[1]*g[1]*Ang, g.h[2]*g[2]*Ang, _Ang);

stat += solve(natoms, xyzZ, g, vtot.data(), 0, 0, 0, echo, 0);

return stat;
} 

status_t test_Hermite_Gauss_normalization(int const echo=5, int const numax=3) {
auto const nSHO = sho_tools::nSHO(numax);
std::vector<std::complex<double>> pzyx(nSHO);
view2D<std::complex<double>> pp(nSHO, nSHO, 0.0); 
double constexpr dg = 0.125, d3g = pow3(dg);
int constexpr ng = 40;
for (int iz = -ng; iz <= ng; ++iz) {
for (int iy = -ng; iy <= ng; ++iy) {
for (int ix = -ng; ix <= ng; ++ix) {
double const gv[3] = {dg*ix, dg*iy, dg*iz};
Hermite_Gauss_projectors(pzyx.data(), numax, 1.0, gv);
for (int iSHO = 0; iSHO < nSHO; ++iSHO) {
add_product(pp[iSHO], nSHO, pzyx.data(), std::conj(pzyx[iSHO]));
} 
}}} 
std::vector<double> p2(nSHO);
for (int iSHO = 0; iSHO < nSHO; ++iSHO) {
p2[iSHO] = pp(iSHO,iSHO).real()*d3g;
} 
if (echo > 5) { std::printf("\n# %s: norms ", __func__); printf_vector(" %g", p2.data(), nSHO); }
double maxdev{0};
for (int iSHO = 0; iSHO < nSHO; ++iSHO) {
for (int jSHO = 0; jSHO < nSHO; ++jSHO) {
auto const dev = std::abs(pp(iSHO,jSHO) * d3g - double(iSHO == jSHO));
maxdev = std::max(maxdev, dev);
p2[jSHO] = dev; 
} 
if (echo > 8) {
std::printf("# %s: orthogonality ", __func__);
printf_vector(" %.1e", p2.data(), nSHO);
} 
} 
if (echo > 3) std::printf("# %s: orthogonality max dev %.1e\n", __func__, maxdev);
return 0;
} 

status_t all_tests(int const echo) {
status_t stat(0);
stat += test_Hamiltonian(echo);
stat += test_Hermite_Gauss_normalization(echo);
return stat;
} 

#endif 

} 
