#pragma once

#include <cstdio> 

#include "status.hxx" 
#include "control.hxx" 

#include "real_space.hxx" 
#include "data_view.hxx" 
#include "debug_output.hxx" 
#include "solid_harmonics.hxx" 
#include "inline_math.hxx" 

#ifdef DEVEL
#include "finite_difference.hxx" 
#include "single_atom.hxx" 
#include "print_tools.hxx" 
#include "radial_r2grid.hxx" 
#include "lossful_compression.hxx" 
#include "simple_timer.hxx" 
#include "poisson_solver.hxx" 
#endif 

namespace potential_generator {

template <typename real_t>
status_t write_array_to_file(
char const *filename 
, real_t const array[]  
, int const nx, int const ny, int const nz 
, int const echo=0 
, char const *arrayname="" 
) {
char title[128]; std::snprintf(title, 128, "%i x %i x %i  %s", nz, ny, nx, arrayname);
auto const size = size_t(nz) * size_t(ny) * size_t(nx);
return dump_to_file(filename, size, array, nullptr, 1, 1, title, echo);
} 


template <typename real_t>
status_t add_smooth_quantities(
real_t values[] 
, real_space::grid_t const & g 
, int const na 
, int32_t const nr2[] 
, float const ar2[] 
, view2D<double> const & center 
, int const n_periodic_images
, view2D<double> const & periodic_images
, double const *const *const atom_qnt 
, int const echo=0 
, int const echo_q=0 
, double const factor=1 
, char const *quantity="???" 
) {
assert(g.is_Cartesian());

status_t stat(0);
for (int ia = 0; ia < na; ++ia) {
#ifdef DEVEL
if (echo > 11) {
std::printf("\n## r, %s of atom #%i\n", quantity, ia);
print_compressed(radial_r2grid::r_axis(nr2[ia], ar2[ia]).data(), atom_qnt[ia], nr2[ia]);
} 
#endif 
double q_added{0};
for (int ii = 0; ii < n_periodic_images; ++ii) {
double cnt[3]; set(cnt, 3, center[ia]); add_product(cnt, 3, periodic_images[ii], 1.0);
double q_added_image{0};
stat += real_space::add_function(values, g, atom_qnt[ia], nr2[ia], ar2[ia], &q_added_image, cnt, factor);
if (echo_q > 11) std::printf("# %g electrons %s of atom #%d added for image #%i\n", q_added_image, quantity, ia, ii);
q_added += q_added_image;
} 
#ifdef DEVEL
if (echo_q > 0) {
std::printf("# after adding %g electrons %s of atom #%d:", q_added, quantity, ia);
print_stats(values, g.all(), g.dV());
} 
if (echo_q > 3) std::printf("# added %s for atom #%d is  %g electrons\n", quantity, ia, q_added);
#endif 
} 
return stat;
} 


template <typename real_t, int debug=0>
status_t add_generalized_Gaussian(
real_t values[] 
, real_space::grid_t const & g 
, double const coeff[] 
, int const ellmax=-1
, double const center[3]=nullptr 
, double const sigma=1 
, int const echo=0
, double *added=nullptr 
, double const factor=1 
, float const r_cut=-1 
) {
status_t stat(0);
if (ellmax < 0) return stat; 
assert(g.is_Cartesian());
assert(sigma > 0);
double c[3] = {0,0,0}; if (center) set(c, 3, center);
double const r_max = 9*sigma; 
double const rcut = (-1 == r_cut) ? r_max : std::min(double(r_cut), r_max);
double const r2cut = rcut*rcut;
int imn[3], imx[3];
size_t nwindow{1};
for (int d = 0; d < 3; ++d) {
imn[d] = std::max(0, int(std::floor((c[d] - rcut)*g.inv_h[d])));
imx[d] = std::min(   int(std::ceil ((c[d] + rcut)*g.inv_h[d])), g[d] - 1);
if (echo > 8) std::printf("# %s window %c = %d elements from %d to %d\n", __func__, 'x'+d, imx[d] + 1 - imn[d], imn[d], imx[d]);
nwindow *= std::max(0, imx[d] + 1 - imn[d]);
} 
int const nlm = pow2(1 + ellmax);
double const sigma2inv = 0.5/pow2(sigma);
std::vector<double> scaled_coefficients(nlm, 0.0);
std::vector<double> scale_factor(debug*nlm, 0.0);
{ 
double radial_norm{constants::sqrt2/(constants::sqrtpi*sigma)};
for (int ell = 0; ell <= ellmax; ++ell) {
radial_norm /= (sigma*sigma*(2*ell + 1));
for (int emm = -ell; emm <= ell; ++emm) {
int const ilm = solid_harmonics::lm_index(ell, emm);
scaled_coefficients[ilm] = coeff[ilm] * radial_norm;
if (debug) scale_factor[ilm] = radial_norm*g.dV();
} 
} 
} 
view2D<double> overlap_ij(debug*nlm, nlm, 0.0);
std::vector<double> rlXlm(nlm); 
double added_charge{0}; 
size_t modified{0};
for (            int iz = imn[2]; iz <= imx[2]; ++iz) {  double const vz = iz*g.h[2] - c[2], vz2 = vz*vz;
for (        int iy = imn[1]; iy <= imx[1]; ++iy) {  double const vy = iy*g.h[1] - c[1], vy2 = vy*vy;
if (vz2 + vy2 < r2cut) {
for (int ix = imn[0]; ix <= imx[0]; ++ix) {  double const vx = ix*g.h[0] - c[0], vx2 = vx*vx;
double const r2 = vz2 + vy2 + vx2;
if (r2 < r2cut) {
int const izyx = (iz*g('y') + iy)*g('x') + ix;
solid_harmonics::rlXlm(rlXlm.data(), ellmax, vx, vy, vz);
double const Gaussian = std::exp(-sigma2inv*r2);
double const value_to_add = Gaussian * dot_product(nlm, scaled_coefficients.data(), rlXlm.data());
values[izyx] += factor*value_to_add;
added_charge += factor*value_to_add;
++modified;
#if 0
if (echo > 13) {
std::printf("#rs %g %g\n", std::sqrt(r2), value_to_add);
} 
#endif 
for (int ilm = 0; ilm < nlm*debug; ++ilm) {
add_product(overlap_ij[ilm], nlm, rlXlm.data(), Gaussian*rlXlm[ilm]);
} 
} 
} 
} 
} 
} 
added_charge *= g.dV(); 
if (echo > 7) std::printf("# %s modified %.3f k inside a window of %.3f k on a grid of %.3f k grid values, added charge= %g\n",
__func__, modified*1e-3, nwindow*1e-3, g('x')*g('y')*g('z')*1e-3, added_charge); 
if (debug) {
double dev[] = {0, 0}; 
for (int ilm = 0; ilm < nlm*debug; ++ilm) {
if (echo > 5) std::printf("# ilm=%3d ", ilm);
for (int jlm = 0; jlm < nlm; ++jlm) {
double const ovl = overlap_ij(ilm,jlm)*scale_factor[ilm];
int const diag = (ilm == jlm);
if (echo > 5) std::printf(diag?" %.9f":" %9.1e", ovl);
dev[diag] = std::max(dev[diag], std::abs(ovl - diag));
} 
if (echo > 5) std::printf("\n");
} 
if (echo > 1) std::printf("# %s(ellmax=%d) is orthogonal to %.1e, normalized to %.1e\n", __func__, ellmax, dev[0], dev[1]);
} 

if (added) *added = added_charge;
return stat;
} 



#ifdef DEVEL

inline status_t potential_projections(
real_space::grid_t const & g 
, double const cell[]
, double const Ves[] 
, double const Vxc[] 
, double const Vtot[] 
, double const rho[] 
, double const cmp[] 
, int const na=0 
, view2D<double> const *const center_ptr=nullptr
, float const rcut=32
, int const echo=0 
) {
status_t stat(0);
assert(g.is_Cartesian());

std::vector<double> Laplace_Ves(g.all(), 0.0);

int const verify_Poisson = control::get("potential_generator.verify.poisson", 0.);
for (int nfd = 1; nfd < verify_Poisson; ++nfd) {
finite_difference::stencil_t<double> const fd(g.h, nfd, -.25/constants::pi);
{ 
SimpleTimer timer(__FILE__, __LINE__, "finite-difference", echo);
stat += finite_difference::apply(Laplace_Ves.data(), Ves, g, fd);
{ 
double res_a{0}, res_2{0};
for (size_t i = 0; i < g.all(); ++i) {
res_a += std::abs(Laplace_Ves[i] - rho[i]);
res_2 +=     pow2(Laplace_Ves[i] - rho[i]);
} 
res_a *= g.dV(); res_2 = std::sqrt(res_2*g.dV());
if (echo > 1) std::printf("# Laplace*Ves - rho: residuals abs %.2e rms %.2e (FD-order=%i)\n", res_a, res_2, nfd);
} 
} 
} 

int const use_Bessel_projection = control::get("potential_generator.use.bessel.projection", 0.);
if (use_Bessel_projection)
{ 

double constexpr Y00sq = pow2(solid_harmonics::Y00);

view2D<double> periodic_images;
int const n_periodic_images = boundary_condition::periodic_images(periodic_images,
cell, g.boundary_conditions(), rcut, echo - 4);


std::vector<radial_grid_t const*> rg(na, nullptr); 
{   
auto const dcpp = reinterpret_cast<double const *const *>(rg.data());
auto const dpp  =       const_cast<double       *const *>(dcpp);
stat += single_atom::atom_update("radial grids", na, 0, 0, 0, dpp);
}

double const* const value_pointers[] = {Ves, Vxc, Vtot, rho, cmp, Laplace_Ves.data()};
char const *  array_names[] = {"Ves", "Vxc", "Vtot", "rho", "cmp", "LVes"};

for (int iptr = 0; iptr < std::min(6, use_Bessel_projection); ++iptr) {
auto const values = value_pointers[iptr];
auto const array_name = array_names[iptr];

if (echo > 1) { std::printf("\n# real-space stats of %s:", array_name); print_stats(values, g.all(), g.dV()); }

for (int ia = 0; ia < na; ++ia) {
float const dq = 1.f/16;
int const nq = int(constants::pi/(g.smallest_grid_spacing()*dq));
std::vector<double> qc(nq, 0.0);

{ 
std::vector<double> qc_image(nq, 0.0);
if (nullptr != center_ptr) { auto const & center = *center_ptr;
for (int ii = 0; ii < n_periodic_images; ++ii) {
double cnt[3]; set(cnt, 3, center[ia]); add_product(cnt, 3, periodic_images[ii], 1.0);
stat += real_space::Bessel_projection(qc_image.data(), nq, dq, values, g, cnt);
add_product(qc.data(), nq, qc_image.data(), 1.0);
} 
} 
} 

scale(qc.data(), nq, Y00sq);

std::vector<double> qcq2(nq, 0.0);
for (int iq = 1; iq < nq; ++iq) { 
qcq2[iq] = 4*constants::pi*qc[iq]/pow2(iq*dq); 
} 

if (echo > 11) {
std::printf("\n# Bessel coeff of %s for atom #%d:\n", array_name, ia);
for (int iq = 0; iq < nq; ++iq) {
std::printf("# %g %g %g\n", iq*dq, qc[iq], qcq2[iq]);
} 
std::printf("\n\n");
} 

if (echo > 3) {
std::vector<double> rs(rg[ia]->n);
bessel_transform::transform_s_function(rs.data(), qc.data(), *rg[ia], nq, dq, true); 
std::printf("\n## Real-space projection of %s for atom #%d:\n", array_names[iptr], ia);
float const compression_threshold = 1e-4;
print_compressed(rg[ia]->r, rs.data(), rg[ia]->n, compression_threshold);

if ((values == rho) || (values == Laplace_Ves.data())) {
bessel_transform::transform_s_function(rs.data(), qcq2.data(), *rg[ia], nq, dq, true); 
std::printf("\n## Electrostatics computed by Bessel transform of %s for atom #%d:\n", array_names[iptr], ia);
print_compressed(rg[ia]->r, rs.data(), rg[ia]->n, compression_threshold);
} 
} 
} 

} 

} 

if (echo > 1) {
if (nullptr != center_ptr) { auto const & center = *center_ptr;
if (control::get("potential_generator.direct.projection", 0.) > 0) {
std::printf("\n## all values of Vtot in %s (unordered) as function of the distance to %s\n",
_eV, (na > 0) ? "atom #0" : "the cell center");
poisson_solver::print_direct_projection(Vtot, g, eV, (na > 0) ? center[0] : nullptr);
} 
} else warn("no coordinates passed for na=%d atoms, center_ptr==nullptr", na);
} 

{ 
auto const Vtot_out_filename = control::get("total.potential.to.file", "vtot.dat");
if (*Vtot_out_filename) stat += write_array_to_file(Vtot_out_filename, Vtot, g[0], g[1], g[2], echo);
} 

return stat;
} 

#endif 

status_t all_tests(int const echo=0); 

} 
