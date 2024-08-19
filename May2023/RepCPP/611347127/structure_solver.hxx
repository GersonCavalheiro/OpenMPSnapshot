#pragma once

#include <cstdio> 
#include <cassert> 
#include <vector> 
#include <complex> 
#include <cmath> 
#include <algorithm> 



#include "display_units.h" 
#include "inline_math.hxx" 
#include "real_space.hxx" 
#include "sho_tools.hxx" 
#include "data_view.hxx" 
#include "data_list.hxx" 
#include "control.hxx" 
#include "constants.hxx" 
#include "debug_tools.hxx" 
#include "debug_output.hxx" 
#include "atom_image.hxx"
#include "real_space.hxx" 
#include "grid_operators.hxx" 
#include "conjugate_gradients.hxx" 
#include "davidson_solver.hxx" 
#include "dense_solver.hxx" 
#include "fermi_distribution.hxx" 
#include "density_generator.hxx" 
#include "multi_grid.hxx" 
#include "brillouin_zone.hxx" 
#include "plane_wave.hxx" 
#include "sho_hamiltonian.hxx" 
#include "print_tools.hxx" 
#include "complex_tools.hxx" 
#include "status.hxx" 

namespace structure_solver {



template <typename wave_function_t>
class KohnShamStates {
public:
using real_wave_function_t = decltype(std::real(wave_function_t(1)));

KohnShamStates() {} 
KohnShamStates(   
real_space::grid_t const & coarse_grid
, std::vector<atom_image::sho_atom_t> const & list_of_atoms
, view2D<double> const & kpoint_mesh
, int const number_of_kpoints=1
, int const number_of_bands=1
, int const run=1 
, int const echo=0 
)
: gc(coarse_grid)
, op(gc, list_of_atoms)
, kmesh(kpoint_mesh)
, nkpoints(number_of_kpoints)
, nbands(number_of_bands)
, nrepeat(int(control::get("grid.eigensolver.repeat", 1.))) 
{
if (echo > 0) std::printf("# %s use  %d x %d x %d  coarse grid points\n", __func__, gc[0], gc[1], gc[2]);


if (echo > 0) std::printf("# real-space grid wave functions are of type %s\n", complex_name<wave_function_t>());
psi = view3D<wave_function_t>(run*nkpoints, nbands, gc.all(), 1.0); 

int const na = list_of_atoms.size();
auto start_wave_file = control::get("start.waves", "");
assert(nullptr != start_wave_file);
if ('\0' != *start_wave_file) {
if (echo > 1) std::printf("# try to read start waves from file \'%s\'\n", start_wave_file);
if (run) {
auto const errors = debug_tools::read_from_file(psi(0,0), start_wave_file, nbands, psi.stride(), gc.all(), "wave functions", echo);
if (errors) {
warn("failed to read start wave functions from file \'%s\'", start_wave_file);
start_wave_file = ""; 
} else {
if (echo > 1) std::printf("# read %d bands x %ld numbers from file \'%s\'\n", nbands, gc.all(), start_wave_file);
for (int ikpoint = 1; ikpoint < nkpoints; ++ikpoint) {
if (echo > 3) { std::printf("# copy %d bands for k-point #%i from k-point #0\n", nbands, ikpoint); std::fflush(stdout); }
if (run) set(psi(ikpoint,0), psi.dim1()*psi.stride(), psi(0,0)); 
} 
} 
} 
} 

if ('\0' == *start_wave_file) {
if (echo > 1) std::printf("# initialize grid wave functions as %d atomic orbitals on %d atoms\n", nbands, na);
float const scale_sigmas = control::get("start.waves.scale.sigma", 10.); 
uint8_t qn[20][4]; 
sho_tools::quantum_number_table(qn[0], 3, sho_tools::order_Ezyx); 
std::vector<int32_t> ncoeff_a(na, 20);
int const na1 = std::max(na, 1); 
data_list<wave_function_t> single_atomic_orbital(ncoeff_a, 0.0); 
for (int ikpoint = 0; ikpoint < nkpoints; ++ikpoint) {
op.set_kpoint(kmesh[ikpoint], echo);
if (na > 0) {
for (int iband = 0; iband < nbands; ++iband) {
int const ia = iband % na1; 
int const io = iband / na1; 
if (io >= 20) error("requested more than 20 start wave functions per atom! bands.per.atom=%g", nbands/double(na));
auto const q = qn[io];
if (echo > 7) std::printf("# initialize band #%i as atomic orbital %x%x%x of atom #%i\n", iband, q[2], q[1], q[0], ia);
int const isho = sho_tools::zyx_index(3, q[0], q[1], q[2]); 
single_atomic_orbital[ia][isho] = 1./std::sqrt((q[3] > 0) ? ( (q[3] > 1) ? 53. : 26.5 ) : 106.); 
if (run) op.get_start_waves(psi(ikpoint,iband), single_atomic_orbital.data(), scale_sigmas, echo);
single_atomic_orbital[ia][isho] = 0; 
} 
} 
} 
op.set_kpoint(); 
} 

} 

status_t solve(
view2D<double> & rho_valence_gc
, data_list<double> atom_rho_new[2]
, view2D<double> & energies
, double charges[]
, fermi_distribution::FermiLevel_t & Fermi
, view2D<double> const & Veff
, data_list<double> const & atom_mat
, char const occupation_method='e'
, char const *grid_eigensolver_method="cg"
, int const scf_iteration=-1
, int const echo=0
) {
status_t stat(0);
int const na = atom_mat.nrows();

op.set_potential(Veff.data(), gc.all(), atom_mat.data(), echo*0); 

auto const export_Hamiltonian = control::get("hamiltonian.export", 0.0);
if (export_Hamiltonian) {
op.write_to_file(echo, control::get("hamiltonian.export.format", "xml"));
if (export_Hamiltonian < 0) abort("Hamiltonian exported, hamiltonian.export= %g < 0", export_Hamiltonian);
} 

for (int ikpoint = 0; ikpoint < nkpoints; ++ikpoint) {
op.set_kpoint(kmesh[ikpoint], echo);
char x_axis[96]; std::snprintf(x_axis, 96, "# %g %g %g spectrum ", kmesh(ikpoint,0),kmesh(ikpoint,1),kmesh(ikpoint,2));
auto psi_k = psi[ikpoint]; 
bool display_spectrum{true};

if ('c' == *grid_eigensolver_method) { 
stat += davidson_solver::rotate(psi_k.data(), energies[ikpoint], nbands, op, echo);
for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
if (echo > 6) { std::printf("# SCF cycle #%i, CG repetition #%i\n", scf_iteration, irepeat); std::fflush(stdout); }
stat += conjugate_gradients::eigensolve(psi_k.data(), energies[ikpoint], nbands, op, echo - 5);
stat += davidson_solver::rotate(psi_k.data(), energies[ikpoint], nbands, op, echo);
} 
} else
if ('d' == *grid_eigensolver_method) { 
for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
if (echo > 6) { std::printf("# SCF cycle #%i, DAV repetition #%i\n", scf_iteration, irepeat); std::fflush(stdout); }
stat += davidson_solver::eigensolve(psi_k.data(), energies[ikpoint], nbands, op, echo);
} 
} else
if ('e' == *grid_eigensolver_method) { 
view3D<wave_function_t> HSm(2, gc.all(), align<4>(gc.all()), 0.0); 
op.construct_dense_operator(HSm(0,0), HSm(1,0), HSm.stride(), echo);
stat += dense_solver::solve(HSm, x_axis, echo, nbands, energies[ikpoint]);
display_spectrum = false; 
wave_function_t const factor = 1./std::sqrt(gc.dV()); 
for (int iband = 0; iband < nbands; ++iband) {
set(psi(ikpoint,iband), gc.all(), HSm(0,iband), factor);
} 
} else
if ('n' == *grid_eigensolver_method) { 
} else {
++stat; error("unknown grid.eigensolver method \'%s\'", grid_eigensolver_method);
} 

if (display_spectrum && echo > 0) dense_solver::display_spectrum(energies[ikpoint], nbands, x_axis, eV, _eV);


} 
op.set_kpoint(); 

here;

if ('e' == (occupation_method | 32)) {
if (echo > 4) std::printf("# fermi.level=exact\n");
view2D<double> kweights(nkpoints, nbands), occupations(nkpoints, nbands);
for (int ikpoint = 0; ikpoint < nkpoints; ++ikpoint) {
double const kpoint_weight = kmesh(ikpoint,brillouin_zone::WEIGHT);
set(kweights[ikpoint], nbands, kpoint_weight);
} 
double const eF = fermi_distribution::Fermi_level(occupations.data(),
energies.data(), kweights.data(), nkpoints*nbands,
Fermi.get_temperature(), Fermi.get_n_electrons(), Fermi.get_spinfactor(), echo);
Fermi.set_Fermi_level(eF, echo);
} 

here;

for (int ikpoint = 0; ikpoint < nkpoints; ++ikpoint) {
op.set_kpoint(kmesh[ikpoint], echo);
double const kpoint_weight = kmesh(ikpoint,brillouin_zone::WEIGHT);
std::vector<uint32_t> coeff_starts;
auto const atom_coeff = density_generator::atom_coefficients(coeff_starts,
psi(ikpoint,0), op, nbands, echo, ikpoint);
stat += density_generator::density(rho_valence_gc[0], atom_rho_new[0].data(), Fermi,
energies[ikpoint], psi(ikpoint,0), atom_coeff.data(),
coeff_starts.data(), na, gc, nbands, kpoint_weight, echo - 4, ikpoint,
rho_valence_gc[1], atom_rho_new[1].data(), charges);
} 
op.set_kpoint(); 

return stat;
} 

status_t store(char const *filename, int const echo=0) const {
return dump_to_file(filename, nbands, psi(0,0), nullptr, psi.stride(), gc.all(), "wave functions", echo);
} 

~KohnShamStates() {
#ifdef    DEBUG
std::printf("# ~KohnShamStates<%s>\n", complex_name<wave_function_t>());
#endif 
psi = view3D<wave_function_t>(0,0,0, 0);
} 

private: 

real_space::grid_t const & gc;
grid_operators::grid_operator_t<wave_function_t, real_wave_function_t> op;
view3D<wave_function_t> psi; 
view2D<double> const & kmesh; 
int nkpoints;
int nbands;
int nrepeat;

}; 





















class RealSpaceKohnSham {
public:

RealSpaceKohnSham(
real_space::grid_t const & g 
, view2D<double> const & xyzZinso
, int const na 
, int const run=1 
, int const echo=0 
)
: gd(g), xyzZ(xyzZinso)
{
basis_method = control::get("basis", "grid");
key = (*basis_method) | 32; 
psi_on_grid = ((*basis_method | 32) == 'g');

nkpoints = brillouin_zone::get_kpoint_mesh(kmesh);
if (echo > 1) std::printf("# k-point mesh has %d points\n", nkpoints);

bool const needs_complex = brillouin_zone::needs_complex(kmesh, nkpoints);
if (echo > 2) std::printf("# k-point mesh %s wavefunctions\n", needs_complex?"needs complex":"allows real");

int const force_complex = control::get("structure_solver.complex", 0.);
if ((echo > 2) && (0 != force_complex)) std::printf("# complex wavefunctions enforced by structure_solver.complex=%d\n", force_complex);

bool const use_complex = needs_complex | (0 != force_complex);

double const nbands_per_atom = control::get("bands.per.atom", 10.); 
int    const nbands_extra    = control::get("bands.extra", 0.);
if (echo > 0) std::printf("# bands.per.atom=%g\n# bands.extra=%d\n", nbands_per_atom, nbands_extra);
nbands = int(nbands_per_atom*na) + nbands_extra;


if (psi_on_grid) {

gc = real_space::grid_t(g[0]/2, g[1]/2, g[2]/2); 
gc.set_grid_spacing(2*g.h[0], 2*g.h[1], 2*g.h[2]); 
gc.set_boundary_conditions(g.boundary_conditions());

if (echo > 1) {
std::printf("# use  %d x %d x %d  coarse grid points\n", gc[0], gc[1], gc[2]);
double const max_grid_spacing = std::max(std::max(gc.h[0], gc.h[1]), gc.h[2]);
std::printf("# use  %g %g %g  %s  coarse grid spacing, corresponds to %.2f Ry\n",
gc.h[0]*Ang, gc.h[1]*Ang, gc.h[2]*Ang, _Ang, pow2(constants::pi/max_grid_spacing));
} 

auto const loa = grid_operators::list_of_atoms(xyzZinso.data(), na, xyzZinso.stride(), gc, echo);


int const floating_point_bits = control::get("hamiltonian.floating.point.bits", 64.); 
if (32 == floating_point_bits) {
key = use_complex ? 'c' : 's';
} else if (64 == floating_point_bits) {
key = use_complex ? 'z' : 'd';
} else {
error("hamiltonian.floating.point.bits=%d must be 32 or 64 (default)", floating_point_bits);
}

if ('z' == key) z = new KohnShamStates<std::complex<double>>(gc, loa, kmesh, nkpoints, nbands, run, echo);
if ('c' == key) c = new KohnShamStates<std::complex<float>> (gc, loa, kmesh, nkpoints, nbands, run, echo);
if ('d' == key) d = new KohnShamStates<double>              (gc, loa, kmesh, nkpoints, nbands, run, echo);
if ('s' == key) s = new KohnShamStates<float>               (gc, loa, kmesh, nkpoints, nbands, run, echo);

solver_method = control::get("grid.eigensolver", "cg");

} else { 

sigma_a.resize(na);
numax.resize(na);
for (int ia = 0; ia < na; ++ia) {
numax[ia]   = xyzZinso(ia,5);
sigma_a[ia] = xyzZinso(ia,6);
} 

} 

energies = view2D<double>(nkpoints, nbands, 0.0); 

} 

~RealSpaceKohnSham() { 
if (z) z->~KohnShamStates();
if (c) c->~KohnShamStates();
if (d) d->~KohnShamStates();
if (s) s->~KohnShamStates();
} 

status_t solve(
view2D<double> & rho_valence_new  
, data_list<double> atom_rho_new[2] 
, double charges[] 
, fermi_distribution::FermiLevel_t & Fermi
, real_space::grid_t const & g 
, double const Vtot[] 
, int const & natoms
, data_list<double> & atom_mat 
, char const occupation_method='e' 
, int const scf=-1 
, int const echo=9 
) {
status_t stat(0);
#ifdef DEVEL
if (echo > 0) {
std::printf("\n\n#\n# Solve Kohn-Sham equation (basis=%s)\n#\n\n", basis_method);
std::fflush(stdout);
} 
#endif 

assert(g.all() == rho_valence_new.stride());

if (psi_on_grid) {
sanity_check();

view2D<double> Veff(1, gc.all());
multi_grid::restrict3D(Veff[0], gc, Vtot, gd, 0); 
if (echo > 1) print_stats(Veff[0], gc.all(), 0, "\n# Total effective potential  (restricted to coarse grid)   ", eV);

view2D<double> rho_valence_gc(2, gc.all(), 0.0); 

if (z) z->solve(rho_valence_gc, atom_rho_new, energies, charges, Fermi, Veff, atom_mat, occupation_method, solver_method, scf, echo);
if (c) c->solve(rho_valence_gc, atom_rho_new, energies, charges, Fermi, Veff, atom_mat, occupation_method, solver_method, scf, echo);
if (d) d->solve(rho_valence_gc, atom_rho_new, energies, charges, Fermi, Veff, atom_mat, occupation_method, solver_method, scf, echo);
if (s) s->solve(rho_valence_gc, atom_rho_new, energies, charges, Fermi, Veff, atom_mat, occupation_method, solver_method, scf, echo);

auto const dcc_coarse = dot_product(gc.all(), rho_valence_gc[0], Veff.data()) * gc.dV();
if (echo > 4) std::printf("\n# double counting (coarse grid) %.9f %s\n", dcc_coarse*eV, _eV);

stat += multi_grid::interpolate3D(rho_valence_new[0], g, rho_valence_gc[0], gc); 
stat += multi_grid::interpolate3D(rho_valence_new[1], g, rho_valence_gc[1], gc); 

} else if ('p' == key) { 

std::vector<plane_wave::DensityIngredients> export_rho;

stat += plane_wave::solve(natoms, xyzZ, g, Vtot, sigma_a.data(), numax.data(), atom_mat.data(), echo, &export_rho);

if ('e' == (occupation_method | 32)) {
view2D<double> kweights(nkpoints, nbands, 0.0), occupations(nkpoints, nbands);
for (int ikpoint = 0; ikpoint < export_rho.size(); ++ikpoint) {
set(kweights[ikpoint], nbands, export_rho[ikpoint].kpoint_weight);
set(energies[ikpoint], nbands, export_rho[ikpoint].energies.data());
} 
double const eF = fermi_distribution::Fermi_level(occupations.data(),
energies.data(), kweights.data(), nkpoints*nbands,
Fermi.get_temperature(), Fermi.get_n_electrons(), Fermi.get_spinfactor(), echo);
Fermi.set_Fermi_level(eF, echo);
} 

for (auto & x : export_rho) {
if (echo > 1) { std::printf("\n# Generate valence density for %s\n", x.tag); std::fflush(stdout); }
stat += density_generator::density(rho_valence_new[0], atom_rho_new[0].data(), Fermi,
x.energies.data(), x.psi_r.data(), x.coeff.data(),
x.offset.data(), x.natoms, g, x.nbands, x.kpoint_weight, echo - 4, x.kpoint_index,
rho_valence_new[1], atom_rho_new[1].data(), charges);
} 

} else if ('n' == key) { 

warn("with basis=%s no new density is generated", basis_method);

} else {

stat += sho_hamiltonian::solve(natoms, xyzZ, g, Vtot, nkpoints, kmesh, natoms, sigma_a.data(), numax.data(), atom_mat.data(), echo);
warn("with basis=%s no new density is generated", basis_method); 

} 

here;
return stat;
} 

status_t store(char const *filename, int const echo=0) {
status_t nerrors(0);
if (echo > 0) std::printf("# write wave functions to file \'%s\'\n", filename);
if (nullptr != filename) {
if ('\0' != filename[0]) {
if (psi_on_grid) {
sanity_check();
if (z) nerrors += z->store(filename, echo);
if (c) nerrors += c->store(filename, echo);
if (d) nerrors += d->store(filename, echo);
if (s) nerrors += s->store(filename, echo);
} 
if (nerrors) warn("%d errors occured writing file \'%s\'", nerrors, filename);
} else {
if (psi_on_grid && echo > 2) std::printf("# filename for storing wave functions is empty\n");
} 
} else warn("filename for storing wave functions is null", 0);
return nerrors;
} 

private:

void sanity_check() {
char str[] = "\0\0\0\0\0\0\0";
int n{0};
if (z) { str[n] = 'z'; ++n; }
if (c) { str[n] = 'c'; ++n; }
if (d) { str[n] = 'd'; ++n; }
if (s) { str[n] = 's'; ++n; }
if (1 != n) error("expect exactly one of four pointers active but found %d {%s}", n, str);
} 

view2D<double> kmesh; 
int nkpoints = 0;
int nbands = 0;
real_space::grid_t const & gd; 
real_space::grid_t gc; 

std::vector<double> sigma_a;
std::vector<int> numax;
view2D<double> const & xyzZ;

KohnShamStates<std::complex<double>> *z = nullptr;
KohnShamStates<std::complex<float>>  *c = nullptr;
KohnShamStates<double>               *d = nullptr;
KohnShamStates<float>                *s = nullptr;

char key = '?';
bool psi_on_grid;
char const *basis_method  = nullptr;
char const *solver_method = nullptr;
public:
view2D<double> energies; 

}; 



status_t all_tests(int const echo=0); 

} 

#ifdef DEBUG
#undef DEBUG
#endif 

#ifdef FULL_DEBUG
#undef FULL_DEBUG
#endif 
