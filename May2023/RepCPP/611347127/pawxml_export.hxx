#pragma once

#include <cstdio> 
#include <vector> 
#include <time.h> 
#include <cmath>  

#include "chemical_symbol.hxx" 
#include "radial_grid.h" 
#include "radial_grid.hxx" 
#include "energy_level.hxx" 
#include "data_view.hxx" 
#include "control.hxx" 

namespace pawxml_export {

inline char c4(int const i) { return (i & 0x3) ? ' ' : '\n'; }

template <class PartialWave>
int write_to_file(
double const Z 
, radial_grid_t const rg[TRU_AND_SMT] 
, std::vector<PartialWave> const & valence_states
, char const valence_states_active[]
, view3D<double> const & kinetic_energy_differences 
, double const n_electrons[3] 
, view2D<double> const spherical_density[TRU_AND_SMT] 
, view2D<double> const & projector_functions 
, view3D<double> const & aHSm 
, double const r_cut=2
, double const sigma_cmp=.5
, double const zero_potential[]=nullptr
, int const echo=0 
, double const core_kinetic_energy=0
, double const kinetic_energy=0
, double const xc_energy=0
, double const es_energy=0
, double const total_energy=0
, char const *custom_configuration_string=nullptr
, char const *xc_functional="LDA"
, char const *pathname="."
, char const *filename=nullptr 
)
{
auto const git_key = control::get("git.key", "");
int  const show_rg = control::get("pawxml_export.show.radial.grid", 0.);  
int  const ir0     = control::get("pawxml_export.start.radial.grid", 0.); 

double constexpr Y00 = .28209479177387817; 
char const ts_label[TRU_AND_SMT][8] = {"ae", "pseudo"};
char const ts_tag[TRU_AND_SMT][4] = {"ae", "ps"};

char Sy[4]; 
int const iZ = chemical_symbol::get(Sy, Z);

char const *const suffix = control::get("pawxml_export.type", "xml"); 

char file_name_buffer[512];
if (nullptr == filename) {
std::snprintf(file_name_buffer, 512, "%s/%s.%s", pathname, Sy, suffix);
filename = file_name_buffer;
} 
if (echo > 0) std::printf("# %s %s Z=%g iZ=%d filename='%s'\n", Sy, __func__, Z, iZ, filename);

auto *const f = std::fopen(filename, "w");
if (nullptr == f) {
if (echo > 0) std::printf("# %s %s: Error opening file '%s'", Sy, __func__, filename);
return __LINE__;
} 

if (std::string("upf") == suffix) {

int l_max{-1};
std::vector<int> iln_index(0);
for (size_t iln = 0; iln < valence_states.size(); ++iln) {
if (valence_states_active[iln]) {
l_max = std::max(l_max, int(valence_states[iln].ell));
iln_index.push_back(iln);
} 
} 
int const n_proj = iln_index.size();

int const n_mesh = 4*(rg[SMT].n/4), n_wfc = 0;
std::printf("# generate UPF file according to pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format\n");
std::fprintf(f, "<UPF version=\"2.0.1\">\n");
std::fprintf(f, "<PP_INFO>\n");
std::fprintf(f, "<PP_INPUTFILE>\n");
std::fprintf(f, "  +element_%s=\"%s\"\n", Sy, custom_configuration_string);
std::fprintf(f, "</PP_INPUTFILE>\n");
std::fprintf(f, "</PP_INFO>\n");
std::fprintf(f, "<!--                               -->\n");
std::fprintf(f, "<!-- END OF HUMAN READABLE SECTION -->\n");
std::fprintf(f, "<!--                               -->\n");
std::fprintf(f, "<PP_HEADER");
std::fprintf(f, "\ngenerated=\"libliveatom %s\"", git_key);
std::fprintf(f, "\nauthor=\"Paul Baumeister\"");
auto const now = std::time(0);
char date[80]; std::strftime(date, sizeof(date), "%Y-%m-%d", std::localtime(&now));
std::fprintf(f, "\ndate=\"%s\"", date);
std::fprintf(f, "\ncomment=\"pseudo_type should be PAW\"");
std::fprintf(f, "\nelement=\"%s\"", Sy);
std::fprintf(f, "\npseudo_type=\"NC\""); 
std::fprintf(f, "\nrelativistic=\"scalar\"");
std::fprintf(f, "\nis_ultrasoft=\"F\"");
std::fprintf(f, "\nis_paw=\"F\"");
std::fprintf(f, "\nis_coulomb=\"F\"");
std::fprintf(f, "\nhas_so=\"F\""); 
std::fprintf(f, "\nhas_wfc=\"F\"");
std::fprintf(f, "\nhas_gipaw=\"F\""); 
std::fprintf(f, "\ncore_correction=\"F\""); 
std::fprintf(f, "\nfunctional=\"%s\"", xc_functional); 
std::fprintf(f, "\nz_valence=\"%.3f\"", n_electrons[valence]);
std::fprintf(f, "\ntotal_psenergy=\"%g\"", 0.0);
std::fprintf(f, "\nrho_cutoff=\"%g\"", 0.0);
std::fprintf(f, "\nl_max=\"%d\"", l_max);
std::fprintf(f, "\nl_local=\"-1\"");
std::fprintf(f, "\nmesh_size=\"%d\"", n_mesh);
std::fprintf(f, "\nnumber_of_wfc=\"%d\"", n_wfc);
std::fprintf(f, "\nnumber_of_proj=\"%d\"", n_proj);
std::fprintf(f, "/>\n"); 
std::fprintf(f, "<PP_MESH>\n");
std::fprintf(f, "<PP_R type=\"real\" size=\"%d\" columns=\"4\">", n_mesh);
for (int ir = 0; ir < n_mesh; ++ir) {
std::fprintf(f, "%c%.12g", c4(ir), rg[SMT].r[ir]);
} 
std::fprintf(f, "\n</PP_R>\n");
std::fprintf(f, "<PP_RAB type=\"real\" size=\"%d\" columns=\"4\">", n_mesh);
for (int ir = 0; ir < n_mesh; ++ir) {
std::fprintf(f, "%c%.12g", c4(ir), rg[SMT].dr[ir]);
} 
std::fprintf(f, "\n</PP_RAB>\n");
std::fprintf(f, "</PP_MESH>\n");

std::fprintf(f, "<PP_LOCAL type=\"real\" size=\"%d\" columns=\"4\">", n_mesh);
for (int ir = 0; ir < n_mesh; ++ir) {
auto const r = rg[SMT].r[std::max(1, ir)];
auto const pp_local = -n_electrons[valence]*std::erf(r/sigma_cmp)/r; 
std::fprintf(f, "%c%.12g", c4(ir), (pp_local + zero_potential[std::max(1, ir)]*Y00)*2); 
} 
std::fprintf(f, "\n</PP_LOCAL>\n");

std::fprintf(f, "<PP_NONLOCAL>\n");
int const ircut = radial_grid::find_grid_index(rg[SMT], r_cut);
std::vector<double> h(n_proj*n_proj, 0.0); 
std::vector<double> s(n_proj*n_proj, 0.0); 
for (int ibeta = 0; ibeta < n_proj; ++ibeta) {
int const iln = iln_index[ibeta];
std::fprintf(f, "<PP_BETA.%d type=\"real\" size=\"%d\" columns=\"4\"\n", ibeta+1, n_mesh);
std::fprintf(f, "index=\"%i\" angular_momentum=\"%d\"\n", ibeta+1, valence_states[iln].ell);
std::fprintf(f, "cutoff_radius_index=\"%i\" cutoff_radius=\"%g\">", ircut, r_cut);
for (int ir = 0; ir < n_mesh; ++ir) {
std::fprintf(f, "%c%.12g", c4(ir), projector_functions(iln,ir));
} 
std::fprintf(f, "\n</PP_BETA.%d>\n", ibeta+1);
for (int jbeta = 0; jbeta < n_proj; ++jbeta) {
int const jln = iln_index[jbeta];
auto const delta_ell = (valence_states[iln].ell == valence_states[jln].ell);
h[ibeta*n_proj + jbeta] = aHSm(0,iln,jln)*delta_ell;
s[ibeta*n_proj + jbeta] = aHSm(1,iln,jln)*delta_ell;
} 
} 
std::fprintf(f, "<PP_DIJ type=\"real\" size=\"%d\" columns=\"4\">", n_proj*n_proj);
for (int ij = 0; ij < n_proj*n_proj; ++ij) {
std::fprintf(f, "%c%.12g", c4(ij), h[ij]*2); 
} 
std::fprintf(f, "\n</PP_DIJ>\n");
std::fprintf(f, "<PP_QIJ type=\"real\" size=\"%d\" columns=\"4\">", n_proj*n_proj);
for (int ij = 0; ij < n_proj*n_proj; ++ij) {
std::fprintf(f, "%c%.12g", c4(ij), s[ij]); 
} 
std::fprintf(f, "\n</PP_QIJ>\n");
std::fprintf(f, "</PP_NONLOCAL>\n");

if (echo > 5) std::printf("# not implemented: PP_PSWFC\n");

std::fprintf(f, "<PP_RHOATOM type=\"real\" size=\"%d\" columns=\"4\">", n_mesh);
for (int ir = 0; ir < n_mesh; ++ir) {
std::fprintf(f, "%c%.12g", c4(ir), spherical_density[SMT](core,ir) + spherical_density[SMT](valence,ir));
} 
std::fprintf(f, "\n</PP_RHOATOM>\n");

std::fprintf(f, "</UPF>\n");

} else {


std::fprintf(f, "<?xml version=\"%.1f\"?>\n", 1.0);
std::fprintf(f, "<paw_setup version=\"%.1f\">\n", 0.6); 
std::fprintf(f, "  <!-- Z=%g %s setup for the Projector Augmented Wave method. -->\n", Z, Sy);
std::fprintf(f, "  <!-- Units: Hartree and Bohr radii. -->\n");


std::fprintf(f, "  <atom symbol=\"%s\" Z=\"%g\" core=\"%g\" semicore=\"%g\" valence=\"%g\"/>\n",
Sy,  Z,  n_electrons[0], n_electrons[1], n_electrons[2]);

std::fprintf(f, "  <pw_ecut low=\"%.2f\" medium=\"%.2f\" high=\"%.2f\"/>\n", 12., 12., 15.); 

std::fprintf(f, "  <xc_functional type=\"LDA\" name=\"%s\"/>\n", xc_functional);
std::fprintf(f, "  <generator type=\"scalar-relativistic\" name=\"A43\" git=\"%s\">\n", git_key);
std::fprintf(f, "     %s\n", custom_configuration_string ? custom_configuration_string : Sy);
std::fprintf(f, "  </generator>\n");
std::fprintf(f, "  <ae_energy kinetic=\"%.6f\" xc=\"%.6f\"\n             electrostatic=\"%.6f\" "
"total=\"%.6f\"/>\n", kinetic_energy, xc_energy, es_energy, total_energy);
std::fprintf(f, "  <core_energy kinetic=\"%.6f\"/>\n", core_kinetic_energy);

std::fprintf(f, "  <valence_states>\n");
for (size_t iln = 0; iln < valence_states.size(); ++iln) {
if (valence_states_active[iln]) {
auto const & vs = valence_states[iln];
std::fprintf(f, "    <state");
std::fprintf(f, " n=\"%d\"", vs.enn); 
std::fprintf(f, " l=\"%d\"", vs.ell);
char occ[32]; occ[0] = '\0'; if (vs.occupation > 1e-24) std::snprintf(occ, 32, " f=\"%g\"", vs.occupation);
std::fprintf(f, "%-7s rc=\"%.3f\" e=\"%9.6f\" id=\"%s-%s\"/>\n", occ, r_cut, vs.energy, Sy,vs.tag);
} 
} 
std::fprintf(f, "  </valence_states>\n");

char ts_grid[TRU_AND_SMT][8]; 
double const prefactor = radial_grid::get_prefactor(rg[TRU]);
for (int ts = TRU; ts <= SMT; ++ts) {
std::snprintf(ts_grid[ts], 7, "g_%s", ts_tag[ts]);
if (ts == TRU || rg[SMT].n < rg[TRU].n) {
{
bool const reci = (radial_grid::equation_reciprocal == rg[ts].equation);
double const a = prefactor;
double const d = rg[TRU].anisotropy;
int const istart = ir0 + rg[TRU].n - rg[ts].n;
int const iend   = rg[TRU].n - 1;
double const n   = reci ? d : rg[ts].n - ir0;
std::fprintf(f, "  <radial_grid eq=\"%s\" a=\"%.15e\" d=\"%g\""
" n=\"%g\" istart=\"%d\" iend=\"%d\" id=\"g_%s\"%c>\n",
radial_grid::get_formula(rg[ts].equation), a, reci?0:d, n, istart, iend, ts_tag[ts], show_rg?' ':'/');
}
if (show_rg) { 
std::fprintf(f, "    <values>\n      ");
for (int ir = ir0; ir < rg[ts].n; ++ir) {
std::fprintf(f, " %.15e\n", rg[ts].r[ir]);
} 
std::fprintf(f, "    </values>\n    <derivative>\n      ");
for (int ir = ir0; ir < rg[ts].n; ++ir) {
std::fprintf(f, " %.15e\n", rg[ts].dr[ir]);
} 
std::fprintf(f, "    </derivative>\n  </radial_grid>\n");
} 
} else { 
std::snprintf(ts_grid[ts], 7, "g_%s", ts_tag[TRU]);
}
} 

std::fprintf(f, "  <shape_function type=\"gauss\" rc=\"%.12e\"/>\n",
sigma_cmp*std::sqrt(2.)); 

auto constexpr m = 1; 

if (n_electrons[0] > 0) {
int constexpr csv = 0; 
for (int rk = 0; rk < 1; ++rk) { 
auto const rk_tag = rk ? "_kinetic_energy" : "";
for (int ts = TRU; ts <= SMT; ++ts) {
std::fprintf(f, "  <%s_core%s_density grid=\"%s\">\n    ", ts_label[ts], rk_tag, ts_grid[ts]);
for (int ir = ir0; ir < rg[ts].n*m; ++ir) {
std::fprintf(f, " %.12e", spherical_density[ts](csv,ir)*Y00);
} 
std::fprintf(f, "\n  </%s_core%s_density>\n", ts_label[ts], rk_tag);
} 
} 
} 

{   int constexpr csv = 1; 
int const ir = radial_grid::find_grid_index(rg[TRU], r_cut);
if (0 != spherical_density[TRU](csv,ir) && echo > 0) {
std::printf("# %s %s Z=%g semicore density must vanish!", Sy, __func__, Z);
} 
} 

if (n_electrons[2] > 0) {
int constexpr csv = 2; 
{   auto const ts = SMT; 
std::fprintf(f, "  <%s_valence_density grid=\"%s\">\n    ", ts_label[ts], ts_grid[ts]);
for (int ir = ir0; ir < rg[ts].n*m; ++ir) {
std::fprintf(f, " %.12e", spherical_density[ts](csv,ir)*Y00);
} 
std::fprintf(f, "\n  </%s_valence_density>\n", ts_label[ts]);
} 
} 

if (zero_potential != nullptr) { 
auto const ts = SMT;
std::fprintf(f, "  <zero_potential grid=\"%s\">\n    ", ts_grid[ts]);
for (int ir = ir0; ir < rg[ts].n*m; ++ir) {
std::fprintf(f, " %.12e", zero_potential[ir]);
} 
std::fprintf(f, "\n  </zero_potential>\n");
} 

for (size_t iln = 0; iln < valence_states.size(); ++iln) {
if (valence_states_active[iln]) {
auto const & vs = valence_states[iln];
for (int ts = TRU; ts <= SMT; ++ts) {
std::fprintf(f, "  <%s_partial_wave state=\"%s-%s\" grid=\"%s\">\n    ", ts_label[ts], Sy,vs.tag, ts_grid[ts]);
if (nullptr == vs.wave[ts]) {
if (echo > 0) std::printf("# %s %s found nullptr in partial wave iln=%li\n", Sy, __func__, iln);
return __LINE__; 
} 
for (int ir = ir0; ir < rg[ts].n*m; ++ir) {
std::fprintf(f, " %.12e", vs.wave[ts][ir]);
} 
std::fprintf(f, "\n  </%s_partial_wave>\n", ts_label[ts]);
} 
{   auto const ts = SMT;
std::fprintf(f, "  <projector_function state=\"%s-%s\" grid=\"%s\">\n    ", Sy,vs.tag, ts_grid[ts]);
for (int ir = ir0; ir < rg[ts].n*m; ++ir) {
std::fprintf(f, " %.12e", projector_functions(iln,ir)*rg[ts].rinv[ir]);
} 
std::fprintf(f, "\n  </projector_function>\n");
} 
} 
} 


{ 
std::fprintf(f, "  <kinetic_energy_differences>\n");
for (size_t iln = 0; iln < valence_states.size(); ++iln) {
if (valence_states_active[iln]) {
std::fprintf(f, "    ");
for (size_t jln = 0; jln < valence_states.size(); ++jln) {
if (valence_states_active[jln]) {
std::fprintf(f, " %.12e", kinetic_energy_differences(TRU,iln,jln)
- kinetic_energy_differences(SMT,iln,jln));
} 
} 
std::fprintf(f, " \n");
} 
} 
std::fprintf(f, "  </kinetic_energy_differences>\n");
} 

std::fprintf(f, "  <!-- exact_exchange_X_matrix not included -->\n");
std::fprintf(f, "  <exact_exchange core-core=\"0\"/>\n");

std::fprintf(f, "</paw_setup>\n"); 

} 

std::fclose(f);

if (echo > 3) std::printf("# %s %s file '%s' written\n", Sy, __func__, filename);
return 0; 
} 


} 
