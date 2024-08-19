
#include <cstdio> 
#include <cassert> 
#include <algorithm> 
#include <cmath> 
#include <cstdint> 
#include <fstream> 
#include <sstream> 
#include <string> 
#include <vector> 

#include "geometry_analysis.hxx" 

#include "boundary_condition.hxx" 
#include "display_units.h" 
#include "inline_math.hxx" 
#include "constants.hxx" 
#include "data_view.hxx" 
#include "vector_math.hxx" 
#include "chemical_symbol.hxx" 
#include "recorded_warnings.hxx" 
#include "simple_timer.hxx" 
#include "simple_stats.hxx" 
#include "control.hxx" 

#ifndef NO_UNIT_TESTS
#include <fstream> 
#endif

#define FULL_DEBUG
#define DEBUG

namespace geometry_analysis {

double constexpr Bohr2Angstrom = 0.52917724924;
double constexpr Angstrom2Bohr = 1./Bohr2Angstrom;

status_t read_xyz_file(
view2D<double> & xyzZ
, int & n_atoms
, char const *filename 
, double cell[] 
, int8_t bc[] 
, int const echo 
) {

std::ifstream infile(filename, std::ifstream::in);
if (infile.fail()) {
warn("unable to open file '%s' for reading coordinates", filename);
return 1; 
}

int natoms{0}, linenumber{2};
infile >> natoms; 
std::string line;
std::getline(infile, line);
std::getline(infile, line);
if (echo > 2) std::printf("# expect %d atoms, comment line in file \'%s\': %s\n", natoms, filename, line.c_str());
{ 
std::istringstream iss(line);
std::string Cell, B[3];
double L[3]; 
iss >> Cell >> L[0] >> L[1] >> L[2] >> B[0] >> B[1] >> B[2];
for (int d = 0; d < 3; ++d) {
if (nullptr != cell) {
cell[d] = L[d] * Angstrom2Bohr;
assert(cell[d] > 0);
}
if (nullptr != bc) bc[d] = boundary_condition::fromString(B[d].c_str(), echo, 120+d);
} 
} 
xyzZ = view2D<double>(natoms, 4);
int na{0}; 
while (std::getline(infile, line)) {
++linenumber;
std::istringstream iss(line);
std::string Symbol;
double px, py, pz; 
if (!(iss >> Symbol >> px >> py >> pz)) {
if (na < natoms) 
std::printf("# failed parsing in %s:%d reads \"%s\", stop\n", filename, linenumber, line.c_str());
break; 
}
char const *Sy = Symbol.c_str();
char const S = Sy[0], y = (Sy[1])? Sy[1] : ' ';
if (echo > 7) std::printf("# %c%c  %16.9f %16.9f %16.9f\n", S,y, px, py, pz);
int const iZ = chemical_symbol::decode(S, y);
xyzZ(na,0) = px*Angstrom2Bohr;
xyzZ(na,1) = py*Angstrom2Bohr;
xyzZ(na,2) = pz*Angstrom2Bohr;
xyzZ(na,3) = iZ;
++na;
assert(na <= natoms);
} 

n_atoms = na;
return na - natoms; 
} 

float default_bond_length(int const Z1, int const Z2=0) {
int16_t const half_bond_in_pm[128] = {0, 
31,  28,                                                             
128,  96,                                                             
84,  76,  73,  69,  71,  66,                                         
57, 141,                                                             
121, 111, 107, 105, 102, 106,                                         
203, 176,                                                             
130, 160, 153, 139, 132, 110, 139, 124, 132, 122,                     
122, 120, 119, 120, 120, 116,                                         
220, 195,                                                             
190, 175, 164, 154, 147, 146, 142, 139, 145, 144,                     
142, 139, 139, 138, 139, 140,                                         
244, 215,                                                             
207, 204, 203, 201, 199, 198, 198, 196, 194, 192, 192, 189, 190, 187, 
187, 175, 170, 162, 151, 144, 141, 136, 136, 132,                     
145, 146, 148, 140, 150, 150,                                         
260, 221,                                                             
215, 206, 200, 196, 190, 187, 180, 169, 166, 168, 165, 167, 173, 176, 
161, 157, 149, 143, 141, 134, 129, 128, 121, 122,                     
136, 143, 162, 175, 165, 157,                                         
159, 160,                                                             
161, 162, 163, 164, 165, 166, 167};                                   
float const picometer2Bohr = 0.01*Angstrom2Bohr;
return (half_bond_in_pm[Z1 & 127] + half_bond_in_pm[Z2 & 127]) * picometer2Bohr;
} 



template <typename int_t>
class BoxStructure {
private:
int static constexpr X=0, Y=1, Z=2;
std::vector<std::vector<int_t>> atom_index; 
std::vector<int> indirection;
std::vector<int16_t> image_index;
int64_t n_all_atoms; 
int nboxes[3];
int nhalo[3];

template <int D> int inline get_center_box(int const j) { return (j + 999*nboxes[D]) % nboxes[D]; }
int inline get_center_index(int const ix, int const iy, int const iz) {
assert(ix >= 0); assert(ix < nboxes[X]);
assert(iy >= 0); assert(iy < nboxes[Y]);
assert(iz >= 0); assert(iz < nboxes[Z]);
return (iz*nboxes[Y] + iy)*nboxes[X] + ix; }

public:

BoxStructure(
double const cell[3]
, int8_t const bc[3]
, double const radius
, size_t const natoms
, view2D<double> const & xyzZ 
, int const echo=0 
) {
assert(radius > 0);
double box_size[3], inv_box_size[3], inv_nboxes[3];
int hnh[3];
int nbx{1}; 
int mbx{1}; 
for (int d = 0; d < 3; ++d) {
assert(cell[d] > 0);
nboxes[d] = std::max(1, int(cell[d]/radius));
inv_nboxes[d] = 1./nboxes[d];
box_size[d] = cell[d]/nboxes[d];
inv_box_size[d] = 1./box_size[d];
nhalo[d] = (Periodic_Boundary == bc[d]) ? int(std::ceil(radius*inv_box_size[d])) : 1;
hnh[d] = nboxes[d] + 2*nhalo[d];
nbx *= hnh[d]; 
mbx *= nboxes[d]; 
} 
if (echo > 2) std::printf("# %s: divide cell %.3f x %.3f x %.3f %s^3 into %d x %d x %d boxes\n", __func__,
cell[X]*Ang, cell[Y]*Ang, cell[Z]*Ang, _Ang, nboxes[X], nboxes[Y], nboxes[Z]);
if (echo > 2) std::printf("# %s: box size %.3f x %.3f x %.3f %s^3 for interaction radius %.3f %s\n", __func__,
box_size[X]*Ang, box_size[Y]*Ang, box_size[Z]*Ang, _Ang, radius*Ang, _Ang);
if (echo > 2) std::printf("# %s: use %d %d %d halo boxes on each side\n", __func__, nhalo[X], nhalo[Y], nhalo[Z]);

indirection.resize(nbx);
image_index.resize(nbx*4);
{ 
for        (int jz = 0; jz < hnh[Z]; ++jz) {  int const iz = get_center_box<Z>(jz - nhalo[Z]);
for    (int jy = 0; jy < hnh[Y]; ++jy) {  int const iy = get_center_box<Y>(jy - nhalo[Y]);
if (echo > 9) std::printf("# %s: indirection(iz=%2d, iy=%2d):", __func__, jz - nhalo[Z], jy - nhalo[Y]);
for (int jx = 0; jx < hnh[X]; ++jx) {  int const ix = get_center_box<X>(jx - nhalo[X]);
int ib = get_center_index(ix, iy, iz);
int const jb = (jz*hnh[Y] + jy)*hnh[X] + jx;
image_index[jb*4 + X] = int(std::floor((jx - nhalo[X])*inv_nboxes[X]));
image_index[jb*4 + Y] = int(std::floor((jy - nhalo[Y])*inv_nboxes[Y]));
image_index[jb*4 + Z] = int(std::floor((jz - nhalo[Z])*inv_nboxes[Z]));
if ((Isolated_Boundary == bc[X]) && (0 != image_index[jb*4 + X])) ib = -1;
if ((Isolated_Boundary == bc[Y]) && (0 != image_index[jb*4 + Y])) ib = -1;
if ((Isolated_Boundary == bc[Z]) && (0 != image_index[jb*4 + Z])) ib = -1;
indirection[jb] = ib;
if (echo > 9) std::printf(" %2d", ib);
} 
if (echo > 9) std::printf("\n");
} 
} 
} 

atom_index.resize(mbx);
size_t const estimate = std::ceil(natoms/(mbx*.875));
if (echo > 2) std::printf("# %s: use %ld atoms as box balance estimate\n", __func__, estimate);
for (int ib = 0; ib < mbx; ++ib) {
atom_index[ib].reserve(estimate);
} 

if (mbx > 1) {
if (echo > 8) std::printf("# %s: inverse box size is %g %g %g\n",
__func__, inv_box_size[X], inv_box_size[Y], inv_box_size[Z]);
double min_coords[3] = {9e99, 9e99, 9e99};
for (size_t ia = 0; ia < natoms; ++ia) {
for (int d = 0; d < 3; ++d) {
min_coords[d] = std::min(min_coords[d], xyzZ[ia][d]);
} 
} 
double const offset[3] = {-min_coords[X], -min_coords[Y], -min_coords[Z]};

for (size_t ia = 0; ia < natoms; ++ia) {
int const iz = get_center_box<Z>(int(std::floor((xyzZ[ia][Z] + offset[Z])*inv_box_size[Z])));
int const iy = get_center_box<Y>(int(std::floor((xyzZ[ia][Y] + offset[Y])*inv_box_size[Y])));
int const ix = get_center_box<X>(int(std::floor((xyzZ[ia][X] + offset[X])*inv_box_size[X])));
if (echo > 9) std::printf("# %s: atom #%ld Z=%.1f at %g %g %g goes into box %d %d %d\n", __func__,
ia, xyzZ[ia][3], xyzZ[ia][X], xyzZ[ia][Y], xyzZ[ia][Z], ix, iy, iz);
int const ib = get_center_index(ix, iy, iz);
atom_index[ib].push_back(ia);
} 
} else {
if (echo > 2) std::printf("# %s: only 1 box, list is trivial\n", __func__);
for (size_t ia = 0; ia < natoms; ++ia) {
atom_index[0].push_back(ia);
} 
} 

{ 
simple_stats::Stats<> stats;
for (size_t ib = 0; ib < atom_index.size(); ++ib) {
stats.add(atom_index[ib].size());
} 
if (echo > 2) std::printf("# %s: box balance min %g avg %.2f rms %.2f max %g\n",
__func__, stats.min(), stats.mean(), stats.dev(), stats.max());
assert(stats.sum() == natoms);
} 

} 

size_t get_atom_list(int_t const **list, int jx, int jy, int jz, int *ii=nullptr) const {
int const hnh[2] = {nboxes[X] + 2*nhalo[X], nboxes[Y] + 2*nhalo[Y]};
assert(jx >= -nhalo[X]); assert(jx < nboxes[X] + nhalo[X]);
assert(jy >= -nhalo[Y]); assert(jy < nboxes[Y] + nhalo[Y]);
assert(jz >= -nhalo[Z]); assert(jz < nboxes[Z] + nhalo[Z]);
int const jb = ((jz + nhalo[Z])*hnh[Y] + (jy + nhalo[Y]))*hnh[X] + (jx + nhalo[X]);
for (int d = 0; d < 3; ++d) {
if (ii != nullptr) ii[d] = image_index[jb*4 + d];
} 
int const ib = indirection[jb];
*list = (ib >= 0) ? atom_index[ib].data() : nullptr;
return  (ib >= 0) ? atom_index[ib].size() : 0;
} 

int get_number_of_boxes() const { return nboxes[X]*nboxes[Y]*nboxes[Z]; }
int get_number_of_boxes(unsigned const D) const { assert(D < 3); return nboxes[D]; }
inline int get_halo_thickness(unsigned const D) const { assert(D < 3); return nhalo[D]; }

}; 




typedef uint32_t index_t;

class atom_image_index_t {
public:
index_t ia; 
int8_t is; 
int8_t ix, iy, iz; 
atom_image_index_t(index_t const atom_index=0, int const species_index=0,
int const iix=0, int const iiy=0, int const iiz=0)
: ia(atom_index), is(species_index), ix(iix), iy(iiy), iz(iiz) {} 
}; 


template <typename real_t, typename int_t=short>
int print_summary(
char string[] 
, real_t values[] 
, int_t const na 
, double const grid_factor=1
, double const display_factor=1
, char const mult_char='_' 
, char const *const fmt1=" %g"
, int const MaxBufferLen=-1
, int const MaxEntryLen=24
) {
auto const string_start = string;
bool const no_length_check = (MaxBufferLen < 0);

std::sort(values, values + na);

char fmtn[16];
int const nfc = std::snprintf(fmtn, 16, "%s%c%%d", fmt1, mult_char);
assert(nfc < 16);

std::vector<int_t> ivalues(na);
std::vector<int_t> occurrence(na, 1); 
for (int_t ia = 0; ia < na; ++ia) {
ivalues[ia] = std::round(values[ia]*grid_factor); 
if (ia > 0) {
if (ivalues[ia - 1] == ivalues[ia]) { 
occurrence[ia] += occurrence[ia - 1];
occurrence[ia - 1] = 0;
} 
} 
} 

int nbytes{0}; 
for (int_t ia = 0; ia < na; ++ia) {
int_t  const count = occurrence[ia];
double const value = ivalues[ia]*display_factor;
if (count > 0) {
if (no_length_check || ((string + MaxEntryLen) < (string_start + MaxBufferLen))) {
int const nchars = (count < 2) ? std::snprintf(string, MaxEntryLen, fmt1, value)
: std::snprintf(string, MaxEntryLen, fmtn, value, count);
assert( nchars <= MaxEntryLen );
nbytes += nchars;
string += nchars;
} 
} 
} 
return nbytes;
} 


template <typename int_t, typename real_t>
int add_to_histogram(
int_t hist[]
, int const hmax
, real_t const data[]
, int const n
, double const factor=1
) {
if ((nullptr == hist) || (hmax < 1) || (nullptr == data)) return 0;
int out_of_range{0};
for (int i = 0; i < n; ++i) {
int const ih = std::round(data[i]*factor);
if ((ih >= 0) && (ih < hmax)) {
++hist[ih];
} else {
++out_of_range;
}
} 
return out_of_range;
} 


template <typename real_t>
void analyze_bond_structure(
char* string
, int const nb 
, real_t const bond_vectors[]
, float const Z

, index_t const ia=-1
, uint32_t*    angle_hist=nullptr
, int    const angle_max=0
, double const angle_fac=0
, uint32_t*    length_hist=nullptr
, int    const length_max=0
, double const length_fac=0
) {
if (nullptr != string) string[0] = '\0';
if (nb < 1) return;
view2D<real_t const> const bv(bond_vectors, 4); 

std::vector<real_t> bond_length(nb);
{ 
for (int ib = 0; ib < nb; ++ib) {    auto const bvi = bv[ib];
auto const d2i = pow2(bvi[0]) + pow2(bvi[1]) + pow2(bvi[2]);
bond_length[ib] = std::sqrt(d2i);
} 
} 


int const na = (nb*(nb - 1))/2; 
std::vector<real_t> bond_angle(na);
{ 
int ia{0};
for (int ib = 0; ib < nb; ++ib) {      auto const bvi = bv[ib];
for (int jb = 0; jb < ib; ++jb) {  auto const bvj = bv[jb];
auto const dot = bvj[0]*bvi[0] + bvj[1]*bvi[1] + bvj[2]*bvi[2];
auto const bliblj = bond_length[ib]*bond_length[jb];
bond_angle[ia] = (-dot < bliblj) ? std::acos(dot/bliblj)
: constants::pi;
++ia;
} 
} 
assert(na == ia);
} 

real_t constexpr Deg = 180/constants::pi; 
if (nullptr != string) {
string += print_summary(string, bond_length.data(), nb, Ang/.01, .01, '_', " %.2f"); 
string += std::snprintf(string, 8, " | "); 
string += print_summary(string, bond_angle.data(), na, Deg/2., 2., '_'); 
} 

{   auto const out = add_to_histogram(angle_hist, angle_max, bond_angle.data(), na, angle_fac);
if (out > 0) warn("%d angles out-of-range for atom #%i", out, ia);
}

{   auto const out = add_to_histogram(length_hist, length_max, bond_length.data(), nb, length_fac);
if (out > 0) warn("%d distances out-of-range for atom #%i", out, ia);
}

} 

template <typename int_t=int32_t>
class SparsifyPlot {
public:

SparsifyPlot(bool const show0=false) {
assert(int_t(-1) < 0); 
last[0] = show0 ? 0 : -1; last[1] = -1; last[2] = -1; last[3] = -1;
} 

int_t previous(int_t const ih, bool const nonzero=true) {
last[(ih - 2) & 0x3] = -1; 
if (nonzero) {
last[(ih - 1) & 0x3] = ih - 1;
last[(ih    ) & 0x3] = ih; 
last[(ih + 1) & 0x3] = ih + 1;
} 
return last[(ih - 1) & 0x3]; 
} 

private:
int_t last[4]; 
}; 


status_t analysis(
view2D<double> const & xyzZ 
, index_t const natoms 
, double const cell[3] 
, int8_t const bc[3] 
, int const echo=6 
) {
status_t stat(0);
if (echo > 1) std::printf("\n# %s:%s\n", __FILE__, __func__);

float const elongation = control::get("geometry_analysis.elongation", 1.25); 
if (echo > 3) std::printf("# Count interatomic distances up to %g %% of the default bond length as bond\n", elongation*100);

double const max_range = control::get("geometry_analysis.max.range", 6.5*Angstrom2Bohr); 
double const bin_width = control::get("geometry_analysis.bin.width", .02*Angstrom2Bohr);
double const inv_bin_width = (bin_width > 0) ? 1./bin_width : 0;
double const minimum_range = 10.; 
double const rcut  = std::max(minimum_range, max_range); 
int const num_bins = std::max(1, int(std::ceil(rcut*inv_bin_width))); 

if (echo > 4) {
std::printf("# Bond search within interaction radius %.3f %s\n", max_range*Ang, _Ang);
if (num_bins > 1) std::printf("# Distance histogram: %d bins of width %.6f %s up to %.3f %s\n",
num_bins, bin_width*Ang, _Ang, num_bins*bin_width*Ang, _Ang);
} 

int const MaxBP        = control::get("geometry_analysis.max.bond.partners", 24.); 
index_t max_natoms_BP  = control::get("geometry_analysis.max.atoms.with.partners", 2000.);
index_t const natoms_BP = std::min(natoms, max_natoms_BP); 
std::vector<std::vector<atom_image_index_t>> bond_partner((MaxBP > 0)*natoms_BP);

std::vector<int8_t> ispecies(natoms, 0);
std::vector<int> occurrence(128, 0);
int8_t Z_of_species[128]; Z_of_species[0] = -1;
int8_t species_of_Z[128];

int nspecies{0}; 
{ 

for (int first = 1; first >= 0; --first) {
for (index_t ia = 0; ia < natoms; ++ia) {
int const Z_ia = std::round(xyzZ[ia][3]);
int const Z_ia_mod = Z_ia & 127; 
if (first) {
++occurrence[Z_ia_mod];
} else {
ispecies[ia] = species_of_Z[Z_ia_mod];
} 
} 
if (first) {
int is{0};
for (int Z = 0; Z < 128; ++Z) {
if (occurrence[Z] > 0) {
species_of_Z[Z] = is;
Z_of_species[is] = Z;
++is; 
} else {
assert(0 == occurrence[Z]); 
} 
} 
nspecies = is;
} 
} 

} 


char  Sy_of_species[128][4];       
char  Sy_of_species_null[128][4];  
char  Sy_of_species_right[128][4]; 
for (int is = 0; is < nspecies; ++is) {
chemical_symbol::get(Sy_of_species_null[is], Z_of_species[is]);
chemical_symbol::get(Sy_of_species[is], Z_of_species[is], ' ');
set(Sy_of_species_right[is], 4, Sy_of_species[is]);
if (' ' == Sy_of_species[is][1]) {
std::swap(Sy_of_species_right[is][0], Sy_of_species_right[is][1]);
} 
Sy_of_species_right[is][2] = '\0';
} 


if (echo > 2) {
std::printf("# Found %d different elements for %d atoms:  ", nspecies, natoms);
for (int is = 0; is < nspecies; ++is) {
std::printf("  %s_%d", Sy_of_species_null[is], occurrence[Z_of_species[is]]);
} 
std::printf("\n");
} 
assert((natoms > 0) == (nspecies > 0));

std::vector<float> half_bond_length(nspecies, 0.f);
{ 


float hypothetical_bond{0};
int half_bond_lengths_modified{0};
for (int is = 0; is < nspecies; ++is) {
char keyword[64];
std::snprintf(keyword, 64, "geometry_analysis.half.bond.%s", Sy_of_species_null[is]);
double const default_half_bond = default_bond_length(Z_of_species[is]);
half_bond_length[is] = control::get(keyword, default_half_bond);
if (default_half_bond != half_bond_length[is]) {
if (echo > 3) std::printf("# customize half bond length for %s from %.3f %s to %.3f %s\n",
Sy_of_species_null[is], default_half_bond*Ang, _Ang, half_bond_length[is]*Ang, _Ang);
++half_bond_lengths_modified;
}
hypothetical_bond = std::max(hypothetical_bond, elongation*1.999999f*half_bond_length[is]);
} 
if (half_bond_lengths_modified) {
if (echo > 0) std::printf("# %d of %d half bond lengths have been customized\n", half_bond_lengths_modified, nspecies);
} 

if (max_range < hypothetical_bond) {
warn("max.range (%g %s) may truncate bonds, better use %g %s or more!",
max_range*Ang, _Ang, hypothetical_bond*Ang, _Ang);
} else if (echo > 3) {
std::printf("# Hint: geometry_analysis.max.range=%g Bohr would be sufficient\n", hypothetical_bond);
} 
} 


int const nspecies2 = pow2(nspecies); 
view3D<uint16_t> dist_hist(num_bins, nspecies, nspecies, 0);
view2D<int> bond_hist(nspecies, nspecies, 0);
std::vector<uint8_t> coordination_number(natoms, 0);
int constexpr MAX_coordination_number = std::numeric_limits<uint8_t>::max();

float const too_large = 188.973; 
view2D<double> smallest_distance(nspecies, nspecies,  too_large);
view2D<simple_stats::Stats<double>> bond_stat(nspecies, nspecies);

view2D<double> image_pos;
view2D<int8_t> image_shift;
typedef vector_math::vec<3,double> vec3;
double const rcut2 = pow2(rcut);
int64_t nzero{0}, nstrange{0}, npairs{0}, nbonds{0}, nfar{0}, near{0}; 
int64_t bp_exceeded{0}, bp_truncated{0}; 

#ifdef  GEO_ORDER_N2
int const nimages = boundary_condition::periodic_images(image_pos, cell, bc, rcut, echo - 9, &image_shift);
if (echo > 7) std::printf("# use N^2-algorithm, expect to visit %.1e atom pairs\n", nimages*pow2(1.*natoms));

{{{{{ 
for (int ii = 0; ii < nimages; ++ii) { 
vec3 const pos_ii = image_pos[ii];
auto const shift = image_shift[ii]; 
for (index_t ia = 0; ia < natoms; ++ia) {
#else  
BoxStructure<index_t> box(cell, bc, rcut, natoms, xyzZ, 0*echo); 

for (int ibz = 0; ibz < box.get_number_of_boxes(2); ++ibz) {
for (int iby = 0; iby < box.get_number_of_boxes(1); ++iby) {
for (int ibx = 0; ibx < box.get_number_of_boxes(0); ++ibx) {
index_t const *list_i;
int const na_i = box.get_atom_list(&list_i, ibx, iby, ibz);
for (int jbz = ibz - box.get_halo_thickness(2); jbz <= ibz + box.get_halo_thickness(2); ++jbz) {
for (int jby = iby - box.get_halo_thickness(1); jby <= iby + box.get_halo_thickness(1); ++jby) {
for (int jbx = ibx - box.get_halo_thickness(0); jbx <= ibx + box.get_halo_thickness(0); ++jbx) {
index_t const *list_j;
int shift[3];
int const na_j = box.get_atom_list(&list_j, jbx, jby, jbz, shift);
double const ppos_i[3] = {shift[0]*cell[0], shift[1]*cell[1], shift[2]*cell[2]}; 
vec3 const pos_ii = ppos_i; 

for (int iia = 0; iia < na_i; ++iia) { index_t const ia = list_i[iia];
#endif 
vec3 const pos_ia = xyzZ[ia];
int const is = ispecies[ia];
if (echo > 8) std::printf("# [ia=%i] pos_ia = %g %g %g Z=%d\n", ia, pos_ia[0],pos_ia[1],pos_ia[2], Z_of_species[is]);
vec3 const pos_ii_minus_ia = pos_ii - pos_ia;
#ifdef  GEO_ORDER_N2
for (index_t ja = 0; ja < natoms; ++ja) {
#else  
for (int ija = 0; ija < na_j; ++ija) { index_t const ja = list_j[ija];
#endif 
vec3 const pos_ja = xyzZ[ja];
int const js = ispecies[ja];
if (echo > 9) std::printf("# [ia=%i, ja=%i] pos_ja = %g %g %g Z=%d\n", ia, ja, pos_ja[0],pos_ja[1],pos_ja[2], Z_of_species[js]);
vec3 const diff = pos_ja + pos_ii_minus_ia;
auto const d2 = norm(diff);
if (d2 >= rcut2) {
++nfar; 
} else if (d2 < 1e-6) {
if (ia == ja) {
++nzero; 
} else {
++nstrange; 
if (echo > 0) std::printf("# %s found a strange atom pair: ia=%i ja=%i shift= %g %g %g a.u. distance= %g %s\n",
__func__, ia, ja, pos_ii[0],pos_ii[1],pos_ii[2], std::sqrt(d2)*Ang, _Ang);
} 
++near;
} else {
++near;
auto const dist = std::sqrt(d2);
{
int const ibin = int(dist*inv_bin_width);
if (ibin < num_bins) ++dist_hist(ibin,is,js);
}
if (Z_of_species[js] > 0) { 
auto const longest_bond = elongation*(half_bond_length[is] + half_bond_length[js]);
if (dist < longest_bond) {
++nbonds;
if (MaxBP > 0) { 
int const cn = coordination_number[ia];
if (ia < bond_partner.size()) {
if (0 == cn) bond_partner[ia].clear();
if (cn < MaxBP) {
bond_partner[ia].push_back(atom_image_index_t(ja, js, shift[0], shift[1], shift[2]));
} else {
++bp_exceeded;
}
} else {
++bp_truncated;
} 
} 
++coordination_number[ia];
assert( coordination_number[ia] <= MAX_coordination_number );
++bond_hist(is,js);
bond_stat(is,js).add(dist);
} 
} 
smallest_distance(is,js) = std::min(smallest_distance(is,js), dist);
} 
++npairs;
} 
} 
} 
}}}}} 

if (echo > 2) std::printf("# checked %.6f M atom-atom pairs, %.3f k near and %.6f M far\n", 1e-6*npairs, 1e-3*near, 1e-6*nfar);
if (natoms != nzero) {
warn("Should find %d exact zero distances but found %ld", natoms, nzero);
++stat;
} 
assert(natoms == nzero);
assert(0 == nstrange);

if ((bp_exceeded > 0) && (MaxBP > 0)) {
warn("In %ld cases, the max. number of bond partners (MaxBP=%d) was exceeded", bp_exceeded, MaxBP);
++stat;
} 
if (bp_truncated > 0) {
warn("Bond partner analysis is performed only for the first %d of %d atoms", natoms_BP, natoms);
++stat;
} 


int const is_start = (0 == Z_of_species[0]); 
if (true) { 
auto const compression = 1/elongation; 
auto const & Sy = Sy_of_species_null;
double minimum_distance{9e37}; int is_min[2] = {-1, -1};
for (int is = is_start; is < nspecies; ++is) {
for (int js = is_start; js < nspecies; ++js) {
if (smallest_distance(is,js) < minimum_distance) {
minimum_distance = smallest_distance(is,js);
is_min[0] = is; is_min[1] = js;
}

auto const shortest_bond_distance = bond_stat(is,js).min();
if (shortest_bond_distance < too_large) {
if (smallest_distance(is,js) < shortest_bond_distance) {
warn("%s-%s distance %g is below the shortest bond %g %s", Sy[is], Sy[js],
smallest_distance(is,js)*Ang, shortest_bond_distance*Ang, _Ang);
++stat;
}
auto const short_bond = compression*(half_bond_length[is] + half_bond_length[js]);
if (shortest_bond_distance < short_bond) {
warn("%s-%s distance %g is below the compressed bond length %g %s", Sy[is], Sy[js],
shortest_bond_distance*Ang, short_bond*Ang, _Ang);
}
}

if (std::abs(smallest_distance(is,js) - smallest_distance(js,is)) > 1e-9) {
warn("smallest distance for %s-%s asymmetric!", Sy[is], Sy[js]);
++stat;
}
if (bond_hist(is,js) != bond_hist(js,is)) {
warn("count of %s-%s bonds asymmetric!", Sy[is], Sy[js]);
++stat;
}
} 
} 

if (minimum_distance < 1) { 
warn("Minimum distance between two atoms (%s-%s) is %.3f %s",
Sy_of_species_null[is_min[0]], Sy_of_species_null[is_min[1]], minimum_distance*Ang, _Ang);
++stat;
} else if (echo > 3) {
std::printf("# Minimum distance between two atoms (%s-%s) is %.3f %s\n",
Sy_of_species_null[is_min[0]], Sy_of_species_null[is_min[1]], minimum_distance*Ang, _Ang);
} 

} 



if (num_bins > 1) {
bool const echoh = (echo > 5);
size_t ij_dev{0};
if (echoh) {
std::printf("\n## distance histogram (in %s):", _Ang);
for (int is = 0; is < nspecies; ++is) {
for (int js = is; js < nspecies; ++js) { 
std::printf(" %s-%s", Sy_of_species_null[is], Sy_of_species_null[js]);
} 
std::printf((is < nspecies - 1) ? " " : "\n");
} 
} 
SparsifyPlot<int> spp(true); 
for (int ibin = 0; ibin < num_bins; ++ibin) {
#ifdef    PLOT_ALL_HISTOGRAM_POINTS
int const jbin = ibin - 1;
#else  
bool nonzero{false}; 
for (int ijs = 0; ijs < nspecies2; ++ijs) {
nonzero = nonzero || (dist_hist(ibin,0)[ijs] > 0); 
} 
int const jbin = spp.previous(ibin, nonzero);
#endif 
if (jbin >= 0) {
float const dist = (jbin + 0.5)*bin_width; 
if (echoh) std::printf("%.3f ", dist*Ang);
auto const hist = dist_hist[jbin];
for (int is = 0; is < nspecies; ++is) {
if (echoh) std::printf("  %d", hist(is,is)); 
for (int js = is + 1; js < nspecies; ++js) { 
if (echoh) std::printf(" %d", hist(is,js) + hist(js,is)); 
ij_dev += (hist(is,js) != hist(js,is));
if (hist(is,js) != hist(js,is)) {
warn("histogram asymmetry at %.3f %s %s-%s: %d and %d", dist*Ang, _Ang,
Sy_of_species_null[is], Sy_of_species_null[js], hist(is,js), hist(js,is));
} 
} 
} 
if (echoh) std::printf("\n");
} 
} 
if (ij_dev > 0) {
warn("histogram has %.3f k asymmetries", ij_dev*.001);
++stat;
} 
dist_hist = view3D<uint16_t>(0,0,0, 0); 
} 



if (echo > 3) {
if (bond_partner.size() > 0) {

int const nhist = control::get("geometry_analysis.bond.num.bins", 184.);
auto const per_degree = 180/constants::pi; 
auto const per_length = 1./control::get("geometry_analysis.bond.bin.width", .0625); 
view3D<uint32_t> bond_angle_length_hist(2, nspecies, nhist, 0);

int const sbond = control::get("geometry_analysis.show.bond.structure", 1.);
bool const show = (echo > 5) && (sbond > 0);
if (show) std::printf("\n# bond structure analysis: "
"bond lengths are in %s | angles in degree\n", _Ang);
for (index_t ia = 0; ia < bond_partner.size(); ++ia) {
int const cn = std::min(int(coordination_number[ia]), MaxBP);
double const xyz_ia[3] = {xyzZ[ia][0], xyzZ[ia][1], xyzZ[ia][2]}; 
assert(cn == bond_partner[ia].size());
view2D<float> coords(cn, 4, 0.0); 
for (int ip = 0; ip < cn; ++ip) {
auto const & partner = bond_partner[ia][ip];
auto const ja = partner.ia; assert(ja >= 0);
coords[ip][0] = xyzZ[ja][0] + partner.ix*cell[0] - xyz_ia[0];
coords[ip][1] = xyzZ[ja][1] + partner.iy*cell[1] - xyz_ia[1];
coords[ip][2] = xyzZ[ja][2] + partner.iz*cell[2] - xyz_ia[2];
coords[ip][3] = xyzZ[ja][3]; 
} 
bond_partner[ia].clear(); 
int const is = ispecies[ia];
char string_buffer[2048];
analyze_bond_structure(show?string_buffer:nullptr, cn, coords.data(), xyzZ[ia][3]
, ia, bond_angle_length_hist(0,is), nhist, per_degree
, bond_angle_length_hist(1,is), nhist, per_length
);
if (show) std::printf("# a#%i %s %s\n", ia, Sy_of_species[is], string_buffer); 
} 

if (nhist > 0) {
if (echo > 3) {
for (int ab = 0; ab < 2; ++ab) {
auto const fac = ab ? Ang/per_length : 1;
std::printf("\n## bond %s in %s for ", ab?"distances":"angles", ab?_Ang:"degree");
for (int is = 0; is < nspecies; ++is) {
std::printf(" %s", Sy_of_species_null[is]);
} 
std::printf("\n");

SparsifyPlot<int> spp(ab); 
for (int ih = 0; ih < nhist; ++ih) {
#ifdef    PLOT_ALL_POINTS_IN_HISTOGRAM
int const jh = ih - 1;
#else  
int nonzero{0}; 
for (int is = 0; is < nspecies; ++is) {
nonzero += bond_angle_length_hist(ab,is,ih); 
} 
int const jh = spp.previous(ih, (nonzero > 0));
#endif 
if (jh >= 0) {
std::printf("%g ", jh*fac);
for (int is = 0; is < nspecies; ++is) {
std::printf(" %d", bond_angle_length_hist(ab,is,jh));
} 
std::printf("\n");
} 
} 
} 
} 
} 

} 

std::printf("\n# Found %d different elements for %d atoms:  ", nspecies, natoms);
for (int is = 0; is < nspecies; ++is) {
std::printf("  %s_%d", Sy_of_species_null[is], occurrence[Z_of_species[is]]);
} 
std::printf("\n");

} 



if (true) {
size_t cn_exceeds{0};
unsigned constexpr max_cn = 24;
size_t total_cn{0};
view2D<int> cn_hist(nspecies, max_cn, 0);
for (index_t ia = 0; ia < natoms; ++ia) {
auto const cni = coordination_number[ia];
total_cn += cni;
if (cni < max_cn) {
++cn_hist(ispecies[ia],cni);
} else {
++cn_exceeds;
}
} 

if (cn_exceeds > 0) {
warn("In %ld cases, the max. coordination (%d) was exceeded", cn_exceeds, max_cn);
++stat;
} 

if (echo > 3) {
std::printf("\n# half bond length and coordination numbers with occurrence\n");
for (int is = 0; is < nspecies; ++is) {
std::printf("#%9.3f %s coordination for %s", default_bond_length(Z_of_species[is])*Ang, _Ang, Sy_of_species[is]);
simple_stats::Stats<> cs;
for (int cn = 0; cn < max_cn; ++cn) {
if (cn_hist(is,cn) > 0) {
std::printf("  %d_%d", cn, cn_hist(is,cn)); 
cs.add(cn, cn_hist(is,cn));
} 
} 
std::printf("\taverage %.2f +/- %.2f\n", cs.mean(), cs.dev());
} 
std::printf("# coordination numbers total= %ld\n\n", total_cn);
} 
} 


if (echo > 2) {
char const not_available[] = "     n/a";
char const no_entry[]      = "        ";

std::printf("\n# bond counts  ");
for (int js = 0; js < nspecies; ++js) {
std::printf("      %s", Sy_of_species_right[js]); 
} 
std::printf("\n");
std::vector<int> spec_sum(nspecies, 0);
int64_t check_nbonds{0};
for (int is = 0; is < nspecies; ++is) {
int row_sum{0};
std::printf("# bond count   ");
for (int js = 0; js < nspecies; ++js) {
check_nbonds += bond_hist(is,js);
spec_sum[js] += bond_hist(is,js);
row_sum      += bond_hist(is,js);
if (js >= is) {
std::printf("%8d", bond_hist(is,js));
} else {
std::printf(no_entry); 
} 
} 
std::printf("  %ssum= %d\n", Sy_of_species[is], row_sum);
} 
std::printf("# bond counts  ");
for (int js = 0; js < nspecies; ++js) {
std::printf("%8d", spec_sum[js]);
} 
std::printf("   total= %lld\n", nbonds);
if (check_nbonds != nbonds) error("Checksum for bonds does not agree: %d vs %d", check_nbonds, nbonds);

char const label[3][16] = {"min. distance", "longest  bond", "shortest bond"};
for (int i3 = 0; i3 < 3; ++i3) {

std::printf("\n# %ss", label[i3]);
for (int js = 0; js < nspecies; ++js) {
std::printf("     %s ", Sy_of_species_right[js]); 
} 
std::printf("\n");
for (int is = 0; is < nspecies; ++is) {
std::printf("# %s", label[i3]);
for (int js = 0; js < nspecies; ++js) {
if (js >= is) {
double const value = (0 == i3) ? smallest_distance(is,js) :
( (1 == i3) ? bond_stat(is,js).max()   :
bond_stat(is,js).min() );
if ((0 <= value) && (value < too_large)) {
std::printf("%8.3f", value*Ang);
} else {
std::printf(not_available); 
}
} else {
std::printf(no_entry); 
}
} 
std::printf("  %sin %s\n", Sy_of_species[is], _Ang);
} 

} 


if (true) {
bool const show_unavailable_pairs = (control::get("geometry_analysis.show.all.pairs", 1.) > 0);
size_t bonds_total{0};
std::printf("\n# bond table:\n# pair, min dist in %s, bond stats", _Ang);
for (int is = 0; is < nspecies; ++is) {
for (int js = is; js < nspecies; ++js) { 
if (smallest_distance(is,js) < too_large) {
std::printf("\n#  %s-%s%8.3f", Sy_of_species_right[is], Sy_of_species[js], smallest_distance(is,js)*Ang);
auto const & s = bond_stat(is,js);
int const nbonds = s.tim();
if (nbonds > 0) {
std::printf("%8.3f +/- %.3f in [%.3f, %.3f]  %d bonds",
s.mean()*Ang, s.dev()*Ang,
s.min()*Ang, s.max()*Ang, nbonds);
bonds_total += nbonds * (1 + (js != is));
} 
} else if (show_unavailable_pairs) {
std::printf("\n#  %s-%s%s", Sy_of_species_right[is], Sy_of_species[js], not_available);
}
} 
} 
std::printf("\n# total= %ld bonds\n", bonds_total);
} 

} 

return stat;
} 

#ifdef  NO_UNIT_TESTS
status_t all_tests(int const echo) { return STATUS_TEST_NOT_INCLUDED; }
#else 

status_t test_analysis(int const echo=9) {
status_t stat(0);
view2D<double> xyzZ;
int natoms{0};
auto const geo_file = control::get("geometry.file", "atoms.xyz");
double cell[3] = {0, 0, 0};
int8_t bc[3] = {-7, -7, -7};
stat += read_xyz_file(xyzZ, natoms, geo_file, cell, bc, 0);
if (0 != stat) {
warn("read_xyz_file failed with status= %i", int(stat));
return stat;
}
if (echo > 2) std::printf("# found %d atoms in file \"%s\" with cell=[%.3f %.3f %.3f] %s and bc=[%d %d %d]\n",
natoms, geo_file, cell[0]*Ang, cell[1]*Ang, cell[2]*Ang, _Ang, bc[0], bc[1], bc[2]);
{ 
stat += analysis(xyzZ, natoms, cell, bc, echo);
} 
return stat;
} 


status_t test_example_file(int const echo=9) {
int const nspecies = std::min(std::max(2, int(control::get("geometry_analysis.test.nspecies", 128.))), 128);
float xi[128];
double x{0};
for (int iZ = 0; iZ < nspecies; ++iZ) {
xi[iZ] = x;
x += default_bond_length(iZ, (iZ + 1)%nspecies);
} 

auto const filename = control::get("geometry_analysis.test.file", "species_test.xyz");
if (echo > 0) {
std::printf("\n# generate example file \'%s\' with %d different species, length= %g %s\n\n",
filename, nspecies, x*Ang, _Ang);
} 
std::ofstream outfile(filename, std::ofstream::out);
if (outfile.fail()) {
warn("Unable to open file '%s' for writing coordinates", filename);
return 1;
}
outfile << nspecies << "\n#cell " << x*Bohr2Angstrom << " 8 8 periodic isolated isolated\n";
for (int iZ = 0; iZ < nspecies; ++iZ) {
char Sy[4]; chemical_symbol::get(Sy, iZ, ' ');
outfile << Sy << " " << xi[iZ]*Bohr2Angstrom << " 0 0\n";
} 
return 0;
} 


status_t test_fcc_hcp_files(int const echo=9) {
auto const alat = control::get("geometry_analysis.test.lattice.constant", 4.0782*Angstrom2Bohr);
char const Sy[] = "Au";
double const cell[] = {alat*std::sqrt(.5), alat*std::sqrt(1.5), alat*std::sqrt(12.)};
int8_t const layer[3][2][2] = {{{0,0}, {3,3}}, {{3,1}, {0,4}}, {{0,2}, {3,5}}};
double constexpr sixth = 1./6.;
for (int hcp2fcc3 = 2; hcp2fcc3 <= 3; ++hcp2fcc3) {
auto const filename = (2 == hcp2fcc3) ? "hcp.xyz" : "fcc.xyz";
if (echo > 0) std::printf("\n# generate \'%s\' geometry files with lattice constant %g %s\n\n",
filename, alat*Ang, _Ang);
std::ofstream outfile(filename, std::ofstream::out);
if (outfile.fail()) {
warn("Unable to open file \'%s\' for writing coordinates", filename);
return 1;
}
outfile << "12\n#cell";
for (int i3 = 0; i3 < 3; ++i3) outfile << " " << cell[i3]*Bohr2Angstrom;
outfile << " periodic periodic periodic\n";
for (int i6 = 0; i6 < 6; ++i6) {
for (int i2 = 0; i2 < 2; ++i2) {
outfile << Sy << "  ";
for (int i3 = 0; i3 < 3; ++i3) {
double const coordinate = cell[i3]*sixth*((2 == i3) ? i6 : layer[i6 % hcp2fcc3][i2][i3]);
outfile   << coordinate*Bohr2Angstrom << ((2 == i3) ? '\n' : ' ');
} 
} 
} 
} 
return 0;
} 


status_t all_tests(int const echo) {
status_t stat(0);
int const t = control::get("geometry_analysis.select.test", 2.); 
if (t & 0x1) stat += test_example_file(echo);
if (t & 0x2) stat += test_analysis(echo);
if (t & 0x4) stat += test_fcc_hcp_files(echo);
return stat;
} 

#endif 

} 
