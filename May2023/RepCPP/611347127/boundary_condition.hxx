#pragma once

#include <cstdio> 
#include <cmath> 
#include <cstdint> 

#include "inline_math.hxx"
#include "data_view.hxx" 
#include "status.hxx" 
#ifndef STANDALONE_TEST
#include "recorded_warnings.hxx" 
#else
#define warn std::printf
#endif

int8_t constexpr Periodic_Boundary =  1;
int8_t constexpr Isolated_Boundary =  0;
int8_t constexpr Mirrored_Boundary = -1;
int8_t constexpr Invalid_Boundary  = -2;

namespace boundary_condition {

inline int periodic_images( 
view2D<double> & ipos 
, double const cell[3]  
, int8_t const bc[3]       
, float  const rcut      
, int    const echo=0      
, view2D<int8_t> *iidx=nullptr 
) {
double const cell_diagonal2 = pow2(rcut)
+ pow2(cell[0]) + pow2(cell[1]) + pow2(cell[2]);
int ni_xyz[3], ni_max{1};
if (rcut < 0) warn("A negative cutoff radius leads to only one image! rcut = %g a.u.", rcut);
for (int d = 0; d < 3; ++d) {
if (Periodic_Boundary == bc[d]) {
ni_xyz[d] = std::max(0, int(std::ceil(rcut/cell[d])));
assert( ni_xyz[d] <= 127 ); 
ni_max *= (ni_xyz[d]*2 + 1);
} else {
ni_xyz[d] = 0;
} 
} 
if (echo > 5) std::printf("# %s: check %d x %d x %d = %d images max.\n",
__func__, 1+2*ni_xyz[0], 1+2*ni_xyz[1], 1+2*ni_xyz[2], ni_max);
view2D<double> pos(ni_max, 4, 0.0); 
view2D<int8_t> idx(ni_max, 4, 0); 
int ni{1}; 
for         (int iz = -ni_xyz[2]; iz <= ni_xyz[2]; ++iz) {  auto const pz = iz*cell[2];
for     (int iy = -ni_xyz[1]; iy <= ni_xyz[1]; ++iy) {  auto const py = iy*cell[1];
for (int ix = -ni_xyz[0]; ix <= ni_xyz[0]; ++ix) {  auto const px = ix*cell[0];
auto const d2 = pow2(px) + pow2(py) + pow2(pz);
#ifdef DEVEL
char mark{' '};
#endif 
if (d2 < cell_diagonal2) {
if (d2 > 0) { 
pos(ni,0) = px;
pos(ni,1) = py;
pos(ni,2) = pz;
pos(ni,3) = d2; 
idx(ni,0) = ix;
idx(ni,1) = iy;
idx(ni,2) = iz;
idx(ni,3) =  0; 
++ni; 
#ifdef DEVEL
mark = 'o';
} else {
mark = 'x';
#endif 
} 
} 
#ifdef DEVEL
if (echo > 6) {
if (ix == -ni_xyz[0]) {
if (iy == -ni_xyz[1]) std::printf("# %s z=%i\n", __func__, iz);
std::printf("#%4i  | ", iy); 
} 
std::printf("%c", mark);
if (ix == ni_xyz[0]) std::printf(" |\n"); 
} 
#endif 
} 
} 
} 
if (echo > 1) std::printf("# %s: found %d of %d images\n", __func__, ni, ni_max);

ipos = view2D<double>(ni, 4); 
set(ipos.data(), ni*4, pos.data()); 

if (iidx) {
*iidx = view2D<int8_t>(ni, 4); 
set(iidx->data(), ni*4, idx.data()); 
} 

return ni;
} 

inline int8_t fromString(
char const *string
, int const echo=0
, char const dir='?'
) {
int8_t bc{Invalid_Boundary};
if (nullptr != string) {
char const first = *string;
switch (first | 32) { 
case 'p': case '1': bc = Periodic_Boundary; break;
case 'i': case '0': bc = Isolated_Boundary; break;
case 'm': case '-': bc = Mirrored_Boundary; break;
} 
} 
if (echo > 0) {
char const bc_names[][12] = {"isolated", "periodic", "invalid", "mirror"};
std::printf("# interpret \"%s\" as %s boundary condition in %c-direction\n",
string, bc_names[bc & 0x3], dir);
} 
return bc;
} 

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_periodic_images(int const echo=0) {
if (echo > 2) std::printf("\n# %s %s \n", __FILE__, __func__);
double const cell[] = {1,2,3};
float  const rcut = 6;
int8_t const bc[] = {Periodic_Boundary, Periodic_Boundary, Isolated_Boundary};
view2D<double> ipos;
view2D<int8_t> iidx;
auto const nai = periodic_images(ipos, cell, bc, rcut, echo, &iidx);
if (echo > 2) std::printf("# found %d periodic images\n", nai);
auto const nai2 = periodic_images(ipos, cell, bc, rcut);
return (nai2 - nai);
} 

inline status_t test_fromString_single(char const bc_strings[][16], int const echo=0) {
if (echo > 2) std::printf("\n# %s %s \n", __FILE__, __func__);
status_t stat(0);
for (int8_t bc = Invalid_Boundary; bc <= Periodic_Boundary; ++bc) {
stat += (bc != fromString(bc_strings[bc & 0x3], echo));
} 
return stat;
} 

inline status_t test_fromString(int const echo=0) {
if (echo > 2) std::printf("\n# %s %s \n", __FILE__, __func__);
status_t stat(0);
{   char const bc_strings[][16] = {"isolated", "periodic", "?invalid", "mirror"}; 
stat += test_fromString_single(bc_strings, echo);   }
{   char const bc_strings[][16] = {"i", "p", "_", "m"}; 
stat += test_fromString_single(bc_strings, echo);   }
{   char const bc_strings[][16] = {"I", "P", "#", "M"}; 
stat += test_fromString_single(bc_strings, echo);   }
{   char const bc_strings[][16] = {"0", "1", "*", "-"}; 
stat += test_fromString_single(bc_strings, echo);   }
return stat;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s %s\n", __FILE__, __func__);
status_t stat(0);
stat += test_periodic_images(echo);
stat += test_fromString(echo);
return stat;
} 

#endif 

} 
