#pragma once

#include <cstdio> 
#include <cmath> 
#include <algorithm> 
#ifndef NO_UNIT_TESTS
#include <cstdint> 
#include <vector> 
#endif 

#include "constants.hxx" 
#include "display_units.h" 
#include "inline_math.hxx" 
#include "status.hxx" 

namespace shift_boundary {












inline status_t test_plane_wave(int const echo=9, int const structure=4) {
status_t stat(0);
char const structure_name[][4] = {"sc\0","bcc","hcp","fcc"};
double const alat = 4.1741; 
double const ahalf = 0.5 * alat;
if (echo > 3) std::printf("\n# structure = %s  lattice constant = %g %s\n", structure_name[structure - 1], alat*Ang, _Ang);
double amat[3][4]; set(amat[0], 3*4, 0.0);
if (1 == structure) { 
for (int d = 0; d < 3; ++d) amat[d][d] = 2*ahalf;
} else
if (2 == structure) { 
amat[0][0] = 2*ahalf; amat[0][1] = ahalf;
amat[1][1] = 2*ahalf; amat[1][2] = ahalf;
amat[2][2] = ahalf;
} else
if (4 == structure) { 
amat[0][0] = 2*ahalf; amat[0][1] = ahalf;
amat[1][1] = ahalf;   amat[1][2] = ahalf;
amat[2][2] = ahalf;
} else
if (3 == structure) { 
double const s34 = std::sqrt(.75), s83=std::sqrt(8/3.);
double const ann = ahalf*std::sqrt(2.); 
amat[0][0] = ann;     amat[0][1] = ann*0.5;
amat[1][1] = ann*s34;
amat[2][2] = ann*s83;
} else {
if (echo > 0) std::printf("\n# %s no such structure, key= %i\n", __func__, structure);
return -1; 
} 

double bmat[3][4]; set(bmat[0], 3*4, 0.0);
double const cell_volume = amat[0][0]*amat[1][1]*amat[2][2];
if (echo > 4) std::printf("# cell volume %g %s^3\n", cell_volume*pow3(Ang), _Ang);
double const detinv = 1./cell_volume;
for (int i = 0; i < 3; ++i) {     int const i1 = (i + 1)%3, i2 = (i + 2)%3;
for (int j = 0; j < 3; ++j) { int const j1 = (j + 1)%3, j2 = (j + 2)%3;
bmat[j][i] = ( amat[i1][j1] * amat[i2][j2]
- amat[i1][j2] * amat[i2][j1] )*detinv;
} 
} 

if (echo > 4) {
for (int i = 0; i < 3; ++i) {
std::printf("#  bmat %c %8.3f%8.3f%8.3f   amat %8.3f%8.3f%8.3f\n", i+'x',
bmat[i][0],bmat[i][1],bmat[i][2],   amat[i][0],amat[i][1],amat[i][2]);
} 
} 

double maxdev[] = {0, 0};
for (int i = 0; i < 3; ++i) {
if (echo > 6) std::printf("# i=%i ", i);
for (int j = 0; j < 3; ++j) {
double uij{0}, uji{0};
for (int k = 0; k < 3; ++k) {
uij += amat[i][k] * bmat[k][j];
uji += bmat[i][k] * amat[k][j];
} 
maxdev[0] = std::max(maxdev[0], std::abs(uij - (i == j)));
maxdev[1] = std::max(maxdev[1], std::abs(uji - (i == j)));
if (echo > 6) std::printf("%8.3f%8.3f ", uji, uij);
if (echo > 8) std::printf("%.1e %.1e ", uji - (i == j), uij - (i == j));
} 
if (echo > 6) std::printf("\n");
} 
if (echo > 3) std::printf("# %s after inversion largest deviation is %.1e (a*b) and %.1e (b*a)\n", 
__func__, maxdev[0], maxdev[1]);


if (echo > 4) {
int const ng = 16;          
double const h = alat/ng;   
for (int i = 0; i < 3; ++i) {
std::printf("# integer amat %c %8.1f%8.1f%8.1f\n", i+'x',
.1*std::round(10*amat[i][0]/h), .1*std::round(10*amat[i][1]/h), .1*std::round(10*amat[i][2]/h));
} 
} 












return stat;
} 

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t all_tests(int const echo=0) {
if (echo > 1) std::printf("\n# %s: %s\n\n", __FILE__, __func__);
status_t stat(0);
for (int structure = 1; structure <= 4; ++structure) {
stat += test_plane_wave(echo, structure);
} 
return stat;
} 

#endif 

} 
