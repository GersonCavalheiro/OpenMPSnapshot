#pragma once

#include <cstdio> 
#include <cassert> 

#include "display_units.h" 
#include "status.hxx" 
#include "recorded_warnings.hxx" 

namespace unit_system {

char constexpr _Rydberg[] = "Ry"; 

inline double energy_unit(char const *which, char const **const symbol) {
char const w = which[0] | 32; 
if ('e' == w) {
*symbol = "eV";     return 27.210282768626; 
} else if ('r' == w) {
*symbol = _Rydberg; return 2; 
} else if ('k' == w) {
*symbol = _Kelvin;  return Kelvin; 
} else {
if ('h' != w) warn("unknown energy unit \'%s\', default to Ha (Hartree)", which);
*symbol = "Ha";     return 1; 
} 
} 

inline double length_unit(char const *which, char const **const symbol) {
char const w = which[0] | 32; 
double constexpr Bohr2nanometer = .052917724924; 
if ('a' == w) {
*symbol = "Ang";        return Bohr2nanometer*10; 
} else if ('n' == w) {
*symbol = "nm";         return Bohr2nanometer; 
} else if ('p' == w) {
*symbol = "pm";         return Bohr2nanometer*1000; 
} else {
if ('b' != w) warn("unknown length unit \'%s\', default to Bohr", which);
*symbol = "Bohr";       return 1; 
} 
} 

inline status_t set(char const *length, char const *energy, int const echo=0) {
if (echo > 5) std::printf("# Set output units to {%s, %s}\n", energy, length);
#ifdef    _Output_Units_Fixed
if ('B' != *length || 'H' != *energy) {
warn("output units cannot be changed to {%s, %s} at runtime", length, energy);
} 
return -1; 
#else  
Ang = length_unit(length, &_Ang);
eV  = energy_unit(energy, &_eV);
assert( eV  > 0 );
assert( Ang > 0 );
return 0;
#endif 
} 

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_all_combinations(int const echo=0) {
status_t stat(0);
char const sy[2][4+1][8] = {{"Bohr", "Ang", "nm", "pm",  "length"},
{"Ha",   "eV",  "Ry", "Kel", "energy"}};
for (int le = 0; le < 2; ++le) {
auto const le_name = sy[le][4];
for (int iu = 0; iu < 4; ++iu) { 
char const *_si;
auto const f = le ? energy_unit(sy[le][iu], &_si)
: length_unit(sy[le][iu], &_si);
if (echo > 2) std::printf("# %s unit factor %.12f for %s\n", le_name, f, _si);
auto const fi = 1./f; 
for (int ou = 0; ou < 4; ++ou) { 
char const *_so;
auto const fo = le ? energy_unit(sy[le][ou], &_so)
: length_unit(sy[le][ou], &_so);
if (echo > 3 + (iu != ou)) {
std::printf("# %s unit factors %.9f * %.9f = %.9f %s/%s\n", 
le_name, fo, fi, fo*fi, _so, _si);
} 

stat += (iu == ou)*((fo*fi - 1)*(fo*fi - 1) > 4e-32);
} 
} 
if (echo > 2) std::printf("#\n");
} 
return stat;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s %s\n", __FILE__, __func__);
status_t stat(0);
stat += test_all_combinations(echo);
return stat;
} 

#endif 

} 
