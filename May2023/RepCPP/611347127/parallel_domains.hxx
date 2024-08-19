#pragma once

#include <cstdio> 
#include <cassert> 

#include "status.hxx" 

namespace parallel_domains {

inline status_t decompose_grid(unsigned const ng, int const echo=0, int const min_ng_per_pe=4) {
status_t stat(0);
for (int npe = 1; npe <= ng + 9; ++npe) {
for (int bc = 0; bc <= 1; ++bc) { 

int ng_more, np_more, ng_less, np_less, np_zero{0}, np_uppr{0};
if (1) {
ng_less = ng/npe; 
ng_more = ng_less + 1;

np_more = (ng - ng_less*npe);
np_less = npe - np_more;

if (0 == bc && ng_less > 0) {
np_uppr = np_more/2;
} 

} 
int const np_lowr = np_more - np_uppr;

int const gs[] = {ng_more, ng_less, ng_more, 0};
int const ps[] = {np_lowr, np_less, np_uppr, np_zero};
int gxp[3];
char dec_str[3][16];
for (int i = 0; i < 3; ++i) {
gxp[i] = gs[i] * ps[i];
if (gxp[i] > 0) {
std::snprintf(dec_str[i], 16, "%dx%d", gs[i], ps[i]);
} else {
std::snprintf(dec_str[i], 16, "0");
} 
} 
if (echo > 7) std::printf("# parallelize %d grid points as %s + %s + %s with %d process elements (BC=%s)\n", 
ng, dec_str[0], dec_str[1], dec_str[2], npe, bc?"periodic":"isolated");

#ifdef DEBUG
assert(ng == ng_more*np_lowr + ng_less*np_less + ng_more*np_uppr);
assert(npe == np_lowr + np_less + np_uppr);
assert(npe == np_more + np_less);
#endif
stat += (ng != ng_more*np_lowr + ng_less*np_less + ng_more*np_uppr);
stat += (npe != np_lowr + np_less + np_uppr);
stat += (npe != np_more + np_less);
} 
} 
return stat;
} 

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_analysis(int const echo=0) {
status_t stat(0);
for (int ng = 1; ng <= 99; ++ng) {
stat += decompose_grid(ng, echo);
} 
return stat;
} 

inline status_t all_tests(int const echo=0) { 
status_t stat(0);
stat += test_analysis(echo);
return stat; 
} 

#endif 

} 
