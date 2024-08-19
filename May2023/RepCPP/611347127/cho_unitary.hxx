#pragma once

#include <cstdio> 
#include <cassert> 
#include <cmath> 
#include <algorithm> 
#include <fstream> 
#include <sstream> 
#include <numeric> 
#include <vector> 

typedef int status_t;

#define warn std::printf

namespace cho_unitary {

template <typename real_t>
real_t signed_sqrt(real_t const x) { return (x < 0)? -std::sqrt(-x) : std::sqrt(x); }

template <typename real_t>
status_t read_unitary_matrix_from_file(
real_t **u
, int const numax
, int &nu_high
, char const filename[]="cho_unitary.dat"
, int const echo=7
) {
std::ifstream infile(filename, std::ios_base::in);
if (infile.fail()) {
warn("file \'%s\' for Unitary_CHO_Transform cannot be opened!", filename);
return -1; 
} 
bool const file_is_nu_ordered = true;

int n_ignored{0}; 
std::string line;
while (std::getline(infile, line)) {
char const c0 = line[0];
if ('#' != c0 && ' ' != c0 && '\n' != c0 && 0 != c0) {
std::istringstream iss(line);
int nx{-1}, ny{-1}, emm{0}, nrn{-1};
int64_t nom{0}, den{1};
if (!(iss >> nx >> ny >> emm >> nrn >> nom >> den)) {
std::printf("# Failed to read integer number from \"%s\"!\n", line.c_str());
break;
} 
assert(nx >= 0);
assert(ny >= 0);
assert(nrn >= 0);
assert(den > 0);
assert(std::abs(nom) <= den);

real_t const u_entry = signed_sqrt(nom*(1./den));
if (echo > 8) std::printf("%d %d    %2d %d  %.15f\n", nx, ny,   emm, nrn, u_entry);
int const ell = std::abs(emm);
int const nu_xy = nx + ny;
int const nu_rad = ell + 2*nrn;
if (nu_xy != nu_rad) {
std::printf("# off-block entry found in file <%s>: nx=%d ny=%d (nu=%d)  emm=%d nrn=%d (nu=%d)\n",
filename, nx, ny, nu_xy, emm, nrn, nu_rad);
return 1; 
} 
int const nu = nu_xy;
nu_high = std::max(nu_high, nu);
if (nu > numax) {
++n_ignored; 
if (file_is_nu_ordered) return 0; 
} else {
int const nb = nu + 1; 
int const nrad = (nu + emm)/2; 
assert(nrad >= 0);
assert(nrad < nb);
assert(nx < nb);
u[nu][nx*nb + nrad] = u_entry; 
} 
} 
} 
if (n_ignored && (echo > 2)) std::printf("# ignored %d lines in file <%s> reading up to nu=%d\n", n_ignored, filename, numax);
return 0;
} 


template <typename real_t=double>
class Unitary_CHO_Transform {
public:

Unitary_CHO_Transform(int const lmax=7, int const echo=8)
: numax_(lmax)
{
u_ = new real_t*[1 + numax_]; 
for (int nu = 0; nu <= numax_; ++nu) { 
int const nb = nu + 1; 
u_[nu] = new real_t[nb*nb]; 
std::fill(u_[nu], u_[nu] + nb*nb, 0); 
} 

int highest_nu{-1};
auto const stat = read_unitary_matrix_from_file(u_, numax_, highest_nu);
if (stat) { 
for (int nu = 0; nu <= numax_; ++nu) { 
int const nb = nu + 1; 
std::fill(u_[nu], u_[nu] + nb*nb, 0); 
for (int ib = 0; ib < nb; ++ib) {
u_[nu][ib*nb + ib] = 1; 
} 
} 
warn("I/O failed with status=%i, Unitary_SHO_Transform was initialized as unit operator!", int(stat));
} 
if (highest_nu < numax_) {
warn("file for Unitary_SHO_Transform provided elements only up to numax=%d, requested %d", highest_nu, numax_);
} 
} 

~Unitary_CHO_Transform() {
for (int nu = 0; nu <= numax_; ++nu) {
delete[] u_[nu];
} 
delete[] u_;
} 

double test_unitarity(int const echo=7) const {
double maxdevall{0};
for (int nu = 0; nu <= numax_; ++nu) {
int const nb = nu + 1; 
double mxd[2][2] = {{0,0},{0,0}}; 
for (int ib = 0; ib < nb; ++ib) { 
for (int jb = 0; jb < nb; ++jb) { 
double uuT{0}, uTu{0};
for (int kb = 0; kb < nb; ++kb) {
uuT += u_[nu][ib*nb + kb] * u_[nu][jb*nb + kb]; 
uTu += u_[nu][kb*nb + ib] * u_[nu][kb*nb + jb]; 
} 
if (echo > 8) std::printf("# nu=%d ib=%d jb=%d uuT=%g uTu=%g\n", nu, ib, jb, uuT, uTu);
int const diag = (ib == jb); 
mxd[0][diag] = std::max(std::abs(uuT - diag), mxd[0][diag]);
mxd[1][diag] = std::max(std::abs(uTu - diag), mxd[1][diag]);
maxdevall = std::max(maxdevall, std::max(mxd[0][diag], mxd[1][diag]));
} 
} 
if (echo > 3) std::printf("# U<nu=%d> deviations: Radial uuT %.1e (%.1e on diagonal), Cartesian uTu %.1e (%.1e on diagonal)\n",
nu, mxd[0][0], mxd[0][1], mxd[1][0], mxd[1][1]);
} 
return maxdevall;
} 

inline int numax() const { return numax_; }

private: 
real_t **u_; 
int numax_;  

}; 

status_t all_tests(int const echo=0); 

} 
