#pragma once

#include <cstdio> 
#include <cmath> 
#include <vector> 

#include "radial_grid.h" 
#include "radial_r2grid.hxx" 
#include "status.hxx" 

namespace bessel_transform {

inline double Bessel_j0(double const x) { return (x*x < 1e-16) ? 1.0 - x*x/6. : std::sin(x)/x; }

template <typename real_t>
inline status_t transform_s_function(
real_t out[] 
, real_t const in[]
, radial_grid_t const & g
, int const nq
, double const dq=.125
, bool const back=false
, int const echo=3 
) {

if (echo > 8) std::printf("# %s(out=%p, in=%p, g=%p, nq=%d, dq=%.3f, back=%d, echo=%d);\n",
__func__, (void*)out, (void*)in, (void*)&g, nq, dq, back, echo);

std::vector<double> q_lin(nq), dq_lin(nq);
for (int iq = 0; iq < nq; ++iq) {
double const q = iq*dq;
q_lin[iq] = q;
dq_lin[iq] = q*q*dq;
} 

int n_out, n_in;
double const *x_out, *x_in, *dx_in;
if (back) {
n_in  = nq;
x_in  = q_lin.data();
dx_in = dq_lin.data();
n_out = g.n;
x_out = g.r;
} else {
n_in  = g.n;
x_in  = g.r;
dx_in = g.r2dr;
n_out = nq;
x_out = q_lin.data();
} 
if (echo > 8) std::printf("# %s    n_in=%d x_in=%p dx_in=%p n_out=%d x_out=%p\n",
__func__, n_in, (void*)x_in, (void*)dx_in, n_out, (void*)x_out);

double const sqrt2overpi = .7978845608028654; 
for (int io = 0; io < n_out; ++io) {
double tmp{0};
for (int ii = 0; ii < n_in; ++ii) {
double const qr = x_in[ii]*x_out[io];
tmp += in[ii] * Bessel_j0(qr) * dx_in[ii];
} 
out[io] = tmp*sqrt2overpi;
} 

return 0;
} 

template <typename real_t>
inline status_t transform_to_r2grid(
real_t out[]
, float const ar2
, int const nr2
, double const in[]
, radial_grid_t const & g
, int const echo=2
) {
if (echo > 8) {
std::printf("\n# %s input:\n", __func__);
for (int ir = 0; ir < g.n; ++ir) {
std::printf("%g %g\n", g.r[ir], in[ir]);
} 
std::printf("\n\n");
} 

int const nq = 256; double const dq = 0.125; 
std::vector<double> bt(nq);
auto stat = transform_s_function(bt.data(), in, g, nq, dq); 

auto r = radial_r2grid::r_axis(nr2, ar2);
radial_grid_t r2g; r2g.n = nr2; r2g.r = r.data();

stat += transform_s_function(out, bt.data(), r2g, nq, dq, true); 

if (echo > 8) {
std::printf("\n# %s output:\n", __func__);
for (int ir2 = 0; ir2 < r2g.n; ++ir2) {
std::printf("%g %g\n", r2g.r[ir2], out[ir2]);
} 
std::printf("\n\n");
} 

return stat;
} 

status_t all_tests(int const echo=0); 

} 
