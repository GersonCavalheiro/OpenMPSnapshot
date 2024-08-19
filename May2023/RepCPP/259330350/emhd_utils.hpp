
#pragma once

#include "decs.hpp"



namespace EMHD {

KOKKOS_INLINE_FUNCTION Real linear_monotonized_cd(Real x1, Real x2, Real x3, Real dx)
{
const Real Dqm = 2 * (x2 - x1) / dx;
const Real Dqp = 2 * (x3 - x2) / dx;
const Real Dqc = 0.5 * (x3 - x1) / dx;

if (Dqm * Dqp <= 0) {
return 0;
} else {
if ((m::abs(Dqm) < m::abs(Dqp)) && (fabs (Dqm) < m::abs(Dqc))) {
return Dqm;
} else if (m::abs(Dqp) < m::abs(Dqc)) {
return Dqp;
} else {
return Dqc;
}
}
}

KOKKOS_INLINE_FUNCTION Real linear_van_leer(Real x1, Real x2, Real x3, Real dx)
{
const Real Dqm = (x2 - x1) / dx;
const Real Dqp = (x3 - x2) / dx;

const Real extrema = Dqm * Dqp;

if (extrema <= 0) {
return 0;
} else {
return (2 * extrema / (Dqm + Dqp)); 
}
}


template<typename Global>
KOKKOS_INLINE_FUNCTION Real slope_calc_scalar(const GRCoordinates& G, const Global& A, const int& dir,
const int& b, const int& k, const int& j, const int& i, 
ReconstructionType recon=ReconstructionType::linear_mc)
{
if (recon != ReconstructionType::linear_vl) {
if (dir == 1) return linear_monotonized_cd(A(b, k, j, i-1), A(b, k, j, i), A(b, k, j, i+1), G.dx1v(i));
if (dir == 2) return linear_monotonized_cd(A(b, k, j-1, i), A(b, k, j, i), A(b, k, j+1, i), G.dx2v(j));
if (dir == 3) return linear_monotonized_cd(A(b, k-1, j, i), A(b, k, j, i), A(b, k+1, j, i), G.dx3v(k));
} else {
if (dir == 1) return linear_van_leer(A(b, k, j, i-1), A(b, k, j, i), A(b, k, j, i+1), G.dx1v(i));
if (dir == 2) return linear_van_leer(A(b, k, j-1, i), A(b, k, j, i), A(b, k, j+1, i), G.dx2v(j));
if (dir == 3) return linear_van_leer(A(b, k-1, j, i), A(b, k, j, i), A(b, k+1, j, i), G.dx3v(k));
}
return 0.;
}


template<typename Global>
KOKKOS_INLINE_FUNCTION Real slope_calc_vector(const GRCoordinates& G, const Global& A, const int& mu,
const int& dir, const int& b, const int& k, const int& j, const int& i, 
ReconstructionType recon=ReconstructionType::linear_mc)
{
if (recon != ReconstructionType::linear_vl) {
if (dir == 1) return linear_monotonized_cd(A(b, mu, k, j, i-1), A(b, mu, k, j, i), A(b, mu, k, j, i+1), G.dx1v(i));
if (dir == 2) return linear_monotonized_cd(A(b, mu, k, j-1, i), A(b, mu, k, j, i), A(b, mu, k, j+1, i), G.dx2v(j));
if (dir == 3) return linear_monotonized_cd(A(b, mu, k-1, j, i), A(b, mu, k, j, i), A(b, mu, k+1, j, i), G.dx3v(k));
} else {
if (dir == 1) return linear_van_leer(A(b, mu, k, j, i-1), A(b, mu, k, j, i), A(b, mu, k, j, i+1), G.dx1v(i));
if (dir == 2) return linear_van_leer(A(b, mu, k, j-1, i), A(b, mu, k, j, i), A(b, mu, k, j+1, i), G.dx2v(j));
if (dir == 3) return linear_van_leer(A(b, mu, k-1, j, i), A(b, mu, k, j, i), A(b, mu, k+1, j, i), G.dx3v(k));
}
return 0.;
}

template<typename Global>
KOKKOS_INLINE_FUNCTION void gradient_calc(const GRCoordinates& G, const Global& P,
const GridVector& ucov_s, const GridScalar& theta_s,
const int& b, const int& k, const int& j, const int& i, 
const bool& do_3d, const bool& do_2d,
Real grad_ucov[GR_DIM][GR_DIM], Real grad_Theta[GR_DIM])
{
DLOOP1 {
grad_ucov[0][mu] = 0;

grad_ucov[1][mu] = slope_calc_vector(G, ucov_s, mu, 1, b, k, j, i);
if (do_2d) {
grad_ucov[2][mu] = slope_calc_vector(G, ucov_s, mu, 2, b, k, j, i);
} else {
grad_ucov[2][mu] = 0.;
}
if (do_3d) {
grad_ucov[3][mu] = slope_calc_vector(G, ucov_s, mu, 3, b, k, j, i);
} else {
grad_ucov[3][mu] = 0.;
}
}
DLOOP3 grad_ucov[mu][nu] -= G.conn(j, i, lam, mu, nu) * ucov_s(lam, k, j, i);

grad_Theta[0] = 0;
grad_Theta[1] = slope_calc_scalar(G, theta_s, 1, b, k, j, i);
if (do_2d) {
grad_Theta[2] = slope_calc_scalar(G, theta_s, 2, b, k, j, i);
} else {
grad_Theta[2] = 0.;
} 
if (do_3d) {
grad_Theta[3] = slope_calc_scalar(G, theta_s, 3, b, k, j, i);
} else {
grad_Theta[3] = 0.;
}
}

} 
