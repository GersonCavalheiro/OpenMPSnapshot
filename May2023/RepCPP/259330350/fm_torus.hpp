#pragma once

#include "decs.hpp"
#include "types.hpp"


TaskStatus InitializeFMTorus(MeshBlockData<Real> *rc, ParameterInput *pin);

TaskStatus PerturbU(MeshBlockData<Real> *rc, ParameterInput *pin);


KOKKOS_INLINE_FUNCTION Real lnh_calc(const GReal a, const Real l, const GReal rin, const GReal r, const GReal th)
{
Real sth = sin(th);
Real cth = cos(th);

Real r2 = m::pow(r, 2);
Real a2 = m::pow(a, 2);
Real DD = r2 - 2. * r + a2;
Real AA = m::pow(r2 + a2, 2) - DD * a2 * sth * sth;
Real SS = r2 + a2 * cth * cth;

Real thin = M_PI / 2.;
Real sthin = sin(thin);
Real cthin = cos(thin);

Real rin2 = m::pow(rin, 2);
Real DDin = rin2 - 2. * rin + a2;
Real AAin = m::pow(rin2 + a2, 2) - DDin * a2 * sthin * sthin;
Real SSin = rin2 + a2 * cthin * cthin;

if (r >= rin) {
return
0.5 *
log((1. +
m::sqrt(1. +
4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth))) /
(SS * DD / AA)) -
0.5 * m::sqrt(1. +
4. * (l * l * SS * SS) * DD /
(AA * AA * sth * sth)) -
2. * a * r * l / AA -
(0.5 *
log((1. +
m::sqrt(1. +
4. * (l * l * SSin * SSin) * DDin /
(AAin * AAin * sthin * sthin))) /
(SSin * DDin / AAin)) -
0.5 * m::sqrt(1. +
4. * (l * l * SSin * SSin) * DDin / (AAin * AAin * sthin * sthin)) -
2. * a * rin * l / AAin);
} else {
return 1.;
}
}


KOKKOS_INLINE_FUNCTION Real lfish_calc(const GReal a, const GReal r)
{
return (((m::pow(a, 2) - 2. * a * m::sqrt(r) + m::pow(r, 2)) *
((-2. * a * r *
(m::pow(a, 2) - 2. * a * m::sqrt(r) +
m::pow(r,
2))) /
m::sqrt(2. * a * m::sqrt(r) + (-3. + r) * r) +
((a + (-2. + r) * m::sqrt(r)) * (m::pow(r, 3) + m::pow(a, 2) *
(2. + r))) /
m::sqrt(1 + (2. * a) / m::pow(r, 1.5) - 3. / r))) /
(m::pow(r, 3) * m::sqrt(2. * a * m::sqrt(r) + (-3. + r) * r) *
(m::pow(a, 2) + (-2. + r) * r)));
}


KOKKOS_INLINE_FUNCTION Real fm_torus_rho(const GReal a, const GReal rin, const GReal rmax, const Real gam,
const Real kappa, const GReal r, const GReal th)
{
Real l = lfish_calc(a, rmax);
Real lnh = lnh_calc(a, l, rin, r, th);
if (lnh >= 0. && r >= rin) {
Real hm1 = exp(lnh) - 1.;
return m::pow(hm1 * (gam - 1.) / (kappa * gam),
1. / (gam - 1.));
} else {
return 0;
}
}
