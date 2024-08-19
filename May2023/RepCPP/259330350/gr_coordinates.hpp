
#pragma once

#include "decs.hpp"

#include "coordinate_embedding.hpp"
#include "kharma_utils.hpp"

#include <coordinates/uniform_cartesian.hpp>
#include <parameter_input.hpp>

#include "Kokkos_Core.hpp"

#define FAST_CARTESIAN 0
#define NO_CACHE 0


class GRCoordinates : public parthenon::UniformCartesian
{
public:
CoordinateEmbedding coords;

int n1, n2, n3;
#if !FAST_CARTESIAN && !NO_CACHE
GeomTensor2 gcon_direct, gcov_direct;
GeomScalar gdet_direct;
GeomTensor3 conn_direct, gdet_conn_direct;
#endif

#pragma hd_warning_disable
GRCoordinates(const parthenon::RegionSize &rs, parthenon::ParameterInput *pin);
#pragma hd_warning_disable
GRCoordinates(const GRCoordinates &src, int coarsen);

KOKKOS_FUNCTION GRCoordinates(): UniformCartesian() {};
KOKKOS_FUNCTION GRCoordinates(const GRCoordinates &src): parthenon::UniformCartesian(src)
{
coords = src.coords;
n1 = src.n1;
n2 = src.n2;
n3 = src.n3;
#if !FAST_CARTESIAN && !NO_CACHE
gcon_direct = src.gcon_direct;
gcov_direct = src.gcov_direct;
gdet_direct = src.gdet_direct;
conn_direct = src.conn_direct;
gdet_conn_direct = src.gdet_conn_direct;
#endif
};
KOKKOS_FUNCTION GRCoordinates operator=(const GRCoordinates& src)
{
UniformCartesian::operator=(src);
coords = src.coords;
n1 = src.n1;
n2 = src.n2;
n3 = src.n3;
#if !FAST_CARTESIAN && !NO_CACHE
gcon_direct = src.gcon_direct;
gcov_direct = src.gcov_direct;
gdet_direct = src.gdet_direct;
conn_direct = src.conn_direct;
gdet_conn_direct = src.gdet_conn_direct;
#endif
return *this;
};

KOKKOS_INLINE_FUNCTION Real gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const;
KOKKOS_INLINE_FUNCTION Real gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const;
KOKKOS_INLINE_FUNCTION Real gdet(const Loci loc, const int& j, const int& i) const;
KOKKOS_INLINE_FUNCTION Real conn(const int& j, const int& i, const int mu, const int nu, const int lam) const;
KOKKOS_INLINE_FUNCTION Real gdet_conn(const int& j, const int& i, const int mu, const int nu, const int lam) const;

KOKKOS_INLINE_FUNCTION void gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const;
KOKKOS_INLINE_FUNCTION void gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const;
KOKKOS_INLINE_FUNCTION void conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const;
KOKKOS_INLINE_FUNCTION void gdet_conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const;

KOKKOS_INLINE_FUNCTION void coord(const int& k, const int& j, const int& i, const Loci& loc, GReal X[GR_DIM]) const;
KOKKOS_INLINE_FUNCTION void coord_embed(const int& k, const int& j, const int& i, const Loci& loc, GReal Xembed[GR_DIM]) const;

KOKKOS_INLINE_FUNCTION void lower(const Real vcon[GR_DIM], Real vcov[GR_DIM],
const int& k, const int& j, const int& i, const Loci loc) const;
KOKKOS_INLINE_FUNCTION void raise(const Real vcov[GR_DIM], Real vcon[GR_DIM],
const int& k, const int& j, const int& i, const Loci loc) const;

};


KOKKOS_INLINE_FUNCTION void GRCoordinates::coord(const int& k, const int& j, const int& i, const Loci& loc, Real X[GR_DIM]) const
{
X[0] = 0;
switch(loc)
{
case Loci::face1:
X[1] = x1f(i);
X[2] = x2v(j);
X[3] = x3v(k);
break;
case Loci::face2:
X[1] = x1v(i);
X[2] = x2f(j);
X[3] = x3v(k);
break;
case Loci::face3:
X[1] = x1v(i);
X[2] = x2v(j);
X[3] = x3f(k);
break;
case Loci::center:
X[1] = x1v(i);
X[2] = x2v(j);
X[3] = x3v(k);
break;
case Loci::corner:
X[1] = x1f(i);
X[2] = x2f(j);
X[3] = x3f(k);
break;
}
}

#if FAST_CARTESIAN
KOKKOS_INLINE_FUNCTION void GRCoordinates::coord_embed(const int& k, const int& j, const int& i, const Loci& loc, GReal Xembed[GR_DIM]) const
{
coord(k, j, i, loc, Xembed);
}
#else

KOKKOS_INLINE_FUNCTION void GRCoordinates::coord_embed(const int& k, const int& j, const int& i, const Loci& loc, GReal Xembed[GR_DIM]) const
{
GReal Xnative[GR_DIM];
coord(k, j, i, loc, Xnative);
coords.coord_to_embed(Xnative, Xembed);
}
#endif

KOKKOS_INLINE_FUNCTION void GRCoordinates::lower(const Real vcon[GR_DIM], Real vcov[GR_DIM],
const int& k, const int& j, const int& i, const Loci loc) const
{
gzero(vcov);
DLOOP2 vcov[mu] += gcov(loc, j, i, mu, nu) * vcon[nu];
}
KOKKOS_INLINE_FUNCTION void GRCoordinates::raise(const Real vcov[GR_DIM], Real vcon[GR_DIM],
const int& k, const int& j, const int& i, const Loci loc) const
{
gzero(vcon);
DLOOP2 vcon[mu] += gcon(loc, j, i, mu, nu) * vcov[nu];
}

#if FAST_CARTESIAN
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{return -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{return -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet(const Loci loc, const int& j, const int& i) const
{return 1;}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
{return 0;}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet_conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
{return 0;}

KOKKOS_INLINE_FUNCTION void GRCoordinates::gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const
{DLOOP2 gcon[mu][nu] = -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const
{DLOOP2 gcov[mu][nu] = -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
{DLOOP3 conn[mu][nu][lam] = 0;}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gdet_conn(const int& j, const int& i, Real gdet_conn[GR_DIM][GR_DIM][GR_DIM]) const
{DLOOP3 gdet_conn[mu][nu][lam] = 0;}
#elif NO_CACHE
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{
GReal X[GR_DIM], gcon[GR_DIM][GR_DIM];
coord(0, j, i, loc, X);
coords.gcon_native(X, gcon);
return gcon[mu][nu];
}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{
GReal X[GR_DIM], gcov[GR_DIM][GR_DIM];
coord(0, j, i, loc, X);
coords.gcov_native(X, gcov);
return gcov[mu][nu];
}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet(const Loci loc, const int& j, const int& i) const
{
GReal X[GR_DIM], gcon[GR_DIM][GR_DIM];
coord(0, j, i, loc, X);
return coords.gcon_native(X, gcon);
}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
{
GReal X[GR_DIM], conn[GR_DIM][GR_DIM][GR_DIM];
coord(0, j, i, Loci::center, X);
coords.conn_native(X, conn);
return conn[mu][nu][lam];
}

KOKKOS_INLINE_FUNCTION void GRCoordinates::gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const
{
GReal X[GR_DIM];
coord(0, j, i, loc, X);
coords.gcon_native(X, gcon);
}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const
{
GReal X[GR_DIM];
coord(0, j, i, loc, X);
coords.gcov_native(X, gcov);
}
KOKKOS_INLINE_FUNCTION void GRCoordinates::conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
{
GReal X[GR_DIM];
coord(0, j, i, Loci::center, X);
coords.conn_native(X, conn);
}
#else
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{return gcon_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{return gcov_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet(const Loci loc, const int& j, const int& i) const
{return gdet_direct(loc, j, i);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
{return conn_direct(j, i, mu, nu, lam);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet_conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
{return gdet_conn_direct(j, i, mu, nu, lam);}

KOKKOS_INLINE_FUNCTION void GRCoordinates::gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const
{DLOOP2 gcon[mu][nu] = gcon_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const
{DLOOP2 gcov[mu][nu] = gcov_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
{DLOOP3 conn[mu][nu][lam] = conn_direct(j, i, mu, nu, lam);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gdet_conn(const int& j, const int& i, Real gdet_conn[GR_DIM][GR_DIM][GR_DIM]) const
{DLOOP3 gdet_conn[mu][nu][lam] = gdet_conn_direct(j, i, mu, nu, lam);}
#endif
