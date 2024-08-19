
#pragma once

#include "decs.hpp"

#define ROOTFIND_TOL 1.e-9


#define ROOT_FIND \
double th = Xembed[2];\
double tha, thb, thc;\
\
double Xa[GR_DIM], Xb[GR_DIM], Xc[GR_DIM], Xtmp[GR_DIM];\
Xa[1] = Xnative[1];\
Xa[3] = Xnative[3];\
\
Xb[1] = Xa[1];\
Xb[3] = Xa[3];\
Xc[1] = Xa[1];\
Xc[3] = Xa[3];\
\
if (Xembed[2] < M_PI / 2.) {\
Xa[2] = 0.;\
Xb[2] = 0.5 + SMALL;\
} else {\
Xa[2] = 0.5 - SMALL;\
Xb[2] = 1.;\
}\
\
coord_to_embed(Xa, Xtmp); tha = Xtmp[2];\
coord_to_embed(Xb, Xtmp); thb = Xtmp[2];\
\
if (m::abs(tha-th) < ROOTFIND_TOL) {\
Xnative[2] = Xa[2]; return;\
} else if (m::abs(thb-th) < ROOTFIND_TOL) {\
Xnative[2] = Xb[2]; return;\
}\
for (int i = 0; i < 1000; i++) {\
Xc[2] = 0.5 * (Xa[2] + Xb[2]);\
coord_to_embed(Xc, Xtmp); thc = Xtmp[2];\
\
if (m::abs(thc - th) < ROOTFIND_TOL) break;\
else if ((thc - th) * (thb - th) < 0.) Xa[2] = Xc[2];\
else Xb[2] = Xc[2];\
}\
Xnative[2] = Xc[2];
