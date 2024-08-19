



#include "dimensions.h"
#include "gauss_seidel.h"
#include <stdio.h>






GS::GS(Dimensions d) {
printf("Creating Gauss Seidel Simulation\n\n");

dims = d;


ncellx = dims.ncell_x;
ncelly = dims.ncell_y;
nu = dims.ncomponents;


v_size = nu*ncellx*ncelly;
a_size = nu*nu*ncellx*ncelly;

#pragma acc enter data copyin(this)


v1 = (real*) malloc(v_size*sizeof(real));
v2 = (real*) malloc(v_size*sizeof(real));
an = (real*) malloc(a_size*sizeof(real));
as = (real*) malloc(a_size*sizeof(real));
ae = (real*) malloc(a_size*sizeof(real));
aw = (real*) malloc(a_size*sizeof(real));


#pragma acc enter data copyin(v1[v_size], v2[v_size]),	\
create(an[a_size], as[a_size],			\
ae[a_size], aw[a_size])
}


void GS::init() {
int iy = 0;
int ix = 0;
int iu = 0;
int ju = 0;


#pragma acc parallel present (v1[v_size], v2[v_size], \
an[a_size], as[a_size], \
ae[a_size], aw[a_size], this)
{

#pragma acc loop independent gang, collapse(2)
for (iy=1; iy<ncelly-1; ++iy)
{
for (ix=1; ix<ncellx-1; ++ix)
{
#pragma acc loop independent vector
for (iu=0; iu<nu; ++iu)
{
#pragma acc loop seq
for (ju=0; ju<nu; ++ju)
{
an[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
as[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
ae[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
aw[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
}


v1[iu+nu*(ix+ncellx*iy)] = 1.0;
v2[iu+nu*(ix+ncellx*iy)] = 1.0;

an[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
as[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
ae[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
aw[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
}
}
}




#pragma acc loop independent gang
for (iy=0; iy<ncelly; ++iy)
{
#pragma acc loop independent vector
for (iu=0; iu<nu; ++iu)
{
v1[iu+nu*(0+ncellx*iy)] = 0.0;
v2[iu+nu*(0+ncellx*iy)] = 0.0;
v1[iu+nu*((ncellx-1)+ncellx*iy)] = 0.0;
v2[iu+nu*((ncellx-1)+ncellx*iy)] = 0.0;
}
}

#pragma acc loop independent gang
for (ix=0; ix<ncellx; ++ix)
{
#pragma acc loop independent vector
for (iu=0; iu<nu; ++iu)
{
v1[iu+nu*(ix+ncellx*0)] = 0.0;
v2[iu+nu*(ix+ncellx*0)] = 0.0;
v1[iu+nu*(ix+ncellx*(ncelly-1))] = 0.0;
v2[iu+nu*(ix+ncellx*(ncelly-1))] = 0.0;
}
}


}
} 



void GS::run(){
int iteration;
for (iteration=0; iteration<dims.niterations; ++iteration) {
const real* const __restrict__ vi = iteration%2 ? v1 : v2;
real* const __restrict__ vo = iteration%2 ? v2 : v1;


GS::solve(nu, ncellx, ncelly, vo, vi, an, as, ae, aw);
}


#pragma acc exit data copyout(v1[v_size], v2[v_size]), \
delete(an[a_size], as[a_size], \
ae[a_size], aw[a_size])


vfinal = dims.niterations%2 ? v1 : v2;


free(an);
free(as);
free(ae);
free(aw);

} 


void GS::print(){
int iy,ix,iu;

for (iy=0; iy<ncelly; ++iy)
for (ix=0; ix<ncellx; ++ix)
printf("vfinal[%d][%d][%d] = %f\n", iy, ix, 0, vfinal[0+nu*(ix+ncellx*iy)]);


free(v1);
free(v2);
}






void GS::solve(int nu, int ncellx, int ncelly, 
real* __restrict__ vo, 
const real* __restrict__ vi, 
const real* __restrict__ an, 
const real* __restrict__ as, 
const real* __restrict__ ae, 
const real* __restrict__ aw) 
{
const int nwavefronts = (ncellx + ncelly) - 1;
int wavefront = 0;


for (wavefront=0; wavefront<nwavefronts; ++wavefront)
{

int v_size = nu*ncellx*ncelly;
int a_size = nu*nu*ncellx*ncelly;

#pragma acc parallel present (vo[v_size], vi[v_size], \
an[a_size], as[a_size], \
ae[a_size], aw[a_size], this)
{

int ix;

#pragma acc loop independent gang
for (ix=1; ix<ncellx-1; ++ix)
{
const int iy = wavefront - ix;
if (iy > 1 && iy < ncelly-1)
process_cell(ix, iy, nu, ncellx, ncelly, vo, vi, an, as, ae, aw);
}
} 
}
} 




#pragma acc routine vector
void GS::process_cell(int ix, int iy, int nu, int ncellx, int ncelly,
real* __restrict__ vo,
const real* __restrict__ vi,
const real* __restrict__ an,
const real* __restrict__ as,
const real* __restrict__ ae,
const real* __restrict__ aw)
{
int iu = 0;
int ju = 0;




#pragma acc loop independent vector
for (iu=0; iu<nu; ++iu)
{

real sum = 0.0;
vo[iu+nu*(ix+ncellx*iy)] = 0.0;

#pragma acc loop seq
for (ju=0; ju<nu; ++ju)
{
sum +=

aw[ju+nu*(iu+nu*(ix+ncellx*iy))] * 
vo[iu+nu*((ix-1)+ncellx*iy)] + 

an[ju+nu*(iu+nu*(ix+ncellx*iy))] * 
vo[iu+nu*(ix+ncellx*(iy-1))] + 

ae[ju+nu*(iu+nu*(ix+ncellx*iy))] * 
vi[iu+nu*((ix+1)+ncellx*iy)] + 

as[ju+nu*(iu+nu*(ix+ncellx*iy))] * 
vi[iu+nu*(ix+ncellx*(iy-1))];
}

vo[iu+nu*(ix+ncellx*iy)] = sum;

}
} 
