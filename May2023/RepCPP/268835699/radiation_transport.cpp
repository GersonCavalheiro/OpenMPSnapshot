



#include "dimensions.h"
#include "radiation_transport.h"
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>






RT::RT(Dimensions d) {
printf("Creating Radiation Transport Simulation\n\n");

dims = d;
NE = dims.ncomponents;


int facexy_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
int facexz_size = dims.ncell_x * dims.ncell_z * NE * NA * NU * NOCTANT;
int faceyz_size = dims.ncell_y * dims.ncell_z * NE * NA * NU * NOCTANT;
int local_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
int v_size = NM * NA * NOCTANT;
int input_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;
int output_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;


facexy = (real*)malloc_host_real(facexy_size);
facexz = (real*)malloc_host_real(facexz_size);
faceyz = (real*)malloc_host_real(faceyz_size);
local = (real*)malloc_host_real(local_size);
m_from_a = (real*)malloc_host_real(v_size);
input = (real*)malloc_host_real(input_size);
output = (real*)malloc_host_real(output_size);


for (int iz=0; iz<dims.ncell_z; ++iz)
for (int iy=0; iy<dims.ncell_y; ++iy)
for (int ix=0; ix<dims.ncell_x; ++ix)
for( int ie=0; ie<NE; ++ie )
for( int iu=0; iu<NU; ++iu )
for ( int im=0; im<NM; ++im )
{
input[ im + NM * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
iz + dims.ncell_z * (
0))))))] = (real) (Quantities_scalefactor_space(ix, iy, iz) * (real) Quantities_scalefactor_energy(ie) * iu) + (real)(im*iu);
}


for (int octant=0; octant<NOCTANT; ++octant)
for (int im=0; im<NM; ++im)
for (int ia=0; ia<NA; ++ia)
{
m_from_a[im + NM * (
ia + NA * (
octant + NOCTANT * (
0)))] = (real)(im+1) + (1.0/(octant+1));
}
}


void RT::init(){


int octant = 0;
int ix = 0;
int iy = 0;
int iz = 0;
int ie = 0;
int iu = 0;
int ia = 0;
int im = 0;


int dim_x = dims.ncell_x;
int dim_y = dims.ncell_y;
int dim_z = dims.ncell_z;


int facexy_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
int facexz_size = dims.ncell_x * dims.ncell_z * NE * NA * NU * NOCTANT;
int faceyz_size = dims.ncell_y * dims.ncell_z * NE * NA * NU * NOCTANT;
int local_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
int v_size = NM * NA * NOCTANT;
int input_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;
int output_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;
{



for (iz=0; iz<dim_z; ++iz)
for (iy=0; iy<dim_y; ++iy)
for (ix=0; ix<dim_x; ++ix)

for( ie=0; ie<NE; ++ie )
for( iu=0; iu<NU; ++iu )
{
output[im + NM * (
iu + NU * (
ie + NE * (
ix + dim_x * (
iy + dim_y * (
iz + dim_z * (
0))))))] = (real)0.0;
}







for( octant=0; octant<NOCTANT; ++octant )
for (iy=0; iy<dim_y; ++iy)
for (ix=0; ix<dim_x; ++ix)

for( ie=0; ie<NE; ++ie )
for( iu=0; iu<NU; ++iu )
for( ia=0; ia<NA; ++ia )
{
iz = -1;
int scalefactor_space = Quantities_scalefactor_space(ix, iy, iz);

facexy[ia + NA * (
iu + NU * (
ie + NE * (
ix + dim_x * (
iy + dim_y * (
octant + NOCTANT * (
0 ))))))]
= Quantities_init_face(ia, ie, iu, scalefactor_space, octant);
}



for( octant=0; octant<NOCTANT; ++octant )
for (iz=0; iz<dim_z; ++iz)
for (ix=0; ix<dim_x; ++ix)

for( ie=0; ie<NE; ++ie )
for( iu=0; iu<NU; ++iu )
for( ia=0; ia<NA; ++ia )
{
iy = -1;
int scalefactor_space = Quantities_scalefactor_space(ix, iy, iz);

facexz[ia + NA * (
iu + NU * (
ie + NE * (
ix + dim_x * (
iz + dim_z * (
octant + NOCTANT * (
0 ))))))]
= Quantities_init_face(ia, ie, iu, scalefactor_space, octant);
}



for( octant=0; octant<NOCTANT; ++octant )
for (iz=0; iz<dim_z; iz++)
for (iy=0; iy<dim_y; iy++)

for( ie=0; ie<NE; ++ie )
for( iu=0; iu<NU; ++iu )
for( ia=0; ia<NA; ++ia )
{
ix = -1;
int scalefactor_space = Quantities_scalefactor_space(ix, iy, iz);

faceyz[ia + NA * (
iu + NU * (
ie + NE * (
iy + dim_y * (
iz + dim_z * (
octant + NOCTANT * (
0 ))))))]
= Quantities_init_face(ia, ie, iu, scalefactor_space, octant);
}
} 
}



void RT::run(){
printf("sweeping...\n\n");



int dim_x = dims.ncell_x;
int dim_y = dims.ncell_y;
int dim_z = dims.ncell_z;


int facexy_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
int facexz_size = dims.ncell_x * dims.ncell_z * NE * NA * NU * NOCTANT;
int faceyz_size = dims.ncell_y * dims.ncell_z * NE * NA * NU * NOCTANT;
int local_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
int v_size = NM * NA * NOCTANT;
int input_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;
int output_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;

auto coordinates=[](int o,int x,int y,int z,int dim_z,int dim_y,int dim_x){ return ((o*dim_z+z)*dim_y+y)*dim_x+x; };
int bw=1,nthreads=1,b=2;
char* bx_str=std::getenv("OMP_BLOCK_DIMX");
char* by_str=std::getenv("OMP_BLOCK_DIMY");
char* bz_str=std::getenv("OMP_BLOCK_DIMZ");
char* bw_str=std::getenv("OMP_BLOCK_DIMW");
char* nthreads_str=std::getenv("OMP_NUM_THREADS");
if(nthreads_str!=nullptr)
if(strlen(nthreads_str)>0)
nthreads=std::stoi(std::string(nthreads_str));
b*=nthreads;
int bx=b,by=b,bz=b;
if(bx_str!=nullptr)
if(strlen(bx_str)>0)
bx=std::stoi(std::string(bx_str));
if(by_str!=nullptr)
if(strlen(by_str)>0)
by=std::stoi(std::string(by_str));
if(bz_str!=nullptr)
if(strlen(bz_str)>0)
bz=std::stoi(std::string(bz_str));
if(bw_str!=nullptr)
if(strlen(bw_str)>0)
bw=std::stoi(std::string(bw_str));
#pragma omp dag coarsening(BLOCK,bw,bz,by,bx)
for( int octant=0; octant<NOCTANT; ++octant ) {
for (int iz=0; iz<dim_z; ++iz)
for (int iy=0; iy<dim_y; ++iy)
for (int ix=0; ix<dim_x; ++ix) {
#pragma omp dag depend({(coordinates(octant,ix,iy,iz+1,dim_z,dim_y,dim_x)),((iz+1)<dim_z)})
#pragma omp dag depend({(coordinates(octant,ix,iy+1,iz,dim_z,dim_y,dim_x)),((iy+1)<dim_y)})
#pragma omp dag depend({(coordinates(octant,ix+1,iy,iz,dim_z,dim_y,dim_x)),((ix+1)<dim_x)})
#pragma omp dag depend({(coordinates(octant,ix+1,iy+1,iz,dim_z,dim_y,dim_x)),(((ix+1)<dim_x)&&((iy+1)<dim_y))})
#pragma omp dag depend({(coordinates(octant,ix+1,iy,iz+1,dim_z,dim_y,dim_x)),(((ix+1)<dim_x)&&((iz+1)<dim_z))})
#pragma omp dag depend({(coordinates(octant,ix,iy+1,iz+1,dim_z,dim_y,dim_x)),(((iz+1)<dim_z)&&((iy+1)<dim_y))})
#pragma omp dag depend({(coordinates(octant,ix+1,iy+1,iz+1,dim_z,dim_y,dim_x)),((((iz+1)<dim_z)&&((iy+1)<dim_y))&&((ix+1)<dim_x))})
#pragma omp dag depend({(coordinates(octant+1,0,0,0,dim_z,dim_y,dim_x)),(((((octant+1)<NOCTANT)&&(ix>=(dim_x-1)))&&(iy>=(dim_y-1))))&&(iz>=(dim_z-1))})
#pragma omp dag task
{
solve(ix, iy, iz, octant);
}
}
} 
}


void RT::print(){
int ie = 0;
int iu = 0;
int im = 0;
for (int iz=0; iz<dims.ncell_z; iz++)
for (int iy=0; iy<dims.ncell_y; iy++)
for (int ix=0; ix<dims.ncell_x; ix++)
{
printf("output[%d][%d][%d] = %f\n", iz, iy, ix,
output[im + NM * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
iz + dims.ncell_z * (
0 )))))) ]
);
}
}




void RT::solve(int ix,
int iy,
int iz,
int octant) {


int ia = 0;
int iu = 0;
int ie = 0;
int im = 0;
int dim_ne = NE;
int dim_nu = NU;
int dim_na = NA;
int dim_nm = NM;




for (ie=0; ie<dim_ne; ++ie)
for( iu=0; iu<dim_nu; ++iu )
for( ia=0; ia<dim_na; ++ia )
{
real result = (real)0.0;


for ( im=0; im<dim_nm; ++im) {

real a_from_m = 1.0/im;


result += input[im + NM * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
iz + dims.ncell_z * (

0 )))))) ] *
m_from_a[im + NM * (
ia + NA * (
octant + NOCTANT * (
0)))];

} 

local[ia + NA * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
octant + NOCTANT * (
0 )))))) ] = result;
} 


for( ie=0; ie<dim_ne; ++ie )
for( ia=0; ia<dim_na; ++ia )
{
compute(ix, iy, iz, ie, ia, octant);
}



for (ie=0; ie<dim_ne; ++ie)
for( iu=0; iu<dim_nu; ++iu )
for ( im=0; im<dim_nm; ++im)
{
real result = (real)0;

for( ia=0; ia<dim_na; ++ia )
{
result += local[ia + NA * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
octant + NOCTANT * (
0 )))))) ] *
m_from_a[im + NM * (
ia + NA * (
octant + NOCTANT * (
0)))];
}

output[im + NM * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
iz + dims.ncell_z * (
0 )))))) ] += result;
} 
}









void RT::compute(int ix, int iy, int iz, int ie, int ia, int octant){




int iu = 0;


const real scalefactor_octant = (real)1.0 + octant;
const real scalefactor_octant_r = ((real)1) / scalefactor_octant;


const real scalefactor_space = (real)Quantities_scalefactor_space(ix, iy, iz);
const real scalefactor_space_r = ((real)1) / scalefactor_space;
const real scalefactor_space_x_r = ((real)1) /
Quantities_scalefactor_space( ix - 1, iy, iz );
const real scalefactor_space_y_r = ((real)1) /
Quantities_scalefactor_space( ix, iy - 1, iz );
const real scalefactor_space_z_r = ((real)1) /
Quantities_scalefactor_space( ix, iy, iz - 1 );


for( iu=0; iu<NU; ++iu )
{

int local_index = ia + NA * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
octant + NOCTANT * (
0))))));

const real result = (real)1.0/( local[local_index] * scalefactor_space_r +
(

facexy[ia + NA * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
octant + NOCTANT * (
0 )))))) ]


* (real) ( 1 / (real) 2 )

* scalefactor_space_z_r


+ facexz[ia + NA * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iz + dims.ncell_z * (
octant + NOCTANT * (
0 )))))) ]


* (real) ( 1 / (real) 4 )

* scalefactor_space_y_r


+ faceyz[ia + NA * (
iu + NU * (
ie + NE * (
iy + dims.ncell_y * (
iz + dims.ncell_z * (
octant + NOCTANT * (
0 )))))) ]


* (real) ( 1 / (real) 4 - 1 / (real) (1 << ( ia & ( (1<<3) - 1 ) )) )

* scalefactor_space_x_r
)
* scalefactor_octant_r ) * scalefactor_space;

local[local_index] = result;

const real result_scaled = result * scalefactor_octant;

facexy[ia + NA * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iy + dims.ncell_y * (
octant + NOCTANT * (
0 )))))) ] = result_scaled;


facexz[ia + NA * (
iu + NU * (
ie + NE * (
ix + dims.ncell_x * (
iz + dims.ncell_z * (
octant + NOCTANT * (
0 )))))) ] = result_scaled;


faceyz[ia + NA * (
iu + NU * (
ie + NE * (
iy + dims.ncell_y * (
iz + dims.ncell_z * (
octant + NOCTANT * (
0 )))))) ] = result_scaled;

} 
}





real RT::Quantities_init_face(int ia, int ie, int iu, int scalefactor_space, int octant)
{



return ( (real) (1 + ia) )


* ( (real) (1 << (ia & ( (1<<3) - 1))) )


* ( (real) scalefactor_space)


* ( (real) (1 << ((( (ie) * 1366 + 150889) % 714025) & ( (1<<2) - 1))) )


* ( (real) (1 << ((( (iu) * 741 + 60037) % 312500) & ( (1<<2) - 1))) )


* ( (real) 1 + octant);
}





int RT::Quantities_scalefactor_space(int ix, int iy, int iz)
{

int scalefactor_space = 0;

scalefactor_space = ( (scalefactor_space+(ix+2))*8121 + 28411 ) % 134456;
scalefactor_space = ( (scalefactor_space+(iy+2))*8121 + 28411 ) % 134456;
scalefactor_space = ( (scalefactor_space+(iz+2))*8121 + 28411 ) % 134456;
scalefactor_space = ( (scalefactor_space+(ix+3*iy+7*iz+2))*8121 + 28411 ) % 134456;
scalefactor_space = ix+3*iy+7*iz+2;
scalefactor_space = scalefactor_space & ( (1<<2) - 1 );
scalefactor_space = 1 << scalefactor_space;

return scalefactor_space;
}





int RT::Quantities_scalefactor_energy(int ie)
{
const int im = 714025;
const int ia = 1366;
const int ic = 150889;

int result = ( (ie)*ia + ic ) % im;
result = result & ( (1<<2) - 1 );
result = 1 << result;

return result;
}
