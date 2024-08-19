#include "Interpolator3D4OrderV.h"

#include <cmath>
#include <iostream>

#include "ElectroMagn.h"
#include "Field3D.h"
#include "Particles.h"

using namespace std;


Interpolator3D4OrderV::Interpolator3D4OrderV( Params &params, Patch *patch ) : Interpolator3D4Order( params, patch )
{
d_inv_[0] = 1.0/params.cell_length[0];
d_inv_[1] = 1.0/params.cell_length[1];
d_inv_[2] = 1.0/params.cell_length[2];

dble_1_ov_384 = 1.0/384.0;
dble_1_ov_48 = 1.0/48.0;
dble_1_ov_16 = 1.0/16.0;
dble_1_ov_12 = 1.0/12.0;
dble_1_ov_24 = 1.0/24.0;
dble_19_ov_96 = 19.0/96.0;
dble_11_ov_24 = 11.0/24.0;
dble_1_ov_4 = 1.0/4.0;
dble_1_ov_6 = 1.0/6.0;
dble_115_ov_192 = 115.0/192.0;
dble_5_ov_8 = 5.0/8.0;
}


void Interpolator3D4OrderV::fieldsWrapper( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, unsigned int scell, int ipart_ref )
{
if( istart[0] == iend[0] ) {
return;    
}

int nparts( ( smpi->dynamics_invgf[ithread] ).size() );

double * __restrict__ Epart[3];
double * __restrict__ Bpart[3];

double *deltaO[3];
deltaO[0] = &( smpi->dynamics_deltaold[ithread][0] );
deltaO[1] = &( smpi->dynamics_deltaold[ithread][nparts] );
deltaO[2] = &( smpi->dynamics_deltaold[ithread][2*nparts] );

for( unsigned int k=0; k<3; k++ ) {
Epart[k]= &( smpi->dynamics_Epart[ithread][k*nparts] );
Bpart[k]= &( smpi->dynamics_Bpart[ithread][k*nparts] );
}

int idx[3], idxO[3];
idx[0]  = round( particles.position( 0, *istart ) * d_inv_[0] );
idxO[0] = idx[0] - i_domain_begin  ;
idx[1]  = round( particles.position( 1, *istart ) * d_inv_[1] );
idxO[1] = idx[1] - j_domain_begin  ;
idx[2]  = round( particles.position( 2, *istart ) * d_inv_[2] );
idxO[2] = idx[2] - k_domain_begin  ;

Field3D *Ex3D = static_cast<Field3D *>( EMfields->Ex_ );
Field3D *Ey3D = static_cast<Field3D *>( EMfields->Ey_ );
Field3D *Ez3D = static_cast<Field3D *>( EMfields->Ez_ );
Field3D *Bx3D = static_cast<Field3D *>( EMfields->Bx_m );
Field3D *By3D = static_cast<Field3D *>( EMfields->By_m );
Field3D *Bz3D = static_cast<Field3D *>( EMfields->Bz_m );

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);
double * __restrict__ position_z = particles.getPtrPosition(2);

double coeff[3][2][5][32];
int dual[3][32]; 

int vecSize = 32;

int cell_nparts( ( int )iend[0]-( int )istart[0] );
int nbVec = ( iend[0]-istart[0]+( cell_nparts-1 )-( ( iend[0]-istart[0]-1 )&( cell_nparts-1 ) ) ) / vecSize;

if( nbVec*vecSize != cell_nparts ) {
nbVec++;
}

for( int iivect=0 ; iivect<nbVec; iivect++ ) {
int ivect = vecSize*iivect;

int np_computed( 0 );
if( cell_nparts > vecSize ) {
np_computed = vecSize;
cell_nparts -= vecSize;
} else {
np_computed = cell_nparts;
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

double delta0, delta;
double delta2, delta3, delta4;


delta0 = position_x[ipart+ivect+istart[0]] *d_inv_[0];
dual [0][ipart] = ( delta0 - ( double )idx[0] >=0. );


delta   = delta0 - ( double )idx[0] ;
delta2  = delta*delta;
delta3  = delta2*delta;
delta4  = delta3*delta;

coeff[0][0][0][ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
coeff[0][0][1][ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[0][0][2][ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4 * delta4;
coeff[0][0][3][ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[0][0][4][ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

deltaO[0][ipart-ipart_ref+ivect+istart[0]] = delta;


delta   = delta0 - ( double )idx[0] + ( 0.5-dual[0][ipart] );
delta2  = delta*delta;
delta3  = delta2*delta;
delta4  = delta3*delta;

coeff[0][1][0][ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
coeff[0][1][1][ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[0][1][2][ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4 * delta4;
coeff[0][1][3][ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[0][1][4][ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;


delta0 = position_y[ipart+ivect+istart[0]] *d_inv_[1];
dual [1][ipart] = ( delta0 - ( double )idx[1] >=0. );


delta   = delta0 - ( double )idx[1];
delta2  = delta*delta;
delta3  = delta2*delta;
delta4  = delta3*delta;

coeff[1][0][0][ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
coeff[1][0][1][ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[1][0][2][ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4 * delta4;
coeff[1][0][3][ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[1][0][4][ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

deltaO[1][ipart-ipart_ref+ivect+istart[0]] = delta;


delta   = delta0 - ( double )idx[1] + ( 0.5-dual[1][ipart] );
delta2  = delta*delta;
delta3  = delta2*delta;
delta4  = delta3*delta;

coeff[1][1][0][ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
coeff[1][1][1][ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[1][1][2][ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4 * delta4;
coeff[1][1][3][ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[1][1][4][ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;


delta0 = position_z[ipart+ivect+istart[0]] *d_inv_[2];
dual [2][ipart] = ( delta0 - ( double )idx[2] >=0. );


delta   = delta0 - ( double )idx[2];
delta2  = delta*delta;
delta3  = delta2*delta;
delta4  = delta3*delta;

coeff[2][0][0][ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
coeff[2][0][1][ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[2][0][2][ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4 * delta4;
coeff[2][0][3][ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[2][0][4][ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

deltaO[2][ipart-ipart_ref+ivect+istart[0]] = delta;


delta   = delta0 - ( double )idx[2] + 1*( 0.5-dual[2][ipart] );
delta2  = delta*delta;
delta3  = delta2*delta;
delta4  = delta3*delta;

coeff[2][1][0][ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
coeff[2][1][1][ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[2][1][2][ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4 * delta4;
coeff[2][1][3][ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[2][1][4][ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

}

double interp_res = 0;

#if defined __INTEL_COMPILER

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

double *coeffyp = &( coeff[1][0][2][ipart] );
double *coeffyd = &( coeff[1][1][2][ipart] );
double *coeffxd = &( coeff[0][1][2][ipart] );
double *coeffxp = &( coeff[0][0][2][ipart] );
double *coeffzp = &( coeff[2][0][2][ipart] );
double *coeffzd = &( coeff[2][1][2][ipart] );

interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxd+iloc*32 ) * *( coeffyp+jloc*32 ) * *( coeffzp+kloc*32 ) *
( ( 1-dual[0][ipart] )*( *Ex3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[0][ipart]*( *Ex3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+kloc ) );
}
}
}
Epart[0][ipart-ipart_ref+ivect+istart[0]] = interp_res;


interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*32 ) * *( coeffyd+jloc*32 ) * *( coeffzp+kloc*32 ) *
( ( 1-dual[1][ipart] )*( *Ey3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[1][ipart]*( *Ey3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) );
}
}
}
Epart[1][ipart-ipart_ref+ivect+istart[0]] = interp_res;


interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*32 ) * *( coeffyp+jloc*32 ) * *( coeffzd+kloc*32 ) *
( ( 1-dual[2][ipart] )*( *Ez3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[2][ipart]*( *Ez3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) );
}
}
}
Epart[2][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}

interp_Bx( idxO, np_computed, &(coeff[0][0][2][0]), &(coeff[1][1][2][0]), &(coeff[2][1][2][0]), &(dual[1][0]), &(dual[2][0]), Bx3D, &(Bpart[0][ivect+istart[0]-ipart_ref]) );
interp_By( idxO, np_computed, &(coeff[0][1][2][0]), &(coeff[1][0][2][0]), &(coeff[2][1][2][0]), &(dual[0][0]), &(dual[2][0]), By3D, &(Bpart[1][ivect+istart[0]-ipart_ref]) );
interp_Bz( idxO, np_computed, &(coeff[0][1][2][0]), &(coeff[1][1][2][0]), &(coeff[2][0][2][0]), &(dual[0][0]), &(dual[1][0]), Bz3D, &(Bpart[2][ivect+istart[0]-ipart_ref]) );


#else

double * __restrict__ coeffxp2 = &( coeff[0][0][2][0] );
double * __restrict__ coeffxd2 = &( coeff[0][1][2][0] );

double * __restrict__ coeffyp2 = &( coeff[1][0][2][0] );
double * __restrict__ coeffyd2 = &( coeff[1][1][2][0] );

double * __restrict__ coeffzp2 = &( coeff[2][0][2][0] );
double * __restrict__ coeffzd2 = &( coeff[2][1][2][0] );

double field_buffer[6][6][6];


for( int iloc=-2 ; iloc<4 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
field_buffer[iloc+2][jloc+2][kloc+2] = Ex3D->data_3D[idxO[0]+iloc][idxO[1]+jloc][idxO[2]+kloc];
}
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

interp_res = 0.;

UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
UNROLL_S(5)
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += coeffxd2[ipart+iloc*32] * coeffyp2[ipart+jloc*32] * coeffzp2[ipart+kloc*32] *
( ( 1-dual[0][ipart] )* field_buffer[iloc+2][jloc+2][kloc+2]
+ dual[0][ipart]*field_buffer[iloc+3][jloc+2][kloc+2] );
}
}
}
Epart[0][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}


for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<4 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
field_buffer[iloc+2][jloc+2][kloc+2] = Ey3D->data_3D[idxO[0]+iloc][idxO[1]+jloc][idxO[2]+kloc];
}
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

interp_res = 0.;

UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
UNROLL_S(5)
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += coeffxp2[ipart+iloc*32] * coeffyd2[ipart+jloc*32] * coeffzp2[ipart+kloc*32] *
( ( 1-dual[1][ipart] )* field_buffer[iloc+2][jloc+2][kloc+2]
+ dual[1][ipart]*field_buffer[2+iloc][jloc+3][kloc+2] );
}
}
}
Epart[1][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}


for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<4 ; kloc++ ) {
field_buffer[iloc+2][jloc+2][kloc+2] = Ez3D->data_3D[idxO[0]+iloc][idxO[1]+jloc][idxO[2]+kloc];
}
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

interp_res = 0.;

UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
UNROLL_S(5)
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += coeffxp2[ipart+iloc*32] * coeffyp2[ipart+jloc*32] * coeffzd2[ipart+kloc*32] *
( ( 1-dual[2][ipart] )* field_buffer[iloc+2][jloc+2][kloc+2] +
dual[2][ipart]*field_buffer[2+iloc][jloc+2][kloc+3] );
}
}
}
Epart[2][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}



for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<4 ; jloc++ ) {
for( int kloc=-2 ; kloc<4 ; kloc++ ) {
field_buffer[iloc+2][jloc+2][kloc+2] = Bx3D->data_3D[idxO[0]+iloc][idxO[1]+jloc][idxO[2]+kloc];
}
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
UNROLL_S(5)
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += coeffxp2[ipart+iloc*32] * coeffyd2[ipart+jloc*32] * coeffzd2[ipart+kloc*32] *
( ( 1-dual[2][ipart] ) * ( ( 1-dual[1][ipart] )*field_buffer[2+iloc][2+jloc][2+kloc] + dual[1][ipart]*field_buffer[2+iloc][3+jloc][2+kloc] )
+    dual[2][ipart]  * ( ( 1-dual[1][ipart] )*field_buffer[2+iloc][2+jloc][3+kloc] + dual[1][ipart]*field_buffer[2+iloc][3+jloc][3+kloc] ) );
}
}
}
Bpart[0][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}



for( int iloc=-2 ; iloc<4 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<4 ; kloc++ ) {
field_buffer[iloc+2][jloc+2][kloc+2] = By3D->data_3D[idxO[0]+iloc][idxO[1]+jloc][idxO[2]+kloc];
}
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
UNROLL_S(5)
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += coeffxd2[ipart+iloc*32] * coeffyp2[ipart+jloc*32] * coeffzd2[ipart+kloc*32] *
( ( 1-dual[2][ipart] ) * ( ( 1-dual[0][ipart] )*field_buffer[2+iloc][2+jloc][2+kloc] + dual[0][ipart]*field_buffer[3+iloc][2+jloc][2+kloc] )
+    dual[2][ipart]  * ( ( 1-dual[0][ipart] )*field_buffer[2+iloc][2+jloc][3+kloc] + dual[0][ipart]*field_buffer[3+iloc][2+jloc][3+kloc] ) );
}
}
}
Bpart[1][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}



for( int iloc=-2 ; iloc<4 ; iloc++ ) {
for( int jloc=-2 ; jloc<4 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
field_buffer[iloc+2][jloc+2][kloc+2] = Bz3D->data_3D[idxO[0]+iloc][idxO[1]+jloc][idxO[2]+kloc];
}
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
UNROLL_S(5)
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += coeffxd2[ipart+iloc*32] * coeffyd2[ipart+jloc*32] * coeffzp2[ipart+kloc*32] *
( ( 1-dual[1][ipart] ) * ( ( 1-dual[0][ipart] )*field_buffer[2+iloc][2+jloc][2+kloc] + dual[0][ipart]*field_buffer[3+iloc][2+jloc][2+kloc] )
+    dual[1][ipart]  * ( ( 1-dual[0][ipart] )*field_buffer[2+iloc][3+jloc][2+kloc] + dual[0][ipart]*field_buffer[3+iloc][3+jloc][2+kloc] ) );
}
}
}
Bpart[2][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}

#endif


}
} 


void Interpolator3D4OrderV::fieldsAndCurrents( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, LocalFields *JLoc, double *RhoLoc )
{

int ipart = *istart;
int nparts( particles.size() );


double *Epart[3], *Bpart[3];
for( unsigned int k=0; k<3; k++ ) {
Epart[k]= &( smpi->dynamics_Epart[ithread][k*nparts] );
Bpart[k]= &( smpi->dynamics_Bpart[ithread][k*nparts] );
}

int idx[3], idxO[3];
idx[0]  = round( particles.position( 0, *istart ) * d_inv_[0] );
idxO[0] = idx[0] - i_domain_begin  ;
idx[1]  = round( particles.position( 1, *istart ) * d_inv_[1] );
idxO[1] = idx[1] - j_domain_begin  ;
idx[2]  = round( particles.position( 2, *istart ) * d_inv_[2] );
idxO[2] = idx[2] - k_domain_begin  ;

Field3D *Ex3D = static_cast<Field3D *>( EMfields->Ex_ );
Field3D *Ey3D = static_cast<Field3D *>( EMfields->Ey_ );
Field3D *Ez3D = static_cast<Field3D *>( EMfields->Ez_ );
Field3D *Bx3D = static_cast<Field3D *>( EMfields->Bx_ );
Field3D *By3D = static_cast<Field3D *>( EMfields->By_ );
Field3D *Bz3D = static_cast<Field3D *>( EMfields->Bz_ );
Field3D *Jx3D = static_cast<Field3D *>( EMfields->Jx_ );
Field3D *Jy3D = static_cast<Field3D *>( EMfields->Jy_ );
Field3D *Jz3D = static_cast<Field3D *>( EMfields->Jz_ );
Field3D *rho3D = static_cast<Field3D *>( EMfields->rho_ );

double coeff[3][2][5];
int dual[3]; 

double delta0, delta;
double delta2, delta3, delta4;

for( int i=0; i<3; i++ ) { 
delta0 = particles.position( i, ipart )*d_inv_[i];
dual [i] = ( delta0 - ( double )idx[i] >=0. );

for( int j=0; j<2; j++ ) { 

delta   = delta0 - ( double )idx[i] + ( double )j*( 0.5-dual[i] );
delta2  = delta*delta;
delta3  = delta2*delta;
delta4  = delta3*delta;

coeff[i][j][0] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
coeff[i][j][1] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[i][j][2] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4 * delta4;
coeff[i][j][3] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4 * delta2  - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
coeff[i][j][4] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

}
}

double *coeffyp = &( coeff[1][0][2] );
double *coeffyd = &( coeff[1][1][2] );
double *coeffxd = &( coeff[0][1][2] );
double *coeffxp = &( coeff[0][0][2] );
double *coeffzp = &( coeff[2][0][2] );
double *coeffzd = &( coeff[2][1][2] );

double interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxd+iloc*1 ) * *( coeffyp+jloc*1 ) * *( coeffzp+kloc*1 ) *
( ( 1-dual[0] )*( *Ex3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[0]*( *Ex3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+kloc ) );
}
}
}
Epart[0][ipart] = interp_res;


interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*1 ) * *( coeffyd+jloc*1 ) * *( coeffzp+kloc*1 ) *
( ( 1-dual[1] )*( *Ey3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[1]*( *Ey3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) );
}
}
}
Epart[1][ipart] = interp_res;

interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*1 ) * *( coeffyp+jloc*1 ) * *( coeffzd+kloc*1 ) *
( ( 1-dual[2] )*( *Ez3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[2]*( *Ez3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) );
}
}
}
Epart[2][ipart] = interp_res;


interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*1 ) * *( coeffyd+jloc*1 ) * *( coeffzd+kloc*1 ) *
( ( 1-dual[2] ) * ( ( 1-dual[1] )*( *Bx3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[1]*( *Bx3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) )
+    dual[2]  * ( ( 1-dual[1] )*( *Bx3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) + dual[1]*( *Bx3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+1+kloc ) ) );
}
}
}
Bpart[0][ipart] = interp_res;

interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxd+iloc*1 ) * *( coeffyp+jloc*1 ) * *( coeffzd+kloc*1 ) *
( ( 1-dual[2] ) * ( ( 1-dual[0] )*( *By3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[0]*( *By3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+kloc ) )
+    dual[2]  * ( ( 1-dual[0] )*( *By3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) + dual[0]*( *By3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) ) );
}
}
}
Bpart[1][ipart] = interp_res;

interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxd+iloc*1 ) * *( coeffyd+jloc*1 ) * *( coeffzp+kloc*1 ) *
( ( 1-dual[1] ) * ( ( 1-dual[0] )*( *Bz3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[0]*( *Bz3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+kloc ) )
+    dual[1]  * ( ( 1-dual[0] )*( *Bz3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) + dual[0]*( *Bz3D )( idxO[0]+1+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) ) );
}
}
}
Bpart[2][ipart] = interp_res;


interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxd+iloc*1 ) * *( coeffyp+jloc*1 ) * *( coeffzp+kloc*1 ) *
( ( 1-dual[0] )*( *Jx3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[0]*( *Jx3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+kloc ) );
}
}
}
JLoc->x = interp_res;


interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*1 ) * *( coeffyd+jloc*1 ) * *( coeffzp+kloc*1 ) *
( ( 1-dual[1] )*( *Jy3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[1]*( *Jy3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) );
}
}
}
JLoc->y = interp_res;


interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*1 ) * *( coeffyp+jloc*1 ) * *( coeffzd+kloc*1 ) *
( ( 1-dual[2] )*( *Jz3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dual[2]*( *Jz3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) );
}
}
}
JLoc->z = interp_res;

interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeffxp+iloc*1 ) * *( coeffyp+jloc*1 ) * *( coeffzp+kloc*1 ) * ( *rho3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc );
}
}
}
( *RhoLoc ) = interp_res;

}



void Interpolator3D4OrderV::fieldsAndEnvelope( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, int ipart_ref )
{
ERROR( "Projection and interpolation for the envelope model are implemented only for interpolation_order = 2" );
}


void Interpolator3D4OrderV::timeCenteredEnvelope( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, int ipart_ref )
{
ERROR( "Projection and interpolation for the envelope model are implemented only for interpolation_order = 2" );
}

void Interpolator3D4OrderV::envelopeAndSusceptibility( ElectroMagn *EMfields, Particles &particles, int ipart, double *Env_A_abs_Loc, double *Env_Chi_Loc, double *Env_E_abs_Loc, double *Env_Ex_abs_Loc )
{
ERROR( "Projection and interpolation for the envelope model are implemented only for interpolation_order = 2" );
}



void Interpolator3D4OrderV::oneField( Field **field, Particles &particles, int *istart, int *iend, double *FieldLoc, double *l1, double *l2, double *l3 )
{
ERROR( "Single field 3D4O interpolator not available in vectorized mode" );
}

void Interpolator3D4OrderV::interp_Bx( int* idxO, int np_computed, double *coeffxp, double *coeffyd, double *coeffzd, int *dualy, int* dualz, Field3D *Bx3D, double *Bpart ) {

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
double *coeff_yd = &( coeffyd[ipart] );
double *coeff_xp = &( coeffxp[ipart] );
double *coeff_zd = &( coeffzd[ipart] );
double interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeff_xp+iloc*32 ) * *( coeff_yd+jloc*32 ) * *( coeff_zd+kloc*32 ) *
( ( 1-dualz[ipart] ) * ( ( 1-dualy[ipart] )*( *Bx3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dualy[ipart]*( *Bx3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) )
+    dualz[ipart]  * ( ( 1-dualy[ipart] )*( *Bx3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) + dualy[ipart]*( *Bx3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+1+kloc ) ) );
}
}
}
Bpart[ipart] = interp_res;
}
}

void Interpolator3D4OrderV::interp_By( int* idxO, int np_computed, double *coeffxd, double *coeffyp, double *coeffzd, int *dualx, int* dualz, Field3D *By3D, double *Bpart )
{
#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
double *coeff_yp = &( coeffyp[ipart] );
double *coeff_xd = &( coeffxd[ipart] );
double *coeff_zd = &( coeffzd[ipart] );
double interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeff_xd+iloc*32 ) * *( coeff_yp+jloc*32 ) * *( coeff_zd+kloc*32 ) *
( ( 1-dualz[ipart] ) * ( ( 1-dualx[ipart] )*( *By3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dualx[ipart]*( *By3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+kloc ) )
+    dualz[ipart]  * ( ( 1-dualx[ipart] )*( *By3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) + dualx[ipart]*( *By3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+1+kloc ) ) );
}
}
}
Bpart[ipart] = interp_res;
}
}

void Interpolator3D4OrderV::interp_Bz( int* idxO, int np_computed, double *coeffxd, double *coeffyd, double *coeffzp, int *dualx, int* dualy, Field3D *Bz3D, double *Bpart )
{
#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
double *coeff_yd = &( coeffyd[ipart] );
double *coeff_xd = &( coeffxd[ipart] );
double *coeff_zp = &( coeffzp[ipart] );
double interp_res = 0.;
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
for( int kloc=-2 ; kloc<3 ; kloc++ ) {
interp_res += *( coeff_xd+iloc*32 ) * *( coeff_yd+jloc*32 ) * *( coeff_zp+kloc*32 ) *
( ( 1-dualy[ipart] ) * ( ( 1-dualx[ipart] )*( *Bz3D )( idxO[0]+iloc, idxO[1]+jloc, idxO[2]+kloc ) + dualx[ipart]*( *Bz3D )( idxO[0]+1+iloc, idxO[1]+jloc, idxO[2]+kloc ) )
+    dualy[ipart]  * ( ( 1-dualx[ipart] )*( *Bz3D )( idxO[0]+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) + dualx[ipart]*( *Bz3D )( idxO[0]+1+iloc, idxO[1]+1+jloc, idxO[2]+kloc ) ) );
}
}
}
Bpart[ipart] = interp_res;
}
}

