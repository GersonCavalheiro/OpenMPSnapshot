#include "Interpolator2D4OrderV.h"

#include <cmath>
#include <iostream>

#include "ElectroMagn.h"
#include "Field2D.h"
#include "Particles.h"

using namespace std;


Interpolator2D4OrderV::Interpolator2D4OrderV( Params &params, Patch *patch ) : Interpolator2D4Order( params, patch )
{

d_inv_[0] = 1.0/params.cell_length[0];
d_inv_[1] = 1.0/params.cell_length[1];

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


void Interpolator2D4OrderV::fieldsWrapper(  ElectroMagn *EMfields,
Particles &particles,
SmileiMPI *smpi,
int *istart, int *iend, int ithread, unsigned int scell, int ipart_ref )
{



if( istart[0] == iend[0] ) {
return;    
}

int nparts( ( smpi->dynamics_invgf[ithread] ).size() );

double * __restrict__ Epart[3];
double * __restrict__ Bpart[3];

double *deltaO[2];
deltaO[0] = &( smpi->dynamics_deltaold[ithread][0] );
deltaO[1] = &( smpi->dynamics_deltaold[ithread][nparts] );

for( unsigned int k=0; k<3; k++ ) {
Epart[k]= &( smpi->dynamics_Epart[ithread][k*nparts] );
Bpart[k]= &( smpi->dynamics_Bpart[ithread][k*nparts] );
}

int idx[2], idxO[2];
idx[0]  = round( particles.position( 0, *istart ) * d_inv_[0] );
idxO[0] = idx[0] - i_domain_begin  ;
idx[1]  = round( particles.position( 1, *istart ) * d_inv_[1] );
idxO[1] = idx[1] - j_domain_begin  ;

Field2D *Ex2D = static_cast<Field2D *>( EMfields->Ex_ );
Field2D *Ey2D = static_cast<Field2D *>( EMfields->Ey_ );
Field2D *Ez2D = static_cast<Field2D *>( EMfields->Ez_ );
Field2D *Bx2D = static_cast<Field2D *>( EMfields->Bx_m );
Field2D *By2D = static_cast<Field2D *>( EMfields->By_m );
Field2D *Bz2D = static_cast<Field2D *>( EMfields->Bz_m );

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);

double coeff[2][2][5][32];

int dual[2][32];

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

double delta0, delta, delta2, delta3, delta4;


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



delta   = delta0 - ( double )idx[1] ;
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



}

double * __restrict__ coeffxp2 = &( coeff[0][0][2][0] );
double * __restrict__ coeffxd2 = &( coeff[0][1][2][0] );

double * __restrict__ coeffyp2 = &( coeff[1][0][2][0] );
double * __restrict__ coeffyd2 = &( coeff[1][1][2][0] );

double field_buffer[6][6];

double interp_res = 0;


for( int iloc=-2 ; iloc<4 ; iloc++ ) {
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
field_buffer[iloc+2][jloc+2] = ( *Ex2D )( idxO[0]+iloc, idxO[1]+jloc );
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
interp_res += coeffxd2[ipart+iloc*32] * coeffyp2[ipart+jloc*32] *
( ( 1-dual[0][ipart] )*field_buffer[iloc+2][jloc+2]
+ dual[0][ipart]*field_buffer[iloc+3][jloc+2] );
}
}
Epart[0][ipart-ipart_ref+ivect+istart[0]] = interp_res;
}

for( int iloc=-2 ; iloc<3 ; iloc++ ) {
for( int jloc=-2 ; jloc<4 ; jloc++ ) {
field_buffer[iloc+2][jloc+2] = ( *Ey2D )( idxO[0]+iloc, idxO[1]+jloc );
}
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
interp_res += coeffxp2[ipart+iloc*32] * coeffyd2[ipart+jloc*32] *
( ( 1-dual[1][ipart] )*( *Ey2D )( idxO[0]+iloc, idxO[1]+jloc)
+ dual[1][ipart]*( *Ey2D )( idxO[0]+iloc, idxO[1]+1+jloc ) );
}
}
Epart[1][ipart-ipart_ref+ivect+istart[0]] = interp_res;

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
interp_res += coeffxp2[ipart+iloc*32] * coeffyp2[ipart+jloc*32]  *
( *Ez2D )( idxO[0]+iloc, idxO[1]+jloc );
}
}
Epart[2][ipart-ipart_ref+ivect+istart[0]] = interp_res;

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
interp_res += coeffxp2[ipart+iloc*32] * coeffyd2[ipart+jloc*32] *
(  ( 1-dual[1][ipart] )*( *Bx2D )( idxO[0]+iloc, idxO[1]+jloc )
+ dual[1][ipart]*( *Bx2D )( idxO[0]+iloc, idxO[1]+1+jloc ) );
}
}
Bpart[0][ipart-ipart_ref+ivect+istart[0]] = interp_res;

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
interp_res += coeffxd2[ipart+iloc*32] * coeffyp2[ipart+jloc*32] *
( ( 1-dual[0][ipart] )*( *By2D )( idxO[0]+iloc, idxO[1]+jloc )
+ dual[0][ipart]*( *By2D )( idxO[0]+1+iloc, idxO[1]+jloc) );
}
}
Bpart[1][ipart-ipart_ref+ivect+istart[0]] = interp_res;

interp_res = 0.;
UNROLL_S(5)
for( int iloc=-2 ; iloc<3 ; iloc++ ) {
UNROLL_S(5)
for( int jloc=-2 ; jloc<3 ; jloc++ ) {
interp_res += coeffxd2[ipart+iloc*32] * coeffyd2[ipart+jloc*32] *
( ( 1-dual[1][ipart] ) * ( ( 1-dual[0][ipart] )*( *Bz2D )( idxO[0]+iloc, idxO[1]+jloc) + dual[0][ipart]*( *Bz2D )( idxO[0]+1+iloc, idxO[1]+jloc) )
+    dual[1][ipart]  * ( ( 1-dual[0][ipart] )*( *Bz2D )( idxO[0]+iloc, idxO[1]+1+jloc) + dual[0][ipart]*( *Bz2D )( idxO[0]+1+iloc, idxO[1]+1+jloc) ) );
}
}
Bpart[2][ipart-ipart_ref+ivect+istart[0]] = interp_res;

}


}

}

void Interpolator2D4OrderV::oneField( Field **field, Particles &particles, int *istart, int *iend, double *FieldLoc, double *l1, double *l2, double *l3 )
{
ERROR( "Single field 2D2O interpolator not available in vectorized mode" );
}

void Interpolator2D4OrderV::fieldsAndEnvelope( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, int ipart_ref )
{
ERROR( "Projection and interpolation for the envelope model are implemented only for interpolation_order = 2" );
}


void Interpolator2D4OrderV::timeCenteredEnvelope( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, int ipart_ref )
{
ERROR( "Projection and interpolation for the envelope model are implemented only for interpolation_order = 2" );
}

void Interpolator2D4OrderV::envelopeAndSusceptibility( ElectroMagn *EMfields, Particles &particles, int ipart, double *Env_A_abs_Loc, double *Env_Chi_Loc, double *Env_E_abs_Loc, double *Env_Ex_abs_Loc )
{
ERROR( "Projection and interpolation for the envelope model are implemented only for interpolation_order = 2" );
}
