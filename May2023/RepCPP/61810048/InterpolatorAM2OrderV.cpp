#include "InterpolatorAM2OrderV.h"

#include <cmath>
#include <iostream>
#include <math.h>
#include "ElectroMagn.h"
#include "ElectroMagnAM.h"
#include "cField2D.h"
#include "Particles.h"
#include "LaserEnvelope.h"
#include <complex>
#include "dcomplex.h"

using namespace std;


InterpolatorAM2OrderV::InterpolatorAM2OrderV( Params &params, Patch *patch ) : InterpolatorAM( params, patch )
{

nmodes_ = params.nmodes;
D_inv_[0] = 1.0/params.cell_length[0];
D_inv_[1] = 1.0/params.cell_length[1];
nscellr_ = params.n_space[1] + 1;
oversize_[0] = params.oversize[0];
oversize_[1] = params.oversize[1];
}


void InterpolatorAM2OrderV::fieldsWrapper( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, unsigned int scell, int ipart_ref )
{
if( istart[0] == iend[0] ) {
return;    
}

int nparts( ( smpi->dynamics_invgf[ithread] ).size() );

double * __restrict__ Epart[3];
double * __restrict__ Bpart[3];

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);
double * __restrict__ position_z = particles.getPtrPosition(2);


double * __restrict__ deltaO[2]; 
std::complex<double> * __restrict__ eitheta_old; 

int idx[2], idxO[2];
idx[0] = scell/nscellr_+oversize_[0]+i_domain_begin_;
idxO[0] = idx[0] - i_domain_begin_ -1 ;
idx[1] = ( scell%nscellr_ )+oversize_[1]+j_domain_begin_;
idxO[1] = idx[1] - j_domain_begin_ -1 ;

double coeff[2][2][3][32];
double dual[2][32]; 

int vecSize = 32;
double delta, delta2; 


int cell_nparts( ( int )iend[0]-( int )istart[0] );

std::vector<complex<double>> exp_m_theta_( vecSize), exp_mm_theta( vecSize) ;                                                          

for( int ivect=0 ; ivect < cell_nparts; ivect += vecSize ) {

int np_computed( min( cell_nparts-ivect, vecSize ) );
deltaO[0]   =  &(   smpi->dynamics_deltaold[ithread][0        + ivect + istart[0] - ipart_ref] );
deltaO[1]   =  &(   smpi->dynamics_deltaold[ithread][nparts   + ivect + istart[0] - ipart_ref] );
eitheta_old =  &( smpi->dynamics_eithetaold[ithread][           ivect + istart[0] - ipart_ref] );


#pragma omp simd private(delta2, delta)
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

int ipart2 = ipart+ivect+istart[0];
double r = sqrt( position_y[ipart2]*position_y[ipart2] + position_z[ipart2]*position_z[ipart2] );
exp_m_theta_[ipart] = ( position_y[ipart2] - Icpx * position_z[ipart2] ) / r ;
exp_mm_theta[ipart] = 1. ;
eitheta_old[ipart] =  2.*std::real(exp_m_theta_[ipart]) - exp_m_theta_[ipart] ;  

delta = position_x[ipart2]*D_inv_[0] - (double)idx[0];
delta2  = delta*delta;
coeff[0][0][0][ipart]    =  0.5 * ( delta2-delta+0.25 );
coeff[0][0][1][ipart]    = ( 0.75 - delta2 );
coeff[0][0][2][ipart]    =  0.5 * ( delta2+delta+0.25 );
deltaO[0][ipart] = delta;

dual [0][ipart] = ( delta >= 0. );
delta   = delta - dual[0][ipart] + 0.5 ;
delta2  = delta*delta;
coeff[0][1][0][ipart]    =  0.5 * ( delta2-delta+0.25 );
coeff[0][1][1][ipart]    = ( 0.75 - delta2 );
coeff[0][1][2][ipart]    =  0.5 * ( delta2+delta+0.25 );

delta = r * D_inv_[1] - (double)idx[1];
delta2  = delta*delta;
coeff[1][0][0][ipart]    =  0.5 * ( delta2-delta+0.25 );
coeff[1][0][1][ipart]    = ( 0.75 - delta2 );
coeff[1][0][2][ipart]    =  0.5 * ( delta2+delta+0.25 );
deltaO[1][ipart] = delta;

dual [1][ipart] = ( delta >= 0. );
delta   = delta - dual[1][ipart] + 0.5 ;
delta2  = delta*delta;
coeff[1][1][0][ipart]    =  0.5 * ( delta2-delta+0.25 );
coeff[1][1][1][ipart]    = ( 0.75 - delta2 );
coeff[1][1][2][ipart]    =  0.5 * ( delta2+delta+0.25 );
}

double interp_res;
double * __restrict__ coeffld = &( coeff[0][1][1][0] );
double * __restrict__ coefflp = &( coeff[0][0][1][0] );
double * __restrict__ coeffrp = &( coeff[1][0][1][0] );
double * __restrict__ coeffrd = &( coeff[1][1][1][0] );

for( unsigned int k=0; k<3; k++ ) {
Epart[k]= &( smpi->dynamics_Epart[ithread][k*nparts-ipart_ref+ivect+istart[0]] );
Bpart[k]= &( smpi->dynamics_Bpart[ithread][k*nparts-ipart_ref+ivect+istart[0]] );
#pragma omp simd 
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
Epart[k][ipart] = 0.;
Bpart[k][ipart] = 0.;
}
}

std::complex<double> field_buffer[4][4];

for( unsigned int imode = 0; imode < nmodes_ ; imode++ ) {
cField2D * __restrict__ El = ( static_cast<ElectroMagnAM *>( EMfields ) )->El_[imode];
cField2D * __restrict__ Er = ( static_cast<ElectroMagnAM *>( EMfields ) )->Er_[imode];
cField2D * __restrict__ Et = ( static_cast<ElectroMagnAM *>( EMfields ) )->Et_[imode];
cField2D * __restrict__ Bl = ( static_cast<ElectroMagnAM *>( EMfields ) )->Bl_m[imode];
cField2D * __restrict__ Br = ( static_cast<ElectroMagnAM *>( EMfields ) )->Br_m[imode];
cField2D * __restrict__ Bt = ( static_cast<ElectroMagnAM *>( EMfields ) )->Bt_m[imode];



for( int iloc=-1 ; iloc<3 ; iloc++ ) {
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
field_buffer[iloc+1][jloc+1] = ( *El )( idxO[0]+1+iloc, idxO[1]+1+jloc );
}
}

#pragma omp simd private(interp_res)
for( int ipart=0 ; ipart<np_computed; ipart++ ) {


interp_res = 0.;
UNROLL_S(3) 
for( int iloc=-1 ; iloc<2 ; iloc++ ) {
UNROLL_S(3) 
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
interp_res += std::real( coeffld[ipart + iloc*32] * coeffrp[ipart + jloc*32] *
( ( 1.-dual[0][ipart] )*field_buffer[1+iloc][1+jloc] 
+ dual[0][ipart]  *field_buffer[2+iloc][1+jloc] ) 
* exp_mm_theta[ipart]) ; 
}
}
Epart[0][ipart] += interp_res;
}

for( int iloc=-1 ; iloc<2 ; iloc++ ) {
for( int jloc=-1 ; jloc<3 ; jloc++ ) {
field_buffer[iloc+1][jloc+1] = ( *Er )( idxO[0]+1+iloc, idxO[1]+1+jloc );
}
}

#pragma omp simd private(interp_res)
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
interp_res = 0.;
UNROLL_S(3) 
for( int iloc=-1 ; iloc<2 ; iloc++ ) {
UNROLL_S(3) 
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
interp_res +=  std::real (*( coefflp+ipart+iloc*32 ) * *( coeffrd+ipart+jloc*32 ) *
( ( 1-dual[1][ipart] )*field_buffer[1+iloc][1+jloc] 
+ dual[1][ipart]  *field_buffer[1+iloc][2+jloc] )
* exp_mm_theta[ipart]);
}
}
Epart[1][ipart] += interp_res;
}

for( int iloc=-1 ; iloc<2 ; iloc++ ) {
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
field_buffer[iloc+1][jloc+1] = ( *Et )( idxO[0]+1+iloc, idxO[1]+1+jloc );
}
}

#pragma omp simd private(interp_res)
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
interp_res = 0.;
UNROLL_S(3) 
for( int iloc=-1 ; iloc<2 ; iloc++ ) {
UNROLL_S(3) 
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
interp_res +=  std::real(  *( coefflp+ipart+iloc*32 ) * *( coeffrp+ipart+jloc*32 ) * 
field_buffer[1+iloc][1+jloc]  
* exp_mm_theta[ipart]);
}
}
Epart[2][ipart] += interp_res;
}

for( int iloc=-1 ; iloc<2 ; iloc++ ) {
for( int jloc=-1 ; jloc<3 ; jloc++ ) {
field_buffer[iloc+1][jloc+1] = ( *Bl )( idxO[0]+1+iloc, idxO[1]+1+jloc );
}
}

#pragma omp simd private(interp_res)
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
interp_res = 0.;
UNROLL_S(3) 
for( int iloc=-1 ; iloc<2 ; iloc++ ) {
UNROLL_S(3) 
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
interp_res +=  std::real(  *( coefflp+ipart+iloc*32 ) * *( coeffrd+ipart+jloc*32 ) *
( ( ( 1-dual[1][ipart] ) * field_buffer[1+iloc][1+jloc] 
+ dual[1][ipart]   * field_buffer[1+iloc][2+jloc] )
)  * exp_mm_theta[ipart] );
}
}
Bpart[0][ipart] += interp_res;
}

for( int iloc=-1 ; iloc<3 ; iloc++ ) {
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
field_buffer[iloc+1][jloc+1] = ( *Br )( idxO[0]+1+iloc, idxO[1]+1+jloc );
}
}
#pragma omp simd private(interp_res)
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
interp_res = 0.;
UNROLL_S(3) 
for( int iloc=-1 ; iloc<2 ; iloc++ ) {
UNROLL_S(3) 
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
interp_res +=  std::real(  *( coeffld+ipart+iloc*32 ) * *( coeffrp+ipart+jloc*32 ) *
( ( ( 1-dual[0][ipart] )* field_buffer[ 1+iloc][1+jloc ] 
+ dual[0][ipart]  * field_buffer[ 2+iloc][1+jloc ] )
)  * exp_mm_theta[ipart]);
}
}
Bpart[1][ipart] += interp_res;
}

for( int iloc=-1 ; iloc<3 ; iloc++ ) {
for( int jloc=-1 ; jloc<3 ; jloc++ ) {
field_buffer[iloc+1][jloc+1] = ( *Bt )( idxO[0]+1+iloc, idxO[1]+1+jloc );
}
}
#pragma omp simd private(interp_res)
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
interp_res = 0.;
UNROLL_S(3) 
for( int iloc=-1 ; iloc<2 ; iloc++ ) {
UNROLL_S(3) 
for( int jloc=-1 ; jloc<2 ; jloc++ ) {
interp_res +=  std::real(  *( coeffld+ipart+iloc*32 ) * *( coeffrd+ipart+jloc*32 ) *
( ( 1-dual[1][ipart] ) * ( ( 1-dual[0][ipart] )*field_buffer[1+iloc][1+jloc] 
+ dual[0][ipart]*field_buffer[2+iloc][1+jloc] )
+    dual[1][ipart]  * ( ( 1-dual[0][ipart] )*field_buffer[1+iloc][2+jloc] 
+ dual[0][ipart]*field_buffer[2+iloc][2+jloc] )
) * exp_mm_theta[ipart] );
}
}
Bpart[2][ipart] += interp_res;
exp_mm_theta[ipart] *= exp_m_theta_[ipart]; 
}
} 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
double delta2 = std::real( exp_m_theta_[ipart] ) * Epart[1][ipart] + std::imag( exp_m_theta_[ipart] ) * Epart[2][ipart];
Epart[2][ipart] = -std::imag( exp_m_theta_[ipart] ) * Epart[1][ipart] + std::real( exp_m_theta_[ipart] ) * Epart[2][ipart];
Epart[1][ipart] = delta2 ;
delta2 = std::real( exp_m_theta_[ipart] ) * Bpart[1][ipart] + std::imag( exp_m_theta_[ipart] ) * Bpart[2][ipart];
Bpart[2][ipart] = -std::imag( exp_m_theta_[ipart] ) * Bpart[1][ipart] + std::real( exp_m_theta_[ipart] ) * Bpart[2][ipart];
Bpart[1][ipart] = delta2 ;
}


} 
}





