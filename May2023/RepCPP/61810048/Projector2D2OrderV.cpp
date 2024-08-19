#include "Projector2D2OrderV.h"

#include <cmath>
#include <iostream>

#include "ElectroMagn.h"
#include "Field2D.h"
#include "Particles.h"
#include "Tools.h"
#include "Patch.h"
#include "Pragma.h"

using namespace std;


Projector2D2OrderV::Projector2D2OrderV( Params &params, Patch *patch ) : Projector2D( params, patch )
{
dx_inv_   = 1.0/params.cell_length[0];
dx_ov_dt_  = params.cell_length[0] / params.timestep;
dy_inv_   = 1.0/params.cell_length[1];
dy_ov_dt_  = params.cell_length[1] / params.timestep;

i_domain_begin_ = patch->getCellStartingGlobalIndex( 0 );
j_domain_begin_ = patch->getCellStartingGlobalIndex( 1 );

nscelly_ = params.n_space[1] + 1;
oversize[0] = params.oversize[0];
oversize[1] = params.oversize[1];
nprimy = nscelly_ + 2*oversize[1];
dq_inv[0] = dx_inv_;
dq_inv[1] = dy_inv_;


DEBUG( "cell_length "<< params.cell_length[0] );

}


Projector2D2OrderV::~Projector2D2OrderV()
{
}

void Projector2D2OrderV::currentsAndDensity( double * __restrict__ Jx,
double * __restrict__ Jy,
double * __restrict__ Jz,
double * __restrict__ rho,
Particles & particles,
unsigned int istart,
unsigned int iend,
double * __restrict__ invgf,
int    * __restrict__ iold,
double * __restrict__ deltaold,
unsigned int buffer_size,
int ipart_ref )
{


int ipo = iold[0];
int jpo = iold[1];
int ipom2 = ipo-2;
int jpom2 = jpo-2;

int vecSize = 8;
unsigned int bsize = 5*5*vecSize;

double bJx[bsize] __attribute__( ( aligned( 64 ) ) );

double Sx1_buff_vect[40] __attribute__( ( aligned( 64 ) ) );
double Sy1_buff_vect[40] __attribute__( ( aligned( 64 ) ) );
double charge_weight[8] __attribute__( ( aligned( 64 ) ) );

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);
double * __restrict__ weight     = particles.getPtrWeight();
short  * __restrict__ charge     = particles.getPtrCharge();

int cell_nparts( ( int )iend-( int )istart );
int nbVec = ( iend-istart+( cell_nparts-1 )-( ( iend-istart-1 )&( cell_nparts-1 ) ) ) / vecSize;
if( nbVec*vecSize != cell_nparts ) {
nbVec++;
}

currents( Jx, Jy, Jz, particles, istart, iend, invgf, iold, deltaold, buffer_size, ipart_ref );

cell_nparts = ( int )iend-( int )istart;
#pragma omp simd
for( unsigned int j=0; j<200; j++ ) {
bJx[j] = 0.;
}

for( int ivect=0 ; ivect < cell_nparts; ivect += vecSize ) {

int np_computed( min( cell_nparts-ivect, vecSize ) );

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

double pos = position_x[ivect+ipart+istart] * dx_inv_;
int cell = round( pos );
int cell_shift = cell-ipo-i_domain_begin_;
double delta  = pos - ( double )cell;
double delta2 = delta*delta;
double deltam =  0.5 * ( delta2-delta+0.25 );
double deltap =  0.5 * ( delta2+delta+0.25 );
delta2 = 0.75 - delta2;
double m1 = ( cell_shift == -1 );
double c0 = ( cell_shift ==  0 );
double p1 = ( cell_shift ==  1 );
Sx1_buff_vect[          ipart] = m1 * deltam;
Sx1_buff_vect[  vecSize+ipart] = c0 * deltam + m1*delta2;
Sx1_buff_vect[2*vecSize+ipart] = p1 * deltam + c0*delta2 + m1*deltap;
Sx1_buff_vect[3*vecSize+ipart] =               p1*delta2 + c0*deltap;
Sx1_buff_vect[4*vecSize+ipart] =                           p1*deltap;
pos = position_y[ivect+ipart+istart] * dy_inv_;
cell = round( pos );
cell_shift = cell-jpo-j_domain_begin_;
delta  = pos - ( double )cell;
delta2 = delta*delta;
deltam =  0.5 * ( delta2-delta+0.25 );
deltap =  0.5 * ( delta2+delta+0.25 );
delta2 = 0.75 - delta2;
m1 = ( cell_shift == -1 );
c0 = ( cell_shift ==  0 );
p1 = ( cell_shift ==  1 );
Sy1_buff_vect[          ipart] = m1 * deltam;
Sy1_buff_vect[  vecSize+ipart] = c0 * deltam + m1*delta2;
Sy1_buff_vect[2*vecSize+ipart] = p1 * deltam + c0*delta2 + m1*deltap;
Sy1_buff_vect[3*vecSize+ipart] =               p1*delta2 + c0*deltap;
Sy1_buff_vect[4*vecSize+ipart] =                           p1*deltap;

charge_weight[ipart] = inv_cell_volume * ( double )( charge[ivect+istart+ipart] )
* weight[ivect+istart+ipart];
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
UNROLL_S(5)
for( unsigned int i=0 ; i<5 ; i++ ) {
UNROLL_S(5)
for( unsigned int j=0 ; j<5 ; j++ ) {
int index( ( i*5 + j )*vecSize+ipart );
bJx [ index ] +=  charge_weight[ipart] * Sx1_buff_vect[i*vecSize+ipart]*Sy1_buff_vect[j*vecSize+ipart];
}
}
} 

}

int iloc0 = ipom2*nprimy+jpom2;
int iloc = iloc0;
for( unsigned int i=0 ; i<5 ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<5 ; j++ ) {
double tmpRho = 0.;
int ilocal = ( ( i )*5+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpRho +=  bJx[ilocal+ipart];
}
rho [iloc + j] +=  tmpRho;
}
iloc += nprimy;
}

} 


void Projector2D2OrderV::basic( double *rhoj, Particles &particles, unsigned int ipart, unsigned int type )
{


int iloc, ny( nprimy );
double charge_weight = inv_cell_volume * ( double )( particles.charge( ipart ) )*particles.weight( ipart );

if( type > 0 ) {
charge_weight *= 1./sqrt( 1.0 + particles.momentum( 0, ipart )*particles.momentum( 0, ipart )
+ particles.momentum( 1, ipart )*particles.momentum( 1, ipart )
+ particles.momentum( 2, ipart )*particles.momentum( 2, ipart ) );

if( type == 1 ) {
charge_weight *= particles.momentum( 0, ipart );
} else if( type == 2 ) {
charge_weight *= particles.momentum( 1, ipart );
ny ++;
} else {
charge_weight *= particles.momentum( 2, ipart );
}
}

double xpn, ypn;
double delta, delta2;
double Sx1[5], Sy1[5]; 

for( unsigned int i=0; i<5; i++ ) {
Sx1[i] = 0.;
Sy1[i] = 0.;
}


xpn = particles.position( 0, ipart ) * dx_inv_;
int ip        = round( xpn + 0.5 * ( type==1 ) );                       
delta  = xpn - ( double )ip;
delta2 = delta*delta;
Sx1[1] = 0.5 * ( delta2-delta+0.25 );
Sx1[2] = 0.75-delta2;
Sx1[3] = 0.5 * ( delta2+delta+0.25 );

ypn = particles.position( 1, ipart ) * dy_inv_;
int jp = round( ypn + 0.5*( type==2 ) );
delta  = ypn - ( double )jp;
delta2 = delta*delta;
Sy1[1] = 0.5 * ( delta2-delta+0.25 );
Sy1[2] = 0.75-delta2;
Sy1[3] = 0.5 * ( delta2+delta+0.25 );

ip -= i_domain_begin_ + 2;
jp -= j_domain_begin_ + 2;

for( unsigned int i=0 ; i<5 ; i++ ) {
iloc = ( i+ip )*ny+jp;
for( unsigned int j=0 ; j<5 ; j++ ) {
rhoj[iloc+j] += charge_weight * Sx1[i]*Sy1[j];
}
}
} 


void Projector2D2OrderV::ionizationCurrents( Field *Jx, Field *Jy, Field *Jz, Particles &particles, int ipart, LocalFields Jion )
{
Field2D *Jx2D  = static_cast<Field2D *>( Jx );
Field2D *Jy2D  = static_cast<Field2D *>( Jy );
Field2D *Jz2D  = static_cast<Field2D *>( Jz );


int ip, id, jp, jd;
double xpn, xpmxip, xpmxip2, xpmxid, xpmxid2;
double ypn, ypmyjp, ypmyjp2, ypmyjd, ypmyjd2;
double Sxp[3], Sxd[3], Syp[3], Syd[3];

double weight = inv_cell_volume * particles.weight( ipart );
double Jx_ion = Jion.x * weight;
double Jy_ion = Jion.y * weight;
double Jz_ion = Jion.z * weight;

xpn    = particles.position( 0, ipart ) * dx_inv_; 
ypn    = particles.position( 1, ipart ) * dy_inv_; 

ip      = round( xpn );                  
xpmxip  = xpn - ( double )ip;            
xpmxip2 = xpmxip*xpmxip;                 

id      = round( xpn+0.5 );              
xpmxid  = xpn - ( double )id + 0.5;      
xpmxid2 = xpmxid*xpmxid;                 

jp      = round( ypn );                  
ypmyjp  = ypn - ( double )jp;            
ypmyjp2 = ypmyjp*ypmyjp;                 

jd      = round( ypn+0.5 );              
ypmyjd  = ypn - ( double )jd + 0.5;      
ypmyjd2 = ypmyjd*ypmyjd;                 

Sxp[0] = 0.5 * ( xpmxip2-xpmxip+0.25 );
Sxp[1] = ( 0.75-xpmxip2 );
Sxp[2] = 0.5 * ( xpmxip2+xpmxip+0.25 );

Sxd[0] = 0.5 * ( xpmxid2-xpmxid+0.25 );
Sxd[1] = ( 0.75-xpmxid2 );
Sxd[2] = 0.5 * ( xpmxid2+xpmxid+0.25 );

Syp[0] = 0.5 * ( ypmyjp2-ypmyjp+0.25 );
Syp[1] = ( 0.75-ypmyjp2 );
Syp[2] = 0.5 * ( ypmyjp2+ypmyjp+0.25 );

Syd[0] = 0.5 * ( ypmyjd2-ypmyjd+0.25 );
Syd[1] = ( 0.75-ypmyjd2 );
Syd[2] = 0.5 * ( ypmyjd2+ypmyjd+0.25 );

ip  -= i_domain_begin_;
id  -= i_domain_begin_;
jp  -= j_domain_begin_;
jd  -= j_domain_begin_;

for( unsigned int i=0 ; i<3 ; i++ ) {
int iploc=ip+i-1;
int idloc=id+i-1;
for( unsigned int j=0 ; j<3 ; j++ ) {
int jploc=jp+j-1;
int jdloc=jd+j-1;
( *Jx2D )( idloc, jploc ) += Jx_ion * Sxd[i]*Syp[j];
( *Jy2D )( iploc, jdloc ) += Jy_ion * Sxp[i]*Syd[j];
( *Jz2D )( iploc, jploc ) += Jz_ion * Sxp[i]*Syp[j];
}
}


} 


void Projector2D2OrderV::currents( double * __restrict__ Jx,
double * __restrict__ Jy,
double * __restrict__ Jz,
Particles &particles,
unsigned int istart,
unsigned int iend,
double * __restrict__ invgf,
int    * __restrict__ iold,
double * __restrict__ deltaold,
unsigned int buffer_size,
int ipart_ref )
{

int ipo = iold[0];
int jpo = iold[1];
int ipom2 = ipo-2;
int jpom2 = jpo-2;

int vecSize = 8;
int bsize = 5*5*vecSize;

double bJx[bsize] __attribute__( ( aligned( 64 ) ) );
double bJy[bsize] __attribute__( ( aligned( 64 ) ) );
double bJz[bsize] __attribute__( ( aligned( 64 ) ) );

double Sx0_buff_vect[40] __attribute__( ( aligned( 64 ) ) );
double Sy0_buff_vect[40] __attribute__( ( aligned( 64 ) ) );
double Sx1_buff_vect[40] __attribute__( ( aligned( 64 ) ) );
double Sy1_buff_vect[40] __attribute__( ( aligned( 64 ) ) );
double DSx[40] __attribute__( ( aligned( 64 ) ) );
double DSy[40] __attribute__( ( aligned( 64 ) ) );
double charge_weight[8] __attribute__( ( aligned( 64 ) ) );
double crz_p[8] __attribute__( ( aligned( 64 ) ) );

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);
double * __restrict__ momentum_z = particles.getPtrMomentum(2);
double * __restrict__ weight     = particles.getPtrWeight();
short  * __restrict__ charge     = particles.getPtrCharge();

#pragma omp simd
for( int j=0; j<bsize; j++ ) {
bJx[j] = 0.;
bJy[j] = 0.;
bJz[j] = 0.;
}

int cell_nparts( ( int )iend-( int )istart );

for( int ivect=0 ; ivect < cell_nparts; ivect += vecSize ) {

int np_computed = min( cell_nparts-ivect, vecSize );

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

double pos = position_x[ivect+ipart+istart] * dx_inv_;
int cell = round( pos );
int cell_shift = cell-ipo-i_domain_begin_;
double delta  = pos - ( double )cell;
double delta2 = delta*delta;
double deltam =  0.5 * ( delta2-delta+0.25 );
double deltap =  0.5 * ( delta2+delta+0.25 );
delta2 = 0.75 - delta2;
double m1 = ( cell_shift == -1 );
double c0 = ( cell_shift ==  0 );
double p1 = ( cell_shift ==  1 );
Sx1_buff_vect[          ipart] = m1 * deltam                                                                                  ;
Sx1_buff_vect[  vecSize+ipart] = c0 * deltam + m1*delta2                                               ;
Sx1_buff_vect[2*vecSize+ipart] = p1 * deltam + c0*delta2 + m1*deltap;
Sx1_buff_vect[3*vecSize+ipart] =               p1*delta2 + c0*deltap;
Sx1_buff_vect[4*vecSize+ipart] =                           p1*deltap;
delta = deltaold[ivect+ipart-ipart_ref+istart];
delta2 = delta*delta;
Sx0_buff_vect[          ipart] = 0;
Sx0_buff_vect[  vecSize+ipart] = 0.5 * ( delta2-delta+0.25 );
Sx0_buff_vect[2*vecSize+ipart] = 0.75-delta2;
Sx0_buff_vect[3*vecSize+ipart] = 0.5 * ( delta2+delta+0.25 );
Sx0_buff_vect[4*vecSize+ipart] = 0;
UNROLL(5)
for( unsigned int i = 0; i < 5 ; i++ ) {
DSx[i*vecSize+ipart] = Sx1_buff_vect[ i*vecSize+ipart] - Sx0_buff_vect[ i*vecSize+ipart];
}
pos = position_y[ivect+ipart+istart] * dy_inv_;
cell = round( pos );
cell_shift = cell-jpo-j_domain_begin_;
delta  = pos - ( double )cell;
delta2 = delta*delta;
deltam =  0.5 * ( delta2-delta+0.25 );
deltap =  0.5 * ( delta2+delta+0.25 );
delta2 = 0.75 - delta2;
m1 = ( cell_shift == -1 );
c0 = ( cell_shift ==  0 );
p1 = ( cell_shift ==  1 );
Sy1_buff_vect[          ipart] = m1 * deltam                                                                                  ;
Sy1_buff_vect[  vecSize+ipart] = c0 * deltam + m1*delta2                                               ;
Sy1_buff_vect[2*vecSize+ipart] = p1 * deltam + c0*delta2 + m1*deltap;
Sy1_buff_vect[3*vecSize+ipart] =               p1*delta2 + c0*deltap;
Sy1_buff_vect[4*vecSize+ipart] =                           p1*deltap;
delta = deltaold[ivect+ipart-ipart_ref+istart+buffer_size];
delta2 = delta*delta;
Sy0_buff_vect[          ipart] = 0;
Sy0_buff_vect[  vecSize+ipart] = 0.5 * ( delta2-delta+0.25 );
Sy0_buff_vect[2*vecSize+ipart] = 0.75-delta2;
Sy0_buff_vect[3*vecSize+ipart] = 0.5 * ( delta2+delta+0.25 );
Sy0_buff_vect[4*vecSize+ipart] = 0;

UNROLL(5)
for( unsigned int i = 0; i < 5 ; i++ ) {
DSy[i*vecSize+ipart] = Sy1_buff_vect[ i*vecSize+ipart] - Sy0_buff_vect[ i*vecSize+ipart];
}
charge_weight[ipart] = inv_cell_volume * ( double )( charge[ivect+istart+ipart] )
* weight[ivect+istart+ipart];

crz_p[ipart] = charge_weight[ipart] * one_third * momentum_z[ivect+istart+ipart]
* invgf[ivect+istart+ipart];
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
double crx_p = charge_weight[ipart]*dx_ov_dt_;

double sum[5];
sum[0] = 0.;
UNROLL_S(4)
for( unsigned int k=1 ; k<5 ; k++ ) {
sum[k] = sum[k-1]-DSx[( k-1 )*vecSize+ipart];
}

double tmp( crx_p * ( 0.5*DSy[ipart] ) );
UNROLL_S(4)
for( unsigned int i=1 ; i<5 ; i++ ) {
bJx [( i*5 )*vecSize+ipart] += sum[i] * tmp;
}

UNROLL_S(4)
for( unsigned int j=1; j<5 ; j++ ) {
tmp =  crx_p * ( Sy0_buff_vect[j*vecSize+ipart] + 0.5*DSy[j*vecSize+ipart] );
UNROLL_S(4)
for( unsigned int i=1 ; i<5 ; i++ ) {
bJx [( i*5+j )*vecSize+ipart] += sum[i] * tmp;
}
}
} 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
double cry_p = charge_weight[ipart]*dy_ov_dt_;

double sum[5];
sum[0] = 0.;
UNROLL_S(4)
for( unsigned int k=1 ; k<5 ; k++ ) {
sum[k] = sum[k-1]-DSy[( k-1 )*vecSize+ipart];
}

double tmp( cry_p * ( 0.5*DSx[ipart] ) );
UNROLL_S(4)
for( unsigned int j=1 ; j<5 ; j++ ) {
bJy [j*vecSize+ipart] += sum[j] * tmp;
}

UNROLL_S(4)
for( unsigned int i=1; i<5 ; i++ ) {
tmp = cry_p * ( Sx0_buff_vect[i*vecSize+ipart] + 0.5*DSx[i*vecSize+ipart] );
UNROLL_S(4)
for( unsigned int j=1 ; j<5 ; j++ ) {
bJy [( i*5+j )*vecSize+ipart] += sum[j] * tmp;
}
}
}
#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
bJz [ipart] += crz_p[ipart] * Sx1_buff_vect[ipart] * Sy1_buff_vect[ipart];
double tmp( crz_p[ipart] * Sy1_buff_vect[ipart] );
UNROLL(4)
for( unsigned int i=1 ; i<5 ; i++ ) {
bJz [( ( i )*5 )*vecSize+ipart] += tmp * ( 0.5*Sx0_buff_vect[i*vecSize+ipart] + Sx1_buff_vect[i*vecSize+ipart] );
}

tmp = crz_p[ipart] * Sx1_buff_vect[ipart];
UNROLL(4)
for( unsigned int j=1; j<5 ; j++ ) {
bJz [j*vecSize+ipart] +=  tmp * ( 0.5*Sy0_buff_vect[j*vecSize+ipart]
+ Sy1_buff_vect[j*vecSize+ipart] );
}

UNROLL(4)
for( unsigned int i=1 ; i<5 ; i++ ) {
double tmp0( crz_p[ipart] * ( 0.5*Sx0_buff_vect[i*vecSize+ipart] + Sx1_buff_vect[i*vecSize+ipart] ) );
double tmp1( crz_p[ipart] * ( 0.5*Sx1_buff_vect[i*vecSize+ipart] + Sx0_buff_vect[i*vecSize+ipart] ) );
UNROLL(4)
for( unsigned int j=1; j<5 ; j++ ) {
bJz [( ( i )*5+j )*vecSize+ipart] += ( Sy0_buff_vect[j*vecSize+ipart]* tmp1 + Sy1_buff_vect[j*vecSize+ipart]* tmp0 );
}
}

} 

} 

int iloc0 = ipom2*nprimy+jpom2;
int iloc = iloc0;
for( unsigned int i=1 ; i<5 ; i++ ) {
iloc += nprimy;
#pragma omp simd
for( unsigned int j=0 ; j<5 ; j++ ) {
double tmpJx( 0. );
int ilocal = ( i*5+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJx += bJx [ilocal+ipart];
}
Jx[iloc+j] += tmpJx;
}
}

iloc  = iloc0 + ipom2;
for( unsigned int i=0 ; i<5 ; i++ ) {
#pragma omp simd
for( unsigned int j=1 ; j<5 ; j++ ) {
double tmpJy( 0. );
int ilocal = ( i*5+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJy += bJy [ilocal+ipart];
}
Jy[iloc+j] += tmpJy;
}
iloc += ( nprimy+1 );
}

iloc = iloc0;
for( unsigned int i=0 ; i<5 ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<5 ; j++ ) {
double tmpJz( 0. );
int ilocal = ( i*5+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJz  +=  bJz [ilocal+ipart];
}
Jz[iloc+j]  +=  tmpJz;
}
iloc += nprimy;
} 

} 


void Projector2D2OrderV::currentsAndDensityWrapper( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int istart, int iend, int ithread,  bool diag_flag, bool is_spectral, int ispec, int scell, int ipart_ref )
{
if( istart == iend ) {
return;    
}

std::vector<double> *delta = &( smpi->dynamics_deltaold[ithread] );
std::vector<double> *invgf = &( smpi->dynamics_invgf[ithread] );
int iold[2];
iold[0] = scell/nscelly_+oversize[0];
iold[1] = ( scell%nscelly_ )+oversize[1];

if( !diag_flag ) {
if( !is_spectral ) {
double *b_Jx =  &( *EMfields->Jx_ )( 0 );
double *b_Jy =  &( *EMfields->Jy_ )( 0 );
double *b_Jz =  &( *EMfields->Jz_ )( 0 );
currents( b_Jx, b_Jy, b_Jz, particles,  istart, iend, invgf->data(), iold, &( *delta )[0], invgf->size(), ipart_ref );
} else {
ERROR( "TO DO with rho" );
}

} else {
double *b_Jx  = EMfields->Jx_s [ispec] ? &( *EMfields->Jx_s [ispec] )( 0 ) : &( *EMfields->Jx_ )( 0 ) ;
double *b_Jy  = EMfields->Jy_s [ispec] ? &( *EMfields->Jy_s [ispec] )( 0 ) : &( *EMfields->Jy_ )( 0 ) ;
double *b_Jz  = EMfields->Jz_s [ispec] ? &( *EMfields->Jz_s [ispec] )( 0 ) : &( *EMfields->Jz_ )( 0 ) ;
double *b_rho = EMfields->rho_s[ispec] ? &( *EMfields->rho_s[ispec] )( 0 ) : &( *EMfields->rho_ )( 0 ) ;
currentsAndDensity( b_Jx, b_Jy, b_Jz, b_rho, particles, istart, iend, invgf->data(), iold, &( *delta )[0], invgf->size(), ipart_ref );
}
}

void Projector2D2OrderV::susceptibility( ElectroMagn *EMfields, Particles &particles, double species_mass, SmileiMPI *smpi, int istart, int iend,  int ithread, int icell, int ipart_ref )
{
ERROR( "Vectorized projection of the susceptibility for the envelope model is not implemented for 2D geometry" );
}
