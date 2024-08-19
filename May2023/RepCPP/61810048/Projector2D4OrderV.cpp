#include "Projector2D4OrderV.h"

#include <cmath>
#include <iostream>

#include "ElectroMagn.h"
#include "Field2D.h"
#include "Particles.h"
#include "Tools.h"
#include "Patch.h"

using namespace std;


Projector2D4OrderV::Projector2D4OrderV( Params &params, Patch *patch ) : Projector2D( params, patch )
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


Projector2D4OrderV::~Projector2D4OrderV()
{
}

void Projector2D4OrderV::currentsAndDensity( double * __restrict__ Jx,
double * __restrict__ Jy,
double * __restrict__ Jz,
double * __restrict__ rho,
Particles &particles,
unsigned int istart,
unsigned int iend,
double * __restrict__ invgf,
int * __restrict__ iold,
double * __restrict__ deltaold,
unsigned int buffer_size,
int ipart_ref )
{

int ipo = iold[0];
int jpo = iold[1];

int ipom2 = ipo-3;
int jpom2 = jpo-3;

int vecSize = 8;
unsigned int bsize = 7*7*vecSize;

double bJx[bsize] __attribute__( ( aligned( 64 ) ) );

double DSx[56] __attribute__( ( aligned( 64 ) ) );
double DSy[56] __attribute__( ( aligned( 64 ) ) );

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
for( unsigned int j=0; j<bsize; j++ ) {
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
double delta3 = delta2*delta;
double delta4 = delta3*delta;
double S0 = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
double S1 = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
double S2 = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
double S3 = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
double S4 = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
double m1 = ( cell_shift == -1 );
double c0 = ( cell_shift ==  0 );
double p1 = ( cell_shift ==  1 );
DSx [          ipart] = m1 * S0                                        ;
DSx [  vecSize+ipart] = c0 * S0 + m1 * S1                              ;
DSx [2*vecSize+ipart] = p1 * S0 + c0 * S1 + m1* S2                     ;
DSx [3*vecSize+ipart] =           p1 * S1 + c0* S2 + m1 * S3           ;
DSx [4*vecSize+ipart] =                     p1* S2 + c0 * S3 + m1 * S4 ;
DSx [5*vecSize+ipart] =                              p1 * S3 + c0 * S4 ;
DSx [6*vecSize+ipart] =                                        p1 * S4 ;
pos = position_y[ivect+ipart+istart] * dy_inv_;
cell = round( pos );
cell_shift = cell-jpo-j_domain_begin_;
delta  = pos - ( double )cell;
delta2 = delta*delta;
delta3 = delta2*delta;
delta4 = delta3*delta;
S0 = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
S1 = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
S2 = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
S3 = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
S4 = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
m1 = ( cell_shift == -1 );
c0 = ( cell_shift ==  0 );
p1 = ( cell_shift ==  1 );
DSy [          ipart] = m1 * S0                                        ;
DSy [  vecSize+ipart] = c0 * S0 + m1 * S1                              ;
DSy [2*vecSize+ipart] = p1 * S0 + c0 * S1 + m1* S2                     ;
DSy [3*vecSize+ipart] =           p1 * S1 + c0* S2 + m1 * S3           ;
DSy [4*vecSize+ipart] =                     p1* S2 + c0 * S3 + m1 * S4 ;
DSy [5*vecSize+ipart] =                              p1 * S3 + c0 * S4 ;
DSy [6*vecSize+ipart] =                                        p1 * S4 ;

charge_weight[ipart] = inv_cell_volume * ( double )( charge[ivect+istart+ipart] )*
weight[ivect+istart+ipart];
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
UNROLL_S(7)
for( unsigned int i=0 ; i<7 ; i++ ) {
UNROLL_S(7)
for( unsigned int j=0 ; j<7 ; j++ ) {
int index( ( i*7 + j )*vecSize+ipart );
bJx [ index ] +=  charge_weight[ipart] * DSx[i*vecSize+ipart]*DSy[j*vecSize+ipart];
}
}
} 

}

int iloc0 = ipom2*nprimy+jpom2;
int iloc = iloc0;
for( unsigned int i=0 ; i<7 ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<7 ; j++ ) {
double tmpRho = 0.;
int ilocal = ( ( i )*7+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpRho +=  bJx[ilocal+ipart];
}
rho [iloc + j] +=  tmpRho;
}
iloc += nprimy;
}

} 


void Projector2D4OrderV::basic( double *rhoj, Particles &particles, unsigned int ipart, unsigned int type )
{

int iloc;
int ny( nprimy );
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
double delta, delta2, delta3, delta4;
double  Sx1[7], Sy1[7];

for( unsigned int i=0; i<7; i++ ) {
Sx1[i] = 0.;
Sy1[i] = 0.;
}

xpn = particles.position( 0, ipart ) * dx_inv_;
int ip        = round( xpn + 0.5 * ( type==1 ) );                       
delta  = xpn - ( double )ip;
delta2 = delta*delta;
delta3 = delta2*delta;
delta4 = delta3*delta;

Sx1[1] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
Sx1[2] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sx1[3] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
Sx1[4] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sx1[5] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

ypn = particles.position( 1, ipart ) * dy_inv_;
int jp = round( ypn + 0.5*( type==2 ) );
delta  = ypn - ( double )jp;
delta2 = delta*delta;
delta3 = delta2*delta;
delta4 = delta3*delta;

Sy1[1] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
Sy1[2] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sy1[3] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
Sy1[4] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sy1[5] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

ip -= i_domain_begin_ + 3;
jp -= j_domain_begin_ + 3;

for( unsigned int i=0 ; i<7 ; i++ ) {
iloc = ( i+ip )*ny+jp;
for( unsigned int j=0 ; j<7 ; j++ ) {
rhoj[iloc+j] += charge_weight * Sx1[i]*Sy1[j];
}
}

} 


void Projector2D4OrderV::ionizationCurrents( Field *Jx, Field *Jy, Field *Jz,
Particles &particles, int ipart, LocalFields Jion )
{
Field2D *Jx2D  = static_cast<Field2D *>( Jx );
Field2D *Jy2D  = static_cast<Field2D *>( Jy );
Field2D *Jz2D  = static_cast<Field2D *>( Jz );


int ip, id, jp, jd;
double xpn, xpmxip, xpmxip2, xpmxip3, xpmxip4, xpmxid, xpmxid2, xpmxid3, xpmxid4;
double ypn, ypmyjp, ypmyjp2, ypmyjp3, ypmyjp4, ypmyjd, ypmyjd2, ypmyjd3, ypmyjd4;
double Sxp[5], Sxd[5], Syp[5], Syd[5];

double weight = inv_cell_volume * particles.weight( ipart );
double Jx_ion = Jion.x * weight;
double Jy_ion = Jion.y * weight;
double Jz_ion = Jion.z * weight;

xpn    = particles.position( 0, ipart ) * dx_inv_; 
ypn    = particles.position( 1, ipart ) * dy_inv_; 

ip      = round( xpn );                  
xpmxip  = xpn - ( double )ip;            
xpmxip2 = xpmxip*xpmxip;                 
xpmxip3 = xpmxip2*xpmxip;                
xpmxip4 = xpmxip2*xpmxip2;               

id      = round( xpn+0.5 );              
xpmxid  = xpn - ( double )id + 0.5;      
xpmxid2 = xpmxid*xpmxid;                 
xpmxid3 = xpmxid2*xpmxid;                
xpmxid4 = xpmxid2*xpmxid2;               

jp      = round( ypn );                  
ypmyjp  = ypn - ( double )jp;            
ypmyjp2 = ypmyjp*ypmyjp;                 
ypmyjp3 = ypmyjp2*ypmyjp;                
ypmyjp4 = ypmyjp2*ypmyjp2;               

jd      = round( ypn+0.5 );              
ypmyjd  = ypn - ( double )jd + 0.5;      
ypmyjd2 = ypmyjd*ypmyjd;                 
ypmyjd3 = ypmyjd2*ypmyjd;                
ypmyjd4 = ypmyjd2*ypmyjd2;               

Sxp[0] = dble_1_ov_384   - dble_1_ov_48  * xpmxip  + dble_1_ov_16 * xpmxip2 - dble_1_ov_12 * xpmxip3 + dble_1_ov_24 * xpmxip4;
Sxp[1] = dble_19_ov_96   - dble_11_ov_24 * xpmxip  + dble_1_ov_4  * xpmxip2 + dble_1_ov_6  * xpmxip3 - dble_1_ov_6  * xpmxip4;
Sxp[2] = dble_115_ov_192 - dble_5_ov_8   * xpmxip2 + dble_1_ov_4  * xpmxip4;
Sxp[3] = dble_19_ov_96   + dble_11_ov_24 * xpmxip  + dble_1_ov_4  * xpmxip2 - dble_1_ov_6  * xpmxip3 - dble_1_ov_6  * xpmxip4;
Sxp[4] = dble_1_ov_384   + dble_1_ov_48  * xpmxip  + dble_1_ov_16 * xpmxip2 + dble_1_ov_12 * xpmxip3 + dble_1_ov_24 * xpmxip4;

Sxd[0] = dble_1_ov_384   - dble_1_ov_48  * xpmxid  + dble_1_ov_16 * xpmxid2 - dble_1_ov_12 * xpmxid3 + dble_1_ov_24 * xpmxid4;
Sxd[1] = dble_19_ov_96   - dble_11_ov_24 * xpmxid  + dble_1_ov_4  * xpmxid2 + dble_1_ov_6  * xpmxid3 - dble_1_ov_6  * xpmxid4;
Sxd[2] = dble_115_ov_192 - dble_5_ov_8   * xpmxid2 + dble_1_ov_4  * xpmxid4;
Sxd[3] = dble_19_ov_96   + dble_11_ov_24 * xpmxid  + dble_1_ov_4  * xpmxid2 - dble_1_ov_6  * xpmxid3 - dble_1_ov_6  * xpmxid4;
Sxd[4] = dble_1_ov_384   + dble_1_ov_48  * xpmxid  + dble_1_ov_16 * xpmxid2 + dble_1_ov_12 * xpmxid3 + dble_1_ov_24 * xpmxid4;

Syp[0] = dble_1_ov_384   - dble_1_ov_48  * ypmyjp  + dble_1_ov_16 * ypmyjp2 - dble_1_ov_12 * ypmyjp3 + dble_1_ov_24 * ypmyjp4;
Syp[1] = dble_19_ov_96   - dble_11_ov_24 * ypmyjp  + dble_1_ov_4  * ypmyjp2 + dble_1_ov_6  * ypmyjp3 - dble_1_ov_6  * ypmyjp4;
Syp[2] = dble_115_ov_192 - dble_5_ov_8   * ypmyjp2 + dble_1_ov_4  * ypmyjp4;
Syp[3] = dble_19_ov_96   + dble_11_ov_24 * ypmyjp  + dble_1_ov_4  * ypmyjp2 - dble_1_ov_6  * ypmyjp3 - dble_1_ov_6  * ypmyjp4;
Syp[4] = dble_1_ov_384   + dble_1_ov_48  * ypmyjp  + dble_1_ov_16 * ypmyjp2 + dble_1_ov_12 * ypmyjp3 + dble_1_ov_24 * ypmyjp4;

Syd[0] = dble_1_ov_384   - dble_1_ov_48  * ypmyjd  + dble_1_ov_16 * ypmyjd2 - dble_1_ov_12 * ypmyjd3 + dble_1_ov_24 * ypmyjd4;
Syd[1] = dble_19_ov_96   - dble_11_ov_24 * ypmyjd  + dble_1_ov_4  * ypmyjd2 + dble_1_ov_6  * ypmyjd3 - dble_1_ov_6  * ypmyjd4;
Syd[2] = dble_115_ov_192 - dble_5_ov_8   * ypmyjd2 + dble_1_ov_4  * ypmyjd4;
Syd[3] = dble_19_ov_96   + dble_11_ov_24 * ypmyjd  + dble_1_ov_4  * ypmyjd2 - dble_1_ov_6  * ypmyjd3 - dble_1_ov_6  * ypmyjd4;
Syd[4] = dble_1_ov_384   + dble_1_ov_48  * ypmyjd  + dble_1_ov_16 * ypmyjd2 + dble_1_ov_12 * ypmyjd3 + dble_1_ov_24 * ypmyjd4;

ip  -= i_domain_begin_;
id  -= i_domain_begin_;
jp  -= j_domain_begin_;
jd  -= j_domain_begin_;

for (unsigned int i=0 ; i<5 ; i++) {
int iploc=ip+i-2;
int idloc=id+i-2;
for (unsigned int j=0 ; j<5 ; j++) {
int jploc=jp+j-2;
int jdloc=jd+j-2;
(*Jx2D)(idloc,jploc) += Jx_ion * Sxd[i]*Syp[j];
(*Jy2D)(iploc,jdloc) += Jy_ion * Sxp[i]*Syd[j];
(*Jz2D)(iploc,jploc) += Jz_ion * Sxp[i]*Syp[j];
}
}

} 


void Projector2D4OrderV::currents( double * __restrict__ Jx,
double * __restrict__ Jy,
double * __restrict__ Jz,
Particles &particles,
unsigned int istart, unsigned int iend,
double * __restrict__ invgf,
int * __restrict__ iold,
double * __restrict__ deltaold,
unsigned int buffer_size,
int ipart_ref )
{



int ipo = iold[0];
int jpo = iold[1];

int ipom2 = ipo-3;
int jpom2 = jpo-3;

int vecSize = 8;
unsigned int bsize = 7*7*vecSize;

double bJx[bsize] __attribute__( ( aligned( 64 ) ) );
double bJy[bsize] __attribute__( ( aligned( 64 ) ) );
double bJz[bsize] __attribute__( ( aligned( 64 ) ) );

double Sx0_buff_vect[48] __attribute__( ( aligned( 64 ) ) );
double Sy0_buff_vect[48] __attribute__( ( aligned( 64 ) ) );

double Sx1_buff_vect[56] __attribute__( ( aligned( 64 ) ) );
double Sy1_buff_vect[56] __attribute__( ( aligned( 64 ) ) );

double DSx[56] __attribute__( ( aligned( 64 ) ) );
double DSy[56] __attribute__( ( aligned( 64 ) ) );

double charge_weight[8] __attribute__( ( aligned( 64 ) ) );
double crz_p[8]         __attribute__( ( aligned( 64 ) ) );

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);
double * __restrict__ momentum_z = particles.getPtrMomentum(2);
double * __restrict__ weight     = particles.getPtrWeight();
short  * __restrict__ charge     = particles.getPtrCharge();

int cell_nparts( ( int )iend-( int )istart );
int nbVec = ( iend-istart+( cell_nparts-1 )-( ( iend-istart-1 )&( cell_nparts-1 ) ) ) / vecSize;
if( nbVec*vecSize != cell_nparts ) {
nbVec++;
}






#pragma omp simd
for( unsigned int j=0; j<bsize; j++ ) {
bJx[j] = 0.;
bJy[j] = 0.;
bJz[j] = 0.;
}

for( int ivect=0 ; ivect < cell_nparts; ivect += vecSize ) {

int np_computed( min( cell_nparts-ivect, vecSize ) );

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

double delta = deltaold[ivect+ipart-ipart_ref+istart];
double delta2 = delta*delta;
double delta3 = delta2*delta;
double delta4 = delta3*delta;

Sx0_buff_vect[          ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
Sx0_buff_vect[  vecSize+ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sx0_buff_vect[2*vecSize+ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
Sx0_buff_vect[3*vecSize+ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sx0_buff_vect[4*vecSize+ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
Sx0_buff_vect[5*vecSize+ipart] = 0.;


delta = deltaold[ivect+ipart-ipart_ref+istart+buffer_size];
delta2 = delta*delta;
delta3 = delta2*delta;
delta4 = delta3*delta;

Sy0_buff_vect[          ipart] = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
Sy0_buff_vect[  vecSize+ipart] = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sy0_buff_vect[2*vecSize+ipart] = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
Sy0_buff_vect[3*vecSize+ipart] = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
Sy0_buff_vect[4*vecSize+ipart] = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
Sy0_buff_vect[5*vecSize+ipart] = 0.;



double pos = position_x[ivect+ipart+istart] * dx_inv_;
int cell = round( pos );
int cell_shift = cell-ipo-i_domain_begin_;
delta  = pos - ( double )cell;
delta2 = delta*delta;
delta3 = delta2*delta;
delta4 = delta3*delta;
double S0 = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 - dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
double S1 = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 + dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
double S2 = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
double S3 = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2 - dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
double S4 = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2 + dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
double m1 = ( cell_shift == -1 );
double c0 = ( cell_shift ==  0 );
double p1 = ( cell_shift ==  1 );

Sx1_buff_vect [          ipart] = m1 * S0                                        ;
Sx1_buff_vect [  vecSize+ipart] = c0 * S0 + m1 * S1                              ;
Sx1_buff_vect [2*vecSize+ipart] = p1 * S0 + c0 * S1 + m1* S2                     ;
Sx1_buff_vect [3*vecSize+ipart] =           p1 * S1 + c0* S2 + m1 * S3           ;
Sx1_buff_vect [4*vecSize+ipart] =                     p1* S2 + c0 * S3 + m1 * S4 ;
Sx1_buff_vect [5*vecSize+ipart] =                              p1 * S3 + c0 * S4 ;
Sx1_buff_vect [6*vecSize+ipart] =                                        p1 * S4 ;

DSx [          ipart] = Sx1_buff_vect [          ipart]                                  ;
DSx [  vecSize+ipart] = Sx1_buff_vect [  vecSize+ipart] - Sx0_buff_vect[0*vecSize+ipart] ;
DSx [2*vecSize+ipart] = Sx1_buff_vect [2*vecSize+ipart] - Sx0_buff_vect[1*vecSize+ipart] ;
DSx [3*vecSize+ipart] = Sx1_buff_vect [3*vecSize+ipart] - Sx0_buff_vect[2*vecSize+ipart] ;
DSx [4*vecSize+ipart] = Sx1_buff_vect [4*vecSize+ipart] - Sx0_buff_vect[3*vecSize+ipart] ;
DSx [5*vecSize+ipart] = Sx1_buff_vect [5*vecSize+ipart] - Sx0_buff_vect[4*vecSize+ipart] ;
DSx [6*vecSize+ipart] = Sx1_buff_vect [6*vecSize+ipart]                                  ;


pos = position_y[ivect+ipart+istart] * dy_inv_;
cell = round( pos );
cell_shift = cell-jpo-j_domain_begin_;
delta  = pos - ( double )cell;
delta2 = delta*delta;
delta3 = delta2*delta;
delta4 = delta3*delta;

S0 = dble_1_ov_384   - dble_1_ov_48  * delta  + dble_1_ov_16 * delta2
- dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;
S1 = dble_19_ov_96   - dble_11_ov_24 * delta  + dble_1_ov_4  * delta2
+ dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
S2 = dble_115_ov_192 - dble_5_ov_8   * delta2 + dble_1_ov_4  * delta4;
S3 = dble_19_ov_96   + dble_11_ov_24 * delta  + dble_1_ov_4  * delta2
- dble_1_ov_6  * delta3 - dble_1_ov_6  * delta4;
S4 = dble_1_ov_384   + dble_1_ov_48  * delta  + dble_1_ov_16 * delta2
+ dble_1_ov_12 * delta3 + dble_1_ov_24 * delta4;

m1 = ( cell_shift == -1 );
c0 = ( cell_shift ==  0 );
p1 = ( cell_shift ==  1 );

Sy1_buff_vect [          ipart] = m1 * S0                                        ;
Sy1_buff_vect [  vecSize+ipart] = c0 * S0 + m1 * S1                              ;
Sy1_buff_vect [2*vecSize+ipart] = p1 * S0 + c0 * S1 + m1* S2                     ;
Sy1_buff_vect [3*vecSize+ipart] =           p1 * S1 + c0* S2 + m1 * S3           ;
Sy1_buff_vect [4*vecSize+ipart] =                     p1* S2 + c0 * S3 + m1 * S4 ;
Sy1_buff_vect [5*vecSize+ipart] =                              p1 * S3 + c0 * S4 ;
Sy1_buff_vect [6*vecSize+ipart] =                                        p1 * S4 ;

DSy [          ipart] = Sy1_buff_vect [          ipart]                                  ;                                                                     ;
DSy [  vecSize+ipart] = Sy1_buff_vect [  vecSize+ipart] - Sy0_buff_vect[0*vecSize+ipart] ;
DSy [2*vecSize+ipart] = Sy1_buff_vect [2*vecSize+ipart] - Sy0_buff_vect[1*vecSize+ipart] ;
DSy [3*vecSize+ipart] = Sy1_buff_vect [3*vecSize+ipart] - Sy0_buff_vect[2*vecSize+ipart] ;
DSy [4*vecSize+ipart] = Sy1_buff_vect [4*vecSize+ipart] - Sy0_buff_vect[3*vecSize+ipart] ;
DSy [5*vecSize+ipart] = Sy1_buff_vect [5*vecSize+ipart] - Sy0_buff_vect[4*vecSize+ipart] ;
DSy [6*vecSize+ipart] = Sy1_buff_vect [6*vecSize+ipart]                                  ;


charge_weight[ipart] = inv_cell_volume * ( double )( charge[ivect+istart+ipart] )*weight[ivect+istart+ipart];

crz_p[ipart] = charge_weight[ipart]*one_third*momentum_z[ivect+istart+ipart]
* invgf[ivect+istart+ipart];

}


#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

double crx_p = charge_weight[ipart]*dx_ov_dt_;

double sum[7];
sum[0] = 0.;
UNROLL_S(6)
for( unsigned int k=1 ; k<7 ; k++ ) {
sum[k] = sum[k-1]-DSx[( k-1 )*vecSize+ipart];
}

double tmp( crx_p * ( 0.5*DSy[ipart] ) ); 
UNROLL_S(6)
for( unsigned int i=1 ; i<7 ; i++ ) {
bJx [( ( i )*7 )*vecSize+ipart] += sum[i]*tmp;
}

UNROLL_S(6)
for( unsigned int j=1 ; j<7 ; j++ ) { 
tmp = crx_p * ( Sy0_buff_vect[(j-1)*vecSize+ipart] + 0.5*DSy[j*vecSize+ipart] );
UNROLL_S(6)
for( unsigned int i=1 ; i<7 ; i++ ) {
bJx [ ( i*7+j )*vecSize+ipart ] += sum[i]*tmp;
}
}

} 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
double cry_p = charge_weight[ipart]*dy_ov_dt_;

double sum[7];
sum[0] = 0.;
UNROLL_S(6)
for( unsigned int k=1 ; k<7 ; k++ ) {
sum[k] = sum[k-1]-DSy[( k-1 )*vecSize+ipart];
}

double tmp( cry_p *0.5*DSx[ipart] );
UNROLL_S(6)
for( unsigned int j=1 ; j<7 ; j++ ) {
bJy [j*vecSize+ipart] += sum[j]*tmp;
}


UNROLL_S(6)
for( unsigned int i=1 ; i<7 ; i++ ) {
tmp = cry_p * ( Sx0_buff_vect[(i-1)*vecSize+ipart] + 0.5*DSx[i*vecSize+ipart] );

UNROLL_S(6)
for( unsigned int j=1 ; j<7 ; j++ ) {
bJy [ ( i*7+j )*vecSize+ipart ] += sum[j]*tmp;
}
}

} 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {

bJz [ipart] += crz_p[ipart] * Sx1_buff_vect[ipart] * Sy1_buff_vect[ipart];
double tmp( crz_p[ipart] * Sy1_buff_vect[ipart] );
UNROLL(6)
for( unsigned int i=1 ; i<7 ; i++ ) {
bJz [( ( i )*7 )*vecSize+ipart] += tmp * ( 0.5*Sx0_buff_vect[(i-1)*vecSize+ipart]
+ Sx1_buff_vect[i*vecSize+ipart] );
}

tmp = crz_p[ipart] * Sx1_buff_vect[ipart];
UNROLL(6)
for( unsigned int j=1; j<7 ; j++ ) {
bJz [j*vecSize+ipart] +=  tmp * ( 0.5*Sy0_buff_vect[(j-1)*vecSize+ipart]
+ Sy1_buff_vect[j*vecSize+ipart] );
}

UNROLL(6)
for( unsigned int i=1 ; i<7 ; i++ ) {
double tmp0( crz_p[ipart] * ( 0.5*Sx0_buff_vect[(i-1)*vecSize+ipart] + Sx1_buff_vect[i*vecSize+ipart] ) );
double tmp1( crz_p[ipart] * ( 0.5*Sx1_buff_vect[i*vecSize+ipart] + Sx0_buff_vect[(i-1)*vecSize+ipart] ) );
UNROLL(6)
for( unsigned int j=1; j<7 ; j++ ) {
bJz [( ( i )*7+j )*vecSize+ipart] += ( Sy0_buff_vect[(j-1)*vecSize+ipart]* tmp1
+ Sy1_buff_vect[j*vecSize+ipart]* tmp0 );
}
}

} 

} 


int iglobal0 = ipom2*nprimy+jpom2;
int iglobal  = iglobal0;
for( unsigned int i=1 ; i<7 ; i++ ) {
iglobal += nprimy;
#pragma omp simd
for( unsigned int j=0 ; j<7 ; j++ ) {
double tmpJx = 0.;
int ilocal = ( ( i )*7+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJx += bJx [ilocal+ipart];
}
Jx[iglobal+j] += tmpJx;
}
}

iglobal = iglobal0+ipom2;
for( unsigned int i=0 ; i<7 ; i++ ) {
#pragma omp simd
for( unsigned int j=1 ; j<7 ; j++ ) {
double tmpJy = 0.;
int ilocal = ( ( i )*7+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJy += bJy [ilocal+ipart];
}
Jy[iglobal+j] += tmpJy;
}
iglobal += ( nprimy+1 );
}




iglobal = iglobal0;
for( unsigned int i=0 ; i<7 ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<7 ; j++ ) {
double tmpJz( 0. );
int ilocal = ( i*7+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJz  +=  bJz [ilocal+ipart];
}
Jz[iglobal+j]  +=  tmpJz;
}
iglobal += nprimy;
} 

} 


void Projector2D4OrderV::currentsAndDensityWrapper( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int istart, int iend, int ithread, bool diag_flag, bool is_spectral, int ispec, int scell, int ipart_ref )
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
currentsAndDensity( b_Jx, b_Jy, b_Jz, b_rho, particles,  istart, iend,
invgf->data(), iold, &( *delta )[0], invgf->size(), ipart_ref );
}
}


void Projector2D4OrderV::susceptibility( ElectroMagn *EMfields, Particles &particles, double species_mass, SmileiMPI *smpi, int istart, int iend,  int ithread, int scell, int ipart_ref )
{
ERROR( "Projection and interpolation for the envelope model are implemented only for interpolation_order = 2" );
}
