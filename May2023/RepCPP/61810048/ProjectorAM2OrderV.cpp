#include "ProjectorAM2OrderV.h"

#include <cmath>
#include <iostream>
#include <complex>
#include "dcomplex.h"
#include "ElectroMagnAM.h"
#include "cField2D.h"
#include "Particles.h"
#include "Tools.h"
#include "Patch.h"
#include "PatchAM.h"

using namespace std;


ProjectorAM2OrderV::ProjectorAM2OrderV( Params &params, Patch *patch ) : ProjectorAM( params, patch )
{
dt = params.timestep;
dr = params.cell_length[1];
dl_inv_   = 1.0/params.cell_length[0];
dl_ov_dt_  = params.cell_length[0] / params.timestep;
one_ov_dt  = 1.0 / params.timestep;
dr_inv_   = 1.0/dr;
dr_ov_dt_  = dr / dt;

i_domain_begin_ = patch->getCellStartingGlobalIndex( 0 );
j_domain_begin_ = patch->getCellStartingGlobalIndex( 1 );

nscellr_ = params.n_space[1] + 1;
oversize_[0] = params.oversize[0];
oversize_[1] = params.oversize[1];
nprimr_ = nscellr_ + 2*oversize_[1];
npriml_ = params.n_space[0] + 1 + 2*oversize_[0];

Nmode_=params.nmodes;
dq_inv_[0] = dl_inv_;
dq_inv_[1] = dr_inv_;

invR_ = &((static_cast<PatchAM *>( patch )->invR)[0]);
invRd_ = &((static_cast<PatchAM *>( patch )->invRd)[0]);

DEBUG( "cell_length "<< params.cell_length[0] );

}


ProjectorAM2OrderV::~ProjectorAM2OrderV()
{
}



void ProjectorAM2OrderV::currentsAndDensity( ElectroMagnAM *emAM,
Particles &particles,
unsigned int istart,
unsigned int iend,
double * __restrict__ invgf,
int * __restrict__ iold,
double * __restrict__ deltaold,
std::complex<double> * __restrict__ array_eitheta_old,
int npart_total,
int ipart_ref )
{

currents( emAM, particles,  istart, iend, invgf, iold, deltaold, array_eitheta_old, npart_total, ipart_ref );

int ipo = iold[0];
int jpo = iold[1];
int ipom2 = ipo-2;
int jpom2 = jpo-2;

int vecSize = 8;
int bsize = 5*5*vecSize*Nmode_;

std::complex<double> brho[bsize] __attribute__( ( aligned( 64 ) ) );

double Sl0_buff_vect[32] __attribute__( ( aligned( 64 ) ) );
double Sr0_buff_vect[32] __attribute__( ( aligned( 64 ) ) );
double DSl[40] __attribute__( ( aligned( 64 ) ) );
double DSr[40] __attribute__( ( aligned( 64 ) ) );
double charge_weight[8] __attribute__( ( aligned( 64 ) ) );
double r_bar[8] __attribute__( ( aligned( 64 ) ) );
complex<double> * __restrict__ rho;

double *invR_local = &(invR_[jpom2]);

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);
double * __restrict__ position_z = particles.getPtrPosition(2);
double * __restrict__ weight     = particles.getPtrWeight();
short  * __restrict__ charge     = particles.getPtrCharge();
int * __restrict__ cell_keys  = particles.getPtrCellKeys();

#pragma omp simd
for( unsigned int j=0; j<200*Nmode_; j++ ) {
brho[j] = 0.;
}

int cell_nparts( ( int )iend-( int )istart );

for( int ivect=0 ; ivect < cell_nparts; ivect += vecSize ) {

int np_computed = min( cell_nparts-ivect, vecSize );
int istart0 = ( int )istart + ivect;
complex<double> e_bar[8], e_delta_m1[8]; 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
compute_distances( position_x, position_y, position_z, cell_keys, npart_total, ipart, istart0, ipart_ref, deltaold, array_eitheta_old, iold, Sl0_buff_vect, Sr0_buff_vect, DSl, DSr, r_bar, e_bar, e_delta_m1 );
charge_weight[ipart] = inv_cell_volume * ( double )( charge[istart0+ipart] )*weight[istart0+ipart];
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
computeRho( ipart, charge_weight, DSl, DSr, Sl0_buff_vect, Sr0_buff_vect, brho, invR_local, e_bar);
} 
}

int iloc0 = ipom2*nprimr_+jpom2;
for( unsigned int imode=0; imode<( unsigned int )Nmode_; imode++ ) {
rho =  &( *emAM->rho_AM_[imode] )( 0 );
int iloc = iloc0;
for( unsigned int i=0 ; i<5 ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<5 ; j++ ) {
complex<double> tmprho( 0. );
int ilocal = ( i*5+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmprho += brho [200*imode + ilocal+ipart];
}
rho[iloc+j] += tmprho;
}
iloc += nprimr_;
}
}

} 

void ProjectorAM2OrderV::basicForComplex( complex<double> *rhoj, Particles &particles, unsigned int ipart, unsigned int type, int imode )
{




int iloc, nr( nprimr_ );
double charge_weight = inv_cell_volume * ( double )( particles.charge( ipart ) )*particles.weight( ipart );
double r = sqrt( particles.position( 1, ipart )*particles.position( 1, ipart )+particles.position( 2, ipart )*particles.position( 2, ipart ) );

if( type > 0 ) { 
charge_weight *= 1./sqrt( 1.0 + particles.momentum( 0, ipart )*particles.momentum( 0, ipart )
+ particles.momentum( 1, ipart )*particles.momentum( 1, ipart )
+ particles.momentum( 2, ipart )*particles.momentum( 2, ipart ) );
if( type == 1 ) { 
charge_weight *= particles.momentum( 0, ipart );
} else if( type == 2 ) { 
charge_weight *= ( particles.momentum( 1, ipart )*particles.position( 1, ipart ) + particles.momentum( 2, ipart )*particles.position( 2, ipart ) )/ r ;
nr++;
} else { 
charge_weight *= ( -particles.momentum( 1, ipart )*particles.position( 2, ipart ) + particles.momentum( 2, ipart )*particles.position( 1, ipart ) ) / r ;
}
}

complex<double> e_theta = ( particles.position( 1, ipart ) + Icpx*particles.position( 2, ipart ) )/r;
complex<double> C_m = 1.;
if( imode > 0 ) {
C_m = 2.;
}
for( unsigned int i=0; i<( unsigned int )imode; i++ ) {
C_m *= e_theta;
}

double xpn, ypn;
double delta, delta2;
double Sl1[5], Sr1[5];


xpn = particles.position( 0, ipart ) * dl_inv_;
int ip = round( xpn + 0.5 * ( type==1 ) );
delta  = xpn - ( double )ip;
delta2 = delta*delta;
Sl1[1] = 0.5 * ( delta2-delta+0.25 );
Sl1[2] = 0.75-delta2;
Sl1[3] = 0.5 * ( delta2+delta+0.25 );
ypn = r * dr_inv_ ;
int jp = round( ypn + 0.5*( type==2 ) );
delta  = ypn - ( double )jp;
delta2 = delta*delta;
Sr1[1] = 0.5 * ( delta2-delta+0.25 );
Sr1[2] = 0.75-delta2;
Sr1[3] = 0.5 * ( delta2+delta+0.25 );

ip -= i_domain_begin_ + 2;
jp -= j_domain_begin_ + 2;

if( type != 2 ) {
for( unsigned int i=1 ; i<4 ; i++ ) {
iloc = ( i+ip )*nr+jp;
for( unsigned int j=1 ; j<4 ; j++ ) {
rhoj [iloc+j] += C_m*charge_weight* Sl1[i]*Sr1[j] * invR_[j+jp];
}
}
} else {
for( unsigned int i=1 ; i<4 ; i++ ) {
iloc = ( i+ip )*nr+jp;
for( unsigned int j=1 ; j<4 ; j++ ) {
rhoj [iloc+j] += C_m*charge_weight* Sl1[i]*Sr1[j] * invRd_[j+jp];
}
}
}
} 

void ProjectorAM2OrderV::axisBC(ElectroMagnAM *emAM, bool diag_flag )
{

for (unsigned int imode=0; imode < Nmode_; imode++){ 

std::complex<double> *rhoj = &( *emAM->rho_AM_[imode] )( 0 );
std::complex<double> *Jl = &( *emAM->Jl_[imode] )( 0 );
std::complex<double> *Jr = &( *emAM->Jr_[imode] )( 0 );
std::complex<double> *Jt = &( *emAM->Jt_[imode] )( 0 );

apply_axisBC(rhoj, Jl, Jr, Jt, imode, diag_flag);
}

if (diag_flag){
unsigned int n_species = emAM->Jl_s.size() / Nmode_;
for( unsigned int imode = 0 ; imode < emAM->Jl_.size() ; imode++ ) {
for( unsigned int ispec = 0 ; ispec < n_species ; ispec++ ) {
unsigned int ifield = imode*n_species+ispec;
complex<double> *Jl  = emAM->Jl_s    [ifield] ? &( * ( emAM->Jl_s    [ifield] ) )( 0 ) : NULL ;
complex<double> *Jr  = emAM->Jr_s    [ifield] ? &( * ( emAM->Jr_s    [ifield] ) )( 0 ) : NULL ;
complex<double> *Jt  = emAM->Jt_s    [ifield] ? &( * ( emAM->Jt_s    [ifield] ) )( 0 ) : NULL ;
complex<double> *rho = emAM->rho_AM_s[ifield] ? &( * ( emAM->rho_AM_s[ifield] ) )( 0 ) : NULL ;
apply_axisBC( rho , Jl, Jr, Jt, imode, diag_flag );
}
}
}
}

void ProjectorAM2OrderV::apply_axisBC(std::complex<double> *rhoj,std::complex<double> *Jl, std::complex<double> *Jr, std::complex<double> *Jt, unsigned int imode, bool diag_flag )
{

double sign = -1.;
for (unsigned int i=0; i< imode; i++) sign *= -1;

if (diag_flag && rhoj) {
for( unsigned int i=2 ; i<npriml_*nprimr_+2; i+=nprimr_ ) {
for( unsigned int j=1 ; j<3; j++ ) {
rhoj[i+j] += sign * rhoj[i-j];
rhoj[i-j]  = sign * rhoj[i+j];
}
if (imode > 0){
rhoj[i] = 0.;
} else {
rhoj[i] = (4.*rhoj[i+1] - rhoj[i+2])/3.;
}
}
}

if (Jl) {
for( unsigned int i=2 ; i<(npriml_+1)*nprimr_+2; i+=nprimr_ ) {
for( unsigned int j=1 ; j<3; j++ ) {
Jl [i+j] +=  sign * Jl[i-j];
Jl[i-j]   =  sign * Jl[i+j];
}
if (imode > 0){
Jl [i] = 0. ;
} else {
Jl [i] =  (4.*Jl [i+1] - Jl [i+2])/3. ;
}
}
}

if (Jt && Jr) {
for( unsigned int i=0 ; i<npriml_; i++ ) {
int iloc = i*nprimr_+2;
int ilocr = i*(nprimr_+1)+3;
for( unsigned int j=1 ; j<3; j++ ) {
Jt [iloc+j] += -sign * Jt[iloc-j];
Jt[iloc-j]   = -sign * Jt[iloc+j];
}
for( unsigned int j=0 ; j<3; j++ ) {
Jr [ilocr+2-j] += -sign * Jr [ilocr-3+j];
Jr[ilocr-3+j]     = -sign * Jr[ilocr+2-j];
}

if (imode == 1){
Jt [iloc]= -Icpx/8.*( 9.*Jr[ilocr]- Jr[ilocr+1]);
Jr [ilocr-1] = 2.*Icpx*Jt[iloc] - Jr [ilocr];
} else{
Jt [iloc] = 0. ;
Jr [ilocr-1] = -Jr [ilocr];
}
}
}
return;
}

void ProjectorAM2OrderV::axisBCEnvChi( double *EnvChi )
{
double sign = 1.;
int imode = 0;
for (int i=0; i< imode; i++) sign *= -1;
if (EnvChi) {
for( unsigned int i=2 ; i<npriml_*nprimr_+2; i+=nprimr_ ) {

EnvChi[i]   = EnvChi[i+1];
for( unsigned int j=1 ; j<3; j++ ) {
EnvChi[i-j]  = sign * EnvChi[i+j];
}

}
}

return;
}

void ProjectorAM2OrderV::ionizationCurrents( Field *Jx, Field *Jy, Field *Jz, Particles &particles, int ipart, LocalFields Jion )
{
return;



} 


void ProjectorAM2OrderV::currents( ElectroMagnAM *emAM,
Particles &particles,
unsigned int istart,
unsigned int iend,
double * __restrict__ invgf,
int * __restrict__ iold,
double * __restrict__ deltaold,
std::complex<double> * __restrict__ array_eitheta_old,
int npart_total,
int ipart_ref )
{

int ipo = iold[0];
int jpo = iold[1];
int ipom2 = ipo-2;
int jpom2 = jpo-2;

int vecSize = 8;
int bsize = 5*5*vecSize*Nmode_;

std::complex<double> bJl[bsize] __attribute__( ( aligned( 64 ) ) );
std::complex<double> bJr[bsize] __attribute__( ( aligned( 64 ) ) );
std::complex<double> bJt[bsize] __attribute__( ( aligned( 64 ) ) );

double Sl0_buff_vect[32] __attribute__( ( aligned( 64 ) ) );
double Sr0_buff_vect[32] __attribute__( ( aligned( 64 ) ) );
double DSl[40] __attribute__( ( aligned( 64 ) ) );
double DSr[40] __attribute__( ( aligned( 64 ) ) );
double charge_weight[8] __attribute__( ( aligned( 64 ) ) );
double r_bar[8] __attribute__( ( aligned( 64 ) ) );
complex<double> * __restrict__ Jl;
complex<double> * __restrict__ Jr;
complex<double> * __restrict__ Jt;

double *invR_local = &(invR_[jpom2]);
double *invRd_local = &(invRd_[jpom2]);

double * __restrict__ position_x = particles.getPtrPosition(0);
double * __restrict__ position_y = particles.getPtrPosition(1);
double * __restrict__ position_z = particles.getPtrPosition(2);
double * __restrict__ momentum_y = particles.getPtrMomentum(1);
double * __restrict__ momentum_z = particles.getPtrMomentum(2);
double * __restrict__ weight     = particles.getPtrWeight();
short  * __restrict__ charge     = particles.getPtrCharge();
int * __restrict__ cell_keys  = particles.getPtrCellKeys();

#pragma omp simd
for( unsigned int j=0; j<200*Nmode_; j++ ) {
bJl[j] = 0.;
bJr[j] = 0.;
bJt[j] = 0.;
}

int cell_nparts( ( int )iend-( int )istart );

for( int ivect=0 ; ivect < cell_nparts; ivect += vecSize ) {

int np_computed = min( cell_nparts-ivect, vecSize );
int istart0 = ( int )istart + ivect;
complex<double> e_bar[8], e_delta_m1[8]; 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
compute_distances( position_x, position_y, position_z, cell_keys, npart_total, ipart, istart0, ipart_ref, deltaold, array_eitheta_old, iold, Sl0_buff_vect, Sr0_buff_vect, DSl, DSr, r_bar, e_bar, e_delta_m1 );
charge_weight[ipart] = inv_cell_volume * ( double )( charge[istart0+ipart] )*weight[istart0+ipart];
}

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
computeJl( ipart, charge_weight, DSl, DSr, Sr0_buff_vect, bJl, dl_ov_dt_, invR_local, e_bar);
} 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
computeJr( ipart, charge_weight, DSl, DSr, Sl0_buff_vect, bJr, one_ov_dt, invRd_local, e_bar, jpom2);
} 

#pragma omp simd
for( int ipart=0 ; ipart<np_computed; ipart++ ) {
computeJt( ipart, &momentum_y[istart0], &momentum_z[istart0], charge_weight, &invgf[istart0-ipart_ref], DSl, DSr, Sl0_buff_vect, Sr0_buff_vect, bJt, invR_local, r_bar, e_bar, e_delta_m1, one_ov_dt);
} 
} 

int iloc0 = ipom2*nprimr_+jpom2;

for( unsigned int imode=0; imode<( unsigned int )Nmode_; imode++ ) {
Jl =  &( *emAM->Jl_[imode] )( 0 );
int iloc = iloc0;
for( unsigned int i=1 ; i<5 ; i++ ) {
iloc += nprimr_;
#pragma omp simd
for( unsigned int j=0 ; j<5 ; j++ ) {
complex<double> tmpJl( 0. );
int ilocal = ( i*5+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJl += bJl [200*imode + ilocal+ipart];
}
Jl[iloc+j] += tmpJl;
}
}
}


for( unsigned int imode=0; imode<( unsigned int )Nmode_; imode++ ) {
Jr =  &( *emAM->Jr_[imode] )( 0 );
int iloc = iloc0 + ipom2 + 1;
for( unsigned int i=0 ; i<5 ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<4 ; j++ ) {
complex<double> tmpJr( 0. );
int ilocal = ( i*5+j+1 )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJr += bJr [200*imode + ilocal+ipart];
}
Jr[iloc+j] += tmpJr;
}
iloc += nprimr_+1;
}
}

for( unsigned int imode=0; imode<( unsigned int )Nmode_; imode++ ) {
Jt =  &( *emAM->Jt_[imode] )( 0 );
int iloc = iloc0;
for( unsigned int i=0 ; i<5 ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<5 ; j++ ) {
complex<double> tmpJt( 0. );
int ilocal = ( i*5+j )*vecSize;
UNROLL(8)
for( int ipart=0 ; ipart<8; ipart++ ) {
tmpJt += bJt [200*imode + ilocal+ipart];
}
Jt[iloc+j] += tmpJt;
}
iloc += nprimr_;
}
}
} 


void ProjectorAM2OrderV::currentsAndDensityWrapper( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int istart, int iend, int ithread,  bool diag_flag, bool is_spectral, int ispec, int scell, int ipart_ref )
{
if( istart == iend ) {
return;    
}

std::vector<double> *delta = &( smpi->dynamics_deltaold[ithread] );
std::vector<double> *invgf = &( smpi->dynamics_invgf[ithread] );
std::vector<std::complex<double>> *array_eitheta_old = &( smpi->dynamics_eithetaold[ithread] );
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( EMfields );


int iold[2];
iold[0] = scell/nscellr_+oversize_[0];
iold[1] = ( scell%nscellr_ )+oversize_[1];


if( !diag_flag ) {
if( !is_spectral ) {
currents( emAM, particles,  istart, iend, invgf->data(), iold, delta->data(), array_eitheta_old->data(), invgf->size(), ipart_ref );
} else {
ERROR( "Vectorized projection is not supported in spectral AM" );
}

} else {
currentsAndDensity( emAM, particles, istart, iend, invgf->data(), iold, delta->data(), array_eitheta_old->data(), invgf->size(), ipart_ref );
}
}

void ProjectorAM2OrderV::susceptibility( ElectroMagn *EMfields, Particles &particles, double species_mass, SmileiMPI *smpi, int istart, int iend,  int ithread, int icell, int ipart_ref )
{
ERROR( "Vectorized projection of the susceptibility for the envelope model is not implemented for AM geometry" );
}

