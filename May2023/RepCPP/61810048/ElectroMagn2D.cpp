#include "ElectroMagn2D.h"

#include <cmath>

#include <iostream>
#include <sstream>

#include "Params.h"
#include "Field2D.h"
#include "FieldFactory.h"

#include "Patch.h"
#include <cstring>

#include "Profile.h"

#include "ElectroMagnBC.h"

using namespace std;

ElectroMagn2D::ElectroMagn2D( Params &params, DomainDecomposition *domain_decomposition, vector<Species *> &vecSpecies, Patch *patch ) :
ElectroMagn( params, domain_decomposition, vecSpecies, patch )
{

initElectroMagn2DQuantities( params, patch );

for( unsigned int ispec=0; ispec<n_species; ispec++ ) {
Jx_s[ispec]  = FieldFactory::create(Tools::merge("Jx_" ,vecSpecies[ispec]->name_).c_str(), dimPrim, params);
Jy_s[ispec]  = FieldFactory::create(Tools::merge("Jy_" ,vecSpecies[ispec]->name_).c_str(), dimPrim, params);
Jz_s[ispec]  = FieldFactory::create(Tools::merge("Jz_" ,vecSpecies[ispec]->name_).c_str(), dimPrim, params);
rho_s[ispec] = new Field2D( Tools::merge( "Rho_", vecSpecies[ispec]->name_ ).c_str(), dimPrim );

if( params.Laser_Envelope_model ) {
Env_Chi_s[ispec] = new Field2D( Tools::merge( "Env_Chi_", vecSpecies[ispec]->name_ ).c_str(), dimPrim );
}

}

}


ElectroMagn2D::ElectroMagn2D( ElectroMagn2D *emFields, Params &params, Patch *patch ) :
ElectroMagn( emFields, params, patch )
{

initElectroMagn2DQuantities( params, patch );

for( unsigned int ispec=0; ispec<n_species; ispec++ ) {
if ( emFields->Jx_s[ispec] != NULL ) {
if ( emFields->Jx_s[ispec]->data_ != NULL )
Jx_s[ispec]  = FieldFactory::create(dimPrim, 0, false, emFields->Jx_s[ispec]->name, params);
else
Jx_s[ispec]  = FieldFactory::create(emFields->Jx_s[ispec]->name, dimPrim, params);
}
if ( emFields->Jy_s[ispec] != NULL ) {
if ( emFields->Jy_s[ispec]->data_ != NULL )
Jy_s[ispec]  = FieldFactory::create(dimPrim, 1, false, emFields->Jy_s[ispec]->name, params);
else
Jy_s[ispec]  = FieldFactory::create(emFields->Jy_s[ispec]->name, dimPrim, params);
}
if ( emFields->Jz_s[ispec] != NULL ) {
if ( emFields->Jz_s[ispec]->data_ != NULL )
Jz_s[ispec]  = FieldFactory::create(dimPrim, 2, false, emFields->Jz_s[ispec]->name, params);
else
Jz_s[ispec]  = FieldFactory::create(emFields->Jz_s[ispec]->name, dimPrim, params);
}
if( emFields->rho_s[ispec] != NULL ) {
if( emFields->rho_s[ispec]->data_ != NULL ) {
rho_s[ispec] = new Field2D( dimPrim, emFields->rho_s[ispec]->name );
} else {
rho_s[ispec]  = new Field2D( emFields->rho_s[ispec]->name, dimPrim );
}
}

if( params.Laser_Envelope_model ) {
if( emFields->Env_Chi_s[ispec] != NULL ) {
if( emFields->Env_Chi_s[ispec]->data_ != NULL ) {
Env_Chi_s[ispec] = new Field2D( dimPrim, emFields->Env_Chi_s[ispec]->name );
} else {
Env_Chi_s[ispec]  = new Field2D( emFields->Env_Chi_s[ispec]->name, dimPrim );
}
}
}


}

}

void ElectroMagn2D::initElectroMagn2DQuantities( Params &params, Patch *patch )
{

dx       = cell_length[0];
dt_ov_dx = timestep/dx;
dx_ov_dt = 1.0/dt_ov_dx;

dy       = cell_length[1];
dt_ov_dy = timestep/dy;
dy_ov_dt = 1.0/dt_ov_dy;


dimPrim.resize( nDim_field );
dimDual.resize( nDim_field );

for( size_t i=0 ; i<nDim_field ; i++ ) {
dimPrim[i] = n_space[i]+1;
dimDual[i] = n_space[i]+2-(params.is_pxr);
dimPrim[i] += 2*oversize[i];
dimDual[i] += 2*oversize[i];
}
nx_p = n_space[0]+1+2*oversize[0];
nx_d = n_space[0]+2+2*oversize[0]-(params.is_pxr);
ny_p = n_space[1]+1+2*oversize[1];
ny_d = n_space[1]+2+2*oversize[1]-(params.is_pxr);

Ex_  = FieldFactory::create( dimPrim, 0, false, "Ex", params );
Ey_  = FieldFactory::create( dimPrim, 1, false, "Ey", params );
Ez_  = FieldFactory::create( dimPrim, 2, false, "Ez", params );
Bx_  = FieldFactory::create( dimPrim, 0, true,  "Bx", params );
By_  = FieldFactory::create( dimPrim, 1, true,  "By", params );
Bz_  = FieldFactory::create( dimPrim, 2, true,  "Bz", params );
Bx_m = FieldFactory::create( dimPrim, 0, true,  "Bx_m", params );
By_m = FieldFactory::create( dimPrim, 1, true,  "By_m", params );
Bz_m = FieldFactory::create( dimPrim, 2, true,  "Bz_m", params );

if( params.Laser_Envelope_model ) {
Env_A_abs_  = new Field2D( dimPrim, "Env_A_abs" );
Env_Chi_    = new Field2D( dimPrim, "Env_Chi" );
Env_E_abs_  = new Field2D( dimPrim, "Env_E_abs" );
Env_Ex_abs_ = new Field2D( dimPrim, "Env_Ex_abs" );
}
if( params.Friedman_filter ) {
Exfilter.resize( 3 );
Exfilter[0] = new Field2D( dimPrim, 0, false, "Ex_f" );
Exfilter[1] = new Field2D( dimPrim, 0, false, "Ex_m1" );
Exfilter[2] = new Field2D( dimPrim, 0, false, "Ex_m2" );
Eyfilter.resize( 3 );
Eyfilter[0] = new Field2D( dimPrim, 1, false, "Ey_f" );
Eyfilter[1] = new Field2D( dimPrim, 1, false, "Ey_m1" );
Eyfilter[2] = new Field2D( dimPrim, 1, false, "Ey_m2" );
Ezfilter.resize( 3 );
Ezfilter[0] = new Field2D( dimPrim, 2, false, "Ez_f" );
Ezfilter[1] = new Field2D( dimPrim, 2, false, "Ez_m1" );
Ezfilter[2] = new Field2D( dimPrim, 2, false, "Ez_m2" );
}

Jx_   = FieldFactory::create( dimPrim, 0, false, "Jx", params );
Jy_   = FieldFactory::create( dimPrim, 1, false, "Jy", params );
Jz_   = FieldFactory::create( dimPrim, 2, false, "Jz", params );
rho_  = new Field2D( dimPrim, "Rho" );

if(params.is_pxr == true) {
rhoold_ = new Field2D( dimPrim, "RhoOld" );
}


index_bc_min.resize( nDim_field, 0 );
index_bc_max.resize( nDim_field, 0 );
for( unsigned int i=0 ; i<nDim_field ; i++ ) {
index_bc_min[i] = oversize[i];
index_bc_max[i] = dimDual[i]-oversize[i]-1;
}



for( unsigned int i=0 ; i<3 ; i++ )
for( unsigned int isDual=0 ; isDual<2 ; isDual++ ) {
istart[i][isDual] = 0;
}
for( unsigned int i=0 ; i<nDim_field ; i++ ) {
for( unsigned int isDual=0 ; isDual<2 ; isDual++ ) {
istart[i][isDual] = oversize[i];
if( patch->Pcoordinates[i]!=0 ) {
istart[i][isDual]+=1;
}
}
}

for( unsigned int i=0 ; i<3 ; i++ )
for( unsigned int isDual=0 ; isDual<2 ; isDual++ ) {
bufsize[i][isDual] = 1;
}

for( unsigned int i=0 ; i<nDim_field ; i++ ) {
for( int isDual=0 ; isDual<2 ; isDual++ ) {
bufsize[i][isDual] = n_space[i] + 1;
}

for( int isDual=0 ; isDual<2 ; isDual++ ) {
bufsize[i][isDual] += isDual;
if( params.number_of_patches[i]!=1 ) {

if( ( !isDual ) && ( patch->Pcoordinates[i]!=0 ) ) {
bufsize[i][isDual]--;
} else if( isDual ) {
bufsize[i][isDual]--;
if( ( patch->Pcoordinates[i]!=0 ) && ( patch->Pcoordinates[i]!=( unsigned int )params.number_of_patches[i]-1 ) ) {
bufsize[i][isDual]--;
}
}

} 
} 
} 
}

ElectroMagn2D::~ElectroMagn2D()
{
}





void ElectroMagn2D::initPoisson( Patch *patch )
{
Field2D *rho2D = static_cast<Field2D *>( rho_ );


index_min_p_.resize( 2, 0 );
index_max_p_.resize( 2, 0 );

index_min_p_[0] = oversize[0];
index_min_p_[1] = oversize[1];
index_max_p_[0] = nx_p - 2 - oversize[0];
index_max_p_[1] = ny_p - 2 - oversize[1];
if( patch->isXmin() ) {
index_min_p_[0] = 0;
}
if( patch->isXmax() ) {
index_max_p_[0] = nx_p-1;
}

phi_ = new Field2D( dimPrim );  
r_   = new Field2D( dimPrim );  
p_   = new Field2D( dimPrim );  
Ap_  = new Field2D( dimPrim );  


for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *phi_ )( i, j )   = 0.0;
( *r_ )( i, j )     = -( *rho2D )( i, j );
( *p_ )( i, j )     = ( *r_ )( i, j );
}
}

} 

double ElectroMagn2D::compute_r()
{
double rnew_dot_rnew_local( 0. );
for( unsigned int i=index_min_p_[0]; i<=index_max_p_[0]; i++ ) {
for( unsigned int j=index_min_p_[1]; j<=index_max_p_[1]; j++ ) {
rnew_dot_rnew_local += ( *r_ )( i, j )*( *r_ )( i, j );
}
}
return rnew_dot_rnew_local;
} 

void ElectroMagn2D::compute_Ap( Patch *patch )
{
double one_ov_dx_sq       = 1.0/( dx*dx );
double one_ov_dy_sq       = 1.0/( dy*dy );
double two_ov_dx2dy2      = 2.0*( 1.0/( dx*dx )+1.0/( dy*dy ) );

for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Ap_ )( i, j ) = one_ov_dx_sq*( ( *p_ )( i-1, j )+( *p_ )( i+1, j ) )
+ one_ov_dy_sq*( ( *p_ )( i, j-1 )+( *p_ )( i, j+1 ) )
- two_ov_dx2dy2*( *p_ )( i, j );
}
}


if( patch->isXmin() ) {
for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Ap_ )( 0, j )      = one_ov_dx_sq*( ( *p_ )( 1, j ) )
+              one_ov_dy_sq*( ( *p_ )( 0, j-1 )+( *p_ )( 0, j+1 ) )
-              two_ov_dx2dy2*( *p_ )( 0, j );
}
( *Ap_ )( 0, 0 )           = one_ov_dx_sq*( ( *p_ )( 1, 0 ) )   
+                   one_ov_dy_sq*( ( *p_ )( 0, 1 ) )
-                   two_ov_dx2dy2*( *p_ )( 0, 0 );
( *Ap_ )( 0, ny_p-1 )      = one_ov_dx_sq*( ( *p_ )( 1, ny_p-1 ) ) 
+                   one_ov_dy_sq*( ( *p_ )( 0, ny_p-2 ) )
-                   two_ov_dx2dy2*( *p_ )( 0, ny_p-1 );
}

if( patch->isXmax() ) {

for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Ap_ )( nx_p-1, j ) = one_ov_dx_sq*( ( *p_ )( nx_p-2, j ) )
+              one_ov_dy_sq*( ( *p_ )( nx_p-1, j-1 )+( *p_ )( nx_p-1, j+1 ) )
-              two_ov_dx2dy2*( *p_ )( nx_p-1, j );
}
( *Ap_ )( nx_p-1, 0 )      = one_ov_dx_sq*( ( *p_ )( nx_p-2, 0 ) )     
+                   one_ov_dy_sq*( ( *p_ )( nx_p-1, 1 ) )
-                   two_ov_dx2dy2*( *p_ )( nx_p-1, 0 );
( *Ap_ )( nx_p-1, ny_p-1 ) = one_ov_dx_sq*( ( *p_ )( nx_p-2, ny_p-1 ) ) 
+                   one_ov_dy_sq*( ( *p_ )( nx_p-1, ny_p-2 ) )
-                   two_ov_dx2dy2*( *p_ )( nx_p-1, ny_p-1 );
}

} 

void ElectroMagn2D::compute_Ap_relativistic_Poisson( Patch *patch, double gamma_mean )
{

double one_ov_dx_sq_ov_gamma_sq       = 1.0/( dx*dx )/( gamma_mean*gamma_mean );
double one_ov_dy_sq                   = 1.0/( dy*dy );
double two_ov_dxgam2dy2               = 2.0*( 1.0/( dx*dx )/( gamma_mean*gamma_mean )+1.0/( dy*dy ) );

for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Ap_ )( i, j ) = one_ov_dx_sq_ov_gamma_sq*( ( *p_ )( i-1, j )+( *p_ )( i+1, j ) )
+ one_ov_dy_sq*( ( *p_ )( i, j-1 )+( *p_ )( i, j+1 ) )
- two_ov_dxgam2dy2*( *p_ )( i, j );
}
}


if( patch->isXmin() ) {
for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Ap_ )( 0, j )      = one_ov_dx_sq_ov_gamma_sq*( ( *p_ )( 1, j ) )
+              one_ov_dy_sq*( ( *p_ )( 0, j-1 )+( *p_ )( 0, j+1 ) )
-              two_ov_dxgam2dy2*( *p_ )( 0, j );
}
( *Ap_ )( 0, 0 )           = one_ov_dx_sq_ov_gamma_sq*( ( *p_ )( 1, 0 ) )   
+                   one_ov_dy_sq*( ( *p_ )( 0, 1 ) )
-                   two_ov_dxgam2dy2*( *p_ )( 0, 0 );
( *Ap_ )( 0, ny_p-1 )      = one_ov_dx_sq_ov_gamma_sq*( ( *p_ )( 1, ny_p-1 ) ) 
+                   one_ov_dy_sq*( ( *p_ )( 0, ny_p-2 ) )
-                   two_ov_dxgam2dy2*( *p_ )( 0, ny_p-1 );
}

if( patch->isXmax() ) {

for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Ap_ )( nx_p-1, j ) = one_ov_dx_sq_ov_gamma_sq*( ( *p_ )( nx_p-2, j ) )
+              one_ov_dy_sq*( ( *p_ )( nx_p-1, j-1 )+( *p_ )( nx_p-1, j+1 ) )
-              two_ov_dxgam2dy2*( *p_ )( nx_p-1, j );
}
( *Ap_ )( nx_p-1, 0 )      = one_ov_dx_sq_ov_gamma_sq*( ( *p_ )( nx_p-2, 0 ) )     
+                   one_ov_dy_sq*( ( *p_ )( nx_p-1, 1 ) )
-                   two_ov_dxgam2dy2*( *p_ )( nx_p-1, 0 );
( *Ap_ )( nx_p-1, ny_p-1 ) = one_ov_dx_sq_ov_gamma_sq*( ( *p_ )( nx_p-2, ny_p-1 ) ) 
+                   one_ov_dy_sq*( ( *p_ )( nx_p-1, ny_p-2 ) )
-                   two_ov_dxgam2dy2*( *p_ )( nx_p-1, ny_p-1 );
}

} 

double ElectroMagn2D::compute_pAp()
{
double p_dot_Ap_local = 0.0;
for( unsigned int i=index_min_p_[0]; i<=index_max_p_[0]; i++ ) {
for( unsigned int j=index_min_p_[1]; j<=index_max_p_[1]; j++ ) {
p_dot_Ap_local += ( *p_ )( i, j )*( *Ap_ )( i, j );
}
}
return p_dot_Ap_local;
} 

void ElectroMagn2D::update_pand_r( double r_dot_r, double p_dot_Ap )
{
double alpha_k = r_dot_r/p_dot_Ap;
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *phi_ )( i, j ) += alpha_k * ( *p_ )( i, j );
( *r_ )( i, j )   -= alpha_k * ( *Ap_ )( i, j );
}
}

} 

void ElectroMagn2D::update_p( double rnew_dot_rnew, double r_dot_r )
{
double beta_k = rnew_dot_rnew/r_dot_r;
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *p_ )( i, j ) = ( *r_ )( i, j ) + beta_k * ( *p_ )( i, j );
}
}
} 

void ElectroMagn2D::initE( Patch *patch )
{
Field2D *Ex2D  = static_cast<Field2D *>( Ex_ );
Field2D *Ey2D  = static_cast<Field2D *>( Ey_ );
Field2D *rho2D = static_cast<Field2D *>( rho_ );


DEBUG( "Computing Ex from scalar potential, Poisson problem" );
for( unsigned int i=1; i<nx_d-1; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( i, j ) = ( ( *phi_ )( i-1, j )-( *phi_ )( i, j ) )/dx;
}
}
DEBUG( "Computing Ey from scalar potential, Poisson problem" );
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=1; j<ny_d-1; j++ ) {
( *Ey2D )( i, j ) = ( ( *phi_ )( i, j-1 )-( *phi_ )( i, j ) )/dy;
}
}

if( patch->isXmin() ) {
DEBUG( "Computing Xmin BC on Ex, Poisson problem" );
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( 0, j ) = ( *Ex2D )( 1, j ) + ( ( *Ey2D )( 0, j+1 )-( *Ey2D )( 0, j ) )*dx/dy  - dx*( *rho2D )( 0, j );
}
}
if( patch->isXmax() ) {
DEBUG( "Computing Xmax BC on Ex, Poisson problem" );
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( nx_d-1, j ) = ( *Ex2D )( nx_d-2, j ) - ( ( *Ey2D )( nx_p-1, j+1 )-( *Ey2D )( nx_p-1, j ) )*dx/dy + dx*( *rho2D )( nx_p-1, j );
}
}

delete phi_;
delete r_;
delete p_;
delete Ap_;

} 

void ElectroMagn2D::initE_relativistic_Poisson( Patch *patch, double gamma_mean )
{

Field2D *Ex2D  = static_cast<Field2D *>( Ex_rel_ );
Field2D *Ey2D  = static_cast<Field2D *>( Ey_rel_ );
Field2D *rho2D = static_cast<Field2D *>( rho_ );




MESSAGE( 1, "Computing Ex from scalar potential, relativistic Poisson problem" );
for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( i, j ) = ( ( *phi_ )( i-1, j )-( *phi_ )( i, j ) )/dx/gamma_mean/gamma_mean;
}
}
MESSAGE( 1, "Ex: done" );
MESSAGE( 1, "Computing Ey from scalar potential, relativistic Poisson problem" );
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Ey2D )( i, j ) = ( ( *phi_ )( i, j-1 )-( *phi_ )( i, j ) )/dy;
}
}
MESSAGE( 1, "Ey: done" );
if( patch->isXmin() ) {
DEBUG( "Computing Xmin BC on Ex, relativistic Poisson problem" );
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( 0, j ) = ( *Ex2D )( 1, j ) + ( ( *Ey2D )( 0, j+1 )-( *Ey2D )( 0, j ) )*dx/dy  - dx*( *rho2D )( 0, j );
}
}
if( patch->isXmax() ) {
DEBUG( "Computing Xmax BC on Ex, relativistic Poisson problem" );
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( nx_d-1, j ) = ( *Ex2D )( nx_d-2, j ) - ( ( *Ey2D )( nx_p-1, j+1 )-( *Ey2D )( nx_p-1, j ) )*dx/dy + dx*( *rho2D )( nx_p-1, j );
}
}


delete phi_;
delete r_;
delete p_;
delete Ap_;

} 

void ElectroMagn2D::initB_relativistic_Poisson( Patch *patch, double gamma_mean )
{

Field2D *Ey2D  = static_cast<Field2D *>( Ey_rel_ );
Field2D *Bz2D  = static_cast<Field2D *>( Bz_rel_ );
Field2D *Bx2D  = static_cast<Field2D *>( Bx_rel_ );

double beta_mean = sqrt( 1.-1./gamma_mean/gamma_mean );
MESSAGE( 0, "In relativistic Poisson solver, gamma_mean = " << gamma_mean );

MESSAGE( 1, "Computing Bx, relativistic Poisson problem" );
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Bx2D )( i, j ) = 0.;
}
}
MESSAGE( 1, "Bx: done" );

MESSAGE( 1, "Computing Bz from scalar potential, relativistic Poisson problem" );
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Bz2D )( i, j ) = beta_mean*( *Ey2D )( i, j );
}
}
MESSAGE( 1, "Bz: done" );


} 

void ElectroMagn2D::center_fields_from_relativistic_Poisson( Patch *patch )
{

Field2D *Bx2Drel  = static_cast<Field2D *>( Bx_rel_ );
Field2D *By2Drel  = static_cast<Field2D *>( By_rel_ );
Field2D *Bz2Drel  = static_cast<Field2D *>( Bz_rel_ );

Field2D *Bx2D  = static_cast<Field2D *>( Bx_rel_t_plus_halfdt_ );
Field2D *By2D  = static_cast<Field2D *>( By_rel_t_plus_halfdt_ );
Field2D *Bz2D  = static_cast<Field2D *>( Bz_rel_t_plus_halfdt_ );
Field2D *Bx2D0  = static_cast<Field2D *>( Bx_rel_t_minus_halfdt_ );
Field2D *By2D0  = static_cast<Field2D *>( By_rel_t_minus_halfdt_ );
Field2D *Bz2D0  = static_cast<Field2D *>( Bz_rel_t_minus_halfdt_ );



for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Bx2D )( i, j )= ( *Bx2Drel )( i, j );
( *Bx2D0 )( i, j )= ( *Bx2Drel )( i, j );
}
}

for( unsigned int i=1; i<nx_d-1; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *By2D )( i, j )= 0.5 * ( ( *By2Drel )( i, j ) + ( *By2Drel )( i-1, j ) );
( *By2D0 )( i, j )= 0.5 * ( ( *By2Drel )( i, j ) + ( *By2Drel )( i-1, j ) );
}
}

for( unsigned int i=1; i<nx_d-1; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Bz2D )( i, j )= 0.5 * ( ( *Bz2Drel )( i, j ) + ( *Bz2Drel )( i-1, j ) );
( *Bz2D0 )( i, j )= 0.5 * ( ( *Bz2Drel )( i, j ) + ( *Bz2Drel )( i-1, j ) );
}
}

}

void ElectroMagn2D::initRelativisticPoissonFields( Patch *patch )
{

Ex_rel_  = new Field2D( dimPrim, 0, false, "Ex_rel" );
Ey_rel_  = new Field2D( dimPrim, 1, false, "Ey_rel" );
Ez_rel_  = new Field2D( dimPrim, 2, false, "Ez_rel" );
Bx_rel_  = new Field2D( dimPrim, 0, true,  "Bx_rel" ); 
By_rel_  = new Field2D( dimPrim, 2, false,  "By_rel" ); 
Bz_rel_  = new Field2D( dimPrim, 1, false,  "Bz_rel" ); 


Bx_rel_t_plus_halfdt_  = new Field2D( dimPrim, 0, true,  "Bx_rel_t_plus_halfdt" );
By_rel_t_plus_halfdt_  = new Field2D( dimPrim, 1, true,  "By_rel_t_plus_halfdt" );
Bz_rel_t_plus_halfdt_  = new Field2D( dimPrim, 2, true,  "Bz_rel_t_plus_halfdt" );
Bx_rel_t_minus_halfdt_  = new Field2D( dimPrim, 0, true,  "Bx_rel_t_plus_halfdt" );
By_rel_t_minus_halfdt_  = new Field2D( dimPrim, 1, true,  "By_rel_t_plus_halfdt" );
Bz_rel_t_minus_halfdt_  = new Field2D( dimPrim, 2, true,  "Bz_rel_t_plus_halfdt" );



} 

void ElectroMagn2D::sum_rel_fields_to_em_fields( Patch *patch )
{
Field2D *Ex2Drel  = static_cast<Field2D *>( Ex_rel_ );
Field2D *Ey2Drel  = static_cast<Field2D *>( Ey_rel_ );
Field2D *Ez2Drel  = static_cast<Field2D *>( Ez_rel_ );

Field2D *Bx_rel_t_plus_halfdt = static_cast<Field2D *>( Bx_rel_t_plus_halfdt_ );
Field2D *By_rel_t_plus_halfdt = static_cast<Field2D *>( By_rel_t_plus_halfdt_ );
Field2D *Bz_rel_t_plus_halfdt = static_cast<Field2D *>( Bz_rel_t_plus_halfdt_ );

Field2D *Bx_rel_t_minus_halfdt = static_cast<Field2D *>( Bx_rel_t_minus_halfdt_ );
Field2D *By_rel_t_minus_halfdt = static_cast<Field2D *>( By_rel_t_minus_halfdt_ );
Field2D *Bz_rel_t_minus_halfdt = static_cast<Field2D *>( Bz_rel_t_minus_halfdt_ );

Field2D *Ex2D  = static_cast<Field2D *>( Ex_ );
Field2D *Ey2D  = static_cast<Field2D *>( Ey_ );
Field2D *Ez2D  = static_cast<Field2D *>( Ez_ );
Field2D *Bx2D  = static_cast<Field2D *>( Bx_ );
Field2D *By2D  = static_cast<Field2D *>( By_ );
Field2D *Bz2D  = static_cast<Field2D *>( Bz_ );
Field2D *Bx2D0  = static_cast<Field2D *>( Bx_m );
Field2D *By2D0  = static_cast<Field2D *>( By_m );
Field2D *Bz2D0  = static_cast<Field2D *>( Bz_m );

for( unsigned int i=0; i<nx_d; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( i, j ) = ( *Ex2D )( i, j ) + ( *Ex2Drel )( i, j );
}
}

for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Ey2D )( i, j ) = ( *Ey2D )( i, j ) + ( *Ey2Drel )( i, j );
}
}

for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ez2D )( i, j ) = ( *Ez2D )( i, j ) + ( *Ez2Drel )( i, j );
}
}




double half_dt_ov_dx = 0.5 * timestep / dx;
double half_dt_ov_dy = 0.5 * timestep / dy;

for( unsigned int i=0 ; i<nx_p;  i++ ) {
for( unsigned int j=1 ; j<ny_d-1 ; j++ ) {
( *Bx_rel_t_plus_halfdt )( i, j ) += -1.* half_dt_ov_dy * ( ( *Ez2Drel )( i, j ) - ( *Ez2Drel )( i, j-1 ) );
( *Bx_rel_t_minus_halfdt )( i, j ) -= -1.* half_dt_ov_dy * ( ( *Ez2Drel )( i, j ) - ( *Ez2Drel )( i, j-1 ) );
( *Bx2D )( i, j ) += ( *Bx_rel_t_plus_halfdt )( i, j );
( *Bx2D0 )( i, j ) += ( *Bx_rel_t_minus_halfdt )( i, j );
}
}

for( unsigned int i=1 ; i<nx_d-1 ; i++ ) {
for( unsigned int j=0 ; j<ny_p ; j++ ) {
( *By_rel_t_plus_halfdt )( i, j ) +=  half_dt_ov_dx * ( ( *Ez2Drel )( i, j ) - ( *Ez2Drel )( i-1, j ) );
( *By_rel_t_minus_halfdt )( i, j ) -=  half_dt_ov_dx * ( ( *Ez2Drel )( i, j ) - ( *Ez2Drel )( i-1, j ) );
( *By2D )( i, j ) += ( *By_rel_t_plus_halfdt )( i, j );
( *By2D0 )( i, j ) += ( *By_rel_t_minus_halfdt )( i, j );
}
}

for( unsigned int i=1 ; i<nx_d-1 ; i++ ) {
for( unsigned int j=1 ; j<ny_d-1 ; j++ ) {
( *Bz_rel_t_plus_halfdt )( i, j )  += -half_dt_ov_dx * ( ( *Ey2Drel )( i, j ) - ( *Ey2Drel )( i-1, j ) ) + half_dt_ov_dy * ( ( *Ex2Drel )( i, j ) - ( *Ex2Drel )( i, j-1 ) );
( *Bz_rel_t_minus_halfdt )( i, j ) -= -half_dt_ov_dx * ( ( *Ey2Drel )( i, j ) - ( *Ey2Drel )( i-1, j ) ) + half_dt_ov_dy * ( ( *Ex2Drel )( i, j ) - ( *Ex2Drel )( i, j-1 ) );
( *Bz2D )( i, j ) += ( *Bz_rel_t_plus_halfdt )( i, j );
( *Bz2D0 )( i, j ) += ( *Bz_rel_t_minus_halfdt )( i, j );
}
}

delete Ex_rel_;
delete Ey_rel_;
delete Ez_rel_;
delete Bx_rel_;
delete By_rel_;
delete Bz_rel_;

delete Bx_rel_t_plus_halfdt;
delete By_rel_t_plus_halfdt;
delete Bz_rel_t_plus_halfdt;
delete Bx_rel_t_minus_halfdt;
delete By_rel_t_minus_halfdt;
delete Bz_rel_t_minus_halfdt;




} 

void ElectroMagn2D::centeringE( std::vector<double> E_Add )
{
Field2D *Ex2D  = static_cast<Field2D *>( Ex_ );
Field2D *Ey2D  = static_cast<Field2D *>( Ey_ );

for( unsigned int i=0; i<nx_d; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( i, j ) += E_Add[0];
}
}
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Ey2D )( i, j ) += E_Add[1];
}
}
} 

void ElectroMagn2D::centeringErel( std::vector<double> E_Add )
{
Field2D *Ex2D  = static_cast<Field2D *>( Ex_rel_ );
Field2D *Ey2D  = static_cast<Field2D *>( Ey_rel_ );

for( unsigned int i=0; i<nx_d; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Ex2D )( i, j ) += E_Add[0];
}
}
for( unsigned int i=0; i<nx_p; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Ey2D )( i, j ) += E_Add[1];
}
}
} 



void ElectroMagn2D::saveMagneticFields( bool is_spectral )
{
if( !is_spectral ) {
Field2D *Bx2D   = static_cast<Field2D *>( Bx_ );
Field2D *By2D   = static_cast<Field2D *>( By_ );
Field2D *Bz2D   = static_cast<Field2D *>( Bz_ );
Field2D *Bx2D_m = static_cast<Field2D *>( Bx_m );
Field2D *By2D_m = static_cast<Field2D *>( By_m );
Field2D *Bz2D_m = static_cast<Field2D *>( Bz_m );

for( unsigned int i=0 ; i<nx_p ; i++ ) {
memcpy( &( ( *Bx2D_m )( i, 0 ) ), &( ( *Bx2D )( i, 0 ) ), ny_d*sizeof( double ) );

memcpy( &( ( *By2D_m )( i, 0 ) ), &( ( *By2D )( i, 0 ) ), ny_p*sizeof( double ) );

memcpy( &( ( *Bz2D_m )( i, 0 ) ), &( ( *Bz2D )( i, 0 ) ), ny_d*sizeof( double ) );
}
memcpy( &( ( *By2D_m )( nx_p, 0 ) ), &( ( *By2D )( nx_p, 0 ) ), ny_p*sizeof( double ) );
memcpy( &( ( *Bz2D_m )( nx_p, 0 ) ), &( ( *Bz2D )( nx_p, 0 ) ), ny_d*sizeof( double ) );
} else {
Bx_m->deallocateDataAndSetTo( Bx_ );
By_m->deallocateDataAndSetTo( By_ );
Bz_m->deallocateDataAndSetTo( Bz_ );
}
}


void ElectroMagn2D::binomialCurrentFilter(unsigned int ipass, std::vector<unsigned int> passes)
{
Field2D *Jx2D = static_cast<Field2D *>( Jx_ );
Field2D *Jy2D = static_cast<Field2D *>( Jy_ );
Field2D *Jz2D = static_cast<Field2D *>( Jz_ );

if (ipass < passes[0]){
for( unsigned int i=0; i<nx_d-1; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Jx2D )( i, j) = ( ( *Jx2D )( i, j) + ( *Jx2D )( i+1, j) )*0.5;
}
}
for( unsigned int i=nx_d-2; i>0; i-- ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Jx2D )( i, j) = ( ( *Jx2D )( i, j) + ( *Jx2D )( i-1, j) )*0.5;
}
}
for( unsigned int i=0; i<nx_p-1; i++ ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Jy2D )( i, j) = ( ( *Jy2D )( i, j) + ( *Jy2D )( i+1, j) )*0.5;
}
}
for( unsigned int i=nx_p-2; i>0; i-- ) {
for( unsigned int j=0; j<ny_d; j++ ) {
( *Jy2D )( i, j) = ( ( *Jy2D )( i, j) + ( *Jy2D )( i-1, j) )*0.5;
}
}
for( unsigned int i=0; i<nx_p-1; i++ ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Jz2D )( i, j) = ( ( *Jz2D )( i, j) + ( *Jz2D )( i+1, j) )*0.5;
}
}
for( unsigned int i=nx_p-2; i>0; i-- ) {
for( unsigned int j=0; j<ny_p; j++ ) {
( *Jz2D )( i, j) = ( ( *Jz2D )( i, j) + ( *Jz2D )( i-1, j) )*0.5;
}
}
}

if (ipass < passes[1]){
for( unsigned int i=1; i<nx_d-1; i++ ) {
for( unsigned int j=0; j<ny_p-1; j++ ) {
( *Jx2D )( i, j) = ( ( *Jx2D )( i, j) + ( *Jx2D )( i, j+1) )*0.5;
}
}
for( unsigned int i=1; i<nx_d-1; i++ ) {
for( unsigned int j=ny_p-2; j>0; j-- ) {
( *Jx2D )( i, j) = ( ( *Jx2D )( i, j) + ( *Jx2D )( i, j-1) )*0.5;
}
}
for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=0; j<ny_d-1; j++ ) {
( *Jy2D )( i, j) = ( ( *Jy2D )( i, j) + ( *Jy2D )( i, j+1) )*0.5;
}
}
for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=ny_d-2; j>0; j-- ) {
( *Jy2D )( i, j) = ( ( *Jy2D )( i, j) + ( *Jy2D )( i, j-1) )*0.5;
}
}
for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=0; j<ny_p-1; j++ ) {
( *Jz2D )( i, j) = ( ( *Jz2D )( i, j) + ( *Jz2D )( i, j+1) )*0.5;
}
}
for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=ny_p-2; j>0; j-- ) {
( *Jz2D )( i, j) = ( ( *Jz2D )( i, j) + ( *Jz2D )( i, j-1) )*0.5;
}
}
}


}

void ElectroMagn2D::customFIRCurrentFilter(unsigned int ipass, std::vector<unsigned int> passes, std::vector<double> filtering_coeff)
{
Field2D *Jx2D = static_cast<Field2D *>( Jx_ );
Field2D *Jy2D = static_cast<Field2D *>( Jy_ );
Field2D *Jz2D = static_cast<Field2D *>( Jz_ );

unsigned int m=1 ;

unsigned int gcfilt=0 ;

if (ipass < passes[0]){
Field2D *tmp   = new Field2D( dimPrim, 0, false );
tmp->copyFrom( Jx2D );
for( unsigned int i=((filtering_coeff.size()-1)/(m*2)+gcfilt); i<nx_d-((filtering_coeff.size()-1)/(m*2)+gcfilt); i++ ) {
for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Jx2D )( i, j ) = 0. ;
for ( unsigned int kernel_idx = 0; kernel_idx < filtering_coeff.size(); kernel_idx+=m) {
( *Jx2D )( i, j ) += filtering_coeff[kernel_idx]*( *tmp )( i - (filtering_coeff.size()-1)/(m*2) + kernel_idx/m, j ) ;
}
( *Jx2D )( i, j ) *= m ;
}
}
delete tmp;
tmp   = new Field2D( dimPrim, 1, false );
tmp->copyFrom( Jy2D );
for( unsigned int i=((filtering_coeff.size()-1)/(m*2)+gcfilt); i<nx_p-((filtering_coeff.size()-1)/(m*2)+gcfilt); i++ ) {
for( unsigned int j=1; j<ny_d-1; j++ ) {
( *Jy2D )( i, j ) = 0. ;
for ( unsigned int kernel_idx = 0; kernel_idx < filtering_coeff.size(); kernel_idx+=m) {
( *Jy2D )( i, j ) += filtering_coeff[kernel_idx]*( *tmp )( i - (filtering_coeff.size()-1)/(m*2) + kernel_idx/m, j ) ;
}
( *Jy2D )( i, j ) *= m ;
}
}
delete tmp;
tmp   = new Field2D( dimPrim, 2, false );
tmp->copyFrom( Jz2D );
for( unsigned int i=((filtering_coeff.size()-1)/(m*2)+gcfilt); i<nx_p-((filtering_coeff.size()-1)/(m*2)+gcfilt); i++ ) {
for( unsigned int j=1; j<ny_p-1; j++ ) {
( *Jz2D )( i, j ) = 0. ;
for ( unsigned int kernel_idx = 0; kernel_idx < filtering_coeff.size(); kernel_idx+=m) {
( *Jz2D )( i, j ) += filtering_coeff[kernel_idx]*( *tmp )( i - (filtering_coeff.size()-1)/(m*2) + kernel_idx/m, j ) ;
}
( *Jz2D )( i, j ) *= m ;
}
}
delete tmp;
}

if (ipass < passes[1]){
Field2D *tmp   = new Field2D( dimPrim, 0, false );
tmp->copyFrom( Jx2D );
for( unsigned int i=1; i<nx_d-1; i++ ) {
for( unsigned int j=((filtering_coeff.size()-1)/(m*2)+gcfilt); j<ny_p-((filtering_coeff.size()-1)/(m*2)+gcfilt); j++ ) {
( *Jx2D )( i, j ) = 0. ;
for ( unsigned int kernel_idx = 0; kernel_idx < filtering_coeff.size(); kernel_idx+=m) {
( *Jx2D )( i, j ) += filtering_coeff[kernel_idx]*( *tmp )( i, j - (filtering_coeff.size()-1)/(m*2) + kernel_idx/m ) ;
}
( *Jx2D )( i, j ) *= m ;
}
}
delete tmp;
tmp   = new Field2D( dimPrim, 1, false );
tmp->copyFrom( Jy2D );
for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=((filtering_coeff.size()-1)/(m*2)+gcfilt); j<ny_d-((filtering_coeff.size()-1)/(m*2)+gcfilt); j++ ) {
( *Jy2D )( i, j ) = 0. ;
for ( unsigned int kernel_idx = 0; kernel_idx < filtering_coeff.size(); kernel_idx+=m) {
( *Jy2D )( i, j ) += filtering_coeff[kernel_idx]*( *tmp )( i, j - (filtering_coeff.size()-1)/(m*2) + kernel_idx/m ) ;
}
( *Jy2D )( i, j ) *= m ;
}
}
delete tmp;
tmp   = new Field2D( dimPrim, 2, false );
tmp->copyFrom( Jz2D );
for( unsigned int i=1; i<nx_p-1; i++ ) {
for( unsigned int j=((filtering_coeff.size()-1)/(m*2)+gcfilt); j<ny_p-((filtering_coeff.size()-1)/(m*2)+gcfilt); j++ ) {
( *Jz2D )( i, j ) = 0. ;
for ( unsigned int kernel_idx = 0; kernel_idx < filtering_coeff.size(); kernel_idx+=m) {
( *Jz2D )( i, j ) += filtering_coeff[kernel_idx]*( *tmp )( i, j - (filtering_coeff.size()-1)/(m*2) + kernel_idx/m ) ;
}
( *Jz2D )( i, j ) *= m ;
}
}
delete tmp;
}

}





void ElectroMagn2D::centerMagneticFields()
{
Field2D *Bx2D   = static_cast<Field2D *>( Bx_ );
Field2D *By2D   = static_cast<Field2D *>( By_ );
Field2D *Bz2D   = static_cast<Field2D *>( Bz_ );
Field2D *Bx2D_m = static_cast<Field2D *>( Bx_m );
Field2D *By2D_m = static_cast<Field2D *>( By_m );
Field2D *Bz2D_m = static_cast<Field2D *>( Bz_m );

for( unsigned int i=0 ; i<nx_p ; i++ ) {
#pragma omp simd
for( unsigned int j=0 ; j<ny_d ; j++ ) {
( *Bx2D_m )( i, j ) = ( ( *Bx2D )( i, j ) + ( *Bx2D_m )( i, j ) )*0.5;
}

#pragma omp simd
for( unsigned int j=0 ; j<ny_p ; j++ ) {
( *By2D_m )( i, j ) = ( ( *By2D )( i, j ) + ( *By2D_m )( i, j ) )*0.5;
}

#pragma omp simd
for( unsigned int j=0 ; j<ny_d ; j++ ) {
( *Bz2D_m )( i, j ) = ( ( *Bz2D )( i, j ) + ( *Bz2D_m )( i, j ) )*0.5;
} 
} 
#pragma omp simd
for( unsigned int j=0 ; j<ny_p ; j++ ) {
( *By2D_m )( nx_p, j ) = ( ( *By2D )( nx_p, j ) + ( *By2D_m )( nx_p, j ) )*0.5;
}
#pragma omp simd
for( unsigned int j=0 ; j<ny_d ; j++ ) {
( *Bz2D_m )( nx_p, j ) = ( ( *Bz2D )( nx_p, j ) + ( *Bz2D_m )( nx_p, j ) )*0.5;
} 


}



Field * ElectroMagn2D::createField( string fieldname, Params& params )
{
if     (fieldname.substr(0,2)=="Ex" ) return FieldFactory::create(dimPrim, 0, false, fieldname, params);
else if(fieldname.substr(0,2)=="Ey" ) return FieldFactory::create(dimPrim, 1, false, fieldname, params);
else if(fieldname.substr(0,2)=="Ez" ) return FieldFactory::create(dimPrim, 2, false, fieldname, params);
else if(fieldname.substr(0,2)=="Bx" ) return FieldFactory::create(dimPrim, 0, true,  fieldname, params);
else if(fieldname.substr(0,2)=="By" ) return FieldFactory::create(dimPrim, 1, true,  fieldname, params);
else if(fieldname.substr(0,2)=="Bz" ) return FieldFactory::create(dimPrim, 2, true,  fieldname, params);
else if(fieldname.substr(0,2)=="Jx" ) return FieldFactory::create(dimPrim, 0, false, fieldname, params);
else if(fieldname.substr(0,2)=="Jy" ) return FieldFactory::create(dimPrim, 1, false, fieldname, params);
else if(fieldname.substr(0,2)=="Jz" ) return FieldFactory::create(dimPrim, 2, false, fieldname, params);
else if(fieldname.substr(0,3)=="Rho") return new Field2D(dimPrim, fieldname );

ERROR("Cannot create field "<<fieldname);
return NULL;
}

void ElectroMagn2D::computeTotalRhoJ()
{
Field2D *Jx2D    = static_cast<Field2D *>( Jx_ );
Field2D *Jy2D    = static_cast<Field2D *>( Jy_ );
Field2D *Jz2D    = static_cast<Field2D *>( Jz_ );
Field2D *rho2D   = static_cast<Field2D *>( rho_ );



for( unsigned int ispec=0; ispec<n_species; ispec++ ) {
if( Jx_s[ispec] ) {
Field2D *Jx2D_s  = static_cast<Field2D *>( Jx_s[ispec] );
for( unsigned int i=0 ; i<Jx2D->dims_[0] ; i++ )
for( unsigned int j=0 ; j<Jx2D->dims_[1] ; j++ ) {
( *Jx2D )( i, j ) += ( *Jx2D_s )( i, j );
}
}
if( Jy_s[ispec] ) {
Field2D *Jy2D_s  = static_cast<Field2D *>( Jy_s[ispec] );
for( unsigned int i=0 ; i<Jy2D->dims_[0] ; i++ )
for( unsigned int j=0 ; j<Jy2D->dims_[1] ; j++ ) {
( *Jy2D )( i, j ) += ( *Jy2D_s )( i, j );
}
}
if( Jz_s[ispec] ) {
Field2D *Jz2D_s  = static_cast<Field2D *>( Jz_s[ispec] );
for( unsigned int i=0 ; i<Jz2D->dims_[0] ; i++ )
for( unsigned int j=0 ; j<Jz2D->dims_[1] ; j++ ) {
( *Jz2D )( i, j ) += ( *Jz2D_s )( i, j );
}
}
if( rho_s[ispec] ) {
Field2D *rho2D_s  = static_cast<Field2D *>( rho_s[ispec] );
for( unsigned int i=0 ; i<rho2D->dims_[0] ; i++ )
for( unsigned int j=0 ; j<rho2D->dims_[1] ; j++ ) {
( *rho2D )( i, j ) += ( *rho2D_s )( i, j );
}
}
}
}

void ElectroMagn2D::computeTotalEnvChi()
{

Field2D *Env_Chi2D   = static_cast<Field2D *>( Env_Chi_ );

for( unsigned int ispec=0; ispec<n_species; ispec++ ) {
if( Env_Chi_s[ispec] ) {
Field2D *Env_Chi2D_s  = static_cast<Field2D *>( Env_Chi_s[ispec] );
for( unsigned int i=0 ; i<nx_p ; i++ ) {
for( unsigned int j=0 ; j<ny_p ; j++ ) {
( *Env_Chi2D )( i, j ) += ( *Env_Chi2D_s )( i, j );
}
}
}
}


} 

void ElectroMagn2D::computePoynting( unsigned int axis, unsigned int side )
{
Field2D *Ex2D     = static_cast<Field2D *>( Ex_ );
Field2D *Ey2D     = static_cast<Field2D *>( Ey_ );
Field2D *Ez2D     = static_cast<Field2D *>( Ez_ );
Field2D *Bx2D_m   = static_cast<Field2D *>( Bx_m );
Field2D *By2D_m   = static_cast<Field2D *>( By_m );
Field2D *Bz2D_m   = static_cast<Field2D *>( Bz_m );

double sign = ( side == 0 ) ? 1. : -1;

if( axis == 0 ) {

unsigned int offset = ( side == 0 ) ? 0 : bufsize[0][Ey2D->isDual( 0 )];

unsigned int iEy = istart[0][Ey2D  ->isDual( 0 )] + offset;
unsigned int iBz = istart[0][Bz2D_m->isDual( 0 )] + offset;
unsigned int iEz = istart[0][Ez2D  ->isDual( 0 )] + offset;
unsigned int iBy = istart[0][By2D_m->isDual( 0 )] + offset;

unsigned int jEy = istart[1][Ey2D  ->isDual( 1 )];
unsigned int jBz = istart[1][Bz2D_m->isDual( 1 )];
unsigned int jEz = istart[1][Ez2D  ->isDual( 1 )];
unsigned int jBy = istart[1][By2D_m->isDual( 1 )];

poynting_inst[side][0] = 0.;
for( unsigned int j=0; j<bufsize[1][Ez2D->isDual( 1 )]; j++ ) {
double Ey__ = 0.5*( ( *Ey2D )( iEy, jEy+j ) + ( *Ey2D )( iEy, jEy+j+1 ) );
double Bz__ = 0.25*( ( *Bz2D_m )( iBz, jBz+j )+( *Bz2D_m )( iBz+1, jBz+j )+( *Bz2D_m )( iBz, jBz+j+1 )+( *Bz2D_m )( iBz+1, jBz+j+1 ) );
double Ez__ = ( *Ez2D )( iEz, jEz+j );
double By__ = 0.5*( ( *By2D_m )( iBy, jBy+j ) + ( *By2D_m )( iBy+1, jBy+j ) );
poynting_inst[side][0] += Ey__*Bz__ - Ez__*By__;
}
poynting_inst[side][0] *= dy*timestep;
poynting[side][0] += sign * poynting_inst[side][0];

} else if( axis == 1 ) {

unsigned int offset = ( side == 0 ) ? 0 : bufsize[1][Ez2D->isDual( 1 )];

unsigned int iEz = istart[0][Ez_ ->isDual( 0 )];
unsigned int iBx = istart[0][Bx_m->isDual( 0 )];
unsigned int iEx = istart[0][Ex_ ->isDual( 0 )];
unsigned int iBz = istart[0][Bz_m->isDual( 0 )];

unsigned int jEz = istart[1][Ez_ ->isDual( 1 )] + offset;
unsigned int jBx = istart[1][Bx_m->isDual( 1 )] + offset;
unsigned int jEx = istart[1][Ex_ ->isDual( 1 )] + offset;
unsigned int jBz = istart[1][Bz_m->isDual( 1 )] + offset;

poynting_inst[side][1] = 0.;
for( unsigned int i=0; i<bufsize[0][Ez2D->isDual( 0 )]; i++ ) {
double Ez__ = ( *Ez2D )( iEz+i, jEz );
double Bx__ = 0.5*( ( *Bx2D_m )( iBx+i, jBx ) + ( *Bx2D_m )( iBx+i, jBx+1 ) );
double Ex__ = 0.5*( ( *Ex2D )( iEx+i, jEx ) + ( *Ex2D )( iEx+i+1, jEx ) );
double Bz__ = 0.25*( ( *Bz2D_m )( iBz+i, jBz )+( *Bz2D_m )( iBz+i+1, jBz )+( *Bz2D_m )( iBz+i, jBz+1 )+( *Bz2D_m )( iBz+i+1, jBz+1 ) );
poynting_inst[side][1] += Ez__*Bx__ - Ex__*Bz__;
}
poynting_inst[side][1] *= dx*timestep;
poynting[side][1] += sign * poynting_inst[side][1];

}

}

void ElectroMagn2D::applyExternalField( Field *my_field,  Profile *profile, Patch *patch )
{
Field2D *field2D=static_cast<Field2D *>( my_field );

vector<double> pos( 2, 0 );
pos[0]      = dx*( ( double )( patch->getCellStartingGlobalIndex( 0 ) )+( field2D->isDual( 0 )?-0.5:0. ) );
double pos1 = dy*( ( double )( patch->getCellStartingGlobalIndex( 1 ) )+( field2D->isDual( 1 )?-0.5:0. ) );

vector<Field *> xyz( 2 );
vector<unsigned int> dims = { field2D->dims_[0], field2D->dims_[1], 1 };
for( unsigned int idim=0 ; idim<2 ; idim++ ) {
xyz[idim] = new Field3D( dims );
}

for( unsigned int i=0 ; i<dims[0] ; i++ ) {
pos[1] = pos1;
for( unsigned int j=0 ; j<dims[1] ; j++ ) {
for( unsigned int idim=0 ; idim<2 ; idim++ ) {
( *xyz[idim] )( i, j ) = pos[idim];
}
pos[1] += dy;
}
pos[0] += dx;
}

vector<double> global_origin = { 
dx * ( ( field2D->isDual( 0 )?-0.5:0. ) - oversize[0] ),
dy * ( ( field2D->isDual( 1 )?-0.5:0. ) - oversize[1] )
};
profile->valuesAt( xyz, global_origin, *field2D, 1 );

for( unsigned int idim=0 ; idim<2 ; idim++ ) {
delete xyz[idim];
}
}

void ElectroMagn2D::applyPrescribedField( Field *my_field,  Profile *profile, Patch *patch, double time )
{

Field2D *field2D=static_cast<Field2D *>( my_field );

vector<double> pos( 2, 0 );
pos[0]      = dx*( ( double )( patch->getCellStartingGlobalIndex( 0 ) )+( field2D->isDual( 0 )?-0.5:0. ) );
double pos1 = dy*( ( double )( patch->getCellStartingGlobalIndex( 1 ) )+( field2D->isDual( 1 )?-0.5:0. ) );

vector<Field *> xyz( 2 );
vector<unsigned int> dims = { field2D->dims_[0], field2D->dims_[1] };
for( unsigned int idim=0 ; idim<2 ; idim++ ) {
xyz[idim] = new Field2D( dims );
}

for( unsigned int i=0 ; i<dims[0] ; i++ ) {
pos[1] = pos1;
for( unsigned int j=0 ; j<dims[1] ; j++ ) {
for( unsigned int idim=0 ; idim<2 ; idim++ ) {
( *xyz[idim] )( i, j ) = pos[idim];
}
pos[1] += dy;
}
pos[0] += dx;
}

vector<double> global_origin = { 
dx * ( ( field2D->isDual( 0 )?-0.5:0. ) - oversize[0] ),
dy * ( ( field2D->isDual( 1 )?-0.5:0. ) - oversize[1] )
};
profile->valuesAt( xyz, global_origin, *field2D, 3, time );

for( unsigned int idim=0 ; idim<2 ; idim++ ) {
delete xyz[idim];
}

}


void ElectroMagn2D::initAntennas( Patch *patch, Params& params )
{

for( unsigned int i=0; i<antennas.size(); i++ ) {
if( antennas[i].fieldName == "Jx" ) {
antennas[i].field = FieldFactory::create( dimPrim, 0, false, "Jx", params );
} else if( antennas[i].fieldName == "Jy" ) {
antennas[i].field = FieldFactory::create( dimPrim, 1, false, "Jy", params );
} else if( antennas[i].fieldName == "Jz" ) {
antennas[i].field = FieldFactory::create( dimPrim, 2, false, "Jz", params );
} else {
ERROR("Antenna cannot be applied to field "<<antennas[i].fieldName);
}

if( ! antennas[i].spacetime && antennas[i].field ) {
applyExternalField( antennas[i].field, antennas[i].space_profile, patch );
}
}

}

