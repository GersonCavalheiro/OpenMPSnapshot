#include "IonizationFromRate.h"

#include <cmath>

#include "Particles.h"
#include "ParticleData.h"
#include "Species.h"

using namespace std;



IonizationFromRate::IonizationFromRate( Params &params, Species *species ) : Ionization( params, species )
{

DEBUG( "Creating the FromRate Ionizaton class" );

maximum_charge_state_ = species->maximum_charge_state_;
ionization_rate_ = species->ionization_rate_;

DEBUG( "Finished Creating the FromRate Ionizaton class" );

}



void IonizationFromRate::operator()( Particles *particles, unsigned int ipart_min, unsigned int ipart_max, vector<double> *Epart, Patch *patch, Projector *Proj, int ipart_ref )
{

unsigned int Z, k_times;
vector<double> rate;

if( ipart_min >= ipart_max ) {
return;
}

#ifdef SMILEI_USE_NUMPY
PyArrayObject *ret;
unsigned int npart = ipart_max - ipart_min;
#pragma omp critical
{
ParticleData particleData( npart );
particleData.startAt( ipart_min );
particleData.set( particles );
ret = ( PyArrayObject * )PyObject_CallFunctionObjArgs( ionization_rate_, particleData.get(), NULL );
PyTools::checkPyError();
if( ret == NULL ) {
ERROR( "ionization_rate profile has not provided a correct result" );
}
double *arr = ( double * ) PyArray_GETPTR1( ret, 0 );
rate.resize( npart );
for( unsigned int i=0; i<npart; i++ ) {
rate[i] = arr[i];
}
Py_DECREF( ret );
}
#endif


for( unsigned int ipart=ipart_min ; ipart<ipart_max; ipart++ ) {

Z = ( unsigned int )( particles->charge( ipart ) );

if( Z==maximum_charge_state_ ) {
continue;
}

k_times = 0;
double ran_p = patch->rand_->uniform();
if( ran_p < 1.0 - exp( -rate[ipart-ipart_min]*dt ) ) {
k_times        = 1;
}

if( k_times!=0 ) {
new_electrons.createParticle();
int idNew = new_electrons.size() - 1;
for( unsigned int i=0; i<new_electrons.dimension(); i++ ) {
new_electrons.position( i, idNew )=particles->position( i, ipart );
}
for( unsigned int i=0; i<3; i++ ) {
new_electrons.momentum( i, idNew ) = particles->momentum( i, ipart )*ionized_species_invmass;
}
new_electrons.weight( idNew )=double( k_times )*particles->weight( ipart );
new_electrons.charge( idNew )=-1;

particles->charge( ipart ) += k_times;
}


} 
}
