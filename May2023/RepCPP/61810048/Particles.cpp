
#include "Particles.h"

#include <cstring>
#include <iostream>

#include "Params.h"
#include "Patch.h"
#include "Species.h"

#include "Particle.h"

using namespace std;



Particles::Particles():
tracked( false )
{
Position.resize( 0 );
Position_old.resize( 0 );
Momentum.resize( 0 );
cell_keys.resize( 0 );
is_test = false;
isQuantumParameter = false;
isMonteCarlo = false;

double_prop_.resize( 0 );
short_prop_.resize( 0 );
uint64_prop_.resize( 0 );
}

Particles::~Particles()
{
clear();
shrinkToFit();
}

void Particles::initialize( unsigned int nParticles, unsigned int nDim, bool keep_position_old )
{
if( Weight.size()==0 ) {
float c_part_max =1.2;
}

resize( nParticles, nDim, keep_position_old );

if( double_prop_.empty() ) {  

Position.resize( nDim );
for( unsigned int i=0 ; i< nDim ; i++ ) {
double_prop_.push_back( &( Position[i] ) );
}

for( unsigned int i=0 ; i< 3 ; i++ ) {
double_prop_.push_back( &( Momentum[i] ) );
}

double_prop_.push_back( &Weight );

if( keep_position_old ) {
Position_old.resize( nDim );
for( unsigned int i=0 ; i< nDim ; i++ ) {
double_prop_.push_back( &( Position_old[i] ) );
}
}

short_prop_.push_back( &Charge );
if( tracked ) {
uint64_prop_.push_back( &Id );
}

if( isQuantumParameter ) {
double_prop_.push_back( &Chi );
}

if( isMonteCarlo ) {
double_prop_.push_back( &Tau );
}

}

}

void Particles::initialize( unsigned int nParticles, Particles &part )
{
is_test=part.is_test;

tracked=part.tracked;

isQuantumParameter=part.isQuantumParameter;

isMonteCarlo=part.isMonteCarlo;

initialize( nParticles, part.Position.size(), part.Position_old.size() > 0 );
}



void Particles::reserve( unsigned int n_part_max, unsigned int nDim, bool keep_position_old  )
{

Position.resize( nDim );
for( unsigned int i=0 ; i< nDim ; i++ ) {
Position[i].reserve( n_part_max );
}

if( keep_position_old ) {
Position_old.resize( nDim );
for( unsigned int i=0 ; i< Position_old.size() ; i++ ) {
Position_old[i].reserve( n_part_max );
}
}

for( unsigned int i=0 ; i< 3 ; i++ ) {
Momentum[i].reserve( n_part_max );
}
Weight.reserve( n_part_max );
Charge.reserve( n_part_max );

if( tracked ) {
Id.reserve( n_part_max );
}

if( isQuantumParameter ) {
Chi.reserve( n_part_max );
}

if( isMonteCarlo ) {
Tau.reserve( n_part_max );
}

cell_keys.reserve( n_part_max );
}

void Particles::reserve( unsigned int n_part_max)
{

for( unsigned int i=0 ; i< Position.size() ; i++ ) {
Position[i].reserve( n_part_max );
}

if (Position_old.size() > 0) {
for( unsigned int i=0 ; i< Position_old.size() ; i++ ) {
Position_old[i].reserve( n_part_max );
}
}

for( unsigned int i=0 ; i< Momentum.size() ; i++ ) {
Momentum[i].reserve( n_part_max );
}
Weight.reserve( n_part_max );
Charge.reserve( n_part_max );

if( tracked ) {
Id.reserve( n_part_max );
}

if( isQuantumParameter ) {
Chi.reserve( n_part_max );
}

if( isMonteCarlo ) {
Tau.reserve( n_part_max );
}

cell_keys.reserve( n_part_max );


}

void Particles::initializeReserve( unsigned int npart_max, Particles &part )
{
initialize( 0, part );
}

void Particles::resize( unsigned int nParticles, unsigned int nDim, bool keep_position_old )
{
Position.resize( nDim );
for( unsigned int i=0 ; i<nDim ; i++ ) {
Position[i].resize( nParticles, 0. );
}

if( keep_position_old ) {
Position_old.resize( nDim );
for( unsigned int i=0 ; i<nDim ; i++ ) {
Position_old[i].resize( nParticles, 0. );
}
}

Momentum.resize( 3 );
for( unsigned int i=0 ; i< 3 ; i++ ) {
Momentum[i].resize( nParticles, 0. );
}

Weight.resize( nParticles, 0. );
Charge.resize( nParticles, 0 );

if( tracked ) {
Id.resize( nParticles, 0 );
}

if( isQuantumParameter ) {
Chi.resize( nParticles, 0. );
}

if( isMonteCarlo ) {
Tau.resize( nParticles, 0. );
}

cell_keys.resize( nParticles, 0. );

}

void Particles::resize( unsigned int nParticles)
{

for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).resize( nParticles, 0. );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).resize( nParticles, 0 );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).resize( nParticles, 0 );
}

cell_keys.resize( nParticles, 0. );

}

void Particles::resizeCellKeys(unsigned int nParticles)
{
cell_keys.resize( nParticles, 0. );
}

void Particles::shrinkToFit()
{

for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
std::vector<double>( *double_prop_[iprop] ).swap( *double_prop_[iprop] );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
std::vector<short>( *short_prop_[iprop] ).swap( *short_prop_[iprop] );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
std::vector<uint64_t>( *uint64_prop_[iprop] ).swap( *uint64_prop_[iprop] );
}


}


void Particles::clear()
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
double_prop_[iprop]->clear();
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
short_prop_[iprop]->clear();
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
uint64_prop_[iprop]->clear();
}


}

void Particles::copyParticle( unsigned int ipart )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
double_prop_[iprop]->push_back( ( *double_prop_[iprop] )[ipart] );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
short_prop_[iprop]->push_back( ( *short_prop_[iprop] )[ipart] );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
uint64_prop_[iprop]->push_back( ( *uint64_prop_[iprop] )[ipart] );
}
}


void Particles::copyParticle( unsigned int ipart, Particles &dest_parts )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
dest_parts.double_prop_[iprop]->push_back( ( *double_prop_[iprop] )[ipart] );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
dest_parts.short_prop_[iprop]->push_back( ( *short_prop_[iprop] )[ipart] );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
dest_parts.uint64_prop_[iprop]->push_back( ( *uint64_prop_[iprop] )[ipart] );
}
}

void Particles::copyParticle( unsigned int ipart, Particles &dest_parts, int dest_id )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
dest_parts.double_prop_[iprop]->insert( dest_parts.double_prop_[iprop]->begin() + dest_id, ( *double_prop_[iprop] )[ipart] );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
dest_parts.short_prop_[iprop]->insert( dest_parts.short_prop_[iprop]->begin() + dest_id, ( *short_prop_[iprop] )[ipart] );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
dest_parts.uint64_prop_[iprop]->insert( dest_parts.uint64_prop_[iprop]->begin() + dest_id, ( *uint64_prop_[iprop] )[ipart] );
}

}

void Particles::copyParticles( unsigned int iPart, unsigned int nPart, Particles &dest_parts, int dest_id )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
dest_parts.double_prop_[iprop]->insert( dest_parts.double_prop_[iprop]->begin() + dest_id, double_prop_[iprop]->begin()+iPart, double_prop_[iprop]->begin()+iPart+nPart );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
dest_parts.short_prop_[iprop]->insert( dest_parts.short_prop_[iprop]->begin() + dest_id, short_prop_[iprop]->begin()+iPart, short_prop_[iprop]->begin()+iPart+nPart );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
dest_parts.uint64_prop_[iprop]->insert( dest_parts.uint64_prop_[iprop]->begin() + dest_id, uint64_prop_[iprop]->begin()+iPart, uint64_prop_[iprop]->begin()+iPart+nPart );
}
}

void Particles::makeParticleAt( Particles &source_particles, unsigned int ipart, double w, short q, double px, double py, double pz )
{
for( unsigned int i=0 ; i<Position.size() ; i++ ) {
Position[i].push_back( source_particles.Position[i][ipart] );
}

if( Position_old.size() > 0. ) {
for( unsigned int i=0 ; i<Position_old.size() ; i++ ) {
Position_old[i].push_back( source_particles.Position_old[i][ipart] );
}
}

Momentum[0].push_back( px );
Momentum[1].push_back( py );
Momentum[2].push_back( pz );

Weight.push_back( w );
Charge.push_back( q );

if( tracked ) {
Id.push_back( 0 );
}

if( isQuantumParameter ) {
Chi.push_back( 0. );
}

if( isMonteCarlo ) {
Tau.push_back( 0. );
}
}


void Particles::eraseParticle( unsigned int ipart )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).erase( ( *double_prop_[iprop] ).begin()+ipart );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).erase( ( *short_prop_[iprop] ).begin()+ipart );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).erase( ( *uint64_prop_[iprop] ).begin()+ipart );
}


}

void Particles::eraseParticleTrail( unsigned int ipart )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).erase( ( *double_prop_[iprop] ).begin()+ipart, ( *double_prop_[iprop] ).end() );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).erase( ( *short_prop_[iprop] ).begin()+ipart, ( *short_prop_[iprop] ).end() );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).erase( ( *uint64_prop_[iprop] ).begin()+ipart, ( *uint64_prop_[iprop] ).end() );
}

}
void Particles::eraseParticle( unsigned int ipart, unsigned int npart )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).erase( ( *double_prop_[iprop] ).begin()+ipart, ( *double_prop_[iprop] ).begin()+ipart+npart );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).erase( ( *short_prop_[iprop] ).begin()+ipart, ( *short_prop_[iprop] ).begin()+ipart+npart );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).erase( ( *uint64_prop_[iprop] ).begin()+ipart, ( *uint64_prop_[iprop] ).begin()+ipart+npart );
}

}

void Particles::print( unsigned int iPart )
{
for( unsigned int i=0; i<Position.size(); i++ ) {
cout << Position[i][iPart] << " ";
}
for( unsigned int i=0; i<3; i++ ) {
cout << Momentum[i][iPart] << " ";
}
cout << Weight[iPart] << " ";
cout << Charge[iPart] << endl;;

if( tracked ) {
cout << Id[iPart] << endl;
}

if( isQuantumParameter ) {
cout << Chi[iPart] << endl;
}

if( isMonteCarlo ) {
cout << Tau[iPart] << endl;
}
}


ostream &operator << ( ostream &out, const Particles &particles )
{
for( unsigned int iPart=0; iPart<particles.Weight.size(); iPart++ ) {

for( unsigned int i=0; i<particles.Position.size(); i++ ) {
out << particles.Position[i][iPart] << " ";
}
for( unsigned int i=0; i<3; i++ ) {
out << particles.Momentum[i][iPart] << " ";
}
out << particles.Weight[iPart] << " ";
out << particles.Charge[iPart] << endl;;

if( particles.tracked ) {
out << particles.Id[iPart] << endl;
}

if( particles.isQuantumParameter ) {
out << particles.Chi[iPart] << endl;
}

if( particles.isMonteCarlo ) {
out << particles.Tau[iPart] << endl;
}
}

return ( out );
}


void Particles::swapParticle( unsigned int part1, unsigned int part2 )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
std::swap( ( *double_prop_[iprop] )[part1], ( *double_prop_[iprop] )[part2] );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
std::swap( ( *short_prop_[iprop] )[part1], ( *short_prop_[iprop] )[part2] );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
std::swap( ( *uint64_prop_[iprop] )[part1], ( *uint64_prop_[iprop] )[part2] );
}
}


void Particles::swapParticle3( unsigned int part1, unsigned int part2, unsigned int part3 )
{
double temp;
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
temp = ( *double_prop_[iprop] )[part1];
( *double_prop_[iprop] )[part1] = ( *double_prop_[iprop] )[part3];
( *double_prop_[iprop] )[part3] = ( *double_prop_[iprop] )[part2];
( *double_prop_[iprop] )[part2] = temp;
}

short stemp;
for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
stemp = ( *short_prop_[iprop] )[part1];
( *short_prop_[iprop] )[part1] = ( *short_prop_[iprop] )[part3];
( *short_prop_[iprop] )[part3] = ( *short_prop_[iprop] )[part2];
( *short_prop_[iprop] )[part2] = stemp;
}

unsigned int uitemp;
for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
uitemp = ( *short_prop_[iprop] )[part1];
( *uint64_prop_[iprop] )[part1] = ( *uint64_prop_[iprop] )[part3];
( *uint64_prop_[iprop] )[part3] = ( *uint64_prop_[iprop] )[part2];
( *uint64_prop_[iprop] )[part2] = uitemp;
}

}


void Particles::swapParticle4( unsigned int part1, unsigned int part2, unsigned int part3, unsigned int part4 )
{
double temp;
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
temp = ( *double_prop_[iprop] )[part1];
( *double_prop_[iprop] )[part1] = ( *double_prop_[iprop] )[part4];
( *double_prop_[iprop] )[part4] = ( *double_prop_[iprop] )[part3];
( *double_prop_[iprop] )[part3] = ( *double_prop_[iprop] )[part2];
( *double_prop_[iprop] )[part2] = temp;
}

short stemp;
for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
stemp = ( *short_prop_[iprop] )[part1];
( *short_prop_[iprop] )[part1] = ( *short_prop_[iprop] )[part4];
( *short_prop_[iprop] )[part4] = ( *short_prop_[iprop] )[part3];
( *short_prop_[iprop] )[part3] = ( *short_prop_[iprop] )[part2];
( *short_prop_[iprop] )[part2] = stemp;
}

unsigned int uitemp;
for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
uitemp = ( *short_prop_[iprop] )[part1];
( *uint64_prop_[iprop] )[part1] = ( *uint64_prop_[iprop] )[part4];
( *uint64_prop_[iprop] )[part4] = ( *uint64_prop_[iprop] )[part3];
( *uint64_prop_[iprop] )[part3] = ( *uint64_prop_[iprop] )[part2];
( *uint64_prop_[iprop] )[part2] = uitemp;
}

}


void Particles::swapParticles( std::vector<unsigned int> parts )
{

copyParticle( parts.back() );
translateParticles( parts );
overwriteParticle( size()-1, parts[0] );
eraseParticle( size()-1 );

}


void Particles::translateParticles( std::vector<unsigned int> parts )
{

for( int icycle = parts.size()-2; icycle >=0; icycle-- ) {
overwriteParticle( parts[icycle], parts[icycle+1] );
}

}


void Particles::overwriteParticle( unsigned int src_particle, unsigned int dest_particle )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] )[dest_particle] = ( *double_prop_[iprop] )[src_particle];
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] )[dest_particle] = ( *short_prop_[iprop] )[src_particle];
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] )[dest_particle] = ( *uint64_prop_[iprop] )[src_particle];
}
}


void Particles::overwriteParticle( unsigned int part1, unsigned int part2, unsigned int N )
{
unsigned int sizepart = N*sizeof( Position[0][0] );
unsigned int sizecharge = N*sizeof( Charge[0] );
unsigned int sizeid = N*sizeof( Id[0] );

for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
memcpy( & ( *double_prop_[iprop] )[part2],  &( *double_prop_[iprop] )[part1], sizepart );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
memcpy( & ( *short_prop_[iprop] )[part2],  &( *short_prop_[iprop] )[part1], sizecharge );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
memcpy( & ( *uint64_prop_[iprop] )[part2],  &( *uint64_prop_[iprop] )[part1], sizeid );
}
}

void Particles::overwriteParticle( unsigned int part1, Particles &dest_parts, unsigned int part2 )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *dest_parts.double_prop_[iprop] )[part2] = ( *double_prop_[iprop] )[part1];
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *dest_parts.short_prop_[iprop] )[part2] = ( *short_prop_[iprop] )[part1];
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *dest_parts.uint64_prop_[iprop] )[part2] = ( *uint64_prop_[iprop] )[part1];
}
}

void Particles::overwriteParticle( unsigned int part1, Particles &dest_parts, unsigned int part2, unsigned int N )
{
unsigned int sizepart = N*sizeof( Position[0][0] );
unsigned int sizecharge = N*sizeof( Charge[0] );
unsigned int sizeid = N*sizeof( Id[0] );

for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
memcpy( & ( *dest_parts.double_prop_[iprop] )[part2],  &( *double_prop_[iprop] )[part1], sizepart );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
memcpy( & ( *dest_parts.short_prop_[iprop] )[part2],  &( *short_prop_[iprop] )[part1], sizecharge );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
memcpy( & ( *dest_parts.uint64_prop_[iprop] )[part2],  &( *uint64_prop_[iprop] )[part1], sizeid );
}

}


void Particles::swapParticle( unsigned int part1, unsigned int part2, unsigned int N )
{
double *buffer[N];

unsigned int sizepart = N*sizeof( Position[0][0] );
unsigned int sizecharge = N*sizeof( Charge[0] );
unsigned int sizeid = N*sizeof( Id[0] );

for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
memcpy( buffer, &( ( *double_prop_[iprop] )[part1] ), sizepart );
memcpy( &( ( *double_prop_[iprop] )[part1] ), &( ( *double_prop_[iprop] )[part2] ), sizepart );
memcpy( &( ( *double_prop_[iprop] )[part2] ), buffer, sizepart );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
memcpy( buffer, &( ( *short_prop_[iprop] )[part1] ), sizecharge );
memcpy( &( ( *short_prop_[iprop] )[part1] ), &( ( *short_prop_[iprop] )[part2] ), sizecharge );
memcpy( &( ( *short_prop_[iprop] )[part2] ), buffer, sizecharge );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
memcpy( buffer, &( ( *uint64_prop_[iprop] )[part1] ), sizeid );
memcpy( &( ( *uint64_prop_[iprop] )[part1] ), &( ( *uint64_prop_[iprop] )[part2] ), sizeid );
memcpy( &( ( *uint64_prop_[iprop] )[part2] ), buffer, sizeid );
}

}

void Particles::pushToEnd( unsigned int iPart )
{

}

void Particles::createParticle()
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).push_back( 0. );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).push_back( 0 );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).push_back( 0 );
}
}

void Particles::createParticles( int n_additional_particles )
{
int nParticles = size();
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).resize( nParticles+n_additional_particles, 0. );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).resize( nParticles+n_additional_particles, 0 );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).resize( nParticles+n_additional_particles, 0 );
}

cell_keys.resize( nParticles+n_additional_particles, 0);

}

void Particles::createParticles( int n_additional_particles, int pstart )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).insert( ( *double_prop_[iprop] ).begin()+pstart, n_additional_particles, 0. );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).insert( ( *short_prop_[iprop] ).begin()+pstart, n_additional_particles, 0 );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).insert( ( *uint64_prop_[iprop] ).begin()+pstart, n_additional_particles, 0 );
}
}

void Particles::eraseParticlesWithMask( int istart, int iend, vector <int> & mask ) {

unsigned int idest = (unsigned int) istart;
unsigned int isrc = (unsigned int) istart;
while (isrc < (unsigned int) iend) {
if (mask[idest] < 0) {
if (mask[isrc] >= 0) {
overwriteParticle( isrc, idest);
cell_keys[idest] = cell_keys[isrc];
mask[idest] = 1;
mask[isrc] = -1;
idest++;
} else {
isrc++;
}
} else {
idest++;
isrc = idest;
}
}

resize(idest);
}

void Particles::eraseParticlesWithMask( int istart, int iend) {

unsigned int idest = (unsigned int) istart;
unsigned int isrc = (unsigned int) istart;
while (isrc < (unsigned int) iend) {
if (cell_keys[idest] < 0) {
if (cell_keys[isrc] >= 0) {
overwriteParticle( isrc, idest);
cell_keys[idest] = cell_keys[isrc];
idest++;
} else {
isrc++;
}
} else {
idest++;
isrc = idest;
}
}

resize(idest);
}



void Particles::moveParticles( int iPart, int new_pos )
{
for( unsigned int iprop=0 ; iprop<double_prop_.size() ; iprop++ ) {
( *double_prop_[iprop] ).insert( ( *double_prop_[iprop] ).begin()+new_pos,( *double_prop_[iprop] )[iPart]  );
}

for( unsigned int iprop=0 ; iprop<short_prop_.size() ; iprop++ ) {
( *short_prop_[iprop] ).insert( ( *short_prop_[iprop] ).begin()+new_pos, ( *short_prop_[iprop] )[iPart] );
}

for( unsigned int iprop=0 ; iprop<uint64_prop_.size() ; iprop++ ) {
( *uint64_prop_[iprop] ).insert( ( *uint64_prop_[iprop] ).begin()+new_pos,( *uint64_prop_[iprop] )[iPart]  );
}

eraseParticle( iPart+1 );
}


bool Particles::isParticleInDomain( unsigned int ipart, Patch *patch )
{
for( unsigned int i=0; i<Position.size(); i++ ) {
if( Position[i][ipart] <  patch->getDomainLocalMin( i )
|| Position[i][ipart] >= patch->getDomainLocalMax( i ) ) {
return false;
}
}
return true;
}


void Particles::sortById()
{
if( !tracked ) {
ERROR( "Impossible" );
return;
}
int nParticles( Weight.size() );

bool stop;
int jPart( 0 );
do {
stop = true;
for( int iPart = nParticles-1 ; iPart > jPart ; --iPart ) {
if( Id[iPart] < Id[iPart-1] ) {
swapParticle( iPart, jPart );
stop = false;
}
}
jPart++;
} while( !stop );

}

void Particles::savePositions() {
unsigned int ndim = Position.size(), npart = size();
double *p[3], *pold[3];
for( unsigned int i = 0 ; i<ndim ; i++ ) {
p[i] =  &( Position[i][0] );
pold[i] =  &( Position_old[i][0] );
}
if (ndim == 1) {
#pragma omp simd
for( unsigned int ipart=0 ; ipart<npart; ipart++ ) {
pold[0][ipart] = p[0][ipart];
}
} else if (ndim == 2) {
#pragma omp simd
for( unsigned int ipart=0 ; ipart<npart; ipart++ ) {
pold[0][ipart] = p[0][ipart];
pold[1][ipart] = p[1][ipart];
}
} else if (ndim == 3) {
#pragma omp simd
for( unsigned int ipart=0 ; ipart<npart; ipart++ ) {
pold[0][ipart] = p[0][ipart];
pold[1][ipart] = p[1][ipart];
pold[2][ipart] = p[2][ipart];
}
}


}

#ifdef __DEBUG
bool Particles::testMove( int iPartStart, int iPartEnd, Params &params )
{
for( int iDim = 0 ; iDim < Position.size() ; iDim++ ) {
double dx2 = params.cell_length[iDim];
for( int iPart = iPartStart ; iPart < iPartEnd ; iPart++ ) {
if( dist( iPart, iDim ) > dx2 ) {
ERROR( "Too large displacment for particle : " << iPart << "\t: " << ( *this )( iPart ) );
return false;
}
}
}
return true;

}
#endif

Particle Particles::operator()( unsigned int iPart )
{
return  Particle( *this, iPart );
}
