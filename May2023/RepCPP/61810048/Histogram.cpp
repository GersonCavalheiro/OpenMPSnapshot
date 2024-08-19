#include "PyTools.h"
#include "Histogram.h"
#include "Patch.h"
#include "ParticleData.h"

#include <algorithm>

using namespace std;

void Histogram::digitize( vector<Species *> species,
vector<double> &double_buffer,
vector<int>    &int_buffer,
SimWindow *simWindow )
{
unsigned int npart = double_buffer.size();

for( unsigned int iaxis=0 ; iaxis < axes.size() ; iaxis++ ) {

HistogramAxis * axis = axes[iaxis];

unsigned int istart = 0;
for( unsigned int ispec=0; ispec < species.size(); ispec++ ) {
unsigned int npart = species[ispec]->getNbrOfParticles();
axis->calculate_locations( species[ispec], &double_buffer[istart], &int_buffer[istart], npart, simWindow );
istart += npart;
}

if( axis->logscale ) {
for( unsigned int ipart = 0 ; ipart < npart ; ipart++ ) {
if( int_buffer[ipart] < 0 ) {
continue;
}
double_buffer[ipart] = log10( abs( double_buffer[ipart] ) );
}
}

double actual_min = axis->logscale ? log10( axis->global_min ) : axis->global_min;
double actual_max = axis->logscale ? log10( axis->global_max ) : axis->global_max;
double coeff = ( ( double ) axis->nbins )/( actual_max - actual_min );

if( iaxis>0 ) {
for( unsigned int ipart = 0 ; ipart < npart ; ipart++ ) {
int_buffer[ipart] *= axis->nbins;
}
}

if( !axis->edge_inclusive ) { 

for( unsigned int ipart = 0 ; ipart < npart ; ipart++ ) {
if( int_buffer[ipart] < 0 ) {
continue;
}
int ind = floor( ( double_buffer[ipart]-actual_min ) * coeff );
if( ind >= 0  &&  ind < axis->nbins ) {
int_buffer[ipart] += ind;
} else {
int_buffer[ipart] = -1;    
}
}

} else { 

for( unsigned int ipart = 0 ; ipart < npart ; ipart++ ) {
if( int_buffer[ipart] < 0 ) {
continue;
}
int ind = floor( ( double_buffer[ipart]-actual_min ) * coeff );
if( ind < 0 ) {
ind = 0;
}
if( ind >= axis->nbins ) {
ind = axis->nbins-1;
}
int_buffer[ipart] += ind;
}

}

} 
}

void Histogram::distribute(
std::vector<double> &double_buffer,
std::vector<int>    &int_buffer,
std::vector<double> &output_array )
{

unsigned int ipart, npart=double_buffer.size();
int ind;

for( ipart = 0 ; ipart < npart ; ipart++ ) {
ind = int_buffer[ipart];
if( ind<0 ) {
continue;    
}
#pragma omp atomic
output_array[ind] += double_buffer[ipart];
}

}



void HistogramAxis::init( string type_, double min_, double max_, int nbins_, bool logscale_, bool edge_inclusive_, vector<double> coefficients_ )
{
type           = type_          ;
min            = min_           ;
max            = max_           ;
nbins          = nbins_         ;
logscale       = logscale_      ;
edge_inclusive = edge_inclusive_;
coefficients   = coefficients_  ;
global_min     = min;
global_max     = max;
}
