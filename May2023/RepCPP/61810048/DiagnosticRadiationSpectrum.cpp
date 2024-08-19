#include "PyTools.h"
#include <iomanip>

#include "DiagnosticRadiationSpectrum.h"
#include "HistogramFactory.h"
#include "RadiationTools.h"
#include "RadiationTables.h"


using namespace std;


DiagnosticRadiationSpectrum::DiagnosticRadiationSpectrum(
Params &params,
SmileiMPI *smpi,
Patch *patch,
RadiationTables *radiation_tables_,
int diagId
) : DiagnosticParticleBinningBase( params, smpi, patch, diagId, "RadiationSpectrum", false, PyUnicode_FromString( "" ), excludedAxes() )
{

ostringstream name( "" );
name << "DiagRadiationSpectrum #" << diagId;
string errorPrefix = name.str();

if (params.reference_angular_frequency_SI<=0.) {
ERROR("DiagRadiationSpectrum requires 'reference_angular_frequency_SI' to be defined.");
}

minimum_chi_continuous_ = radiation_tables_->getMinimumChiContinuous();

two_third = 2./3.;
double squared_fine_structure_constant = 5.325135447834466e-5;
double normalized_classical_electron_time = 9.399637140638142e-24*params.reference_angular_frequency_SI;
double factor  = two_third*squared_fine_structure_constant/normalized_classical_electron_time;
factor *= sqrt(3.)/2./M_PI;

vector<string> excluded_axes( 0 );
excluded_axes.push_back( "a" );
excluded_axes.push_back( "b" );
excluded_axes.push_back( "theta" );
excluded_axes.push_back( "phi" );

PyObject* photon_energy_axis = PyTools::extract_py( "photon_energy_axis", "DiagRadiationSpectrum", diagId );
ostringstream t("");
t << errorPrefix << "photon_energy_axis : ";
photon_axis = HistogramFactory::createAxis( photon_energy_axis, params, species_indices, patch, excluded_axes, t.str(), false );
total_axes++;
dims.push_back( photon_axis->nbins );

if( std::isnan( photon_axis->min ) || std::isnan( photon_axis->max ) ) {
ERROR( errorPrefix << "photon_energy_axis cannot have `auto` limits" );
}

photon_energies.resize( photon_axis->nbins );
delta_energies.resize( photon_axis->nbins );
emin = photon_axis->logscale ? log10( photon_axis->min ) : photon_axis->min;
emax = photon_axis->logscale ? log10( photon_axis->max ) : photon_axis->max;
double spacing = (emax-emin) / photon_axis->nbins;
for( int i=0; i<photon_axis->nbins; i++ ) {
photon_energies[i] = emin + (i+0.5)*spacing;
if( photon_axis->logscale ) {
photon_energies[i] = pow(10., photon_energies[i]);
delta_energies[i] = pow(10., emin+i*spacing) * ( pow(10., spacing) - 1. );
} else {
delta_energies[i] = spacing;
}
delta_energies[i] *= factor;
}

uint64_t total_size = (uint64_t)output_size * photon_axis->nbins;
if( total_size > 2147483648 ) { 
ERROR( errorPrefix << ": too many points (" << total_size << " > 2^31)" );
}
output_size = ( unsigned int ) total_size;

if( smpi->isMaster() ) {
MESSAGE( 2, photon_axis->info( "photon energy" ) );
}

} 


DiagnosticRadiationSpectrum::~DiagnosticRadiationSpectrum()
{
delete photon_axis;
} 


void DiagnosticRadiationSpectrum::openFile( Params& params, SmileiMPI* smpi )
{
if( !smpi->isMaster() || file_ ) {
return;
}

DiagnosticParticleBinningBase::openFile( params, smpi );

string str1 = "photon_energy_axis";
ostringstream mystream( "" );
mystream << photon_axis->min << " " << photon_axis->max << " "
<< photon_axis->nbins << " " << photon_axis->logscale << " " << photon_axis->edge_inclusive;
string str2 = mystream.str();
file_->attr( str1, str2 );

file_->flush();
}

void DiagnosticRadiationSpectrum::run( Patch* patch, int itime, SimWindow* simWindow )
{

unsigned int npart = 0;
vector<Species *> species;
for( unsigned int ispec=0 ; ispec < species_indices.size() ; ispec++ ) {
Species *s = patch->vecSpecies[species_indices[ispec]];
species.push_back( s );
npart += s->getNbrOfParticles();
}
vector<int> int_buffer( npart, 0 );
vector<double> double_buffer( npart );

histogram->digitize( species, double_buffer, int_buffer, simWindow );

unsigned int istart = 0;
for( unsigned int ispec=0 ; ispec < species_indices.size() ; ispec++ ) {


double gamma_inv, gamma, chi, xi, zeta, nu, cst;
double two_third_ov_chi, increment0, increment;
int iphoton_energy_max;
double coeff = ( ( double ) photon_axis->nbins )/( emax - emin );

Species *s = patch->vecSpecies[species_indices[ispec]];
unsigned int npart = s->getNbrOfParticles();
int *index = &int_buffer[istart];
for( unsigned int ipart = 0 ; ipart < npart ; ipart++ ) {
int ind = index[ipart];
if( ind < 0 ) continue; 
ind *= photon_axis->nbins;

chi = s->particles->chi( ipart );

if( chi <= minimum_chi_continuous_ ) continue;

gamma = s->particles->LorentzFactor( ipart );
gamma_inv = 1./gamma;
two_third_ov_chi = two_third/chi;
increment0 = gamma_inv * s->particles->weight( ipart );

if( photon_axis->logscale ) {
gamma = log10( gamma );
}
iphoton_energy_max = int( (gamma - emin) * coeff );
iphoton_energy_max = min( iphoton_energy_max, photon_axis->nbins );

for( int i=0; i<iphoton_energy_max; i++ ) {
xi   = photon_energies[i] * gamma_inv;
zeta = xi / (1.-xi); 
nu   = two_third_ov_chi * zeta;
cst  = xi * zeta;
increment = increment0 * delta_energies[i] * xi * RadiationTools::computeBesselPartsRadiatedPower(nu,cst);
#pragma omp atomic
data_sum[ind+i] += increment;
}
}

istart += npart;

}

} 



