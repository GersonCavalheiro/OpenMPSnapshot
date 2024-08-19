#include "PyTools.h"
#include "DiagnosticScalar.h"
#include "VectorPatch.h"
#include "Tools.h"

#include <iomanip>
#include <algorithm>
#include <limits>

using namespace std;


DiagnosticScalar::DiagnosticScalar( Params &params, SmileiMPI *smpi, Patch *patch = NULL ):
latest_timestep( -1 )
{

if( PyTools::nComponents( "DiagScalar" ) > 1 ) {
ERROR( "Only one DiagScalar can be specified" );
}

if( PyTools::nComponents( "DiagScalar" ) > 0 ) {

timeSelection = new TimeSelection(
PyTools::extract_py( "every", "DiagScalar", 0 ),
"Scalars"
);

precision = 10;
PyTools::extract( "precision", precision, "DiagScalar"  );
PyTools::extractV( "vars", vars, "DiagScalar" );

res_time       = params.res_time;
dt             = params.timestep;
cell_volume    = params.cell_volume;
n_space        = params.n_space;
n_space_global = params.n_space_global;

filename = "scalars.txt";
} else {
timeSelection = new TimeSelection();
filename = "";
}

} 


DiagnosticScalar::~DiagnosticScalar()
{
for( unsigned int i=0; i<allScalars.size(); i++ ) {
delete allScalars[i];
}
delete timeSelection;
} 


void DiagnosticScalar::openFile( Params &params, SmileiMPI *smpi )
{
if( !smpi->isMaster() || fout.is_open() ) {
return;
}

fout.open( "scalars.txt" );

if( !fout.is_open() ) {
ERROR( "Can't open scalar.txt file" );
}

} 


void DiagnosticScalar::closeFile()
{
if( fout.is_open() ) {
fout.close();
}

} 


unsigned int DiagnosticScalar::calculateWidth( string key )
{
return 2 + max( ( ( unsigned int )key.length() ), precision+8 );
}

Scalar_value *DiagnosticScalar::newScalar_SUM( string name )
{
bool allow = allowedKey( name );
unsigned int width = calculateWidth( name );
Scalar_value *scalar = new Scalar_value( name, width, allow, &values_SUM );
values_SUM.push_back( 0. );
allScalars.push_back( scalar );
return scalar;
}
Scalar_value_location *DiagnosticScalar::newScalar_MINLOC( string name )
{
bool allow = allowedKey( name );
unsigned int width = calculateWidth( name );
val_index default_;
default_.index = -1;
default_.val = 0.;
Scalar_value_location *scalar = new Scalar_value_location( name, name+"Cell", width, allow, &values_MINLOC, numeric_limits<double>::max() );
values_MINLOC.push_back( default_ );
allScalars.push_back( scalar );
return scalar;
}
Scalar_value_location *DiagnosticScalar::newScalar_MAXLOC( string name )
{
bool allow = allowedKey( name );
unsigned int width = calculateWidth( name );
val_index default_;
default_.index = -1;
default_.val = 0.;
Scalar_value_location *scalar = new Scalar_value_location( name, name+"Cell", width, allow, &values_MAXLOC, numeric_limits<double>::lowest() );
values_MAXLOC.push_back( default_ );
allScalars.push_back( scalar );
return scalar;
}


void DiagnosticScalar::init( Params &params, SmileiMPI *smpi, VectorPatch &vecPatches )
{

ElectroMagn *EMfields = vecPatches( 0 )->EMfields;
vector<string> fields;
unsigned int nmodes = 1;
if( params.geometry != "AMcylindrical" ) {
fields.push_back( EMfields->Ex_ ->name );
fields.push_back( EMfields->Ey_ ->name );
fields.push_back( EMfields->Ez_ ->name );
fields.push_back( EMfields->Bx_m->name );
fields.push_back( EMfields->By_m->name );
fields.push_back( EMfields->Bz_m->name );
fields.push_back( EMfields->Jx_ ->name );
fields.push_back( EMfields->Jy_ ->name );
fields.push_back( EMfields->Jz_ ->name );
fields.push_back( EMfields->rho_->name );
if( params.Laser_Envelope_model ) {
fields.push_back( EMfields->Env_A_abs_->name );
fields.push_back( EMfields->Env_Chi_->name );
fields.push_back( EMfields->Env_E_abs_->name );
fields.push_back( EMfields->Env_Ex_abs_->name );
}
} else {
ElectroMagnAM *emfields = static_cast<ElectroMagnAM *>( EMfields );
nmodes = emfields->El_.size();
for( unsigned int imode=0; imode < nmodes; imode++ ) {
fields.push_back( emfields->El_[imode] ->name );
fields.push_back( emfields->Er_[imode] ->name );
fields.push_back( emfields->Et_[imode] ->name );
fields.push_back( emfields->Bl_m[imode]->name );
fields.push_back( emfields->Br_m[imode]->name );
fields.push_back( emfields->Bt_m[imode]->name );
}
for( unsigned int imode=0; imode < nmodes; imode++ ) {
fields.push_back( emfields->Jl_[imode]->name );
fields.push_back( emfields->Jr_[imode]->name );
fields.push_back( emfields->Jt_[imode]->name );
fields.push_back( emfields->rho_AM_[imode]->name );
}

}


necessary_Ubal_norm  = allowedKey( "Ubal_norm" );
necessary_Ubal       = necessary_Ubal_norm || allowedKey( "Ubal" );
necessary_Utot       = necessary_Ubal || allowedKey( "Utot" );
necessary_Uexp       = necessary_Ubal || allowedKey( "Uexp" );
necessary_Ukin       = necessary_Utot || allowedKey( "Ukin" );
necessary_Uelm       = necessary_Utot || allowedKey( "Uelm" );
necessary_Urad       = necessary_Utot || allowedKey( "Urad" );
necessary_UmBWpairs  = allowedKey( "UmBWpairs" );
necessary_Ukin_BC = necessary_Uexp || allowedKey( "Ukin_bnd" ) || allowedKey( "Ukin_out_mvw" ) || allowedKey( "Ukin_inj_mvw" );
necessary_Uelm_BC = necessary_Uexp || allowedKey( "Uelm_bnd" ) || allowedKey( "Uelm_out_mvw" ) || allowedKey( "Uelm_inj_mvw" );
unsigned int nspec = vecPatches( 0 )->vecSpecies.size();
necessary_species.resize( nspec, false );
string species_name;
for( unsigned int ispec=0; ispec<nspec; ispec++ ) {
if( ! vecPatches( 0 )->vecSpecies[ispec]->particles->is_test ) {
species_name = vecPatches( 0 )->vecSpecies[ispec]->name_;
necessary_species[ispec] = necessary_Ukin || allowedKey( Tools::merge( "Dens_", species_name ) )
|| allowedKey( Tools::merge( "Ntot_", species_name ) )
|| allowedKey( Tools::merge( "Zavg_", species_name ) )
|| allowedKey( Tools::merge( "Ukin_", species_name ) )
|| allowedKey( Tools::merge( "Urad_", species_name ) )
|| allowedKey( "UmBWpairs" );
}
}
unsigned int nfield = nmodes * 6;
necessary_fieldUelm.resize( nfield, false );
for( unsigned int ifield=0; ifield<nfield; ifield++ ) {
necessary_fieldUelm[ifield] = necessary_Uelm || allowedKey( Tools::merge( "Uelm_", fields[ifield] ) );
}
nfield = fields.size();
necessary_fieldMinMax.resize( nfield, false );
necessary_fieldMinMax_any = false;
for( unsigned int ifield=0; ifield<nfield; ifield++ ) {
necessary_fieldMinMax[ifield] =
allowedKey( Tools::merge( fields[ifield], "Min" ) )
|| allowedKey( Tools::merge( fields[ifield], "MinCell" ) )
|| allowedKey( Tools::merge( fields[ifield], "Max" ) )
|| allowedKey( Tools::merge( fields[ifield], "MaxCell" ) );
if( necessary_fieldMinMax[ifield] ) {
necessary_fieldMinMax_any = true;
}
}
unsigned int npoy  = EMfields->poynting[0].size() * EMfields->poynting[1].size();
necessary_poy.resize( npoy );
string poy_name;
unsigned int k = 0;
for( unsigned int j=0; j<2; j++ ) {
for( unsigned int i=0; i<EMfields->poynting[j].size(); i++ ) {
poy_name = Tools::merge( "Poy", Tools::xyz[i], j==0?"min":"max" );
necessary_poy[k] = necessary_Uelm_BC || allowedKey( poy_name ) || allowedKey( poy_name+"Inst" );
k++;
}
}


values_SUM   .reserve( 13 + nspec*5 + 6 + 2*npoy );

if( !params.Laser_Envelope_model ) {
values_MINLOC.reserve( 10 );
values_MAXLOC.reserve( 10 );
} else {
values_MINLOC.reserve( 14 );
values_MAXLOC.reserve( 14 );
}

Ubal_norm    = newScalar_SUM( "Ubal_norm" );
Ubal         = newScalar_SUM( "Ubal" );
Utot         = newScalar_SUM( "Utot" );
Uexp         = newScalar_SUM( "Uexp" );
Ukin         = newScalar_SUM( "Ukin" );
Urad         = newScalar_SUM( "Urad" );          
UmBWpairs    = newScalar_SUM( "UmBWpairs" );     
Uelm         = newScalar_SUM( "Uelm" );
Ukin_bnd     = newScalar_SUM( "Ukin_bnd" );
Ukin_out_mvw = newScalar_SUM( "Ukin_out_mvw" );
Ukin_inj_mvw = newScalar_SUM( "Ukin_inj_mvw" );
Ukin_new     = newScalar_SUM( "Ukin_new" );
Uelm_bnd     = newScalar_SUM( "Uelm_bnd" );
Uelm_out_mvw = newScalar_SUM( "Uelm_out_mvw" );
Uelm_inj_mvw = newScalar_SUM( "Uelm_inj_mvw" );

sDens.resize( nspec, NULL );
sNtot.resize( nspec, NULL );
sZavg.resize( nspec, NULL );
sUkin.resize( nspec, NULL );
sUrad.resize( nspec, NULL );
for( unsigned int ispec=0; ispec<nspec; ispec++ ) {
if( ! vecPatches( 0 )->vecSpecies[ispec]->particles->is_test ) {
species_name = vecPatches( 0 )->vecSpecies[ispec]->name_;
sDens[ispec] = newScalar_SUM( Tools::merge( "Dens_", species_name ) );
sNtot[ispec] = newScalar_SUM( Tools::merge( "Ntot_", species_name ) );
sZavg[ispec] = newScalar_SUM( Tools::merge( "Zavg_", species_name ) );
sUkin[ispec] = newScalar_SUM( Tools::merge( "Ukin_", species_name ) );
sUrad[ispec] = newScalar_SUM( Tools::merge( "Urad_", species_name ) );
}
}

nfield = nmodes * 6;
fieldUelm.resize( nfield, NULL );
for( unsigned int ifield=0; ifield<nfield; ifield++ ) {
fieldUelm[ifield] = newScalar_SUM( Tools::merge( "Uelm_", fields[ifield] ) );
}

nfield = ( params.geometry == "AMcylindrical" ) ? 0 : fields.size();
fieldMin.resize( nfield, NULL );
fieldMax.resize( nfield, NULL );
for( unsigned int ifield=0; ifield<nfield; ifield++ ) {
if( necessary_fieldMinMax[ifield] ) {
fieldMin[ifield] = newScalar_MINLOC( Tools::merge( fields[ifield], "Min" ) );
fieldMax[ifield] = newScalar_MAXLOC( Tools::merge( fields[ifield], "Max" ) );
}
}

poy    .resize( 2*npoy, NULL );
poyInst.resize( 2*npoy, NULL );
k = 0;
for( unsigned int j=0; j<2; j++ ) {
for( unsigned int i=0; i<EMfields->poynting[j].size(); i++ ) {
poy_name = Tools::merge( "Poy", Tools::xyz[i], j==0?"min":"max" );
poy    [k] = newScalar_SUM( poy_name );
poyInst[k] = newScalar_SUM( Tools::merge( poy_name, "Inst" ) );
k++;
}
}
}



bool DiagnosticScalar::prepare( int itime )
{
if( timeSelection->theTimeIsNow( itime ) ) {
for( unsigned int iscalar=0 ; iscalar<allScalars.size() ; iscalar++ ) {
allScalars[iscalar]->reset();
}
}

return true;
} 


void DiagnosticScalar::run( Patch *patch, int itime, SimWindow *simWindow )
{

if( ! filename.empty() ) {
patch->computePoynting();
}

if( timeSelection->theTimeIsNow( itime ) && itime>latest_timestep ) {
compute( patch, itime );
}

} 


void DiagnosticScalar::write( int itime, SmileiMPI *smpi )
{
if( smpi->isMaster() ) {

if( timeSelection->theTimeIsNow( itime ) && itime>latest_timestep ) {

unsigned int j, k, s = allScalars.size();

fout << std::scientific << setprecision( precision );
if( fout.tellp()==ifstream::pos_type( 0 ) ) { 
fout << "# " << 1 << " time" << endl;
j = 2;
for( k=0; k<s; k++ ) {
if( allScalars[k]->allowed_ ) {
fout << "# " << j << " " << allScalars[k]->name_ << endl;
j++;
if( ! allScalars[k]->secondname_.empty() ) {
fout << "# " << j << " " << allScalars[k]->secondname_ << endl;
j++;
}
}
}
fout << "#\n#" << setw( precision+9 ) << "time";
for( k=0; k<s; k++ ) {
if( allScalars[k]->allowed_ ) {
fout << setw( allScalars[k]->width_ ) << allScalars[k]->name_;
if( ! allScalars[k]->secondname_.empty() ) {
fout << setw( allScalars[k]->width_ ) << allScalars[k]->secondname_;
}
}
}
fout << endl;
}
fout << setw( precision+10 ) << itime/res_time;
for( k=0; k<s; k++ ) {
if( allScalars[k]->allowed_ ) {
fout << setw( allScalars[k]->width_ ) << ( double )*allScalars[k];
if( ! allScalars[k]->secondname_.empty() ) {
fout << setw( allScalars[k]->width_ ) << ( int )*static_cast<Scalar_value_location *>( allScalars[k] );
}
}
}
fout << endl;

}

} 

latest_timestep = itime;

} 


void DiagnosticScalar::compute( Patch *patch, int itime )
{
ElectroMagn *EMfields = patch->EMfields;
std::vector<Species *> &vecSpecies = patch->vecSpecies;

ElectroMagnAM *AMfields = dynamic_cast<ElectroMagnAM *>( patch->EMfields );
bool AM = AMfields;

double Ukin_=0.;             
double Urad_=0.;             
double UmBWpairs_=0;         
double Ukin_bnd_=0.;         
double Ukin_out_mvw_=0.;     
double Ukin_inj_mvw_=0.;     
double Ukin_new_=0.;         

for( unsigned int ispec=0; ispec<vecSpecies.size(); ispec++ ) {
if( vecSpecies[ispec]->particles->is_test ) {
continue;    
}

if( necessary_species[ispec] ) {
double density=0.0;  
double charge=0.0;   
double ener_tot=0.0; 

unsigned int nPart=vecSpecies[ispec]->getNbrOfParticles(); 

if( vecSpecies[ispec]->mass_ > 0 ) {

for( unsigned int iPart=0 ; iPart<nPart; iPart++ ) {

density  += vecSpecies[ispec]->particles->weight( iPart );
charge   += vecSpecies[ispec]->particles->weight( iPart )
* ( double )vecSpecies[ispec]->particles->charge( iPart );
ener_tot += vecSpecies[ispec]->particles->weight( iPart )
* ( vecSpecies[ispec]->particles->LorentzFactor( iPart )-1.0 );
}
ener_tot *= vecSpecies[ispec]->mass_;
} else if( vecSpecies[ispec]->mass_ == 0 ) {
for( unsigned int iPart=0 ; iPart<nPart; iPart++ ) {

density  += vecSpecies[ispec]->particles->weight( iPart );
ener_tot += vecSpecies[ispec]->particles->weight( iPart )
* ( vecSpecies[ispec]->particles->momentumNorm( iPart ) );
}
}

*sNtot[ispec] += ( double )nPart;
*sDens[ispec] += density;
*sZavg[ispec] += charge;
*sUkin[ispec] += ener_tot;

Ukin_ += ener_tot;

if( vecSpecies[ispec]->Radiate ) {
*sUrad[ispec] += vecSpecies[ispec]->nrj_radiated_;
Urad_         += vecSpecies[ispec]->nrj_radiated_;
}

if( vecSpecies[ispec]->Multiphoton_Breit_Wheeler_process ) {
UmBWpairs_ += vecSpecies[ispec]->nrj_radiated_;
}
}

if( necessary_Ukin_BC ) {
Ukin_bnd_     += vecSpecies[ispec]->nrj_bc_lost;
Ukin_out_mvw_ += vecSpecies[ispec]->nrj_mw_out;
Ukin_inj_mvw_ += vecSpecies[ispec]->nrj_mw_inj;
}

Ukin_new_ += vecSpecies[ispec]->nrj_new_part_;

} 

if( necessary_Ukin ) {
*Ukin += Ukin_;
}
if( necessary_Urad ) {
*Urad += Urad_;
}
if( necessary_UmBWpairs ) {
*UmBWpairs += UmBWpairs_;
}
if( necessary_Ukin_BC ) {
*Ukin_bnd     += Ukin_bnd_     ;
*Ukin_out_mvw += Ukin_out_mvw_ ;
*Ukin_inj_mvw += Ukin_inj_mvw_ ;
}
*Ukin_new += Ukin_new_;


vector<Field *> fields;

if( !AM ) {
fields.push_back( EMfields->Ex_ );
fields.push_back( EMfields->Ey_ );
fields.push_back( EMfields->Ez_ );
fields.push_back( EMfields->Bx_m );
fields.push_back( EMfields->By_m );
fields.push_back( EMfields->Bz_m );
} else {
unsigned int nmodes = AMfields->El_.size();
for( unsigned int imode=0 ; imode < nmodes ; imode++ ) {
fields.push_back( AMfields->El_[imode] );
fields.push_back( AMfields->Er_[imode] );
fields.push_back( AMfields->Et_[imode] );
fields.push_back( AMfields->Bl_m[imode] );
fields.push_back( AMfields->Br_m[imode] );
fields.push_back( AMfields->Bt_m[imode] );
}
}

double Uelm_ = 0.0; 

unsigned int nfield = fields.size();
for( unsigned int ifield=0; ifield<nfield; ifield++ ) {
if( necessary_fieldUelm[ifield] ) {
Field *field = fields[ifield];

double Uem = 0.;
if( ! AM ) {
Uem = field->norm2( EMfields->istart, EMfields->bufsize );
} else {
Uem = 0.5 * AMfields->dr * dynamic_cast<cField2D*>( field )->norm2_cylindrical( AMfields->istart, AMfields->bufsize, AMfields->j_glob_ );
if( ifield < 6 ) {
Uem *= 2.; 
}
}
Uem *= 0.5*cell_volume;

*fieldUelm[ifield] += Uem;
Uelm_ += Uem;
}
}

if( necessary_Uelm ) {
*Uelm += Uelm_;
}

if( necessary_Uelm_BC ) {
*Uelm_out_mvw += EMfields->nrj_mw_out * 0.5*cell_volume;

*Uelm_inj_mvw += EMfields->nrj_mw_inj * 0.5*cell_volume;
}



fields.push_back( EMfields->Jx_ );
fields.push_back( EMfields->Jy_ );
fields.push_back( EMfields->Jz_ );
fields.push_back( EMfields->rho_ );

if( EMfields->Env_A_abs_ != NULL ) {
fields.push_back( EMfields->Env_A_abs_ );
fields.push_back( EMfields->Env_Chi_ );
fields.push_back( EMfields->Env_E_abs_ );
fields.push_back( EMfields->Env_Ex_abs_ );
}

double fieldval;
unsigned int i_min, j_min, k_min;
unsigned int i_max, j_max, k_max;
val_index minloc, maxloc;

nfield = fields.size();

if( AM ){
nfield = 0;
}

for( unsigned int ifield=0; ifield<nfield; ifield++ ) {

if( necessary_fieldMinMax[ifield] ) {

Field *field = fields[ifield];

vector<unsigned int> iFieldStart( 3, 0 ), iFieldEnd( 3, 1 ), iFieldGlobalSize( 3, 1 );
for( unsigned int i=0 ; i<field->isDual_.size() ; i++ ) {
iFieldStart[i] = EMfields->istart[i][field->isDual( i )];
iFieldEnd [i] = iFieldStart[i] + EMfields->bufsize[i][field->isDual( i )];
iFieldGlobalSize [i] = field->dims_[i];
}

unsigned int iifield= iFieldStart[2] + iFieldStart[1]*iFieldGlobalSize[2] +iFieldStart[0]*iFieldGlobalSize[1]*iFieldGlobalSize[2];
minloc.val = maxloc.val = ( *field )( iifield );
i_min = iFieldStart[0];
j_min = iFieldStart[1];
k_min = iFieldStart[2];
i_max = iFieldStart[0];
j_max = iFieldStart[1];
k_max = iFieldStart[2];

for( unsigned int k=iFieldStart[2]; k<iFieldEnd[2]; k++ ) {
for( unsigned int j=iFieldStart[1]; j<iFieldEnd[1]; j++ ) {
for( unsigned int i=iFieldStart[0]; i<iFieldEnd[0]; i++ ) {
unsigned int ii = k+ ( j + i*iFieldGlobalSize[1] ) *iFieldGlobalSize[2];
fieldval = ( *field )( ii );
if( minloc.val > fieldval ) {
minloc.val = fieldval;
i_min=i;
j_min=j;
k_min=k;
}
if( maxloc.val < fieldval ) {
maxloc.val = fieldval;
i_max=i;
j_max=j;
k_max=k;
}
}
}
}

i_min += patch->Pcoordinates[0]*n_space[0] - iFieldStart[0];
j_min += patch->Pcoordinates[1]*n_space[1] - iFieldStart[1];
k_min += patch->Pcoordinates[2]*n_space[2] - iFieldStart[2];
minloc.index = ( int )( i_min*n_space_global[1]*n_space_global[2] + j_min*n_space_global[2] + k_min );
i_max += patch->Pcoordinates[0]*n_space[0] - iFieldStart[0];
j_max += patch->Pcoordinates[1]*n_space[1] - iFieldStart[1];
k_max += patch->Pcoordinates[2]*n_space[2] - iFieldStart[2];
maxloc.index = ( int )( i_max*n_space_global[1]*n_space_global[2] + j_max*n_space_global[2] + k_max );

#pragma omp critical
{
if( minloc.val < ( double )*fieldMin[ifield] ) {
*fieldMin[ifield] = minloc;
}
if( maxloc.val > ( double )*fieldMax[ifield] ) {
*fieldMax[ifield] = maxloc;
}
}
}
}


double Uelm_bnd_=0.0;
unsigned int k=0;
for( unsigned int j=0; j<2; j++ ) { 
for( unsigned int i=0; i<EMfields->poynting[j].size(); i++ ) { 
if( necessary_poy[k] ) {
*poy    [k] += EMfields->poynting     [j][i];
*poyInst[k] += EMfields->poynting_inst[j][i];
k++;
}

Uelm_bnd_ += EMfields->poynting[j][i];
}
}

if( necessary_Uelm_BC ) {
*Uelm_bnd += Uelm_bnd_;
}

} 


double DiagnosticScalar::getScalar( std::string key )
{
unsigned int k, s=allScalars.size();
for( k=0; k<s; k++ ) {
if( allScalars[k]->name_      ==key ) {
return ( double )*allScalars[k];
}
if( allScalars[k]->secondname_==key ) {
return ( int )   *allScalars[k];
}
}
DEBUG( "key not found " << key );
return 0.0;

} 

bool DiagnosticScalar::allowedKey( string key )
{
if( key.empty() ) {
return false;
}
int s=vars.size();
if( s==0 ) {
return true;
}
for( int i=0; i<s; i++ ) {
if( key==vars[i] ) {
return true;
}
}
return false;
}


bool DiagnosticScalar::needsRhoJs( int itime )
{
return timeSelection->theTimeIsNow( itime );
}

uint64_t DiagnosticScalar::getDiskFootPrint( int istart, int istop, Patch *patch )
{
uint64_t footprint = 0;

uint64_t ndumps = timeSelection->howManyTimesBefore( istop ) - timeSelection->howManyTimesBefore( istart );

if( ndumps == 0 ) {
return 0;
}

vector<string> scalars = {"Ubal_norm", "Ubal", "Utot", "Uexp", "Ukin", "Urad", "UmBWpairs", "Uelm", "Ukin_bnd", "Ukin_out_mvw", "Ukin_inj_mvw", "Ukin_new", "Uelm_bnd", "Uelm_out_mvw", "Uelm_inj_mvw"};
unsigned int nspec = patch->vecSpecies.size();
for( unsigned int ispec=0; ispec<nspec; ispec++ ) {
if( ! patch->vecSpecies[ispec]->particles->is_test ) {
string species_name = patch->vecSpecies[ispec]->name_;
scalars.push_back( Tools::merge( "Dens_", species_name ) );
scalars.push_back( Tools::merge( "Ntot_", species_name ) );
scalars.push_back( Tools::merge( "Zavg_", species_name ) );
scalars.push_back( Tools::merge( "Ukin_", species_name ) );
scalars.push_back( Tools::merge( "Urad_", species_name ) );
}
}
if( !dynamic_cast<ElectroMagnAM *>( patch->EMfields ) ) {
scalars.push_back( Tools::merge( "Uelm_", patch->EMfields->Ex_ ->name ) );
scalars.push_back( Tools::merge( "Uelm_", patch->EMfields->Ey_ ->name ) );
scalars.push_back( Tools::merge( "Uelm_", patch->EMfields->Ez_ ->name ) );
scalars.push_back( Tools::merge( "Uelm_", patch->EMfields->Bx_m->name ) );
scalars.push_back( Tools::merge( "Uelm_", patch->EMfields->By_m->name ) );
scalars.push_back( Tools::merge( "Uelm_", patch->EMfields->Bz_m->name ) );
} else {
ElectroMagnAM *emfields = static_cast<ElectroMagnAM *>( patch->EMfields );
unsigned int nmodes = emfields->El_.size();
for( unsigned int imode=0 ; imode < nmodes ; imode++ ) {
scalars.push_back( Tools::merge( "Uelm_", emfields->El_[imode] ->name ) );
scalars.push_back( Tools::merge( "Uelm_", emfields->Er_[imode] ->name ) );
scalars.push_back( Tools::merge( "Uelm_", emfields->Et_[imode] ->name ) );
scalars.push_back( Tools::merge( "Uelm_", emfields->Bl_m[imode]->name ) );
scalars.push_back( Tools::merge( "Uelm_", emfields->Br_m[imode]->name ) );
scalars.push_back( Tools::merge( "Uelm_", emfields->Bt_m[imode]->name ) );
}
}
#ifdef _AM__TODO
for( unsigned int i=0; i<2; i++ ) {
string minmax = ( i==0 ) ? "Min" : "Max";
for( unsigned int j=0; j<2; j++ ) {
string cell = ( j==0 ) ? "" : "Cell";
scalars.push_back( Tools::merge( patch->EMfields->Ex_ ->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Ey_ ->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Ez_ ->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Bx_m->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->By_m->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Bz_m->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Jx_ ->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Jy_ ->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Jz_ ->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->rho_->name, minmax, cell ) );
if( params.Laser_Envelope_model ) {
scalars.push_back( Tools::merge( patch->EMfields->Env_A_abs_->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Env_Chi_->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Env_E_abs_->name, minmax, cell ) );
scalars.push_back( Tools::merge( patch->EMfields->Env_Ex_abs_->name, minmax, cell ) );
}
}
}
#endif
unsigned int k = 0;
string poy_name;
for( unsigned int j=0; j<2; j++ ) {
for( unsigned int i=0; i<patch->EMfields->poynting[j].size(); i++ ) {
poy_name = Tools::merge( "Poy", Tools::xyz[i], j==0?"min":"max" );
scalars.push_back( poy_name );
scalars.push_back( Tools::merge( poy_name, "Inst" ) );
k++;
}
}
int nscalars = 1;
for( k=0; k<scalars.size(); k++ ) {
if( allowedKey( scalars[k] ) ) {
nscalars ++;
}
}

string s = "";
uint64_t linesize = nscalars * calculateWidth( s );

footprint += ( uint64_t )( nscalars * 10 + linesize );

footprint += ndumps * linesize;

return footprint;
}
