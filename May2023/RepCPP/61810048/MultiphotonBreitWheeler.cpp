
#include "MultiphotonBreitWheeler.h"

void MultiphotonBreitWheeler::createTables(int argc, std::string * arguments)
{

int rank;
int number_of_ranks;

MPI_Comm_size( MPI_COMM_WORLD, &number_of_ranks );
MPI_Comm_rank( MPI_COMM_WORLD, &rank );

if( rank==0 ) {
std::cout << "\n You have selected the creation of tables"
<< " for the multiphoton Breit Wheeler process." << std::endl;
}


double t0;
double t1;

double min_photon_chi;
double max_photon_chi;

int size_particle_chi;
int size_photon_chi;

double particle_chi;
double photon_chi;

double log10_min_photon_chi;
double log10_max_photon_chi;

double log10_min_particle_chi;

double delta_particle_chi;
double delta_photon_chi;
double inverse_delta_photon_chi;

double xi_power;
double xi_threshold;

double xi;

int i_photon_chi;
int i_particle_chi;

int number_of_draws;

bool verbose;

std::vector <double> table_1d;
std::vector <double> table_2d;

size_particle_chi      = 128;
size_photon_chi        = 128;
min_photon_chi         = 1e-2;
max_photon_chi         = 1e2;
xi_power               = 5;
xi_threshold           = 1e-9;
number_of_draws        = 0;
verbose                = false;

std::string help_message;
help_message =  "\n Help page specific to the multiphoton Breit-Wheeler:\n";
help_message += "\n";
help_message += " List of available commands:\n";
help_message += " -h, --help                       print a help message and exit.\n";
help_message += " -s, --size       int int         respective size of the photon and particle chi axis. (default 128 128)\n";
help_message += " -b, --boundaries double double   min and max of the photon chi axis. (default 1e-2 1e2)\n";
help_message += " -e, --error      int             compute error due to discretization and use the provided int as a number of draws. (default 0)\n";
help_message += " -t, --threshold  double          Minimum targeted value of xi in the computation the minimum photon quantum parameter. (default 1e-3)\n";
help_message += " -p, --power      int             Maximum decrease in order of magnitude for the search for the minimum photon quantum parameter. (default 4)\n";
help_message += " -v, --verbose                    Dump the tables\n";

int i_arg = 2;
while(i_arg < argc) {
if (arguments[i_arg] == "-s" || arguments[i_arg] == "--size") {
size_photon_chi = std::stoi(arguments[i_arg+1]);
size_particle_chi = std::stoi(arguments[i_arg+2]);
i_arg+=3;
} else if (arguments[i_arg] == "-b" || arguments[i_arg] == "--boundaries") {
min_photon_chi = std::stod(arguments[i_arg+1]);
max_photon_chi = std::stod(arguments[i_arg+2]);
i_arg+=3;
} else if (arguments[i_arg] == "-e" || arguments[i_arg] == "--error") {
number_of_draws = std::stoi(arguments[i_arg+1]);
i_arg+=2;
} else if (arguments[i_arg] == "-t" || arguments[i_arg] == "--threshold") {
xi_threshold = std::stod(arguments[i_arg+1]);
i_arg+=2;
} else if (arguments[i_arg] == "-p" || arguments[i_arg] == "--power") {
xi_power = std::stod(arguments[i_arg+1]);
i_arg+=2;
} else if (arguments[i_arg] == "-v" || arguments[i_arg] == "--verbose") {
verbose = true;
i_arg+=1;
} else if (arguments[i_arg] == "-h" || arguments[i_arg] == "--help") {
if (rank == 0) {
std::cerr << help_message << std::endl;
}
exit(0);
i_arg+=2;
} else {
ERROR("Keywork " << arguments[i_arg] << " not recognized");
}
}

log10_min_photon_chi = std::log10(min_photon_chi);
log10_max_photon_chi = std::log10(max_photon_chi);

delta_photon_chi = (log10_max_photon_chi - log10_min_photon_chi) / (size_photon_chi - 1);

inverse_delta_photon_chi = 1.0 / delta_photon_chi;

table_1d.resize(size_photon_chi);
table_2d.resize(size_photon_chi*size_particle_chi);

if( rank==0 ) {
std::cout << "\n Size photon chi axis: " << size_photon_chi << "\n"
<< " Size particle chi axis: " << size_particle_chi << "\n"
<< " Min photon chi axis: " << min_photon_chi << "\n"
<< " Max photon chi axis: " << max_photon_chi << "\n"
<< " Power: " << xi_power << "\n"
<< " Threshold: " << xi_threshold
<< std::endl;
}


int *rank_first_index = new int[number_of_ranks];
int *rank_indexes = new int[number_of_ranks];

Tools::distributeArray( number_of_ranks,
size_photon_chi,
rank_first_index,
rank_indexes );

if( rank==0 ) {
std::cout << std::endl;
std::cout << " MPI load distribution:" << std::endl;
for( int i =0 ; i < number_of_ranks ; i++ ) {
std::cout << " - Rank " << i
<< " - 1st index: " << rank_first_index[i]
<< " - length: " << rank_indexes[i]
<< std::endl;
}
}


hid_t       fileId;
std::string path;

path = "./multiphoton_Breit_Wheeler_tables.h5";

if (rank == 0) {
remove(path.c_str());
}


double * buffer = new double [rank_indexes[rank]];

if( rank==0 ) {
std::cout << std::endl;
std::cout << " Computation of T"
<< std::endl;
}

double delta_percentage = std::max( delta_percentage, 100.0/rank_indexes[rank] );
double percentage = 0;

t0 = MPI_Wtime();

for( i_photon_chi = 0 ; i_photon_chi < rank_indexes[rank] ; i_photon_chi++ ) {

photon_chi = std::pow( 10.0, ( rank_first_index[rank] + i_photon_chi )*delta_photon_chi
+ log10_min_photon_chi );

buffer[i_photon_chi] = 2.0*MultiphotonBreitWheeler::computeIntegrationRitusDerivative( photon_chi, 0.5*photon_chi, 200, 1e-15 );

t1 = MPI_Wtime();

if( rank==0 ) {
if( 100.0*(i_photon_chi+1) >= rank_indexes[rank]*percentage ) {
percentage += delta_percentage;
std::cout << " - " << i_photon_chi + 1<< "/" << rank_indexes[rank]
<< " - " << ( int )( std::round( percentage ) ) << "%"
<< " - " << t1 - t0 << " s"
<< std::endl;
}
}
}

MPI_Allgatherv( &buffer[0], rank_indexes[rank], MPI_DOUBLE,
&table_1d[0], &rank_indexes[0], &rank_first_index[0],
MPI_DOUBLE, MPI_COMM_WORLD );

t1 = MPI_Wtime();
if (rank==0) {
std::cout << " Total time: " << t1 - t0 << " s" << std::endl;
}

if (number_of_draws > 0) {
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(1e-1,max_photon_chi);
double value;
double interpolated_value;
double error;
double local_max_error = 0;
double max_error = 0;
double value_for_max_error;
double interpolated_value_for_max_error;
double distance;
int index;

if (rank==0) std::cout << " Error computation: " << std::endl;

for(int i = 0 ; i < number_of_draws; i++) {
photon_chi = distribution(generator);
value = MultiphotonBreitWheeler::computeIntegrationRitusDerivative( photon_chi, 0.5*photon_chi, 200, 1e-15 );
i_photon_chi = int((log10(photon_chi) - log10_min_photon_chi) * inverse_delta_photon_chi);
distance = std::abs(log10(photon_chi) - (i_photon_chi*delta_photon_chi + log10_min_photon_chi)) * inverse_delta_photon_chi;
interpolated_value = table_1d[i_photon_chi]*(1 - distance) + table_1d[i_photon_chi+1]*distance;
error = std::abs(value - interpolated_value)/value;
if (error > local_max_error) {
local_max_error = error;
index = i;
value_for_max_error = value;
interpolated_value_for_max_error = interpolated_value;
}
}

MPI_Reduce(&local_max_error,&max_error,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

if (rank==0) {
std::cout << " - Maximal relative error: " << max_error << " at index " << index
<< " " << value_for_max_error << " " << interpolated_value_for_max_error
<< std::endl;
}
}

if (verbose && rank == 0) {
std:: cout << "\n table integration_dt_dchi: " << std::setprecision(std::numeric_limits<double>::digits10 + 1) <<std::endl;
for( i_photon_chi = 0 ; i_photon_chi < size_photon_chi  ; i_photon_chi += 8 ) {
std::cout << " ";
for (int i = i_photon_chi ; i< std::min(i_photon_chi+8,size_photon_chi) ; i++) {
std::cout << table_1d[i] << ", ";
}
std::cout << std::endl;
}
}

if (rank==0) {
fileId  = H5Fcreate( path.c_str(),
H5F_ACC_TRUNC,
H5P_DEFAULT,
H5P_DEFAULT );

std::string vect_name("integration_dt_dchi");

H5::vect( fileId, vect_name, table_1d, 0 );

std::string attr_name("min_photon_chi");
H5::attr( fileId, vect_name, attr_name, min_photon_chi);

attr_name = "max_photon_chi";
H5::attr( fileId, vect_name, attr_name, max_photon_chi);

attr_name = "size_photon_chi";
H5::attr( fileId, vect_name, attr_name, size_photon_chi);

H5Fclose( fileId );
}


if( rank==0 ) {
std::cout << std::endl;
std::cout << " Computation of the minimum particle chi value for the xi table"
<< std::endl;
}

double denominator;
double numerator;
int k;

percentage = 0;
t0 = MPI_Wtime();

for( i_photon_chi = 0 ; i_photon_chi < rank_indexes[rank] ; i_photon_chi++ ) {

xi = 1;

log10_min_particle_chi = ( rank_first_index[rank] + i_photon_chi )*delta_photon_chi
+ log10_min_photon_chi;

photon_chi = std::pow( 10.0, log10_min_particle_chi );

log10_min_particle_chi = std::log10( 0.5*photon_chi );

denominator = MultiphotonBreitWheeler::computeIntegrationRitusDerivative( photon_chi,
0.5*photon_chi, 200, 1e-15 );

k = 0;
while( k < xi_power ) {
log10_min_particle_chi -= std::pow( 0.1, k );
particle_chi = std::pow( 10.0, log10_min_particle_chi );
numerator = MultiphotonBreitWheeler::computeIntegrationRitusDerivative( photon_chi,
particle_chi, 200, 1e-15 );

if( numerator == 0 || denominator == 0 ) {
xi = 0;
} else {
xi = numerator/( 2.0*denominator );
}

if( xi < xi_threshold ) {
log10_min_particle_chi += std::pow( 0.1, k );
k += 1;
}
}
buffer[i_photon_chi] = log10_min_particle_chi;

t1 = MPI_Wtime();

if( rank==0 ) {
if( 100.0*(i_photon_chi+1) >= rank_indexes[rank]*percentage ) {
percentage += delta_percentage;
std::cout << " - " << i_photon_chi + 1<< "/" << rank_indexes[rank]
<< " - " << ( int )( std::round( percentage ) ) << "%"
<< " - " << t1 - t0 << " s"
<< std::endl;
}
}

}

MPI_Allgatherv( &buffer[0], rank_indexes[rank], MPI_DOUBLE,
&table_1d[0], &rank_indexes[0], &rank_first_index[0],
MPI_DOUBLE, MPI_COMM_WORLD );

t1 = MPI_Wtime();
if (rank==0) {
std::cout << " Total time: " << t1 - t0 << " s" << std::endl;
}

if (verbose && rank == 0) {
std:: cout << "\n table min_particle_chi_for_xi: " << std::setprecision(std::numeric_limits<double>::digits10 + 1) <<std::endl;
for( i_photon_chi = 0 ; i_photon_chi < size_photon_chi  ; i_photon_chi += 8 ) {
std::cout << " ";
for (int i = i_photon_chi ; i< std::min(i_photon_chi+8,size_photon_chi) ; i++) {
std::cout << table_1d[i] << ", ";
}
std::cout << std::endl;
}
}

if (rank==0) {
fileId = H5Fopen( path.c_str(),
H5F_ACC_RDWR,
H5P_DEFAULT );

std::string vect_name("min_particle_chi_for_xi");

H5::vect( fileId, vect_name, table_1d, 0 );

std::string attr_name("min_photon_chi");
H5::attr( fileId, vect_name, attr_name, min_photon_chi);

attr_name = "max_photon_chi";
H5::attr( fileId, vect_name, attr_name, max_photon_chi);

attr_name = "size_photon_chi";
H5::attr( fileId, vect_name, attr_name, size_photon_chi);

attr_name = "power";
H5::attr( fileId, vect_name, attr_name, xi_power);

attr_name = "threshold";
H5::attr( fileId, vect_name, attr_name, xi_threshold);

H5Fclose( fileId );
}


if( rank==0 ) {
std::cout << std::endl;
std::cout << " Computation of the xi table"
<< std::endl;
}

buffer = new double [rank_indexes[rank]*size_particle_chi];

percentage = 0;
t0 = MPI_Wtime();

for( i_photon_chi = 0 ; i_photon_chi < rank_indexes[rank] ; i_photon_chi++ ) {

photon_chi = ( rank_first_index[rank] + i_photon_chi )*delta_photon_chi
+ log10_min_photon_chi;

photon_chi = std::pow( 10.0, photon_chi );

log10_min_particle_chi = std::log10( 0.5*photon_chi );

delta_particle_chi = ( log10_min_particle_chi - table_1d[rank_first_index[rank] + i_photon_chi] )
/ ( size_particle_chi - 1 );

denominator = MultiphotonBreitWheeler::computeIntegrationRitusDerivative( photon_chi,
0.5*photon_chi, 200, 1e-15 );

for( i_particle_chi = 0 ; i_particle_chi < size_particle_chi ; i_particle_chi ++ ) {
particle_chi = std::pow( 10.0, i_particle_chi*delta_particle_chi +
table_1d[rank_first_index[rank] + i_photon_chi] );

numerator = MultiphotonBreitWheeler::computeIntegrationRitusDerivative( photon_chi,
particle_chi, 200, 1e-15 );

buffer[i_photon_chi*size_particle_chi + i_particle_chi] = numerator / ( 2.0*denominator );
}

t1 = MPI_Wtime();

if( rank==0 ) {
if( 100.0*(i_photon_chi+1) >= rank_indexes[rank]*percentage ) {
percentage += delta_percentage;
std::cout << " - " << i_photon_chi + 1<< "/" << rank_indexes[rank]
<< " - " << ( int )( std::round( percentage ) ) << "%"
<< " - " << t1 - t0 << " s"
<< std::endl;
}
}
}

t1 = MPI_Wtime();
if (rank==0) {
std::cout << " Total time: " << t1 - t0 << " s" << std::endl;
}

for( int i = 0 ; i < number_of_ranks ; i++ ) {
rank_indexes[i] *= size_particle_chi;
rank_first_index[i] *= size_particle_chi;
}

MPI_Allgatherv( &buffer[0], rank_indexes[rank], MPI_DOUBLE,
&table_2d[0], &rank_indexes[0], &rank_first_index[0],
MPI_DOUBLE, MPI_COMM_WORLD );

if (verbose && rank == 0) {
std:: cout << "\n table xi: " << std::setprecision(std::numeric_limits<double>::digits10 + 1) <<std::endl;
for( i_photon_chi = 0 ; i_photon_chi < size_photon_chi  ; i_photon_chi++ ) {
std::cout << " ";
for( i_particle_chi = 0 ; i_particle_chi < size_particle_chi ; i_particle_chi ++ ) {
std::cout << table_2d[i_photon_chi*size_particle_chi + i_particle_chi] << ", ";
}
std::cout << std::endl;
}
}

if (rank==0) {

fileId = H5Fopen( path.c_str(),
H5F_ACC_RDWR,
H5P_DEFAULT );

std::string vect_name("xi");

int size[2];
size[0] = size_photon_chi;
size[1] = size_particle_chi;
H5::H5Vector2D( fileId, vect_name, &size[0], table_2d);

std::string attr_name("min_photon_chi");
H5::attr( fileId, vect_name, attr_name, min_photon_chi);

attr_name = "max_photon_chi";
H5::attr( fileId, vect_name, attr_name, max_photon_chi);

attr_name = "size_particle_chi";
H5::attr( fileId, vect_name, attr_name, size_particle_chi);

attr_name = "size_photon_chi";
H5::attr( fileId, vect_name, attr_name, size_photon_chi);

H5Fclose( fileId );
}


delete buffer;
delete rank_first_index;
delete rank_indexes;


}


double MultiphotonBreitWheeler::computeIntegrationRitusDerivative( double photon_chi,
double particle_chi,
int discretization,
double eps )
{

double *gauleg_x = new double[discretization];
double *weights = new double[discretization];
int i;
double u;
double T;

Tools::GaussLegendreQuadrature( std::log10( particle_chi )-50., std::log10( particle_chi ),
gauleg_x, weights, discretization, eps );

T = 0;
for( i=0 ; i< discretization ; i++ ) {
u = std::pow( 10.0, gauleg_x[i] );
T += weights[i]*computeRitusDerivative( photon_chi, u, discretization, eps )*u*std::log( 10. );
}

delete[] gauleg_x;
delete[] weights;

return T;

}

double MultiphotonBreitWheeler::computeRitusDerivative( double photon_chi,
double particle_chi, int discretization, double eps )
{

double *gauleg_x = new double[discretization];
double *weights = new double[discretization];
double y, u;
double p1, p2;
int i;

y = ( photon_chi/( 3.0*particle_chi*( photon_chi-particle_chi ) ) );


p1 = ( 2.0 - 3.0*photon_chi*y )*Tools::BesselK( 2.0/3.0, 2.0*y);

Tools::GaussLegendreQuadrature( std::log10( 2*y ), std::log10( 2*y )+50.0,
gauleg_x, weights, discretization, eps );

p2 = 0;
for( i=0 ; i< discretization ; i++ ) {
u = std::pow( 10.0, gauleg_x[i] );
p2 += weights[i]*Tools::BesselK( 1.0/3.0, u )*u*std::log( 10.0 );
}

delete[] gauleg_x;
delete[] weights;

return ( p2 - p1 ); 
}

double MultiphotonBreitWheeler::computeErberT( double photon_chi)
{
double K;

K = Tools::BesselK( 1.0/3.0, 4.0/( 3.0*photon_chi ));

return 0.16*K*K/photon_chi;
}
