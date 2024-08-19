

#include "n_body.h"

n_body::n_body(const std::string& file_path_, double delta_t_, int T_, int is_each_step_, int number_of_threads_, int process_id_, int process_master_, int comm_sz_) {


file_path = file_path_;
delta_t = delta_t_;
T = T_;
is_each_step = is_each_step_;
number_of_threads = number_of_threads_;
process_id = process_id_;
process_master = process_master_;
comm_sz = comm_sz_;

read_data_from_txt();
}

void n_body::initialize_arrays() {

local_n = n/comm_sz;

masses = (double *)malloc(n* sizeof(double));
velocities = (double *)malloc(n* 2 * sizeof(double));
positions = (double *)malloc(n* 2 * sizeof(double));

local_velocities = (double *)malloc(local_n* 2 * sizeof(double));
local_positions = (double *)malloc(local_n* 2 * sizeof(double));
local_forces = (double *)malloc(local_n * 2 * sizeof(double));

memset(local_forces, 0, local_n * 2 * sizeof(double));
}

void n_body::actualize_speed_pos() {



for(int timestep=0; timestep<T; timestep++){
if(is_each_step == 1){
if(process_id==process_master){
write_data_to_txt(timestep*delta_t);
}
}
#pragma omp parallel for num_threads(number_of_threads)
for(int q=0; q<local_n; q++){
for(int k = 0; k<n ;k++){
if(k!=(q + process_id*local_n )){

double x_diff = *(positions + process_id*local_n*2 + q*2 + 0) - *(positions + k*2 + 0);
double y_diff = *(positions + process_id*local_n*2+ q*2 + 1) - *(positions + k*2 + 1);

double dist = sqrt(x_diff*x_diff + y_diff*y_diff);

double dist_cubed = dist*dist*dist;
*(local_forces + q*2 + 0) -= G*masses[process_id*local_n + q]*masses[k]/dist_cubed * x_diff;
*(local_forces + q*2 + 1) -= G*masses[process_id*local_n + q]*masses[k]/dist_cubed * y_diff;

}
}
}

#pragma omp parallel for num_threads(number_of_threads)
for(int q = 0; q < local_n ; ++q){
*(local_positions + q*2 + 0) += delta_t * (*(local_velocities + q*2 + 0));
*(local_positions + q*2 + 1) += delta_t * (*(local_velocities + q*2 + 1));
*(local_velocities + q*2 + 0) += delta_t/masses[process_id*local_n + q] * (*(local_forces + q*2 + 0));
*(local_velocities + q*2 + 1) += delta_t/masses[process_id*local_n + q] * (*(local_forces + q*2 + 1));
}

MPI_Allgather(local_positions, local_n*2, MPI_DOUBLE, positions, local_n*2, MPI_DOUBLE, MPI_COMM_WORLD);
MPI_Allgather(local_velocities,local_n*2, MPI_DOUBLE, velocities,local_n*2, MPI_DOUBLE, MPI_COMM_WORLD);
}

if(process_id==process_master){
write_data_to_txt(T*delta_t);
}
}

void n_body::write_data_to_txt(double timestep) {

std::string outfile_name = std::string("data_")+ std::to_string(timestep);
std::ofstream outfile (outfile_name);
outfile<<n<<std::endl;
for (int q = 0; q < n ; ++q) {
outfile<<*(positions +q*2 +0)<<","<<*(positions +q*2 +1)<<","<<*(velocities +q*2 +0)<<","<<*(velocities +q*2 +1)<<","<<masses[q]<<std::endl;
}
outfile.close();
}

void n_body::read_data_from_txt() {

int was_read = 0;

if(process_id == process_master){
std::ifstream file(file_path.c_str());
if(file.is_open()) {
was_read = 1; 
std::string line;
getline(file, line); 

n = std::stoi(line); 

initialize_arrays(); 

int i = 0;
while (getline(file, line)) {
std::stringstream linestream(line);
std::string value;

getline(linestream, value, ',');
*(positions + i*2 + 0) = std::stod(value); 

getline(linestream, value, ',');
*(positions + i*2 + 1) = std::stod(value); 

getline(linestream, value, ',');
*(velocities + i*2 + 0) = std::stod(value); 

getline(linestream, value, ',');
*(velocities + i*2 + 1) = std::stod(value); 

getline(linestream, value, ',');
masses[i] = std::stod(value);

i++;
}

file.close();
}
}

MPI_Bcast(&was_read,1, MPI_INT, process_master, MPI_COMM_WORLD);

if(was_read==0){
std::cerr<<"Error while opening the file"<<std::endl;
exit(EXIT_FAILURE);
}
else{

MPI_Bcast(&n,1, MPI_INT, process_master, MPI_COMM_WORLD);
if(process_id!=process_master){
initialize_arrays();
}

MPI_Bcast(positions,n* 2, MPI_DOUBLE, process_master, MPI_COMM_WORLD);
MPI_Bcast(masses,n, MPI_DOUBLE, process_master, MPI_COMM_WORLD);
MPI_Scatter(velocities, local_n* 2, MPI_DOUBLE, local_velocities, local_n* 2, MPI_DOUBLE, process_master,MPI_COMM_WORLD);

MPI_Scatter(positions, local_n* 2, MPI_DOUBLE, local_positions, local_n* 2, MPI_DOUBLE, process_master,MPI_COMM_WORLD);


}
}
