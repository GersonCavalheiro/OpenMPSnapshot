#include <iostream>
#include <complex>
#include <fstream>
#include <ctime>
#include <chrono>

#include "omp.h"
#include "mpi.h"

using namespace std;

#define MAX_ITER 1000
#define N_TRIALS 7
#define SEQUENTIAL_SMALL_TIME 58.3892
#define SEQUENTIAL_MEDIUM_TIME 129.778
#define SEQUENTIAL_LARGE_TIME 567.201

void printMatrixToFile(int *T, int n, int m, char *output_image) {
ofstream image;
image.open(output_image, ios::out);
for (int i = 0; i < n; i++) {
for (int j = 0; j < m; j++) {
image << j << " " << i << " " << T[i * m + j] << endl;
}
}
}

int isInMandelbrot(std::complex<double> point) {
bool isInside = true;
std::complex<double> Z = point;
int num_iter = 0;
for (int iter = 0; iter <= MAX_ITER - 1; ++iter) {
if (std::abs(Z) > 2) {
isInside = false;
num_iter = iter;
break;
}
Z = std::pow(Z, 2) + point;
}
if (isInside == true)
return 0;
return num_iter;
}

int main(int argc, char **argv) { 

if (argc < 8) {
cout << "Incorrect number of parameters" << endl;
return -1;
}

char *output_image;

int real_axys_length = atoi(argv[1]);      
int imaginary_axys_length = atoi(argv[2]); 
double real_part_root = stod(argv[3]);     
double img_part_root = stod(argv[4]);      
double L = stod(argv[5]);                  
int task_size = atoi(argv[6]);             
output_image = argv[7];


std::complex<double> root(real_part_root, img_part_root);
std::complex<double> aux_img(0.0, L / imaginary_axys_length);
std::complex<double> aux_real(L / real_axys_length, 0.0);
std::complex<double> aux_img_task(0.0, (L / imaginary_axys_length) * task_size);

int multiple = (int) imaginary_axys_length / task_size;
imaginary_axys_length = multiple * task_size;

int mpi_num_nodes, mpi_my_rank;
MPI_Status status;

chrono::high_resolution_clock::time_point t1, t2;
chrono::duration<double> diff;
double best_time;

int err = MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &mpi_num_nodes);
MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank);

int num_task = imaginary_axys_length / task_size;
int num_slaves = mpi_num_nodes - 1;

for (int iter = 0; iter < N_TRIALS; iter++) {

int flag = 0;

if (mpi_my_rank == 0) { 

MPI_Request requests[mpi_num_nodes];
for (int i = 1; i <= num_slaves; i++)
requests[i] = MPI_REQUEST_NULL;

int num_sent_tasks = 0;
int num_received_tasks = 0;

std::complex<double> buffer_send[2];

int *output_matrix = new int[real_axys_length * imaginary_axys_length];

std::complex<double> aux = root;

t1 = chrono::high_resolution_clock::now();

for (int dest = 1; dest < mpi_num_nodes; dest++) {
buffer_send[0] = aux;
aux = aux + aux_img_task;
buffer_send[1] = aux;
MPI_Send(&buffer_send, 2, MPI_DOUBLE_COMPLEX, dest, num_sent_tasks, MPI_COMM_WORLD);
aux = aux + aux_img_task;
num_sent_tasks = num_sent_tasks + 2;
}

bool go = true;
while (go) {
MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
if (flag) {
MPI_Irecv(&output_matrix[status.MPI_TAG * (real_axys_length * task_size)], (real_axys_length * task_size), MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &requests[status.MPI_SOURCE]);
num_received_tasks++;
if (num_sent_tasks < num_task) {
buffer_send[0] = aux;
MPI_Send(buffer_send, 1, MPI_DOUBLE_COMPLEX, status.MPI_SOURCE, num_sent_tasks, MPI_COMM_WORLD);
aux = aux + aux_img_task;
num_sent_tasks++;
}
if (num_received_tasks == num_task) {
go = false;
break;
}
}
}

for (int dest = 1; dest < mpi_num_nodes; dest++) {
MPI_Send(buffer_send, 1, MPI_DOUBLE_COMPLEX, dest, 99999, MPI_COMM_WORLD);
MPI_Wait(&requests[dest], MPI_STATUS_IGNORE);
}

t2 = chrono::high_resolution_clock::now();
diff = t2 - t1;

if (iter == 0)
best_time = diff.count();
if (diff.count() < best_time)
best_time = diff.count();
if (iter == (N_TRIALS-1))
cout << task_size << " " << SEQUENTIAL_SMALL_TIME / best_time << " (" << best_time << ")" << endl;


}

else {
int task;
std::complex<double> buffer_rcv[2];
MPI_Request request;
int *buffer_send = new int[real_axys_length * task_size];
MPI_Recv(buffer_rcv, 2, MPI_DOUBLE_COMPLEX, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
std::complex<double> aux = buffer_rcv[0];
std::complex<double> root = buffer_rcv[0];
task = status.MPI_TAG;
int i,j;
#pragma omp parallel for private(i,j) num_threads(omp_get_num_threads()) 
for (i = 0; i < task_size; i++) {
for (j = 0; j < real_axys_length; j++) {
buffer_send[i * real_axys_length + j] = isInMandelbrot(aux);
aux += aux_real;
}
aux += aux_img;
aux = complex<double>(real(root), imag(aux));
}

MPI_Send(buffer_send, (real_axys_length * task_size), MPI_INT, 0, task, MPI_COMM_WORLD);

buffer_rcv[0] = buffer_rcv[1];
task = task+1;

while (true) {
aux = buffer_rcv[0];
root = buffer_rcv[0];
MPI_Irecv(buffer_rcv, 1, MPI_DOUBLE_COMPLEX, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
#pragma omp parallel for private(i,j) num_threads(omp_get_num_threads())
for (i = 0; i < task_size; i++) {
for (j = 0; j < real_axys_length; j++) {
buffer_send[i * real_axys_length + j] = isInMandelbrot(aux);
aux += aux_real;
}
aux += aux_img;
aux = complex<double>(real(root), imag(aux));
}
MPI_Send(buffer_send, (real_axys_length * task_size), MPI_INT, 0, task, MPI_COMM_WORLD);
MPI_Test(&request, &flag, &status);
if (!flag)
MPI_Wait(&request, &status);
if (status.MPI_TAG == 99999) 
break;
task = status.MPI_TAG;
}
}
}

err = MPI_Finalize();

return 0;
}
