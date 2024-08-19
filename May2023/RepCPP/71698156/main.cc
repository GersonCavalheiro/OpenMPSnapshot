

#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>

#include <mpi.h>
#include <omp.h>

typedef double real;

int main(int argc, char** argv) {

int rank = 0;
int size = 0;
int provided;

MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

if (provided < MPI_THREAD_FUNNELED) {
MPI_Abort(MPI_COMM_WORLD, 1);
}

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

const size_t max_num_bytes = (1 << 22);
size_t message_size = 0;

std::stringstream string_buffer;

for (int i = 1; i < argc; i++) {
string_buffer << argv[i] << " ";
}

string_buffer >> message_size;

char* send_buffer = (char*) malloc(max_num_bytes);
char* recv_buffer = (char*) malloc(max_num_bytes);

#pragma omp parallel for
for (int i = 0; i < max_num_bytes; i++) {
recv_buffer[i] = send_buffer[i] = (char) (i & 0xff);
}

MPI_Status wait_status;
MPI_Request send_request;
MPI_Request recv_request;

const int tag = 9;
const int num_mpi_repetitions = 100;
const int num_stream_repetitions = 200;

const int next = (rank + 1) % size;
const int prev = (rank + size - 1) % size;

const real scalar = 3.0f;

const size_t stream_array_size = 64000000;
size_t stream_num_bytes = sizeof(real) * stream_array_size;

real* a = (real*) malloc(stream_num_bytes);
real* b = (real*) malloc(stream_num_bytes);
real* c = (real*) malloc(stream_num_bytes);

#pragma omp parallel for
for (int i = 0; i < stream_array_size; i++) {
a[i] = b[i] = c[i] = scalar;
}

double start_time = 0.f;
double end_time = 0.f;
double stream_timer = 0.f;

std::vector<double> stream_timings;

omp_set_dynamic(0);
omp_set_nested(1);

MPI_Barrier(MPI_COMM_WORLD);

if (rank == 0) {
std::cout << "\n";
std::cout << "====================== MPI Contention Benchmark v0.2 =====================\n";
std::cout << "Bytes\t     Time\t  Bandwidth (MB/s)\t\n";
}

#pragma omp parallel default(shared) num_threads(2)
{
int tid = omp_get_thread_num();
int max_threads = omp_get_max_threads();

if (tid == 0) {

start_time = MPI_Wtime();

for (int t = 0; t < num_mpi_repetitions; t++) {

if (rank == 0) {
MPI_Isend(send_buffer, message_size, MPI_CHAR, next, tag, MPI_COMM_WORLD, &send_request);
MPI_Irecv(recv_buffer, message_size, MPI_CHAR, prev, tag, MPI_COMM_WORLD, &recv_request);
} else {
MPI_Irecv(recv_buffer, message_size, MPI_CHAR, prev, tag, MPI_COMM_WORLD, &recv_request);
MPI_Isend(send_buffer, message_size, MPI_CHAR, next, tag, MPI_COMM_WORLD, &send_request);
}

MPI_Wait(&send_request, &wait_status);
MPI_Wait(&recv_request, &wait_status);

end_time = MPI_Wtime() - start_time;

if (rank == 0) {
double bandwidth = message_size * num_mpi_repetitions / end_time / 1024. / 1024.;

std::cout << message_size << "\t";
std::cout << std::fixed << std::setw(11) << std::setprecision(6) << end_time << "\t";
std::cout << std::fixed << std::setw(11) << std::setprecision(6) << bandwidth << "\t\n";
}
}

} else {

for (int t = 0; t < num_stream_repetitions; t++) {
stream_timer = 0.;

stream_timer -= omp_get_wtime();

#pragma omp parallel num_threads(max_threads-1)
{
#pragma omp for
for (int i = 0; i < stream_array_size; i++) {
a[i] = b[i] + scalar * c[i];
}
}

stream_timer += omp_get_wtime();

stream_timings.push_back(stream_timer);
}
}
}

if (rank == 0) {

auto min_time = *std::min_element(stream_timings.begin(), stream_timings.end());
auto max_time = *std::max_element(stream_timings.begin(), stream_timings.end());

double sum_time = 0.;

for (auto const& time : stream_timings) {
sum_time += time;
}

double data_size = 3 * sizeof(real) * stream_array_size;

double stream_bandwidth = 1.0e-06 * data_size / min_time;

std::cout << "\n";
std::cout << "================================ STREAM ==================================\n";
std::cout << "Kernel      Best Rate (MB/s)    Avg time        Min time        Max time\n";
std::cout << "Triad         " << std::setw(11) << std::setprecision(6)
<< stream_bandwidth << "       " << sum_time / (num_stream_repetitions - 1) << "        "
<< min_time << "        " << max_time << std::endl;
std::cout << "\n";
}

free(send_buffer);
free(recv_buffer);

free(a);
free(b);
free(c);

MPI_Finalize();

return 0;
}