#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#define CHUNK_SIZE 5000
typedef struct { double real, imag; } Compl;
inline int calc(double real, double imag) {
Compl z, c;
z.real = 0.0;
z.imag = 0.0;
c.real = real;
c.imag = imag;
double lengthc = c.real * c.real + c.imag * c.imag;
if (lengthc < 0.0625) { return 100000; }
int repeats = 0;
double lengthsq = 0.0;
double temp;
while(repeats < 100000 && lengthsq <= 4.0) {
temp = z.real * z.real - z.imag * z.imag + c.real;
z.imag = 2 * z.real * z.imag + c.imag;
z.real = temp;
lengthsq = z.real * z.real + z.imag * z.imag; 
repeats++;
}
return repeats;
}
int main(int argc, char* argv[]) {
MPI_Init(&argc, &argv);
omp_set_nested(1);
int rank, size;
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
int thread_num = atoi(argv[1]); 
double realaxis_left = atof(argv[2]);
double realaxis_right = atof(argv[3]);
double imageaxis_lower = atof(argv[4]);
double imageaxis_upper = atof(argv[5]);
int width = atoi(argv[6]);
int height = atoi(argv[7]);
char *output = argv[8];
int len;
int msg;
int *BUF;
MPI_File fout;
MPI_File_open(MPI_COMM_WORLD, output, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
if (rank == 0) {
int msg2 = -1;
#pragma omp parallel sections num_threads(thread_num) shared(msg2) private(len)
{
#pragma omp section
{
MPI_File_write_at(fout, 0, &width, 1, MPI_INT, MPI_STATUS_IGNORE);
MPI_File_write_at(fout, sizeof(int), &height, 1, MPI_INT, MPI_STATUS_IGNORE);
len = height * width;
int now = 0;
MPI_Status stat;
MPI_Request req;
while (now < len) {
#pragma omp flush(msg2)
if (msg2 == -1) {
msg2 = now;
#pragma omp flush(msg2)
now = now + CHUNK_SIZE;
continue;
}
if (size != 1) {
MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
msg = now;
MPI_Isend(&msg, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &req);
now = now + CHUNK_SIZE;
}
}
msg = -1;
#pragma omp flush(msg2) 
while (msg2 != -1) {
#pragma omp flush(msg2)
}
msg2 = -2;
#pragma omp flush(msg2)
int current_process;
for (current_process = 1; current_process < size; current_process++) {
MPI_Isend(&msg, 1, MPI_INT, current_process, 0, MPI_COMM_WORLD, &req);
}
}
#pragma omp section
{
BUF = (int*)malloc(CHUNK_SIZE * sizeof(int));
while (1) {
#pragma omp flush(msg2)
if (msg2 == -2) break;
if (msg2 == -1) continue;
len = CHUNK_SIZE;
if (msg2 + len > height * width) len = height * width - msg2;
int k;
#pragma omp parallel for num_threads(thread_num) schedule(dynamic)
for(k = 0; k < len; k++) {
int i = (k + msg2) / height;
int j = (k + msg2) % height;
double real = realaxis_left + (double)i * ((realaxis_right - realaxis_left) / (double)width);
double imag = imageaxis_lower + (double)j * ((imageaxis_upper - imageaxis_lower) / (double)height);
BUF[k] = calc(real, imag);
}
MPI_File_write_at(fout, (msg2 + 2) * sizeof(int), BUF, len, MPI_INT, MPI_STATUS_IGNORE);
msg2 = -1;
#pragma omp flush(msg2)
}
}
}
}
else {
MPI_Request req;
BUF = (int*)malloc(CHUNK_SIZE * sizeof(int));
while (1) {
MPI_Isend(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
MPI_Recv(&msg, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
if (msg == -1) break;
len = CHUNK_SIZE;
if (msg + len > height * width) len = height * width - msg;
int k;
#pragma omp parallel for num_threads(thread_num) schedule(dynamic)
for(k = 0; k < len; k++) {
int i = (k + msg) / height;
int j = (k + msg) % height;
double real = realaxis_left + (double)i * ((realaxis_right - realaxis_left) / (double)width);
double imag = imageaxis_lower + (double)j * ((imageaxis_upper - imageaxis_lower) / (double)height);
BUF[k] = calc(real, imag);
}
MPI_File_write_at(fout, (msg + 2) * sizeof(int), BUF, len, MPI_INT, MPI_STATUS_IGNORE);
}
}
free(BUF);
MPI_File_close(&fout);
MPI_Finalize();
return 0;
}
