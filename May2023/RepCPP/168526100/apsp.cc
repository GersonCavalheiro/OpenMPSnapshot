#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include "mpi.h"
#include <time.h>
#include <vector>

#define MAX_BUF 200000000

MPI_Comm comm;
int rank, size;
int v, e;
int d[MAX_BUF];
int buf[MAX_BUF];

int main(int argc, char **argv)
{


MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

MPI_Status status;

comm = MPI_COMM_WORLD;

MPI_File ifs;
MPI_File ofs;

MPI_File_open(comm, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL,
&ifs);
MPI_File_read_at(ifs, 0, buf, 2, MPI_INT, &status);

v = buf[0];
e = buf[1];

memset(d, 10001, sizeof(int) * v * v);
MPI_File_read_at(ifs, sizeof(int) * 2, buf, e * 3, MPI_INT, &status);

#pragma omp parallel for simd
for (int i = 0; i < e; i++)
{
int s = buf[i * 3];
int r = buf[i * 3 + 1];
int w_e = buf[i * 3 + 2];
d[s * v + r] = w_e;
}
MPI_File_close(&ifs);

#pragma omp parallel for simd
for (int i = 0; i < v; i++)
d[i * v + i] = 0;

int slice = v / size;

int i_begin = slice * rank;
int i_end = rank < size - 1 ? slice * (rank + 1) : v;

std::vector<int> group;
int root[v];
MPI_File pfs;

if (argc >= 4)
{
std::ifstream ifs(argv[3], std::ios::in);

for (int i = 0; i < v; i++)
{
ifs >> root[i];
if (root[i] == rank)
group.push_back(i);
}
i_begin = group.front();
i_end = group.back() + 1;
ifs.close();
}
else
#pragma omp parallel for simd
for (int i = 0; i < v; i++)
root[i] = i / slice < size ? i / slice : size - 1;

for (int k = 0; k < v; k++)
{
MPI_Request req;
if (k < v - 1 && root[k + 1] != rank)
MPI_Ibcast(d + (k + 1) * v, v, MPI_INT, root[k + 1], comm, &req);

#pragma omp parallel for schedule(guided)
for (int i = i_begin; i < i_end; i++)
{
for (int j = 0; j < k; j++)
{
if (d[i * v + j] > d[i * v + k] + d[k * v + j])
d[i * v + j] = d[i * v + k] + d[k * v + j];
}
for (int j = k; j < v; j++)
{
if (d[i * v + j] > d[i * v + k] + d[k * v + j])
d[i * v + j] = d[i * v + k] + d[k * v + j];
}
}
if (k < v - 1 && root[k + 1] == rank)
MPI_Ibcast(d + (k + 1) * v, v, MPI_INT, root[k + 1], comm, &req);
MPI_Wait(&req, &status);
}


MPI_File_open(comm, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
&ofs);
MPI_File_write_at(ofs, sizeof(int) * i_begin * v, d + i_begin * v,
(i_end - i_begin) * v, MPI_INT, &status);
MPI_File_close(&ofs);






MPI_Finalize();
}