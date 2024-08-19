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
clock_t t_comp = 0;
clock_t t_comm = 0;
clock_t t_io = 0;
clock_t comp_beg, comp_end, comm_beg, comm_end, io_beg, io_end;

comp_beg = clock();

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

MPI_Status status;

comm = MPI_COMM_WORLD;

MPI_File ifs;
MPI_File ofs;

io_beg = clock();
MPI_File_open(comm, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL,
&ifs);
MPI_File_read_at(ifs, 0, buf, 2, MPI_INT, &status);

v = buf[0];
e = buf[1];

memset(d, 10001, sizeof(int) * v * v);
MPI_File_read_at(ifs, sizeof(int) * 2, buf, e * 3, MPI_INT, &status);
io_end = clock();
t_io += io_end - io_beg;

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
io_beg = clock();
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
io_end = clock();
t_io += io_end - io_beg;
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
for (int j = 0; j < v; j++)
{
if (d[i * v + j] > d[i * v + k] + d[k * v + j])
d[i * v + j] = d[i * v + k] + d[k * v + j];
}
}
if (k < v - 1 && root[k + 1] == rank)
MPI_Ibcast(d + (k + 1) * v, v, MPI_INT, root[k + 1], comm, &req);
comm_beg = clock();
MPI_Wait(&req, &status);
comm_end = clock();
t_comm += comm_end - comm_beg;
}

io_beg = clock();
MPI_File_open(comm, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
&ofs);
MPI_File_write_at(ofs, sizeof(int) * i_begin * v, d + i_begin * v,
(i_end - i_begin) * v, MPI_INT, &status);
MPI_File_close(&ofs);

io_end = clock();
t_io += io_end - io_beg;

comp_end = clock();
t_comp = comp_end - comp_beg - t_io - t_comm;

clock_t time_io, time_comm, time_comp;

MPI_Reduce(&t_io, &time_io, 1, MPI_LONG, MPI_MAX, 0, comm);
MPI_Reduce(&t_comm, &time_comm, 1, MPI_LONG, MPI_MAX, 0, comm);
MPI_Reduce(&t_comp, &time_comp, 1, MPI_LONG, MPI_MAX, 0, comm);

if (rank == 0)
std::cout << (double)time_comp / CLOCKS_PER_SEC << " "
<< (double)time_io / CLOCKS_PER_SEC << " "
<< (double)time_comm / CLOCKS_PER_SEC << " ";

MPI_Finalize();
}