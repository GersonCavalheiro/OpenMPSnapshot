#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>
#include <math.h>


using namespace std;

void init_gol_map(ifstream &init_data, int **mat);
void print_matrix(int **mat, int r, int c);
void fill_matrix(int **mat, int r, int c, int val);
void write_matrix_inline(int **mat, int r, int c, ofstream &f);
void copy_matrix(int **dest_mat, int **copy_mat, int r, int c);
void allocate_gen_matrices(int r, int c);
void allocate_prx_matrices(int r, int c);

void pop_nxt_gol_map(int ptx_rows, int ptx_cols, int th_cnt, int mpi_chunk_sz);
int get_map_rows(ifstream &init_data);
int get_map_cols(ifstream &init_data);


int **cur_gol_map, **prx_gol_map, **nxt_gol_map;

int main(int argc, char* argv[])
{
MPI_Init(NULL, NULL);
int world_size, world_rank;
MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 


if(argc != 5 && world_rank == 0)
{
cout << "HYBRID_ERR_ARG:ARGC => Wrong number of command line arguments.\nUse \"./<executable> <in_file> <threads> <gens> <out_file>\" as format.\n";
return -5;
}


double start_time, init_time, gol_time; 




char* cmd_in_file = argv[1]; 
int thread_count = atoi(argv[2]); 
int gens = atoi(argv[3]); 
char* cmd_out_file = argv[4]; 


ifstream in_file; 
ofstream out_file; 

in_file.open(cmd_in_file);
out_file.open(cmd_out_file, ofstream::out | ofstream::trunc); 


int map_rows = get_map_rows(in_file); 
int map_cols = get_map_cols(in_file); 


if(world_rank == 0)
{
cout << "Map Rows: " << map_rows << endl;
cout << "Map Columns: " << map_cols << endl;
cout << "Generations: " << gens << endl;
cout << "Input File: " << cmd_in_file << endl;
cout << "Output File: " << cmd_out_file << endl;
cout << "Processor Count: " << endl;
cout << "Thread Count: " << thread_count << endl;
}


allocate_gen_matrices(map_rows, map_cols);
init_gol_map(in_file, cur_gol_map);

int prx_rows;
int prx_cols = map_cols + 2;
int mpi_chunk_sz = floor(map_rows/world_size);
int btm_mpi_chunk_sz = map_rows - ((world_size - 1) * mpi_chunk_sz);

if(world_rank == world_size - 1)
prx_rows = btm_mpi_chunk_sz + 2;
else
prx_rows = mpi_chunk_sz + 2;

allocate_prx_matrices(prx_rows, prx_cols);
fill_matrix(prx_gol_map, prx_rows, prx_cols, -3);


int row_start = mpi_chunk_sz * world_rank;
int row_end = (row_start + mpi_chunk_sz - 1);
if(world_rank == (world_size - 1))
{
row_end = map_rows - 1;
}

int prx_i = 1;
for(int i = row_start; i <= row_end; i++)
{
int prx_j = 1;
for(int j = 0; j < map_cols; j++)
{
prx_gol_map[prx_i][prx_j] = cur_gol_map[i][j];
prx_j++;
}
prx_i++;
}


for(int proc_i = 0; proc_i < world_size; proc_i++)
{
if(world_rank == proc_i)
{
MPI_Recv(&prx_gol_map[prx_rows - 1][1], map_cols, MPI_INT, (proc_i + 1) % world_size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
}

if(world_rank == (proc_i + 1) % world_size)
{
MPI_Send(&cur_gol_map[row_start][0], map_cols, MPI_INT, proc_i, 1, MPI_COMM_WORLD); 
}


if(world_rank == proc_i)
{
if(world_rank - 1 < 0)
proc_i = world_size;
MPI_Recv(&prx_gol_map[0][1], map_cols, MPI_INT, (proc_i - 1) % world_size, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
if(world_rank - 1 < 0)
proc_i = 0;
}

if((proc_i == 0) && (world_rank == world_size - 1)) 
{
MPI_Send(&cur_gol_map[row_start + prx_rows - 3][0], map_cols, MPI_INT, proc_i, 99, MPI_COMM_WORLD); 
}

if(world_rank == ((proc_i - 1) % world_size)) 
{
MPI_Send(&cur_gol_map[row_start + prx_rows - 3][0], map_cols, MPI_INT, proc_i, 99, MPI_COMM_WORLD); 
}
}


prx_gol_map[0][0] = prx_gol_map[0][prx_cols - 2];
prx_gol_map[0][prx_cols - 1] = prx_gol_map[0][1];
prx_gol_map[prx_rows - 1][0] = prx_gol_map[prx_rows - 1][prx_cols - 2];
prx_gol_map[prx_rows - 1][prx_cols - 1] = prx_gol_map[prx_rows - 1][1];


for(int prx_lr = 0; prx_lr < prx_rows - 2; prx_lr++)
{
prx_gol_map[prx_lr + 1][0] = prx_gol_map[prx_lr + 1][prx_cols - 2];
prx_gol_map[prx_lr + 1][prx_cols - 1] = prx_gol_map[prx_lr + 1][1];
}


pop_nxt_gol_map(prx_rows, prx_cols, thread_count, mpi_chunk_sz);


for(int proc_i = 0; proc_i < world_size; proc_i++)
{

}




int curr_rank = 0;
while(curr_rank < world_size)
{
MPI_Barrier(MPI_COMM_WORLD);
if(world_rank == curr_rank)
{
cout << world_rank << ": CURRENT" << endl;
print_matrix(cur_gol_map, map_rows, map_cols);
cout << world_rank << ": NEXT" << endl;
print_matrix(nxt_gol_map, map_rows, map_cols);
cout << world_rank << ": prx_rows: " << prx_rows << endl;
cout << world_rank << ": prx_cols: " << prx_cols << endl;
print_matrix(prx_gol_map, prx_rows, prx_cols);
}
MPI_Barrier(MPI_COMM_WORLD); 
curr_rank++;
}
MPI_Barrier(MPI_COMM_WORLD);









in_file.close();
out_file.close();

MPI_Finalize(); 

return 0;
}










void pop_nxt_gol_map(int prx_rows, int prx_cols, int th_cnt, int mpi_chunk_sz)
{
int world_size, world_rank;
MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 


int data_rows = prx_rows - 2;


if(th_cnt > data_rows)
th_cnt = data_rows;


#pragma omp parallel num_threads(th_cnt)
{
int cell_sum = 0; 
int thread_ID, th_cnt_per_proc;
thread_ID = omp_get_thread_num(); 
th_cnt_per_proc = omp_get_num_threads(); 

int omp_chunk_sz = floor(data_rows / th_cnt_per_proc);
if(omp_chunk_sz == 0)
omp_chunk_sz = 1;



int omp_thread_row_start = (omp_chunk_sz * thread_ID) + 1;
int omp_thread_row_end = omp_thread_row_start + (omp_chunk_sz - 1);
if(thread_ID == th_cnt_per_proc - 1)
{
omp_thread_row_end = data_rows;
}


for(int i = omp_thread_row_start; i <= omp_thread_row_end; i++)
{
for(int j = 1; j <= prx_cols - 2; j++)
{
cell_sum = prx_gol_map[i+1][j] + prx_gol_map[i-1][j] + prx_gol_map[i][j+1] + prx_gol_map[i][j-1] +
prx_gol_map[i-1][j+1] + prx_gol_map[i+1][j-1] + prx_gol_map[i+1][j+1] + prx_gol_map[i-1][j-1];

if((cell_sum < 2) && (prx_gol_map[i][j] == 1)) 
{
nxt_gol_map[(i-1) + (mpi_chunk_sz * world_rank)][j-1] = 0; 
}
else if((cell_sum >= 2 && cell_sum <= 3) && (prx_gol_map[i][j] == 1)) 
{
nxt_gol_map[(i-1) + (mpi_chunk_sz * world_rank)][j-1] = 1; 
}
else if(cell_sum > 3 && (prx_gol_map[i][j] == 1)) 
{
nxt_gol_map[(i-1) + (mpi_chunk_sz * world_rank)][j-1] = 0; 
}
else if(cell_sum == 3 && (prx_gol_map[i][j] == 0)) 
{
nxt_gol_map[(i-1) + (mpi_chunk_sz * world_rank)][j-1] = 1; 
}
else
{
nxt_gol_map[(i-1) + (mpi_chunk_sz * world_rank)][j-1] = 0; 
}

cell_sum = 0; 
}
}
}
}



void print_matrix(int **mat, int r, int c)
{
for(int i = 0; i < r; i++)
{
for(int j = 0; j < c; j++)
{
cout << mat[i][j] << "\t";
}
cout << endl;
}	
}


void fill_matrix(int **mat, int r, int c, int val)
{
for(int i = 0; i < r; i++)
{
for(int j = 0; j < c; j++)
{
mat[i][j] = val;
}
}	
}


void write_matrix_inline(int **mat, int r, int c, ofstream &f)
{
for(int i = 0; i < r; i++)
{
for(int j = 0; j < c; j++)
{
if((i+1)*(j+1) == (r*c))
{
f << mat[i][j];
break;
}
f << mat[i][j] << " ";
}
}
}


void init_gol_map(ifstream &init_data, int **mat)
{
int gol_rows, gol_cols, val;

if(init_data)
{
init_data.clear();
init_data.seekg(0, ios::beg);
init_data >> gol_rows >> gol_cols;
}

for(int i = 0; i < gol_rows; i++)
{
for(int j = 0; j < gol_cols; j++)
{
init_data >> val;
mat[i][j] = val;
}
}
}


void copy_matrix(int **dest_mat, int **copy_mat, int r, int c)
{
for(int i = 0; i < r; i++)
{
for(int j = 0; j < c; j++)
{
dest_mat[i][j] = copy_mat[i][j];
}
}
}



void allocate_gen_matrices(int r, int c)
{
cur_gol_map = new int*[r]; 
nxt_gol_map = new int*[r]; 
for (int i = 0; i < r; i++)
{
cur_gol_map[i] = new int[c];
nxt_gol_map[i] = new int[c];
}
}


void allocate_prx_matrices(int r, int c)
{
prx_gol_map = new int*[r]; 
for (int i = 0; i < r; i++)
{
prx_gol_map[i] = new int[c];
}
}


int get_map_rows(ifstream &init_data)
{
int r, c;

if(init_data)
{
init_data.clear();
init_data.seekg(0, ios::beg);
init_data >> r >> c;
}

return r;
}

int get_map_cols(ifstream &init_data)
{
int r, c;

if(init_data)
{
init_data.clear();
init_data.seekg(0, ios::beg);
init_data >> r >> c;
}

return c;
}

