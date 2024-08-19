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

void pop_org_gol_map(ifstream &init_data, int **org_mat);
void const_ext_gol_map(int **curr_mat, int org_rows, int org_cols, int **ext_mat);
void print_matrix(int **mat, int r, int c);
void fill_matrix(int **mat, int r, int c, int val);
void write_matrix_inline(int **mat, int r, int c, ofstream &f);

int pop_nxt_gol_map(int** ext_mat, int ext_rows, int ext_cols, int **nxt_mat, int th_cnt);
int get_org_gol_rows(ifstream &init_data);
int get_org_gol_cols(ifstream &init_data);

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


if(world_rank == 0)
{
start_time = MPI_Wtime(); 
}


char* cmd_in_file = argv[1]; 
int thread_count = atoi(argv[2]); 
int gens = atoi(argv[3]); 
char* cmd_out_file = argv[4]; 


ifstream in_file; 
ofstream out_file; 

in_file.open(cmd_in_file);
out_file.open(cmd_out_file, ofstream::out | ofstream::trunc); 


int org_gol_rows = get_org_gol_rows(in_file); 
int org_gol_cols = get_org_gol_cols(in_file); 
int org_mat_size = org_gol_rows*org_gol_cols; 

int ext_gol_rows = org_gol_rows + 2;
int ext_gol_cols = org_gol_cols + 2;
int ext_mat_size = ext_gol_rows*ext_gol_cols;

int nxt_gol_rows = org_gol_rows;
int nxt_gol_cols = org_gol_cols;
int nxt_mat_size = nxt_gol_rows*nxt_gol_cols;




if(world_rank == 0)
{
cout << "ROWS:COLS:GENS " << org_gol_rows << ":" << org_gol_cols << ":" << gens<< endl;
}


int **org_gol_map, **ext_gol_map, **nxt_gol_map;

org_gol_map = (int**)malloc(org_gol_rows * sizeof(int*));
ext_gol_map = (int**)malloc(ext_gol_rows * sizeof(int*));
nxt_gol_map = (int**)malloc(nxt_gol_rows * sizeof(int*));

for(int d = 0; d < org_gol_rows; d++)
{
org_gol_map[d] = (int*)malloc(org_gol_cols * sizeof(int));
}
for(int r = 0; r < ext_gol_rows; r++)
{
ext_gol_map[r] = (int*)malloc(ext_gol_cols * sizeof(int));
}
for(int n = 0; n < org_gol_rows; n++)
{
nxt_gol_map[n] = (int*)malloc(nxt_gol_cols * sizeof(int));
}


if(world_rank == 0)
{
cout << "Original Game of Life Map (BEFORE population):" << endl;
fill_matrix(org_gol_map, org_gol_rows, org_gol_cols, -1);
print_matrix(org_gol_map, org_gol_rows, org_gol_cols);


cout << "Extended Game of Life Map (BEFORE population):" << endl;
fill_matrix(ext_gol_map, ext_gol_rows, ext_gol_cols, -2);
print_matrix(ext_gol_map, ext_gol_rows, ext_gol_cols);


cout << "Next Generation Game of Life Map (BEFORE population):" << endl;
fill_matrix(nxt_gol_map, nxt_gol_rows, nxt_gol_cols, -3);
print_matrix(nxt_gol_map, nxt_gol_rows, nxt_gol_cols);	
}


pop_org_gol_map(in_file, org_gol_map); 


if(world_rank == 0)
{
cout << "Original Game of Life Map (Generation 1):" << endl;
print_matrix(org_gol_map, org_gol_rows, org_gol_cols);

init_time = MPI_Wtime() - start_time; 
}


MPI_Barrier(MPI_COMM_WORLD); 


const_ext_gol_map(org_gol_map, org_gol_rows, org_gol_cols, ext_gol_map); 


if(world_rank == 0)
{
cout << "Extended Game of Life Map (Generation 1):" << endl;
print_matrix(ext_gol_map, ext_gol_rows, ext_gol_cols);
}
MPI_Barrier(MPI_COMM_WORLD); 


pop_nxt_gol_map(ext_gol_map, ext_gol_rows, ext_gol_cols, nxt_gol_map, thread_count);


if(world_rank == 0)
{
cout << "Next Generation Game of Life Map (Generation 1):" << endl;
print_matrix(nxt_gol_map, nxt_gol_rows, nxt_gol_cols);
}


MPI_Barrier(MPI_COMM_WORLD); 



for(int g = 2; g <= gens; g++)
{
const_ext_gol_map(nxt_gol_map, nxt_gol_rows, nxt_gol_cols, ext_gol_map); 

if(world_rank == 0)
{
cout << "Extended Game of Life Map: (Generation " << g << ")" << endl;
print_matrix(ext_gol_map, ext_gol_rows, ext_gol_cols);
}
MPI_Barrier(MPI_COMM_WORLD); 


int dead_cells = pop_nxt_gol_map(ext_gol_map, ext_gol_rows, ext_gol_cols, nxt_gol_map, thread_count);


if(world_rank == 0)
{
cout << "Next Generation Game of Life Map: (Generation " << g << ")" << endl;
print_matrix(nxt_gol_map, nxt_gol_rows, nxt_gol_cols);
}
MPI_Barrier(MPI_COMM_WORLD); 


int curr_rank = 0;
while(curr_rank < world_size)
{
if(world_rank == curr_rank)
{
cout << world_rank << ": dead count: " << dead_cells << endl;
}
curr_rank++;
MPI_Barrier(MPI_COMM_WORLD); 
}
MPI_Barrier(MPI_COMM_WORLD);


if(dead_cells == org_gol_rows * org_gol_cols)
{
if(world_rank == 0)
cout << "All cells died at generation " << g << "." << endl;
break;
}

MPI_Barrier(MPI_COMM_WORLD); 
}


if(world_rank == 0)
{
out_file << org_gol_rows << " " << org_gol_cols << endl;
write_matrix_inline(nxt_gol_map, nxt_gol_rows, nxt_gol_cols, out_file);
}

in_file.close();
out_file.close();

MPI_Finalize(); 

return 0;
}





void const_ext_gol_map(int **curr_mat, int org_rows, int org_cols, int **ext_mat)
{
ext_mat[0][0] = curr_mat[org_rows-1][org_cols-1]; 
ext_mat[0][org_cols+1] = curr_mat[org_rows-1][0]; 
ext_mat[org_rows+1][org_cols+1] = curr_mat[0][0]; 
ext_mat[org_rows+1][0] = curr_mat[0][org_cols-1]; 

for(int i = 0; i < org_rows; i++)
{
for(int j = 0; j < org_cols; j++)
{
ext_mat[i+1][j+1] = curr_mat[i][j];
}
}

for(int j = 0; j < org_cols; j++)
{
ext_mat[0][j+1] = curr_mat[org_rows-1][j];
ext_mat[org_rows+1][j+1] = curr_mat[0][j];
}

for(int i = 0; i < org_rows; i++)
{
ext_mat[i+1][0] = curr_mat[i][org_cols-1];
ext_mat[i+1][org_cols+1] = curr_mat[i][0];
}
}


int pop_nxt_gol_map(int** ext_mat, int ext_rows, int ext_cols, int **nxt_mat, int th_cnt)
{
int world_size, world_rank;
MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 


int dead_cnt = 0; 
int org_rows = ext_rows-2;


int mpi_max = floor(org_rows / world_size);
if(th_cnt > mpi_max)
th_cnt = mpi_max;


#pragma omp parallel num_threads(th_cnt)
{
int curr_sum = 0; 
int thread_ID, th_cnt_per_proc;
thread_ID = omp_get_thread_num(); 
th_cnt_per_proc = omp_get_num_threads(); 

int mpi_chunk_sz = floor(org_rows / world_size);
int omp_chunk_sz = floor(mpi_chunk_sz / th_cnt_per_proc);


if(world_rank == world_size - 1)
{
int btm_mpi_chunk_sz = org_rows - ((world_size - 1) * mpi_chunk_sz);
omp_chunk_sz = btm_mpi_chunk_sz / th_cnt_per_proc;
}

int omp_thread_row_start=(mpi_chunk_sz * world_rank) + (omp_chunk_sz * thread_ID);
int omp_thread_row_end = omp_thread_row_start + (omp_chunk_sz-1);


for(int i = omp_thread_row_start + 1; i <= omp_thread_row_end; i++)
{
for(int j = 1; j <= ext_cols-2; j++)
{
curr_sum = ext_mat[i+1][j] + ext_mat[i-1][j] + ext_mat[i][j+1] + ext_mat[i][j-1] + ext_mat[i-1][j+1] + ext_mat[i+1][j-1] + ext_mat[i+1][j+1] + ext_mat[i-1][j-1];

if((curr_sum < 2) && (ext_mat[i][j] == 1)) 
{
nxt_mat[i-1][j-1] = 0; 
dead_cnt++;
}
else if((curr_sum >= 2 && curr_sum <= 3) && (ext_mat[i][j] == 1)) 
{
nxt_mat[i-1][j-1] = 1; 
}
else if(curr_sum > 3 && (ext_mat[i][j] == 1)) 
{
nxt_mat[i-1][j-1] = 0; 
dead_cnt++;
}
else if(curr_sum == 3 && (ext_mat[i][j] == 0)) 
{
nxt_mat[i-1][j-1] = 1; 
}
else
{
nxt_mat[i-1][j-1] = 0; 
dead_cnt++;
}

curr_sum = 0; 
}
}
}

if(world_rank == 0)
return dead_cnt;
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


void pop_org_gol_map(ifstream &init_data, int **org_mat)
{
int org_rows, org_cols, val;

if(init_data)
{
init_data.clear();
init_data.seekg(0, ios::beg);
init_data >> org_rows >> org_cols;
}

for(int i = 0; i < org_rows; i++)
{
for(int j = 0; j < org_cols; j++)
{
init_data >> val;
org_mat[i][j] = val;
}
}
}



int get_org_gol_rows(ifstream &init_data)
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

int get_org_gol_cols(ifstream &init_data)
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

