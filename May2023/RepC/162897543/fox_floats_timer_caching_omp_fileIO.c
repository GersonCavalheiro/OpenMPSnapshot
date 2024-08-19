#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#define PRINT_A 1
#define PRINT_B 1
#define PRINT_C 1
#define PRINT_LOCAL_A 1
#define PRINT_LOCAL_B 1
#define PRINT_LOCAL_C 1
#define FLOAT double
#define FLOAT_MPI MPI_DOUBLE
#define NUM_THREADS 2
#define AFFINITY "KMP_AFFINITY = compact"
typedef struct {
int       p;             
MPI_Comm  comm;          
MPI_Comm  row_comm;      
MPI_Comm  col_comm;      
int       q;             
int       my_row;        
int       my_col;        
int       my_rank;       
} GRID_INFO_T;             
#define MAX 65536  
typedef struct {
int     n_bar;
#define Order(A) ((A)->n_bar)                                        
FLOAT  entries[MAX];
#define Entry(A,i,j) (*(((A)->entries) + ((A)->n_bar)*(i) + (j)))    
} LOCAL_MATRIX_T;
LOCAL_MATRIX_T*  Local_matrix_allocate(int n_bar);
void             Free_local_matrix(LOCAL_MATRIX_T** local_A);
void             Read_matrix_A(char* prompt, LOCAL_MATRIX_T* local_A, 
GRID_INFO_T* grid, int n);                          
void             Read_matrix_B(char* prompt, LOCAL_MATRIX_T* local_B,    
GRID_INFO_T* grid, int n);                          
void             Print_matrix_A(char* title, LOCAL_MATRIX_T* local_A,     
GRID_INFO_T* grid, int n);                          
void             Print_matrix_B(char* title, LOCAL_MATRIX_T* local_B,    
GRID_INFO_T* grid, int n);                          
void             Print_matrix_C(char* title, LOCAL_MATRIX_T* local_C,     
GRID_INFO_T* grid, int n);                          
void             Set_to_zero(LOCAL_MATRIX_T* local_A);
void             Local_matrix_multiply(LOCAL_MATRIX_T* local_A,
LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C);
void             Build_matrix_type(LOCAL_MATRIX_T* local_A);
MPI_Datatype     local_matrix_mpi_t;    
LOCAL_MATRIX_T*  temp_mat;       
void             Print_local_matrices_A(char* title, LOCAL_MATRIX_T* local_A, 
GRID_INFO_T* grid);
void             Print_local_matrices_B(char* title, LOCAL_MATRIX_T* local_B, 
GRID_INFO_T* grid);
void             Print_local_matrices_C(char* title, LOCAL_MATRIX_T* local_B, 
GRID_INFO_T* grid);
void             Write_matrix_C(char* title, LOCAL_MATRIX_T* local_C, 
GRID_INFO_T* grid, int n);                               
void             Write_local_matrices_A(char* title, LOCAL_MATRIX_T* local_A, 
GRID_INFO_T* grid);                                      
void             Write_local_matrices_B(char* title, LOCAL_MATRIX_T* local_B, 
GRID_INFO_T* grid);                                      
void             Write_local_matrices_C(char* title, LOCAL_MATRIX_T* local_A, 
GRID_INFO_T* grid);                                      
main(int argc, char* argv[]) {
FILE             *fp;
int              p;
int              my_rank;
GRID_INFO_T      grid;
LOCAL_MATRIX_T*  local_A;
LOCAL_MATRIX_T*  local_B;
LOCAL_MATRIX_T*  local_C;
int              n;
int              n_bar;
double           timer_start;
double           timer_end;
int              content;
void Setup_grid(GRID_INFO_T*  grid);
void Fox(int n, GRID_INFO_T* grid, LOCAL_MATRIX_T* local_A,
LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C);
MPI_Init(&argc, &argv);                              
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);             
omp_set_num_threads(NUM_THREADS);
kmp_set_defaults(AFFINITY);
Setup_grid(&grid);                                   
if (my_rank == 0) {
fp = fopen("A.dat","r");
n = 0;
while((content = fgetc(fp)) != EOF)
{
if(content != 0x20 && content != 0x0A) n++;
}
fclose(fp);
n = (int) sqrt((double) n); 
printf("We read the order of the matrices from A.dat is\n %d\n", n);
}
MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);        
n_bar = n/grid.q;                                    
local_A = Local_matrix_allocate(n_bar);              
Order(local_A) = n_bar;                              
Read_matrix_A("Read A from A.dat", local_A, &grid, n);  
if (PRINT_A == 1)
Print_matrix_A("We read A =", local_A, &grid, n);
local_B = Local_matrix_allocate(n_bar);              
Order(local_B) = n_bar;                              
Read_matrix_B("Read B from B.dat", local_B, &grid, n);         
if (PRINT_B == 1)
Print_matrix_B("We read B =", local_B, &grid, n);
Build_matrix_type(local_A);                          
temp_mat = Local_matrix_allocate(n_bar);             
local_C = Local_matrix_allocate(n_bar);              
Order(local_C) = n_bar;                              
MPI_Barrier(MPI_COMM_WORLD);                         
timer_start = MPI_Wtime();                           
Fox(n, &grid, local_A, local_B, local_C);            
timer_end = MPI_Wtime();                             
MPI_Barrier(MPI_COMM_WORLD);                         
Write_matrix_C("Write C into the C.dat", local_C, &grid, n); 
if (PRINT_C == 1)
Print_matrix_C("The product is", local_C, &grid, n); 
Write_local_matrices_A("Write split of local matrix A into local_A.dat", 
local_A, &grid);                
if (PRINT_LOCAL_A == 1)
Print_local_matrices_A("Split of local matrix A", 
local_A, &grid);                
Write_local_matrices_B("Write split of local matrix B into local_B.dat", 
local_B, &grid);                
if (PRINT_LOCAL_B == 1)
Print_local_matrices_B("Split of local matrix B", 
local_B, &grid);                
Write_local_matrices_C("Write split of local matrix C into local_C.dat", 
local_C, &grid);                
if (PRINT_LOCAL_C == 1)
Print_local_matrices_C("Split of local matrix C", 
local_C, &grid);                
Free_local_matrix(&local_A);                         
Free_local_matrix(&local_B);                         
Free_local_matrix(&local_C);                         
if(my_rank == 0)  
printf("Parallel Fox Matrix Multiplication Elapsed time:\n %30.20E seconds\n", timer_end-timer_start);
MPI_Finalize();                                      
}  
void Setup_grid(
GRID_INFO_T*  grid  ) {
int old_rank;
int dimensions[2];
int wrap_around[2];
int coordinates[2];
int free_coords[2];
MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);
grid->q = (int) sqrt((double) grid->p); 
dimensions[0] = dimensions[1] = grid->q; 
wrap_around[0] = wrap_around[1] = 1;
MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, 
wrap_around, 1, &(grid->comm));
MPI_Comm_rank(grid->comm, &(grid->my_rank));
MPI_Cart_coords(grid->comm, grid->my_rank, 2, 
coordinates);
grid->my_row = coordinates[0];
grid->my_col = coordinates[1];
free_coords[0] = 0; 
free_coords[1] = 1;
MPI_Cart_sub(grid->comm, free_coords, 
&(grid->row_comm));
free_coords[0] = 1; 
free_coords[1] = 0;
MPI_Cart_sub(grid->comm, free_coords, 
&(grid->col_comm));
} 
void Fox(
int              n         , 
GRID_INFO_T*     grid      , 
LOCAL_MATRIX_T*  local_A   ,
LOCAL_MATRIX_T*  local_B   ,
LOCAL_MATRIX_T*  local_C   ) {
LOCAL_MATRIX_T*  temp_A; 
int              stage;
int              bcast_root;
int              n_bar;  
int              source;
int              dest;
MPI_Status       status;
n_bar = n/grid->q;
Set_to_zero(local_C);
source = (grid->my_row + 1) % grid->q;
dest = (grid->my_row + grid->q - 1) % grid->q;
temp_A = Local_matrix_allocate(n_bar);
for (stage = 0; stage < grid->q; stage++) {
bcast_root = (grid->my_row + stage) % grid->q;
if (bcast_root == grid->my_col) {
MPI_Bcast(local_A, 1, local_matrix_mpi_t,
bcast_root, grid->row_comm);
Local_matrix_multiply(local_A, local_B, 
local_C);
} else {
MPI_Bcast(temp_A, 1, local_matrix_mpi_t,
bcast_root, grid->row_comm);
Local_matrix_multiply(temp_A, local_B, 
local_C);
}
MPI_Sendrecv_replace(local_B, 1, local_matrix_mpi_t,
dest, 0, source, 0, grid->col_comm, &status);
} 
} 
LOCAL_MATRIX_T* Local_matrix_allocate(int local_order) {
LOCAL_MATRIX_T* temp;
temp = (LOCAL_MATRIX_T*) malloc(sizeof(LOCAL_MATRIX_T));
return temp;
}  
void Free_local_matrix(
LOCAL_MATRIX_T** local_A_ptr  ) {
free(*local_A_ptr);
}  
void Read_matrix_A(
char*            prompt   , 
LOCAL_MATRIX_T*  local_A  ,
GRID_INFO_T*     grid     ,
int              n        ) {
FILE *fp;
int        mat_row, mat_col;
int        grid_row, grid_col;
int        dest;
int        coords[2];
FLOAT*     temp;
MPI_Status status;
if (grid->my_rank == 0) {  
fp = fopen("A.dat","r");
temp = (FLOAT*) malloc(Order(local_A)*sizeof(FLOAT));
printf("%s\n", prompt);
fflush(stdout);
for (mat_row = 0;  mat_row < n; mat_row++) {
grid_row = mat_row/Order(local_A);
coords[0] = grid_row;
for (grid_col = 0; grid_col < grid->q; grid_col++) {
coords[1] = grid_col;
MPI_Cart_rank(grid->comm, coords, &dest);
if (dest == 0) {
for (mat_col = 0; mat_col < Order(local_A); mat_col++)
fscanf(fp, "%lf", 
(local_A->entries)+mat_row*Order(local_A)+mat_col);
} else {
for(mat_col = 0; mat_col < Order(local_A); mat_col++)
fscanf(fp,"%lf", temp + mat_col);
MPI_Send(temp, Order(local_A), FLOAT_MPI, dest, 0,
grid->comm);
}
}
}
free(temp);
fclose(fp);
} else {  
for (mat_row = 0; mat_row < Order(local_A); mat_row++) 
MPI_Recv(&Entry(local_A, mat_row, 0), Order(local_A), 
FLOAT_MPI, 0, 0, grid->comm, &status);
}
}  
void Read_matrix_B(
char*            prompt   , 
LOCAL_MATRIX_T*  local_B  ,
GRID_INFO_T*     grid     ,
int              n        ) {
FILE       *fp;
int        mat_row, mat_col;
int        grid_row, grid_col;
int        dest;
int        coords[2];
FLOAT      *temp;
MPI_Status status;
if (grid->my_rank == 0) {  
fp = fopen("B.dat","r");
temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));
printf("%s\n", prompt);
fflush(stdout);
for (mat_row = 0;  mat_row < n; mat_row++) {
grid_row = mat_row/Order(local_B);
coords[0] = grid_row;
for (grid_col = 0; grid_col < grid->q; grid_col++) {
coords[1] = grid_col;
MPI_Cart_rank(grid->comm, coords, &dest);
if (dest == 0) {                                                    
for (mat_col = 0; mat_col < Order(local_B); mat_col++)
fscanf(fp, "%lf", 
(local_B->entries)+mat_col*Order(local_B)+mat_row);       
} else {
for(mat_col = 0; mat_col < Order(local_B); mat_col++)
fscanf(fp, "%lf", temp + mat_col);
MPI_Send(temp, Order(local_B), FLOAT_MPI, dest, 0,
grid->comm);
}
}
}
free(temp);
fclose(fp);
} else {  
temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));               
for (mat_col = 0; mat_col < Order(local_B); mat_col++) { 
MPI_Recv(temp, Order(local_B), 
FLOAT_MPI, 0, 0, grid->comm, &status);                      
for(mat_row = 0; mat_row < Order(local_B); mat_row++)
Entry(local_B, mat_row, mat_col) = *(temp + mat_row);       
}
free(temp);
}
}  
void Print_matrix_A(
char*            title    ,  
LOCAL_MATRIX_T*  local_A  ,
GRID_INFO_T*     grid     ,
int              n        ) {
int        mat_row, mat_col;
int        grid_row, grid_col;
int        source;
int        coords[2];
FLOAT*     temp;
MPI_Status status;
if (grid->my_rank == 0) {
temp = (FLOAT*) malloc(Order(local_A)*sizeof(FLOAT));
printf("%s\n", title);
for (mat_row = 0;  mat_row < n; mat_row++) {
grid_row = mat_row/Order(local_A);
coords[0] = grid_row;
for (grid_col = 0; grid_col < grid->q; grid_col++) {
coords[1] = grid_col;
MPI_Cart_rank(grid->comm, coords, &source);
if (source == 0) {
for(mat_col = 0; mat_col < Order(local_A); mat_col++)
printf("%20.15E ", Entry(local_A, mat_row, mat_col));
} else {
MPI_Recv(temp, Order(local_A), FLOAT_MPI, source, 0,
grid->comm, &status);
for(mat_col = 0; mat_col < Order(local_A); mat_col++)
printf("%20.15E ", temp[mat_col]);
}
}
printf("\n");
}
free(temp);
} else {
for (mat_row = 0; mat_row < Order(local_A); mat_row++) 
MPI_Send(&Entry(local_A, mat_row, 0), Order(local_A), 
FLOAT_MPI, 0, 0, grid->comm);
}
}  
void Print_matrix_B(
char*            title    ,  
LOCAL_MATRIX_T*  local_B  ,
GRID_INFO_T*     grid     ,
int              n        ) {
int        mat_row, mat_col;
int        grid_row, grid_col;
int        source;
int        coords[2];
FLOAT*     temp;
MPI_Status status;
if (grid->my_rank == 0) {
temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));
printf("%s\n", title);
for (mat_row = 0;  mat_row < n; mat_row++) {
grid_row = mat_row/Order(local_B);
coords[0] = grid_row;
for (grid_col = 0; grid_col < grid->q; grid_col++) {
coords[1] = grid_col;
MPI_Cart_rank(grid->comm, coords, &source);
if (source == 0) {
for(mat_col = 0; mat_col < Order(local_B); mat_col++)
printf("%20.15E ", Entry(local_B, mat_col, mat_row));      
} else {
MPI_Recv(temp, Order(local_B), FLOAT_MPI, source, 0,
grid->comm, &status);
for(mat_col = 0; mat_col < Order(local_B); mat_col++)
printf("%20.15E ", temp[mat_col]);
}
}
printf("\n");
}
free(temp);
} else {
temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));
for (mat_col = 0; mat_col < Order(local_B); mat_col++) { 
for(mat_row = 0; mat_row < Order(local_B); mat_row++)
*(temp+mat_row) = Entry(local_B, mat_row, mat_col);       
MPI_Send(temp, Order(local_B), FLOAT_MPI, 0, 0, grid->comm);
}
free(temp);
}
}  
void Print_matrix_C(
char*            title    ,  
LOCAL_MATRIX_T*  local_C  ,
GRID_INFO_T*     grid     ,
int              n        ) {
int        mat_row, mat_col;
int        grid_row, grid_col;
int        source;
int        coords[2];
FLOAT*     temp;
MPI_Status status;
if (grid->my_rank == 0) {
temp = (FLOAT*) malloc(Order(local_C)*sizeof(FLOAT));
printf("%s\n", title);
for (mat_row = 0;  mat_row < n; mat_row++) {
grid_row = mat_row/Order(local_C);
coords[0] = grid_row;
for (grid_col = 0; grid_col < grid->q; grid_col++) {
coords[1] = grid_col;
MPI_Cart_rank(grid->comm, coords, &source);
if (source == 0) {
for(mat_col = 0; mat_col < Order(local_C); mat_col++)
printf("%20.15E ", Entry(local_C, mat_row, mat_col));
} else {
MPI_Recv(temp, Order(local_C), FLOAT_MPI, source, 0,
grid->comm, &status);
for(mat_col = 0; mat_col < Order(local_C); mat_col++)
printf("%20.15E ", temp[mat_col]);
}
}
printf("\n");
}
free(temp);
} else {
for (mat_row = 0; mat_row < Order(local_C); mat_row++) 
MPI_Send(&Entry(local_C, mat_row, 0), Order(local_C), 
FLOAT_MPI, 0, 0, grid->comm);
}
}  
void Write_matrix_C(
char*            title    ,  
LOCAL_MATRIX_T*  local_C  ,
GRID_INFO_T*     grid     ,
int              n        ) {
FILE      *fp;
int        mat_row, mat_col;
int        grid_row, grid_col;
int        source;
int        coords[2];
FLOAT*     temp;
MPI_Status status;
if (grid->my_rank == 0) {
fp = fopen("C.dat", "w+");
temp = (FLOAT*) malloc(Order(local_C)*sizeof(FLOAT));
printf("%s\n", title);
for (mat_row = 0;  mat_row < n; mat_row++) {
grid_row = mat_row/Order(local_C);
coords[0] = grid_row;
for (grid_col = 0; grid_col < grid->q; grid_col++) {
coords[1] = grid_col;
MPI_Cart_rank(grid->comm, coords, &source);
if (source == 0) {
for(mat_col = 0; mat_col < Order(local_C); mat_col++)
fprintf(fp, "%20.15E ", Entry(local_C, mat_row, mat_col));
} else {
MPI_Recv(temp, Order(local_C), FLOAT_MPI, source, 0,
grid->comm, &status);
for(mat_col = 0; mat_col < Order(local_C); mat_col++)
fprintf(fp, "%20.15E ", temp[mat_col]);
}
}
fprintf(fp,"\n");
}
free(temp);
fclose(fp);
} else {
for (mat_row = 0; mat_row < Order(local_C); mat_row++) 
MPI_Send(&Entry(local_C, mat_row, 0), Order(local_C), 
FLOAT_MPI, 0, 0, grid->comm);
}
}  
void Set_to_zero(
LOCAL_MATRIX_T*  local_A  ) {
int i, j;
for (i = 0; i < Order(local_A); i++)
for (j = 0; j < Order(local_A); j++)
Entry(local_A,i,j) = 0.0E0;
}  
void Build_matrix_type(
LOCAL_MATRIX_T*  local_A  ) {
MPI_Datatype  temp_mpi_t;
int           block_lengths[2];
MPI_Aint      displacements[2];
MPI_Datatype  typelist[2];
MPI_Aint      start_address;
MPI_Aint      address;
MPI_Type_contiguous(Order(local_A)*Order(local_A), 
FLOAT_MPI, &temp_mpi_t);                         
block_lengths[0] = block_lengths[1] = 1;
typelist[0] = MPI_INT;
typelist[1] = temp_mpi_t;
MPI_Address(local_A, &start_address);                 
MPI_Address(&(local_A->n_bar), &address);
displacements[0] = address - start_address;
MPI_Address(local_A->entries, &address);
displacements[1] = address - start_address;
MPI_Type_struct(2, block_lengths, displacements,
typelist, &local_matrix_mpi_t);                   
MPI_Type_commit(&local_matrix_mpi_t);                 
}  
void Local_matrix_multiply(
LOCAL_MATRIX_T*  local_A  ,
LOCAL_MATRIX_T*  local_B  , 
LOCAL_MATRIX_T*  local_C  ) {
int i, j, k;
#pragma omp parallel for private(i, j, k) shared(local_A, local_B, local_C) num_threads(NUM_THREADS)       
for (i = 0; i < Order(local_A); i++) {
for (j = 0; j < Order(local_A); j++)              
for (k = 0; k < Order(local_B); k++)
Entry(local_C,i,j) = Entry(local_C,i,j)             
+ Entry(local_A,i,k)*Entry(local_B,j,k);        
}
}  
void Print_local_matrices_A(
char*            title    ,
LOCAL_MATRIX_T*  local_A  , 
GRID_INFO_T*     grid     ) {
int         coords[2];
int         i, j;
int         source;
MPI_Status  status;
if (grid->my_rank == 0) {
printf("%s\n", title);
printf("Process %d > grid_row = %d, grid_col = %d\n",
grid->my_rank, grid->my_row, grid->my_col);
for (i = 0; i < Order(local_A); i++) {
for (j = 0; j < Order(local_A); j++)
printf("%20.15E ", Entry(local_A,i,j));
printf("\n");
}
for (source = 1; source < grid->p; source++) {
MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
grid->comm, &status);
MPI_Cart_coords(grid->comm, source, 2, coords);
printf("Process %d > grid_row = %d, grid_col = %d\n",
source, coords[0], coords[1]);
for (i = 0; i < Order(temp_mat); i++) {
for (j = 0; j < Order(temp_mat); j++)
printf("%20.15E ", Entry(temp_mat,i,j));
printf("\n");
}
}
fflush(stdout);
} else {
MPI_Send(local_A, 1, local_matrix_mpi_t, 0, 0, grid->comm);
}
}  
void Print_local_matrices_B(
char*            title    ,
LOCAL_MATRIX_T*  local_B  , 
GRID_INFO_T*     grid     ) {
int         coords[2];
int         i, j;
int         source;
MPI_Status  status;
if (grid->my_rank == 0) {
printf("%s\n", title);
printf("Process %d > grid_row = %d, grid_col = %d\n",
grid->my_rank, grid->my_row, grid->my_col);
for (i = 0; i < Order(local_B); i++) {
for (j = 0; j < Order(local_B); j++)
printf("%20.15E ", Entry(local_B,j,i));                   
printf("\n");
}
for (source = 1; source < grid->p; source++) {
MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
grid->comm, &status);
MPI_Cart_coords(grid->comm, source, 2, coords);
printf("Process %d > grid_row = %d, grid_col = %d\n",
source, coords[0], coords[1]);
for (i = 0; i < Order(temp_mat); i++) {
for (j = 0; j < Order(temp_mat); j++)
printf("%20.15E ", Entry(temp_mat,j,i));             
printf("\n");
}
}
fflush(stdout);
} else {
MPI_Send(local_B, 1, local_matrix_mpi_t, 0, 0, grid->comm);
}
}  
void Print_local_matrices_C(
char*            title    ,
LOCAL_MATRIX_T*  local_C  , 
GRID_INFO_T*     grid     ) {
int         coords[2];
int         i, j;
int         source;
MPI_Status  status;
if (grid->my_rank == 0) {
printf("%s\n", title);
printf("Process %d > grid_row = %d, grid_col = %d\n",
grid->my_rank, grid->my_row, grid->my_col);
for (i = 0; i < Order(local_C); i++) {
for (j = 0; j < Order(local_C); j++)
printf("%20.15E ", Entry(local_C,i,j));
printf("\n");
}
for (source = 1; source < grid->p; source++) {
MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
grid->comm, &status);
MPI_Cart_coords(grid->comm, source, 2, coords);
printf("Process %d > grid_row = %d, grid_col = %d\n",
source, coords[0], coords[1]);
for (i = 0; i < Order(temp_mat); i++) {
for (j = 0; j < Order(temp_mat); j++)
printf("%20.15E ", Entry(temp_mat,i,j));
printf("\n");
}
}
fflush(stdout);
} else {
MPI_Send(local_C, 1, local_matrix_mpi_t, 0, 0, grid->comm);
}
}  
void Write_local_matrices_A(
char*            title    ,
LOCAL_MATRIX_T*  local_A  , 
GRID_INFO_T*     grid     ) {
FILE        *fp;
int         coords[2];
int         i, j;
int         source;
MPI_Status  status;
if (grid->my_rank == 0) {
fp = fopen("local_A.dat","w+");
printf("%s\n", title);
fprintf(fp,"Process %d > grid_row = %d, grid_col = %d\n",
grid->my_rank, grid->my_row, grid->my_col);
for (i = 0; i < Order(local_A); i++) {
for (j = 0; j < Order(local_A); j++)
fprintf(fp,"%20.15E ", Entry(local_A,i,j));
fprintf(fp, "\n");
}
for (source = 1; source < grid->p; source++) {
MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
grid->comm, &status);
MPI_Cart_coords(grid->comm, source, 2, coords);
fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
source, coords[0], coords[1]);
for (i = 0; i < Order(temp_mat); i++) {
for (j = 0; j < Order(temp_mat); j++)
fprintf(fp, "%20.15E ", Entry(temp_mat,i,j));
fprintf(fp, "\n");
}
}
fflush(stdout);
fclose(fp);
} else {
MPI_Send(local_A, 1, local_matrix_mpi_t, 0, 0, grid->comm);
}
}  
void Write_local_matrices_B(
char*            title    ,
LOCAL_MATRIX_T*  local_B  , 
GRID_INFO_T*     grid     ) {
FILE        *fp;
int         coords[2];
int         i, j;
int         source;
MPI_Status  status;
if (grid->my_rank == 0) {
fp = fopen("local_B.dat","w+");
printf("%s\n", title);
fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
grid->my_rank, grid->my_row, grid->my_col);
for (i = 0; i < Order(local_B); i++) {
for (j = 0; j < Order(local_B); j++)
fprintf(fp, "%20.15E ", Entry(local_B,j,i));                   
fprintf(fp, "\n");
}
for (source = 1; source < grid->p; source++) {
MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
grid->comm, &status);
MPI_Cart_coords(grid->comm, source, 2, coords);
fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
source, coords[0], coords[1]);
for (i = 0; i < Order(temp_mat); i++) {
for (j = 0; j < Order(temp_mat); j++)
fprintf(fp, "%20.15E ", Entry(temp_mat,j,i));             
fprintf(fp, "\n");
}
}
fflush(stdout);
fclose(fp);
} else {
MPI_Send(local_B, 1, local_matrix_mpi_t, 0, 0, grid->comm);
}
}  
void Write_local_matrices_C(
char*            title    ,
LOCAL_MATRIX_T*  local_C  , 
GRID_INFO_T*     grid     ) {
FILE        *fp;
int         coords[2];
int         i, j;
int         source;
MPI_Status  status;
if (grid->my_rank == 0) {
fp = fopen("local_C.dat","w+");
printf("%s\n", title);
fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
grid->my_rank, grid->my_row, grid->my_col);
for (i = 0; i < Order(local_C); i++) {
for (j = 0; j < Order(local_C); j++)
fprintf(fp, "%20.15E ", Entry(local_C,i,j));
fprintf(fp, "\n");
}
for (source = 1; source < grid->p; source++) {
MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
grid->comm, &status);
MPI_Cart_coords(grid->comm, source, 2, coords);
fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
source, coords[0], coords[1]);
for (i = 0; i < Order(temp_mat); i++) {
for (j = 0; j < Order(temp_mat); j++)
fprintf(fp, "%20.15E ", Entry(temp_mat,i,j));
fprintf(fp, "\n");
}
}
fflush(stdout);
fclose(fp);
} else {
MPI_Send(local_C, 1, local_matrix_mpi_t, 0, 0, grid->comm);
}
}  
