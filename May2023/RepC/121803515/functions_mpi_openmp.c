#include "header_mpi_openmp.h"
void initializeRepresentation(int argc, char** argv, char** filename, int* rows, int* columns) {
if (argc == 1) {  
*rows = DEFAULT_ROWS;
*columns = DEFAULT_COLUMNS;
} else if (argc == 7) {
int i, rflag = 0, cflag = 0, fflag = 0;
for (i = 1; i < argc-1; i += 2) {   
if (!fflag && !strcmp(argv[i], "-f")) { 
fflag = 1;
*filename = malloc(strlen(argv[i+1])+1);
strcpy(*filename, argv[i+1]);
} else if (!rflag && !strcmp(argv[i], "-r")) { 
rflag = 1;
*rows = atoi(argv[i+1]);
} else if (!cflag && !strcmp(argv[i], "-c")) { 
cflag = 1;
*columns = atoi(argv[i+1]);
} else {
fprintf(stderr, "ERROR! Bad argument formation!\n");
MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
exit(EXIT_FAILURE);
}
}
} else {
fprintf(stderr, "%s: Error: Insufficient number of arguments\n", argv[0]);
MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
exit(EXIT_FAILURE);
}
}
int divideWorkload(int rows, int columns, int workers) {
int perimeter, r, c, best = 0;          
int min_perimeter = rows + columns + 1; 
for (r = 1; r <= workers; ++r) {
if ((workers % r != 0) || (rows % r != 0)) continue;
c = workers / r;
if (columns % c != 0) continue;
perimeter = rows / r + columns / c;
if (perimeter < min_perimeter) {
min_perimeter = perimeter;
best = r;
}
}
return best;  
}
void readInitialState(char* filename, char* matrix, int starting_row, int starting_column, int proc_rows, int proc_columns, int rows, int columns) {
FILE *fd;
if (filename)
if ((fd = fopen(filename, "r")) == NULL) {
perror("Error: Failed to open file for input");
MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
exit(EXIT_FAILURE);
}
int i,j;
for (i = 1; i <= proc_rows; i++)
for (j = 1; j <= proc_columns; j++)
matrix[(proc_columns+2) * i + j] = 0;
if (filename) {
while (fscanf(fd, "%d %d\n", &i, &j) != EOF) {
if ((i > starting_row) && (i <= starting_row + proc_rows) && (j > starting_column) && (j <= starting_column + proc_columns)) {
matrix[(proc_columns+2) * i + j] = 1;
} else if (i > rows || j > columns) {
fprintf(stderr, "Error: Bad input data at file %s", filename);
MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
exit(EXIT_FAILURE); 
}
}
fclose(fd);
} else {
perror("Error: Failed to open file for input");
MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
exit(EXIT_FAILURE);
}
}
void setRandomInitialState(char* matrix, int proc_rows, int proc_columns) {
int i, j;
for(i = 1; i <= proc_rows; i++)   
for(j = 1; j <= proc_columns; j++)
matrix[(proc_columns+2) * i + j] = DEAD;
int num_of_organisms = rand() % (proc_rows + proc_columns) + 1;
printf("initial number of organisms = %d\n", num_of_organisms);
int counter = 0;
for (counter = 0; counter < num_of_organisms; counter++) { 
do {
i = rand() % proc_rows + 1;
j = rand() % proc_columns + 1;
} while (matrix[(proc_columns+2)* i + j] == ALIVE);  
matrix[(proc_columns+2) * i + j] = ALIVE;      
}
}
int nextGeneration(char* before, char* after, int first_row, int last_row, int first_column, int last_column, int columns) {
int i, j, changed = 0;
#pragma omp parallel for shared(gen1, gen2) schedule(static) collapse(3)
for (i = first_row; i <= last_row; i++) {
for (j = first_column; j <= last_column; j++) {
int neighbors = before[columns*(i-1)+j] + before[columns*i+(j-1)] + before[columns*(i+1)+j] + before[columns*i+(j+1)]
+ before[columns*(i-1)+(j+1)] + before[columns*(i+1)+(j-1)] + before[columns*(i-1)+(j-1)] + before[columns*(i+1)+(j+1)];
if (before[columns*i+j] == ALIVE) {     
if (neighbors <= 1) after[columns*i+j] = DEAD;  
else if (neighbors <= 3) after[columns*i+j] = ALIVE; 
else after[columns*i+j] = DEAD;   
} else {               
if (neighbors == 3) after[columns*i+j] = ALIVE;  
else after[columns*i+j] = DEAD;   
}
if (before[columns*i+j] != after[columns*i+j]) changed++;
}
}
return changed; 
}
char noneAlive(char* matrix, int proc_rows, int proc_columns) {
int i, j;
for (i = 1; i <= proc_rows; i++)
for (j = 1; j <= proc_columns; j++)
if (matrix[(proc_columns+2) * i + j] == ALIVE)  
return 0;
return 1;   
}
char sameGenerations(char* gen1, char* gen2, int proc_rows, int proc_columns) {
int i, j;
#pragma omp parallel for shared(gen1, gen2) schedule(static) collapse(3)
for (i = 1; i <= proc_rows; i++)
for (j = 1; j <= proc_columns; j++)
if (gen1[(proc_columns+2) * i + j] != gen2[(proc_columns+2) * i + j]) 
return 0;
return 1;   
}
char *locate(char *matrix, int row, int column, int columns) {
return &matrix[columns * row + column];
}
void printSubmatrix(int my_rank, char* matrix, int proc_rows, int proc_columns) {
printf("my_rank = %d\n", my_rank);
int i, j;
for (i = 1; i <= proc_rows; i++) {
for (j = 1; j <= proc_columns; j++)
printf("%d ", matrix[(proc_columns+2) * i + j]);
printf("\n");
}
}