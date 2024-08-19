#include "main.h"
#include "gifenc.h" 
int main(int argc, char const *argv[]) {
gettimeofday(&start, NULL); 
int *columns = malloc(sizeof(int));
int *rows = malloc(sizeof(int));
int iterations;
int **matrix;
int **temp_matrix;
bool use_pgm = false;
FILE *file_pointer;
GIF *gif; 
if (argc < 3) {
printf("Include initial state file path and number of iterations\n");
exit(1);
}
file_pointer = fopen(argv[1], "r");
if (file_pointer == NULL) {
printf("Error opening file '%s'\n", argv[1]);
exit(1);
}
if (argc == 4 && strncmp("-pgm", argv[3], 4) == 0) {
use_pgm = true;
}
iterations = atoi(argv[2]);
read_size(columns, rows, file_pointer);
matrix = init_matrix(*columns, *rows);
temp_matrix = init_matrix(*columns, *rows);
read_content(columns, rows, file_pointer, matrix);
if (use_pgm) {
generate_pgm(matrix, 0, *columns, *rows);
} else {
gif = new_gif(
"biography.gif",  
*columns, *rows,           
(uint8_t []) {  
0x00, 0x00, 0x00,   
0xFF, 0xFF, 0xFF,   
},
1,              
0               
);
}
int count = 0;
while (count < iterations) {
count++;
#pragma omp parallel for collapse(2) shared(temp_matrix,matrix)
for (int x = 0; x < *rows; x++) {
for (int y = 0; y < *columns; y++) {
int neighbours = 0;
if (x > 0) {
neighbours+= matrix[x-1][y];
if (y > 0) neighbours+= matrix[x-1][y-1];
if (y < *(rows)-1) neighbours+= matrix[x-1][y+1];
}
if (y > 0) neighbours+= matrix[x][y-1];
if (x < *(rows)-1) {
neighbours+= matrix[x+1][y];
if (y < *(rows)-1) neighbours+= matrix[x+1][y+1];
if (y > 0) neighbours+= matrix[x+1][y-1];
}
if (y < *(rows)-1) neighbours+= matrix[x][y+1];
if (neighbours < 2) temp_matrix[x][y] = 0; 
if (matrix[x][y] == 1 && neighbours == 2 || matrix[x][y] == 1 && neighbours == 3) temp_matrix[x][y] = 1; 
if (neighbours > 3) temp_matrix[x][y] = 0; 
if (neighbours == 3) temp_matrix[x][y] = 1; 
}
}
if (use_pgm) {
generate_pgm(temp_matrix, count, *columns, *rows); 
} else {
for (int j = 0; j < (*columns)*(*rows); j++) {
gif->frame[j] = (uint8_t)temp_matrix[j%(*columns)][j/(*columns)];
}
add_frame(gif, 10);
}
copy_matrix(temp_matrix, matrix, *columns, *rows);
}
if (!use_pgm) {
close_gif(gif);
}
free_matrix(matrix, *rows);
free_matrix(temp_matrix, *rows);
free_dimensions(columns, rows);
gettimeofday(&end, NULL);
double delta = difftime(end.tv_sec, start.tv_sec) + ((double)end.tv_usec - (double)start.tv_usec) / 1000000.0;
printf("Finished in: %f\n",delta);
return 0;
}
void free_dimensions(int * columns, int * rows) {
free(columns);
free(rows);
}
int ** init_matrix(int columns, int rows) {
int ** matrix  = (int **) malloc(sizeof(int *)*columns);
for (int i = 0; i < columns; i++) {
matrix[i] = (int *) malloc(sizeof(int)*rows);
}
return matrix;
}
void free_matrix(int ** matrix, int rows) {
for (int i = 0; i < rows; i++) {
free(matrix[i]);
}
free(matrix);
}
void copy_matrix(int ** origin_matrix, int ** dest_matrix, int rows, int columns) {
#pragma omp parallel for collapse(2) shared(origin_matrix,dest_matrix)
for (int x = 0; x < rows; x++) {
for (int y = 0; y < columns; y++) {
dest_matrix[x][y] = origin_matrix[x][y];
}
}
}
void generate_pgm(int ** matrix, int iteration, int columns, int rows) {
char filename[50];
sprintf(filename, "./build/state-%d.pgm", iteration);
FILE *fp;
fp = fopen(filename, "w");
fprintf(fp, "P2\n# GAME OF LIFE - ITERATION; %d\n", iteration);
fprintf(fp, "%d %d\n1\n", columns, rows);
for (int i = 0; i < (rows); i++) {
for (int j = 0; j < (columns); j++) {
fprintf(fp, "%d ", matrix[j][i]);
}
fputs("\n", fp);
}
fclose(fp);
}
