#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
const int MAX_LINES = 1000;
const int MAX_LINE = 80;
void Usage(char* prog_name);
void Get_text(char* lines[], int* line_count_p);
void Tokenize(char* lines[], int line_count, int thread_count); 
int main(int argc, char* argv[]) {
int thread_count, i;
char* lines[1000];
int line_count;
if (argc != 2) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
printf("Enter text\n");
Get_text(lines, &line_count);
Tokenize(lines, line_count, thread_count);
for (i = 0; i < line_count; i++)
if (lines[i] != NULL) free(lines[i]);
return 0;
}  
void Usage(char* prog_name) {
fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
exit(0);
}  
void Get_text(char* lines[], int* line_count_p) {
char* line = malloc(MAX_LINE*sizeof(char));
int i = 0;
char* fg_rv;
fg_rv = fgets(line, MAX_LINE, stdin);
while (fg_rv != NULL) {
lines[i++] = line;
line = malloc(MAX_LINE*sizeof(char));
fg_rv = fgets(line, MAX_LINE, stdin);
}
*line_count_p = i;
}  
void Tokenize(
char*  lines[]       , 
int    line_count    , 
int    thread_count  ) {
int my_rank, i, j;
char *my_token, *saveptr;
#pragma omp parallel num_threads(thread_count) default(none) private(my_rank, i, j, my_token, saveptr) shared(lines, line_count)
{
my_rank = omp_get_thread_num();
#pragma omp for schedule(static, 1)
for (i = 0; i < line_count; i++) {
printf("Thread %d > line %d = %s", my_rank, i, lines[i]);
j = 0; 
my_token = strtok_r(lines[i], " \t\n", &saveptr);
while ( my_token != NULL ) {
printf("Thread %d > token %d = %s\n", my_rank, j, my_token);
my_token = strtok_r(NULL, " \t\n", &saveptr);
j++;
} 
if (lines[i] != NULL) 
printf("Thread %d > After tokenizing, my line = %s\n",
my_rank, lines[i]);
} 
}  
}  
