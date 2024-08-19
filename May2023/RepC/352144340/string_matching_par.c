#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <omp.h>
void initialize_match(int match[], long match_size);
void find_total_matches(int match[], char* pattern, char* buffer, long match_size, long pattern_size, long* total_matches);
void show_total_matches(int match[], long match_size, long total_matches);
int main (int argc, char *argv[]) {
FILE *pFile;
long file_size, match_size, pattern_size, total_matches;
char * buffer;
char * filename, *pattern;
size_t result;
int i, j, *match;
double start, end;
if (argc != 3) {
printf ("Usage : %s <file_name> <string>\n", argv[0]);
return 1;
}
filename = argv[1];
pattern = argv[2];
pFile = fopen ( filename , "rb" );
if (pFile==NULL) {printf ("File error\n"); return 2;}
fseek (pFile , 0 , SEEK_END);
file_size = ftell (pFile);
rewind (pFile);
printf("file size is %ld\n", file_size);
buffer = (char*) malloc (sizeof(char)*file_size);
if (buffer == NULL) {printf ("Memory error\n"); return 3;}
result = fread (buffer,1,file_size,pFile);
if (result != file_size) {printf ("Reading error\n"); return 4;} 
pattern_size = strlen(pattern);
match_size = file_size - pattern_size + 1;
match = (int *) malloc (sizeof(int)*match_size);
if (match == NULL) {printf ("Malloc error\n"); return 5;}
total_matches = 0;
initialize_match(match, match_size);
start = omp_get_wtime();
find_total_matches(match, pattern, buffer, match_size, pattern_size, &total_matches);
end = omp_get_wtime();
printf("\nTotal matches = %ld\n", total_matches);
printf("Time: %f\n",(end-start));
fclose (pFile);
free (buffer);
free (match);
return 0;
}
void initialize_match(int match[], long match_size) {
int j;
for (j = 0; j < match_size; j++){
match[j]=0;
}
}
void find_total_matches(int match[], char* pattern, char* buffer, long match_size, long pattern_size, long* total_matches) {
int i, j;
#pragma omp parallel for default(shared) private(i,j)
for (i = 0; i < match_size; i++) {
for (j = 0; j < pattern_size; j++){
if (buffer[i + j] != pattern[j])
break;
}
if (j == pattern_size) {
match[j] = 1;
(*total_matches)++;
}		
}
}
void show_total_matches(int match[], long match_size, long total_matches) {
int j;
for (j = 0; j < match_size; j++){
printf("%d", match[j]);
}	
printf("\nTotal matches = %ld\n", total_matches);
}
