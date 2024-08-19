#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <omp.h>
#include "dictionary-util.c"
#include "../globals.h"
int compare_candidates(FILE **file, char *password_hash, int verbose);
int run_chunk(char *password_hash, char **candidate_array, int chunk_size, int verbose);
int dictionary_crack(char *password_hash, char *dictionary_path, int verbose)
{
if( verbose )
{
printf("\n>>> Using dictionary path: %s\n", dictionary_path);
print_password_hash(password_hash);
}
FILE *file = fopen(dictionary_path, "r");
int result = compare_candidates(&file, password_hash, verbose);
if(result == NOT_FOUND)
print_not_found(verbose);
fclose(file);
return result;
}
int compare_candidates(FILE **file, char *password_hash, int verbose)
{
char *line = NULL;
size_t len = 0;
ssize_t read;
int result = NOT_FOUND;
int counter = 0;
char * candidate_array[CHUNK_SIZE];
while ((read = getline(&line, &len, *file) != -1) && result == NOT_FOUND)
{
remove_new_line(line, &candidate_array[counter]);
if (counter++ == CHUNK_SIZE)
{
result = run_chunk(password_hash, candidate_array, CHUNK_SIZE, verbose);
counter=0;
}
}
if (counter > 0)
{
result = run_chunk(password_hash, candidate_array, counter, verbose);
}
return result;
}
int run_chunk(char *password_hash, char **candidate_array, int array_size, int verbose)
{
int result = NOT_FOUND;
int i;
#pragma omp parallel
#pragma omp for schedule(auto)
for (i=0; i<array_size; i++) 
{
int tempResult = do_comparison(password_hash, candidate_array[i], verbose);
if (tempResult == FOUND) {
#pragma omp critical
result = FOUND;
} 
} 
return result;
}
