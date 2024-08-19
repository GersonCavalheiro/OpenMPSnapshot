#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h>
#define N 128
#define base 0
void show_characters_frequency(int freq[]);
void calculate_character_frequency(int freq[], char* buffer, long file_size, omp_lock_t locks[]);
void initialize_frequency_array(int freq[]);
void create_locks(omp_lock_t locks[], int n);
void destroy_locks(omp_lock_t locks[], int n);
int main (int argc, char *argv[]) {
FILE *pFile;
long file_size;
char * buffer;
char * filename;
size_t result;
int freq[N];
omp_lock_t locks[N];
if (argc != 2) {
printf ("Usage : %s <file_name>\n", argv[0]);
return 1;
}
filename = argv[1];
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
create_locks(locks, N);
initialize_frequency_array(freq);
double start = omp_get_wtime();	
calculate_character_frequency(freq, buffer, file_size, locks);
double end = omp_get_wtime();
show_characters_frequency(freq);	
printf("Time for counting: %f", (end-start));
fclose (pFile);
free (buffer);
destroy_locks(locks, N);
return 0;
}
void show_characters_frequency(int freq[]) {
int j;
for ( j=0; j<N; j++){
printf("%d = %d\n", j+base, freq[j]);
}
}
void calculate_character_frequency(int freq[], char* buffer, long file_size, omp_lock_t locks[]) {
int i;
#pragma omp parallel for private(i) 
for (i=0; i<file_size; i++){
omp_set_lock(&(locks[buffer[i] - base]));
freq[buffer[i] - base]++;
omp_unset_lock(&(locks[buffer[i] - base]));
}
}
void initialize_frequency_array(int freq[]) {
int j;
for (j=0; j<N; j++){
freq[j]=0;
}
}
void create_locks(omp_lock_t locks[], int n){
int i;
for (int i=0; i<n; ++i)
omp_init_lock(&(locks[i]));
}
void destroy_locks(omp_lock_t locks[], int n){
int i;
for (i=0; i<N; ++i)
omp_destroy_lock(&(locks[i]));
}
