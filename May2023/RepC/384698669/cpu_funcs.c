#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include "cpu_funcs.h"
#include "def.h"
#include "program_data.h"
#include "mutant.h"
#include "cuda_funcs.h"
#include "mpi_funcs.h"
extern int cuda_percentage;
extern MPI_Datatype program_data_type;
extern MPI_Datatype mutant_type;
char hashtable_cpu[NUM_CHARS][NUM_CHARS];
char conservatives_cpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
char semi_conservatives_cpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
double initiate_program(int pid, int num_processes)
{
ProgramData data;
double time;
if (pid == ROOT)
{
FILE* input_file;
input_file = fopen(INPUT_FILE, "r");
if (!input_file)
{
printf("Error open input file `%s`\n", INPUT_FILE);
MPI_Abort(MPI_COMM_WORLD, 2);
}
if (!read_seq_and_weights_from_file(input_file, &data))
{
printf("Error reading input file `%s`\n", INPUT_FILE);
MPI_Abort(MPI_COMM_WORLD, 2);
}
fclose(input_file);
}
if (num_processes > 1)      
{
MPI_Bcast(&data, 1, program_data_type, ROOT, MPI_COMM_WORLD);
}
if (pid == ROOT)
printf("threads=%2d, processes=%2d\n", omp_get_max_threads(), num_processes);
time = MPI_Wtime();
Mutant mutant;
double score = divide_execute_tasks(&data, num_processes, pid, &mutant);
time = MPI_Wtime() - time;
double my_best[2] = { 0 };          
my_best[0] = score;      
my_best[1] = pid;                   
double gloabl_best[2] = { 0 };      
if (num_processes > 1)
{
if (data.is_max)
MPI_Allreduce(my_best, gloabl_best, 1, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, MPI_COMM_WORLD);
else
MPI_Allreduce(my_best, gloabl_best, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_WORLD);
}
int sender = gloabl_best[1];    
if (sender != ROOT && pid == sender)
{
MPI_Send(&mutant, 1, mutant_type, ROOT, 0, MPI_COMM_WORLD);
}
if (pid == ROOT)
{   
MPI_Status status;
if (sender != ROOT)     
{
score = gloabl_best[0];        
MPI_Recv(&mutant, 1, mutant_type, sender, 0, MPI_COMM_WORLD, &status);
}
char mut[SEQ2_MAX_LEN];
strcpy(mut, data.seq2);
mut[mutant.char_offset] = mutant.ch;
FILE* out_file = fopen(OUTPUT_FILE, "w");
if (!out_file)
{
printf("Error open or write to the output file %s\n", OUTPUT_FILE);
MPI_Abort(MPI_COMM_WORLD, 2);
exit(1);
}
if (!write_results_to_file(out_file, mut, mutant.offset, score))
{
printf("Error write to the output file %s\n", OUTPUT_FILE);
MPI_Abort(MPI_COMM_WORLD, 2);
exit(1);
}
fclose(out_file);
#ifdef DEBUG_PRINT
pretty_print(&data, mut, mutant.offset, mutant.char_offset);
#endif
}
return time;
}
double divide_execute_tasks(ProgramData* data, int num_processes, int pid, Mutant* mt)
{
int total_chars = strlen(data->seq2);
int total_offsets = strlen(data->seq1) - total_chars + 1;
int per_proc_offset = total_offsets / num_processes;
int first_offset = per_proc_offset * pid;  
int last_offset = per_proc_offset + first_offset;
if (pid == num_processes - 1)    
last_offset += total_offsets % num_processes;
double percentage = (double)total_chars * (double)total_offsets * 100;
percentage /= ((double)(SEQ2_MAX_LEN - 1) * (double)MAX_OFFSETS);
if (cuda_percentage == -1 && percentage >= 20)
cuda_percentage = 100;
else if (cuda_percentage == -1)
cuda_percentage = 0;
int gpu_tasks = (last_offset - first_offset) * cuda_percentage / 100;
int gpu_first_offset = first_offset;
int gpu_last_offset = gpu_first_offset + gpu_tasks;
int cpu_tasks = (last_offset - first_offset) - gpu_tasks;
int cpu_first_offset = gpu_last_offset;
int cpu_last_offset = cpu_first_offset + cpu_tasks;
if (pid == ROOT)
printf("CUDA percentage set to %d\n", cuda_percentage);
#ifdef DEBUG_PRINT
printf("pid %2d, total=%4d, per_proc=%4d, cuda start=%4d, cuda end=%4d, cpu start=%4d, cpu end=%4d\n",
pid,
total_tasks,
last_offset - first_offset,
gpu_first_offset,
gpu_last_offset,
cpu_first_offset,
cpu_last_offset);
#endif
double gpu_best_score = data->is_max ? -INFINITY : INFINITY;
double cpu_best_score = data->is_max ? -INFINITY : INFINITY;
Mutant gpu_mutant = { -1, -1, NOT_FOUND_CHAR };
Mutant cpu_mutant = { -1, -1, NOT_FOUND_CHAR };
if (cpu_tasks > 0)
fill_hash(data->weights, pid);
#pragma omp parallel
{
int num_threads = omp_get_num_threads();
int tid = omp_get_thread_num();
if (gpu_tasks > 0 && tid == ROOT)
{
gpu_best_score = gpu_run_program(data, &gpu_mutant, gpu_first_offset, gpu_last_offset);
}
if (cpu_tasks > 0)
{   
if (gpu_tasks > 0)      
{                       
num_threads = 3;    
tid--;              
}
if (tid != -1)
{
int tasks_per_thread = (cpu_last_offset - cpu_first_offset) / num_threads;          
int thread_start = cpu_first_offset + tasks_per_thread * tid;                       
int thread_end = thread_start + tasks_per_thread;                                   
if ((cpu_last_offset - cpu_first_offset) % num_threads != 0 && tid == num_threads - 1)  
thread_end += (cpu_last_offset - cpu_first_offset) % num_threads;
find_best_mutant_cpu(pid, data, &cpu_mutant, thread_start, thread_end, &cpu_best_score);   
}
}
}
#ifdef DEBUG_PRINT
printf("cpu tasks=%3d, cpu best score=%lf\ngpu tasks=%3d, gpu best score=%lf\n", cpu_tasks, cpu_best_score, gpu_tasks, gpu_best_score);
if (cpu_tasks == 0) 
fill_hash(data->weights, pid);
#endif
Mutant final_best_mutant = gpu_mutant;                      
double final_best_score = gpu_best_score;
if ((data->is_max && cpu_best_score > gpu_best_score) ||     
(!data->is_max && cpu_best_score < gpu_best_score))
{
final_best_mutant = cpu_mutant;
final_best_score = cpu_best_score;
}
*mt = final_best_mutant;
return final_best_score;
}
void find_best_mutant_cpu(int pid, ProgramData* data, Mutant* return_mutant, int first_offset, int last_offset, double* cpu_score)
{    
double _best_score = data->is_max ? -INFINITY : INFINITY;      
double _curr_score;          
Mutant _best_mutant, _temp_mutant;         
for (int curr_offset = first_offset; curr_offset < last_offset; curr_offset++)    
{     
_curr_score = find_best_mutant_offset(data, curr_offset, &_temp_mutant);
if (is_swapable(&_best_mutant, &_temp_mutant, _best_score, _curr_score, data->is_max))
{
_best_mutant = _temp_mutant;
_best_mutant.offset = curr_offset;
_best_score = _curr_score;
}
}
#pragma omp critical
{
if (is_swapable(return_mutant, &_best_mutant, *cpu_score, _best_score, data->is_max))
{
*cpu_score = _best_score;
*return_mutant = _best_mutant;
}
}
}
double find_best_mutant_offset(ProgramData* data, int offset, Mutant* mt)
{
int idx1, idx2;
double total_score = 0;
double pair_score, mutant_diff;
double best_mutant_diff = data->is_max ? -INFINITY : INFINITY;
int iterations = strlen_gpu(data->seq2);
char c1, c2, sub;
mt->offset = -1;            
mt->char_offset = -1;
mt->ch = NOT_FOUND_CHAR;
for (int i = 0; i < iterations; i++)        
{
idx1 = offset + i;                      
idx2 = i;                               
c1 = data->seq1[idx1];                  
c2 = data->seq2[idx2];                  
pair_score = get_weight(get_hashtable_sign(c1, c2), data->weights);    
total_score += pair_score;
sub = get_substitute(c1, c2, data->weights, data->is_max);
if (sub == NOT_FOUND_CHAR)   
continue;
mutant_diff = get_weight(get_hashtable_sign(c1, sub), data->weights) - pair_score;    
if ((data->is_max && mutant_diff > best_mutant_diff) || 
(!data->is_max && mutant_diff < best_mutant_diff))
{
best_mutant_diff = mutant_diff;
mt->ch = sub;
mt->char_offset = i;        
mt->offset = offset;
}
}
if (mt->ch == NOT_FOUND_CHAR)       
return best_mutant_diff;
return total_score + best_mutant_diff;     
}
void fill_hash(double* weights, int pid)
{
char c1, c2;
#pragma omp parallel for
for (int i = 0; i < NUM_CHARS; i++)
{
c1 = FIRST_CHAR + i;                    
for (int j = 0; j <= i; j++)            
{
c2 = FIRST_CHAR + j;
hashtable_cpu[i][j] = get_pair_sign(c1, c2);
}
hashtable_cpu[i][NUM_CHARS] = SPACE;    
}
}
void print_hash()
{
char last_char = FIRST_CHAR + NUM_CHARS;
printf("   ");
for (int i = FIRST_CHAR; i < last_char; i++)
printf("%c ", i);
printf("%c\n", HYPHEN);
printf("   ");
for (int i = FIRST_CHAR; i < last_char + 1; i++)
printf("__");
printf("\n");
for (int i = FIRST_CHAR; i < last_char; i++)
{
printf("%c |", i);
for (int j = FIRST_CHAR; j < last_char; j++)
{
printf("%c ", get_hashtable_sign(i, j));
}
printf("%c \n", get_hashtable_sign(i, HYPHEN));
}
printf("%c |", HYPHEN);
for (int i = FIRST_CHAR; i < last_char; i++)
{
printf("%c ", get_hashtable_sign(HYPHEN, i));
}
printf("%c ", get_hashtable_sign(HYPHEN, HYPHEN));
printf("\n");
}
ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data)
{
if (!file || !data)  return NULL;   
if (fscanf(file, "%lf %lf %lf %lf", &data->weights[0], &data->weights[1], &data->weights[2], &data->weights[3]) != 4)   return NULL;
if (fscanf(file, "%s", data->seq1) != 1)   return NULL;
if (fscanf(file, "%s", data->seq2) != 1)   return NULL;
char func_type[FUNC_NAME_LEN];
if (fscanf(file, "%s", func_type) != 1)   return NULL;
data->is_max = strcmp(func_type, MAXIMUM_STR) == 0 ? MAXIMUM_FUNC : MINIMUM_FUNC;    
return data;
}
int write_results_to_file(FILE* file, char* mutant, int offset, double score)
{
if (!file || !mutant)	return 0;
return fprintf(file, "%s\n%d %g", mutant, offset, score) > 0;	
}
void pretty_print(ProgramData* data, char* mut, int offset, int char_offset)
{
if (!data || !mut)  return;
#ifdef PRINT_SIGN_MAT
print_hash();
#endif
printf("\033[0;31m%s\033[0m problem\n", data->is_max ? "Maximum" : "Minimum");
printf("Weights: ");
for (int i = 0; i < WEIGHTS_COUNT; i++)
printf("%g ", data->weights[i]);
if (offset == -1)
{
printf("\n\033[0;31mThere are no mutations found\033[0m\n");
printf("%s\n%s\n", data->seq1, data->seq2);
return;
}
char signs[SEQ2_MAX_LEN] = { '\0' };
double score = get_score_and_signs(data->seq1, data->seq2, data->weights, offset, signs);    
printf("\nOriginal Score: %g\n", score);
print_with_offset(signs, offset, char_offset);
printf("\n");
print_with_offset(data->seq2, offset, char_offset);
printf("\n");
printf("%s\n", data->seq1);       
print_with_offset(mut, offset, char_offset);
printf("\n");
score = get_score_and_signs(data->seq1, mut, data->weights, offset, signs);    
print_with_offset(signs, offset, char_offset);
printf("\n");
printf("Mutation Score: %g\n", score);
printf("Seq offset=%3d, Char offset=%3d\n", offset, char_offset);
}
double get_score_and_signs(char* seq1, char* seq2, double* weights, int offset, char* signs)
{
int idx1 = offset;
int idx2 = 0;
int num_chars = strlen(seq2);
double score = 0;
for (   ; idx2 < num_chars; idx1++, idx2++)
{   
signs[idx2] = get_hashtable_sign(seq1[idx1], seq2[idx2]);
score += get_weight(signs[idx2], weights);
}
return score;
}
void print_with_offset(char* chrs, int offset, int char_offset)
{
if (char_offset < 0)
char_offset = 0;
for (int i = 0; i < offset; i++)
printf(" ");    
for (int i = 0; i < char_offset; i++)
printf("%c", chrs[i]);       
printf("\033[0;31m");
printf("%c", chrs[char_offset]);
printf("\033[0m");
for (uint i = char_offset + 1; i < strlen(chrs); i++)
printf("%c", chrs[i]);       
}