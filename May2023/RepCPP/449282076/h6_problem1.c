#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define ITERATION 10000
#define ANT_NUM   500
#define ALPHA     0.5     
#define BETA      3.0     
#define RHO       0.5     

#define MAX_CITY  50
#define THREAD_NUM 2

typedef struct ant {
int at_city, tour_step;
char visited[MAX_CITY];
int tour[MAX_CITY];
} ant_t;


void reset_ants(ant_t *ants, int ant_num, int city_num);
void update_probability(double *probability, int *map, \
double *pheromone, int city_num);
void travel(ant_t *ant, double *probability, int city_num);
void update_pheromone(int *map, double *pheromone, int city_num, int *best_tour, ant_t *ant);

void error(char *mess);
void print_arr(char* mess, double *arr, int width);
void print_map(int *arr, int width);
void print_tour(ant_t *ant);

#define DEBUG
#define INT_MAX   2147483647
#ifdef DEBUG
#define debug(...) printf(__VA_ARGS__)
#define debug_arr(...) print_arr(__VA_ARGS__)
#define debug_map(...) print_map(__VA_ARGS__)
#define debug_tour(...) print_tour(__VA_ARGS__)
#else
#define debug(...) do {} while (0)
#define debug_arr(...) do {} while (0)
#define debug_map(...) do {} while (0)
#define debug_tour(...) do {} while (0)
#endif



int main(int argc, char *argv[])
{
double start_time, end_time;
int id, comm_sz;
srand(time(NULL));
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
MPI_Comm_rank(MPI_COMM_WORLD, &id);
omp_set_num_threads(THREAD_NUM);


int city_num = 0, map_sz;    
int local_ant_num = ANT_NUM / comm_sz;
ant_t ants[local_ant_num];
int best_tour = INT_MAX;


if (id == 0) {
start_time = MPI_Wtime();
if (argc < 2) {
error("No city file provided.");
}
FILE *fp = fopen(argv[1], "r");
if (!fp) {
error("Cannot read the file");
}
char buffer[1000]; 

char *num_ptr;
num_ptr = fgets(buffer, 1000, fp);
while (*num_ptr != '\n') {
strtol(num_ptr, &num_ptr, 10);
city_num++;
}
fclose(fp);
}
MPI_Bcast(&city_num, 1, MPI_INT, 0, MPI_COMM_WORLD);


map_sz = city_num * city_num;
int map[map_sz];                    
double pheromone[map_sz];           
double delta_pheromone[map_sz];     
double pheromone_buf[map_sz];       
double probability[map_sz];         


if (id == 0) {
FILE *fp = fopen(argv[1], "r");
if (!fp) {
error("Cannot read the file");
}
int i, j;
for (i = 0; i < city_num; i++) {
char buffer[1000];
char *num_ptr;
num_ptr = fgets(buffer, 1000, fp);
int offset = i * city_num;
for (j = 0; j < city_num; j++) {
map[offset + j] = strtol(num_ptr, &num_ptr, 10);
}
}
}
MPI_Bcast(map, map_sz, MPI_INT, 0, MPI_COMM_WORLD);
int iter, i, j, k;


for (i = 0; i < map_sz; i++) {
pheromone[i] = 1.0;
}
double init_prob = 1.0 / (city_num - 1);
for (i = 0; i < city_num; i++) {
int offset = i * city_num;
for (j = 0; j < city_num; j++) {
if (i == j) {
probability[offset + j] = 0.0;
} else {
probability[offset + j] = init_prob;
}
}
}


for (iter = 0; iter < ITERATION; iter++) {
if (id == 0 && iter % 100 == 0) {
printf("Iteration %d\n", iter);
}
reset_ants(ants, local_ant_num, city_num);

for (j = 0; j < city_num - 1; j++) {
for (i = 0; i < local_ant_num; i++) {
travel(&ants[i], probability, city_num);
}
}

#pragma omp parallel for
for (i = 0; i < map_sz; i++) {
pheromone[i] = pheromone[i] * RHO;  
delta_pheromone[i] = 0.0;
}

for (i = 0; i < local_ant_num; i++) {
int start = ants[i].tour[city_num - 1], sum = 0;
for (j = 0; j < city_num; j++) {
int end = ants[i].tour[j];
sum += map[(start * city_num) + end];
start = end;
}
if (sum < best_tour) {
best_tour = sum;
}

start = ants[i].tour[city_num - 1];
double new_pheromone = 1.0 / sum;
for (j = 0; j < city_num; j++) {
pheromone[(start * city_num) + ants[i].tour[j]] += new_pheromone;
start = ants[i].tour[j];
}
}


MPI_Barrier(MPI_COMM_WORLD);
MPI_Allreduce(delta_pheromone, pheromone_buf, map_sz, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#pragma omp parallel for
for (i = 0; i < map_sz; i++) {
pheromone[i] += pheromone_buf[i];
}


for (i = 0; i < city_num; i++) {
double sum = 0.0;
int offset = i * city_num;
#pragma omp parallel for
for (j = 0; j < city_num; j++) {
if (i == j) continue;
double tmp = pow(pheromone[offset + j], ALPHA) * pow((1.0 / map[offset + j]), BETA);
probability[offset + j] = tmp;
sum += tmp;
}
#pragma omp parallel for
for (j = 0; j < city_num; j++) {
if (i == j) continue;
probability[offset + j] /= sum;
}
}
}

end_time = MPI_Wtime();
MPI_Finalize();
if (id == 0) {
printf("Best tour: %d\n", best_tour);
printf("Time: %lf\n", end_time - start_time);
}    
return 0;
}



void error(char *mess)
{
puts(mess);
MPI_Finalize();
exit(0);
}


void print_arr(char *mess, double *arr, int width)
{
int i, j;
puts(mess);
for (i = 0; i < width; i++) {
for (j = 0; j < width; j++) {
printf("%.2lf ", arr[i * width + j]);
}
printf("\n");
}
printf("\n");
}


void print_map(int *arr, int width)
{
int i, j;
puts("MAP:");
for (i = 0; i < width; i++) {
for (j = 0; j < width; j++) {
printf("%5d", arr[i * width + j]);
}
printf("\n");
}
printf("\n");
}


void print_tour(ant_t *ant)
{
int i;
for (i = 0; i <= ant->tour_step; i++) {
printf("%d ", ant->tour[i]);
}
puts("");
}


void reset_ants(ant_t* ants, int ant_num, int city_num)
{
int i, j;
for (i = 0; i < ant_num; i++) {
ants[i].tour_step = 0;
memset(ants[i].visited, '\0', city_num);
ants[i].at_city = rand() % city_num;    
ants[i].visited[ants[i].at_city] = '1';
ants[i].tour[0] = ants[i].at_city;
}
}


void update_probability(double *probability, int *map, double *pheromone, int city_num)
{
int i, j;
for (i = 0; i < city_num; i++) {
double sum = 0.0;
int offset = i * city_num;
for (j = 0; j < city_num; j++) {
if (i == j) continue;
double tmp = pow(pheromone[offset + j], ALPHA) * pow((1.0 / map[offset + j]), BETA);
probability[offset + j] = tmp;
sum += tmp;
}

for (j = 0; j < city_num; j++) {
if (i == j) continue;
probability[offset + j] /= sum;
}
}
}


void travel(ant_t *ant, double *probability, int city_num)
{
int avail_city[city_num];
int offset = (ant->at_city) * city_num, index = 0, i;
double city_prob[city_num], previous = 0.0, sum = 0.0;

for (i = 0; i < city_num; i++) {
if (ant->visited[i] != '\0')
continue;
sum += probability[offset + i];
avail_city[index] = i;
city_prob[index] = probability[offset + i] + previous; 
previous = city_prob[index];
index++;
}
for (i = 0; i < index; i++) {
city_prob[i] /= sum;
}

double x = (double) rand() / RAND_MAX;  
for (i = 0; i < index; i++) {
if (x < city_prob[i]) {
ant->at_city = avail_city[i];
ant->tour_step++;
ant->tour[ant->tour_step] = ant->at_city;
ant->visited[ant->at_city] = '1';
break;
}
}
}


void update_pheromone(int *map, double *pheromone, int city_num, int *best_tour, ant_t *ant)
{
int i, start = ant->tour[city_num - 1], sum = 0;
for (i = 0; i < city_num; i++) {
int end = ant->tour[i];
sum += map[(start * city_num) + end];
start = end;
}
if (sum < *best_tour) {
*best_tour = sum;
}

start = ant->tour[city_num - 1];
double new_pheromone = 1.0 / sum;
for (i = 0; i < city_num; i++) {
pheromone[(start * city_num) + ant->tour[i]] += new_pheromone;
start = ant->tour[i];
}

}
