#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


#define NUM_CITIES 10000
#define MAX_KM 1000
#define MAX_SWAPS 10000000

float CitiesX[NUM_CITIES+1];	
float CitiesY[NUM_CITIES+1];
int Sequence[NUM_CITIES+1];


void cities_init(void);

void print_cities(void);
void print_sequence(void);
void print_results(float initial_dist, float final_dist);

void pick_cities(int *swapped_indices);
void swap(int *swapped_indices);

void update_sequence(int *swapped_indices);

void calc_adjacent_dists(int *swapped_indices, float *prev_dists);
float update_distance(int *swapped_indices, float *prev_dists, float old_tot_dist);
float total_distance(void);
float distance(int city1, int city2);


int main(){
int swapped_indices[2];
float prev_dists[4];	

cities_init();

float initial_tot_dist = total_distance();
float old_tot_dist = initial_tot_dist;
float new_tot_dist = old_tot_dist;

for (int i=0;i<MAX_SWAPS;i++){
pick_cities(swapped_indices);
calc_adjacent_dists(swapped_indices, prev_dists);
swap(swapped_indices);
new_tot_dist = update_distance(swapped_indices, prev_dists, old_tot_dist);

if(new_tot_dist >= old_tot_dist){	
swap(swapped_indices);
}else{
update_sequence(swapped_indices);
old_tot_dist = new_tot_dist;	
}
}

print_results(initial_tot_dist, old_tot_dist);
return 0;
}

void cities_init(void){
for (int i=0;i<NUM_CITIES;i++){
CitiesX[i] = MAX_KM*1.0*rand()/RAND_MAX;
CitiesY[i] = MAX_KM*1.0*rand()/RAND_MAX;
Sequence[i] = i;
}
CitiesX[NUM_CITIES] = CitiesX[0];
CitiesY[NUM_CITIES] = CitiesY[0];
Sequence[NUM_CITIES] = 0;
}

void print_cities(void){
for (int i=0;i<NUM_CITIES;i++){
printf("City %d: (%.3f, %.3f)\n", i, CitiesX[i], CitiesY[i]);
}
}

void print_results(float initial_dist, float final_dist){
int cities_per_line = 10;


printf("\nInitial distance: %.4f\n", initial_dist);
printf("Final distance: %.4f\n", final_dist);
printf("Improvement: %.4f%%\n", 100*(initial_dist - final_dist)/initial_dist);
}

void pick_cities(int *swapped_indices){
int ind1 = 1 + rand()%(NUM_CITIES-1);	
int ind2;
float temp_x, temp_y;

do{
ind2 = 1 + rand()%(NUM_CITIES-1);
}while(ind2==ind1);

swapped_indices[0] = ind1;
swapped_indices[1] = ind2;
}

void calc_adjacent_dists(int *swapped_indices, float *adj_dists){
#pragma omp parallel for schedule(static,2)
for(int i=0;i<2;i++){
*adj_dists++ = distance(swapped_indices[i], swapped_indices[i]-1);
*adj_dists++ = distance(swapped_indices[i], swapped_indices[i]+1);
}
}

void swap(int *swapped_indices){
int ind1 = swapped_indices[0];
int ind2 = swapped_indices[1];
int tempX, tempY;

tempX = CitiesX[ind1];
tempY = CitiesY[ind1];
CitiesX[ind1] = CitiesX[ind2];
CitiesY[ind1] = CitiesY[ind2];
CitiesX[ind2] = tempX;
CitiesY[ind2] = tempY;
}

void update_sequence(int *swapped_indices){
int ind1 = swapped_indices[0];
int ind2 = swapped_indices[1];
int temp;

temp = Sequence[ind1];
Sequence[ind1] = Sequence[ind2];
Sequence[ind2] = temp;
}

float update_distance(int *swapped_indices, float *prev_dists, float old_tot_dist){
float new_dists[4];
calc_adjacent_dists(swapped_indices, new_dists);	


#pragma omp parallel for simd reduction(+:old_tot_dist) schedule(static,4)
for(int i=0;i<4;i++){
old_tot_dist += new_dists[i] - prev_dists[i];	
}
return old_tot_dist;
}

float total_distance(void){
float tot_dist = 0.0;

#pragma omp parallel for reduction(+:tot_dist)
for (int i=0;i<NUM_CITIES;i++){
tot_dist += distance(i, i+1);
}
return tot_dist;
}

float distance(int city1, int city2){
float x_sq, y_sq;
#pragma omp parallel sections num_threads(2)
{
#pragma omp section
{
x_sq = (CitiesX[city1] - CitiesX[city2])*(CitiesX[city1] - CitiesX[city2]);
}
#pragma omp section
{
y_sq = (CitiesY[city1] - CitiesY[city2])*(CitiesY[city1] - CitiesY[city2]);
}
}
return sqrtf(x_sq+y_sq);
}
