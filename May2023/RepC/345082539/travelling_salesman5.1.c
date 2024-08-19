#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>


#define NUM_CITIES 10
#define MAX_KM 1000
#define PROBABILITY_THRESHOLD 0.1


float CitiesX[NUM_CITIES+1];	
float CitiesY[NUM_CITIES+1];
int Sequence[NUM_CITIES+1];
bool Visited[NUM_CITIES+1];		


void cities_init(void);

void print_cities(void);
void print_sequence(void);
void print_results(float final_dist);

bool calc_prob(void);
bool is_not_visited(int city, int to_visit);

float find_nearest(int to_visit);
float find_second_nearest(int to_visit);
float distance(int city1, int city2);


struct Nearest { float dist; int city; };
typedef struct Nearest nearest;   



#pragma omp declare reduction(my_min : nearest : omp_out = omp_in.dist < omp_out.dist ? omp_in : omp_out) initializer (omp_priv=(nearest){.dist = 1e6, .city = -1})


int main(){
float tot_dist = 0.0;

cities_init();
print_cities();
for(int to_visit=1;to_visit<NUM_CITIES;to_visit++){
if(calc_prob()){
tot_dist += find_nearest(to_visit);
}else{
tot_dist += find_second_nearest(to_visit);	
}

}

tot_dist += distance(Sequence[NUM_CITIES-1], NUM_CITIES);
print_results(tot_dist);

return 0;
}

void cities_init(void){
Sequence[0] = 0;
for(int i=0;i<NUM_CITIES;i++){
CitiesX[i] = MAX_KM*1.0*rand()/RAND_MAX;
CitiesY[i] = MAX_KM*1.0*rand()/RAND_MAX;
}
CitiesX[NUM_CITIES] = CitiesX[0];
CitiesY[NUM_CITIES] = CitiesY[0];
Sequence[NUM_CITIES] = 0;
}

void print_cities(void){
for(int i=0;i<NUM_CITIES;i++){
printf("City %d: (%.3f, %.3f)\n", i, CitiesX[i], CitiesY[i]);
}
}

void print_results(float final_dist){
int cities_per_line = 15;

for(int i=0;i<NUM_CITIES;i++){
if((i+1)%cities_per_line != 0){
printf("%d-->", Sequence[i]);
}else{
printf("%d-->\n", Sequence[i]);
}

}
printf("%d\n", Sequence[NUM_CITIES]);
printf("\nFinal distance: %.4f\n", final_dist);
}

bool calc_prob(void){
float prob = 1.0*rand()/RAND_MAX;
return (prob > PROBABILITY_THRESHOLD)?true:false;
}

float find_nearest(int to_visit){
float dist;

nearest nearest_city;
nearest_city.dist = 1e6;
nearest_city.city = -1;

printf("\n\nVISIT CLOSEST FROM: %d\n\n", to_visit-1);

#pragma omp parallel for reduction(my_min:nearest_city) private(dist)
for(int i=1;i<NUM_CITIES;i++){
if(!Visited[i]){
dist = distance(i, Sequence[to_visit-1]); 
printf("Distance between %d and %d: %f\n",i, Sequence[to_visit-1], dist);
if(dist < nearest_city.dist){
nearest_city.dist = dist;
nearest_city.city = i;
}
}
}

printf("Closest: %d with distance: %f\n", nearest_city.city, nearest_city.dist);
Sequence[to_visit] = nearest_city.city;
Visited[nearest_city.city] = true;
return nearest_city.dist;
}

float find_second_nearest(int to_visit){
float dist;

nearest nearest_city1, nearest_city2;
nearest_city1.dist = 1e6;
nearest_city1.city = -1;
nearest_city2.dist = 1e6;
nearest_city2.city = -1;

printf("\n\nVISIT SECOND CLOSEST FROM: %d\n\n", to_visit-1);

#pragma omp parallel for reduction(my_min:nearest_city1) private(dist)
for(int i=1;i<NUM_CITIES;i++){
if(!Visited[i]){
dist = distance(i, Sequence[to_visit-1]); 
printf("Distance between %d and %d: %f\n",i, Sequence[to_visit-1], dist);
if(dist < nearest_city1.dist){
nearest_city1.dist = dist;
nearest_city1.city = i;
}
}
}


if(to_visit == NUM_CITIES-1){
printf("Second closest is closest: %d with distance: %f\n", nearest_city1.city, nearest_city1.dist);
Sequence[to_visit] = nearest_city1.city;
Visited[nearest_city1.city] = true;
return nearest_city1.dist;
}else{

#pragma omp parallel for reduction(my_min:nearest_city2) private(dist)
for(int i=1;i<NUM_CITIES;i++){
if(!Visited[i] && i != nearest_city1.city){
dist = distance(i, Sequence[to_visit-1]); 
if(dist < nearest_city2.dist){
nearest_city2.dist = dist;
nearest_city2.city = i;
}
}
}

printf("Second closest: %d with distance: %f\n", nearest_city2.city, nearest_city2.dist);
Sequence[to_visit] = nearest_city2.city;
Visited[nearest_city2.city] = true;
return nearest_city2.dist;
}



}


float distance(int city1, int city2){
return sqrtf((CitiesX[city1] - CitiesX[city2])*(CitiesX[city1] - CitiesX[city2])+\
(CitiesY[city1] - CitiesY[city2])*(CitiesY[city1] - CitiesY[city2]));
}
