#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 100000
#define NV 1000
#define NC 100
#define THRESHOLD 1.0e-6

float Vec[N][NV];
float Center[NC][NV];
int Class[N];
int unique_vector_indices[NC];
float vector_sum[NV];

void vec_init(void);
void center_init(void);
void unique_vects(void);
void print_vec(void);
float estimate_class(void);
float distance(int n_vector, int n_center);
void estimate_center(void);
void vector_sum_init(void);
void vector_add(int n_vector);
void vector_average(int n_center, int divisor);
void print_centers(void);




int main() {
float tot_dist = 1.0e30;
float prev_dist;
int step = 1;

vec_init();
center_init();
do {
prev_dist = tot_dist;
tot_dist = estimate_class();
printf("Step: %d\tTotal distance: %f\n", step++, tot_dist);
estimate_center();
} while (((prev_dist-tot_dist)/tot_dist) > THRESHOLD);
return 0;
}

void vec_init(void) {
int i, j;
for (i = 0; i < N;i++) {
for (j = 0; j < NV; j++) {
Vec[i][j] = (1.0*rand()) / RAND_MAX;
}
}
}

void center_init(void) {
int i,j;
unique_vects();
for(i=0;i<NC;i++){
for(j=0;j<NV;j++){
Center[i][j] = Vec[unique_vector_indices[i]][j];
}
}

}

void unique_vects(void){
int i,j;
bool is_unique;
unique_vector_indices[0] = rand() % N;
for(i=1;i<NC;i++){
while(1){
unique_vector_indices[i] = rand() % N;
is_unique = true;
for(j=0;j<i;j++){
if(unique_vector_indices[i] == unique_vector_indices[j]){	
is_unique = false;
break;
}
}
if(is_unique) break;
}
}
}

void print_vec(void) {
int i, j;
for (i = 0; i < N; i++) {
for (j = 0; j < NV; j++) {
printf("%f ", Vec[i][j]);
}
printf("\n");
}
}

float estimate_class(void) {
float tot_dist = 0.0;
float min_dist, dist;
int n_vector, n_center, class_num;
for(n_vector=0;n_vector<N;n_vector++){
min_dist = distance(n_vector, 0);	
class_num = 0;						
for(n_center=1;n_center<NC;n_center++){
dist = distance(n_vector, n_center);
if(dist < min_dist){
min_dist = dist;			
class_num = n_center;		
}
}
tot_dist += min_dist;				
Class[n_vector] = class_num;		
}
return tot_dist;
}

float distance(int n_vector, int n_center){
float dist = 0.0;
int i;
for(i=0;i<NV;i++){
dist+=(Vec[n_vector][i] - Center[n_center][i])*(Vec[n_vector][i] - Center[n_center][i]);
}
return dist;
}

void estimate_center(void) {
int n_center,n_vector,vector_dim;
int cnt = 0;
for(n_center=0;n_center<NC;n_center++){
vector_sum_init();		
cnt = 0;				
for(n_vector=0;n_vector<N;n_vector++){
if(Class[n_vector] == n_center){
vector_add(n_vector);	
cnt++;					
}
}
vector_average(n_center, cnt);
}
}

void vector_sum_init(void){
int i;
for(i=0;i<NV;i++){
vector_sum[i] = 0;
}
}

void vector_add(int n_vector){
int i;
for(i=0;i<NV;i++){
vector_sum[i] += Vec[n_vector][i];
}
}

void vector_average(int n_center, int divisor){
int i;
for(i=0;i<NV;i++){
Center[n_center][i] = vector_sum[i] / divisor;
}
}

void print_centers(void){
int i, j;
for (i = 0; i < NC; i++) {
for (j = 0; j < NV; j++) {
printf("%f ", Center[i][j]);
}
printf("\n");
}
}
