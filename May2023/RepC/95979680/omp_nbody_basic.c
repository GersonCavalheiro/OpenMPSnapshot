#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#define DIM 2  
#define X 0    
#define Y 1    
const double G = 6.673e-11;  
typedef double vect_t[DIM];  
struct particle_s {
double m;  
vect_t s;  
vect_t v;  
};
void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* thread_count_p, int* n_p, 
int* n_steps_p, double* delta_t_p, int* output_freq_p, char* g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[], 
int n, double delta_t);
int main(int argc, char* argv[]) {
int n;                      
int n_steps;                
int step;                   
int part;                   
int output_freq;            
double delta_t;             
double t;                   
struct particle_s* curr;    
vect_t* forces;             
int thread_count;           
char g_i;                   
double start, finish;       
Get_args(argc, argv, &thread_count, &n, &n_steps, &delta_t, 
&output_freq, &g_i);
curr = malloc(n*sizeof(struct particle_s));
forces = malloc(n*sizeof(vect_t));
if (g_i == 'i')
Get_init_cond(curr, n);
else
Gen_init_cond(curr, n);
start = omp_get_wtime();
#  ifndef NO_OUTPUT
Output_state(0, curr, n);
#  endif
#pragma omp parallel num_threads(thread_count) default(none) shared(curr, forces, thread_count, delta_t, n, n_steps, output_freq) private(step, part, t)
for (step = 1; step <= n_steps; step++) {
t = step*delta_t;
#pragma omp for
for (part = 0; part < n; part++)
Compute_force(part, forces, curr, n);
#pragma omp for
for (part = 0; part < n; part++)
Update_part(part, forces, curr, n, delta_t);
#     ifndef NO_OUTPUT
#pragma omp single
if (step % output_freq == 0)
Output_state(t, curr, n);
#     endif
}
finish = omp_get_wtime();
printf("Elapsed time = %e seconds\n", finish-start);
free(curr);
free(forces);
return 0;
}  
void Usage(char* prog_name) {
fprintf(stderr, "usage: %s <number of threads> <number of particles>\n",
prog_name);
fprintf(stderr, "   <number of timesteps> <size of timestep>\n"); 
fprintf(stderr, "   <output frequency> <g|i>\n");
fprintf(stderr, "   'g': program should generate init conds\n");
fprintf(stderr, "   'i': program should get init conds from stdin\n");
exit(0);
}  
void Get_args(int argc, char* argv[], int* thread_count_p, int* n_p, 
int* n_steps_p, double* delta_t_p, int* output_freq_p, char* g_i_p) {
if (argc != 7) Usage(argv[0]);
*thread_count_p = strtol(argv[1], NULL, 10);
*n_p = strtol(argv[2], NULL, 10);
*n_steps_p = strtol(argv[3], NULL, 10);
*delta_t_p = strtod(argv[4], NULL);
*output_freq_p = strtol(argv[5], NULL, 10);
*g_i_p = argv[6][0];
if (*thread_count_p < 0 || *n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0) 
Usage(argv[0]);
if (*g_i_p != 'g' && *g_i_p != 'i') Usage(argv[0]);
#  ifdef DEBUG
printf("thread_count = %d\n", *thread_count_p);
printf("n = %d\n", *n_p);
printf("n_steps = %d\n", *n_steps_p);
printf("delta_t = %e\n", *delta_t_p);
printf("output_freq = %d\n", *output_freq_p);
printf("g_i = %c\n", *g_i_p);
#  endif
}  
void Get_init_cond(struct particle_s curr[], int n) {
int part;
printf("For each particle, enter (in order):\n");
printf("   its mass, its x-coord, its y-coord, ");
printf("its x-velocity, its y-velocity\n");
for (part = 0; part < n; part++) {
scanf("%lf", &curr[part].m);
scanf("%lf", &curr[part].s[X]);
scanf("%lf", &curr[part].s[Y]);
scanf("%lf", &curr[part].v[X]);
scanf("%lf", &curr[part].v[Y]);
}
}  
void Gen_init_cond(struct particle_s curr[], int n) {
int part;
double mass = 5.0e24;
double gap = 1.0e5;
double speed = 3.0e4;
srandom(1);
for (part = 0; part < n; part++) {
curr[part].m = mass;
curr[part].s[X] = part*gap;
curr[part].s[Y] = 0.0;
curr[part].v[X] = 0.0;
if (part % 2 == 0)
curr[part].v[Y] = speed;
else
curr[part].v[Y] = -speed;
}
}  
void Output_state(double time, struct particle_s curr[], int n) {
int part;
printf("%.2f\n", time);
for (part = 0; part < n; part++) {
printf("%3d %10.3e ", part, curr[part].s[X]);
printf("  %10.3e ", curr[part].s[Y]);
printf("  %10.3e ", curr[part].v[X]);
printf("  %10.3e\n", curr[part].v[Y]);
}
printf("\n");
}  
void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
int n) {
int k;
double mg; 
vect_t f_part_k;
double len, len_3, fact;
#  ifdef DEBUG
printf("Current total force on particle %d = (%.3e, %.3e)\n",
part, forces[part][X], forces[part][Y]);
#  endif
forces[part][X] = forces[part][Y] = 0.0;
for (k = 0; k < n; k++) {
if (k != part) {
f_part_k[X] = curr[part].s[X] - curr[k].s[X];
f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
len_3 = len*len*len;
mg = -G*curr[part].m*curr[k].m;
fact = mg/len_3;
f_part_k[X] *= fact;
f_part_k[Y] *= fact;
#     ifdef DEBUG
printf("Force on particle %d due to particle %d = (%.3e, %.3e)\n",
part, k, f_part_k[X], f_part_k[Y]);
#     endif
forces[part][X] += f_part_k[X];
forces[part][Y] += f_part_k[Y];
}
}
}  
void Update_part(int part, vect_t forces[], struct particle_s curr[], 
int n, double delta_t) {
double fact = delta_t/curr[part].m;
#  ifdef DEBUG
printf("Before update of %d:\n", part);
printf("   Position  = (%.3e, %.3e)\n", curr[part].s[X], curr[part].s[Y]);
printf("   Velocity  = (%.3e, %.3e)\n", curr[part].v[X], curr[part].v[Y]);
printf("   Net force = (%.3e, %.3e)\n", forces[part][X], forces[part][Y]);
#  endif
curr[part].s[X] += delta_t * curr[part].v[X];
curr[part].s[Y] += delta_t * curr[part].v[Y];
curr[part].v[X] += fact * forces[part][X];
curr[part].v[Y] += fact * forces[part][Y];
#  ifdef DEBUG
printf("Position of %d = (%.3e, %.3e), Velocity = (%.3e,%.3e)\n",
part, curr[part].s[X], curr[part].s[Y],
curr[part].v[X], curr[part].v[Y]);
#  endif
}  
void Compute_energy(struct particle_s curr[], int n, double* kin_en_p,
double* pot_en_p) {
int i, j;
vect_t diff;
double pe = 0.0, ke = 0.0;
double dist, speed_sqr;
for (i = 0; i < n; i++) {
speed_sqr = curr[i].v[X]*curr[i].v[X] + curr[i].v[Y]*curr[i].v[Y];
ke += curr[i].m*speed_sqr;
}
ke *= 0.5;
for (i = 0; i < n-1; i++) {
for (j = i+1; j < n; j++) {
diff[X] = curr[i].s[X] - curr[j].s[X];
diff[Y] = curr[i].s[Y] - curr[j].s[Y];
dist = sqrt(diff[X]*diff[X] + diff[Y]*diff[Y]);
pe += -G*curr[i].m*curr[j].m/dist;
}
}
*kin_en_p = ke;
*pot_en_p = pe;
}  