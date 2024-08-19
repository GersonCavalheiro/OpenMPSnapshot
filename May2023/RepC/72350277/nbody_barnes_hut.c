#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
#include <omp.h>
#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif
#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"
FILE* f_out=NULL;
int nparticles=10;      
float T_FINAL=1.0;     
particle_t*particles;
node_t *root;
double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;
int num_threads = 4;
void init() {
init_alloc(4*nparticles);
root = malloc(sizeof(node_t));
init_node(root, NULL, XMIN, XMAX, YMIN, YMAX);
}
#ifdef DISPLAY
Display *theDisplay;  
GC theGC;             
Window theMain;       
#endif
void compute_force(particle_t*p, double x_pos, double y_pos, double mass) {
double x_sep, y_sep, dist_sq, grav_base;
x_sep = x_pos - p->x_pos;
y_sep = y_pos - p->y_pos;
dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);
grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;
p->x_force += grav_base*x_sep;
p->y_force += grav_base*y_sep;
}
void compute_force_on_particle(node_t* n, particle_t *p) {
if(! n || n->n_particles==0) {
return;
}
if(n->particle) {
assert(n->children == NULL);
compute_force(p, n->x_center, n->y_center, n->mass);
} else {
#define THRESHOLD 2
double size = n->x_max - n->x_min; 
double diff_x = n->x_center - p->x_pos;
double diff_y = n->y_center - p->y_pos;
double distance = sqrt(diff_x*diff_x + diff_y*diff_y);
#if BRUTE_FORCE
int i;
for(i=0; i<4; i++) {
compute_force_on_particle(&n->children[i], p);
}
#else
if(size / distance < THRESHOLD) {
compute_force(p, n->x_center, n->y_center, n->mass);
} else {
int i;
for(i=0; i<4; i++) {
compute_force_on_particle(&n->children[i], p);
}
}
#endif
}
}
void compute_force_in_node(node_t *n) {
if(!n) return;
if(n->particle) {
particle_t*p = n->particle;
p->x_force = 0;
p->y_force = 0;
compute_force_on_particle(root, p);
}
if(n->children) {
int i;
#pragma omp parallel for schedule(dynamic)
for(i=0; i<4; i++) {
compute_force_in_node(&n->children[i]);
}
}
}
void move_particle(particle_t*p, double step, node_t* new_root) {
p->x_pos += (p->x_vel)*step;
p->y_pos += (p->y_vel)*step;
double x_acc = p->x_force/p->mass;
double y_acc = p->y_force/p->mass;
p->x_vel += x_acc*step;
p->y_vel += y_acc*step;
double cur_acc = (x_acc*x_acc + y_acc*y_acc);
cur_acc = sqrt(cur_acc);
double speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
double cur_speed = sqrt(speed_sq);
sum_speed_sq += speed_sq;
max_acc = MAX(max_acc, cur_acc);
max_speed = MAX(max_speed, cur_speed);
p->node = NULL;
if(p->x_pos < new_root->x_min ||
p->x_pos > new_root->x_max ||
p->y_pos < new_root->y_min ||
p->y_pos > new_root->y_max) {
free(p);
nparticles--;
} else {
insert_particle(p, new_root);
}
}
void move_particles_in_node(node_t*n, double step, node_t *new_root) {
if(!n) return;
if(n->particle) {
particle_t*p = n->particle;
move_particle(p, step, new_root);
}
if(n->children) {
int i;
for(i=0; i<4; i++) {
move_particles_in_node(&n->children[i], step, new_root);
}
}
}
void all_move_particles(double step)
{
compute_force_in_node(root);
node_t* new_root = malloc(sizeof(node_t));
init_node(new_root, NULL, XMIN, XMAX, YMIN, YMAX);
move_particles_in_node(root, step, new_root);
free_node(root);
free(root);
root = new_root;
}
void run_simulation() {
double t = 0.0, dt = 0.01;
while (t < T_FINAL && nparticles>0) {
t += dt;
all_move_particles(dt);
dt = 0.1*max_speed/max_acc;
#if DISPLAY
node_t *n = root;
clear_display();
draw_node(n);
flush_display();
#endif
}
}
void insert_all_particles(int nparticles, particle_t*particles, node_t*root) {
int i;
for(i=0; i<nparticles; i++) {
insert_particle(&particles[i], root);
}
}
int main(int argc, char**argv)
{
if(argc >= 2) {
nparticles = atoi(argv[1]);
}
if(argc == 3) {
T_FINAL = atof(argv[2]);
}
init();
particles = malloc(sizeof(particle_t)*nparticles);
all_init_particles(nparticles, particles);
insert_all_particles(nparticles, particles, root);
#ifdef DISPLAY
simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
#endif
struct timeval t1, t2;
gettimeofday(&t1, NULL);
run_simulation();
gettimeofday(&t2, NULL);
double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
#ifdef DUMP_RESULT
FILE* f_out = fopen("particles.log", "w");
assert(f_out);
print_particles(f_out, root);
fclose(f_out);
#endif
printf("-----------------------------\n");
printf("nparticles: %d\n", nparticles);
printf("T_FINAL: %f\n", T_FINAL);
printf("-----------------------------\n");
printf("Simulation took %lf s to complete\n", duration);
#ifdef DISPLAY
node_t *n = root;
clear_display();
draw_node(n);
flush_display();
printf("Hit return to close the window.");
getchar();
XCloseDisplay(theDisplay);
#endif
return 0;
}
