#include "../include/twoBody.h"
TwoBody* newTwoBody (int argc, char *argv[])
{
if (argc-1 != 1) Usage(argv[0]);
TwoBody *model = (TwoBody*)malloc(sizeof(TwoBody));
model->num_threads = atoi(argv[1]);
readInput(model);
return model;
}
void readInput (TwoBody *m)
{
if (!scanf("%lf %d %d %d",&m->dt,&m->num_steps,&m->out_freq,&m->n)) printError("Reading input");
m->p = (Particle*)malloc(sizeof(Particle)*m->n);
for (int i = 0; i < m->n; i++)
{
if (!scanf("%lf %lf %lf %lf %lf %lf %lf",&m->p[i].mass,&m->p[i].pos[X],&m->p[i].pos[Y],&m->p[i].pos[Z],\
&m->p[i].vel[X],&m->p[i].vel[Y],&m->p[i].vel[Z])) printError("Reading input");
}
}
void printError (const char msg[])
{
fprintf(stderr,"[-] ERROR ! %s\n",msg);
exit(EXIT_FAILURE);
}
void printParticle (const Particle *p, int N)
{
for (int i = 0; i < N; i++)
{
printf("%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n",p[i].pos[X],p[i].pos[Y],p[i].pos[Z],p[i].vel[X],p[i].vel[Y],p[i].vel[Z]);
}
}
void Compute_pos_vel (Particle *p, vect_t *forces, double dt, int q, int n)
{
p[q].pos[X] += dt*p[q].vel[X];
p[q].pos[Y] += dt*p[q].vel[Y];
p[q].pos[Z] += dt*p[q].vel[Z];
p[q].vel[X] += dt/p[q].mass*forces[q][X];
p[q].vel[Y] += dt/p[q].mass*forces[q][Y];
p[q].vel[Z] += dt/p[q].mass*forces[q][Z];
}
void Compute_force_phase1 (Particle *p, vect_t *loc_forces, vect_t *forces, int q, int n)
{
int k;
double x_diff, y_diff, z_diff, dist, dist_cubed;
vect_t forces_qk;
int my_rank = omp_get_thread_num();
for (k = q+1; k < n; k++)
{
x_diff = p[q].pos[X] - p[k].pos[X];
y_diff = p[q].pos[Y] - p[k].pos[Y];
z_diff = p[q].pos[Z] - p[k].pos[Z];
dist = sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff);
dist_cubed = dist*dist*dist;
forces_qk[X] = -G*p[q].mass*p[k].mass / dist_cubed * x_diff;
forces_qk[Y] = -G*p[q].mass*p[k].mass / dist_cubed * y_diff;
forces_qk[Z] = -G*p[q].mass*p[k].mass / dist_cubed * z_diff;
loc_forces[my_rank*n+q][X] += forces_qk[X];
loc_forces[my_rank*n+q][Y] += forces_qk[Y];
loc_forces[my_rank*n+q][Z] += forces_qk[Z];
loc_forces[my_rank*n+k][X] -= forces_qk[X];
loc_forces[my_rank*n+k][Y] -= forces_qk[Y];
loc_forces[my_rank*n+k][Z] -= forces_qk[Z];
}
}
void Compute_force_phase2 (vect_t *loc_forces, vect_t *forces, int n)
{
int k, thread;
#pragma omp for
for (k = 0; k < n; k++)
{
forces[k][X] = 0; forces[k][Y] = 0; forces[k][Z] = 0;
for (thread = 0; thread < omp_get_num_threads(); thread++)
{
forces[k][X] += loc_forces[thread*n+k][X];
forces[k][Y] += loc_forces[thread*n+k][Y];
forces[k][Z] += loc_forces[thread*n+k][Z];
}
}
}
void solveModel (TwoBody *m)
{
int step, part;
int n = m->n;
int thread_count = m->num_threads;
int output_freq = m->out_freq;
int nsteps = m->num_steps;
double dt = m->dt;
Particle *p = m->p;
vect_t *forces = (vect_t*)malloc(sizeof(vect_t)*n);
vect_t *loc_forces = (vect_t*)malloc(sizeof(vect_t)*n*thread_count);
#pragma omp parallel num_threads(thread_count) default(none) shared(p,forces,loc_forces,thread_count,dt,n,nsteps,output_freq) private(step,part)
for (step = 0; step < nsteps; step++)
{
#pragma omp for
for (part = 0; part < n*thread_count; part++)
loc_forces[part][X] = 0, loc_forces[part][Y] = 0, loc_forces[part][Z] = 0;
#pragma omp for schedule(static,1)                   
for (part = 0; part < n; part++)
Compute_force_phase1(p,loc_forces,forces,part,n);
Compute_force_phase2(loc_forces,forces,n);
#pragma omp for
for (part = 0; part < n; part++)
Compute_pos_vel(p,forces,dt,part,n);
#pragma omp single
if (step % output_freq == 0)
writeVTK(step,p,n);
}
free(forces);
}
void writeVTK (int step, Particle *p, int n)
{
int i;
char filename[500];
sprintf(filename,"VTK/step%d.vtk",step);
FILE *vtk = fopen(filename,"w+");
fprintf(vtk,"# vtk DataFile Version 3.0\n");
fprintf(vtk,"Vtk output\n");
fprintf(vtk,"ASCII\n");
fprintf(vtk,"DATASET POLYDATA\n");
fprintf(vtk,"POINTS %d float\n",n);
for (i = 0; i < n; i++)
fprintf(vtk,"%e %e %e\n",p[i].pos[X],p[i].pos[Y],p[i].pos[Z]);
fprintf(vtk,"VERTICES %d %d\n",n,n*2);
for (i = 0; i < n; i++)
fprintf(vtk,"1 %d\n",i);
fclose(vtk);
}
void Usage (const char pName[])
{
printf("=================================================\n");
printf("Usage:> %s <num_threads>\n",pName);
printf("<num_threads> = Number of threads\n");
printf("=================================================\n");
exit (EXIT_FAILURE);
}