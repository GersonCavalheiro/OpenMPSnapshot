
#ifndef COMPUTE_INTEGRAL_H
#define COMPUTE_INTEGRAL_H

#include "integrals.h"

class compute_integral{

private:
uint64 cnt4=0;
int nbin, mbin;


public:
int particle_list(int id_1D, Particle* &part_list, int* &id_list, Grid *grid){

Cell cell = grid->c[id_1D]; 
int no_particles = 0;
for (int i = cell.start; i<cell.start+cell.np; i++, no_particles++){
part_list[no_particles]=grid->p[i];
id_list[no_particles]=i;
}
return no_particles;
}

private:
int draw_particle(integer3 id_3D, Particle &particle, int &pid, Float3 shift, Grid *grid, int &n_particles, gsl_rng* locrng, int &n_particles1, int &n_particles2){

int id_1D = grid-> test_cell(id_3D);
if(id_1D<0) return 1; 
Cell cell = grid->c[id_1D];
if(cell.np==0) return 1; 
pid = floor(gsl_rng_uniform(locrng)*cell.np) + cell.start; 
particle = grid->p[pid]; 
n_particles = cell.np; 
n_particles1 = cell.np1; 
n_particles2 = cell.np2;
#ifdef PERIODIC
particle.pos+=shift;
#endif
return 0;
}

public:
int draw_particle_without_class(integer3 id_3D, Particle &particle, int &pid, Float3 shift, Grid *grid, int &n_particles, gsl_rng* locrng){
int id_1D = grid-> test_cell(id_3D);
if(id_1D<0) return 1; 
Cell cell = grid->c[id_1D];
if(cell.np==0) return 1; 
pid = floor(gsl_rng_uniform(locrng)*cell.np) + cell.start; 
particle = grid->p[pid]; 
n_particles = cell.np; 
#ifdef PERIODIC
particle.pos+=shift;
#endif
return 0;
}
public:
void check_threads(Parameters *par,int print){
#ifdef OPENMP
cpu_set_t mask[par->nthread+1];
int tnum=0;
sched_getaffinity(0, sizeof(cpu_set_t), &mask[par->nthread]);
if(print==1) fprintf(stderr, " CPUs used are: ");
for(int ii=0;ii<64;ii++){
if(CPU_ISSET(ii, &mask[par->nthread])){
if(print==1) fprintf(stderr,"%d ", ii);
CPU_ZERO(&mask[tnum]);
CPU_SET(ii,&mask[tnum]);
tnum++;
}
}
fprintf(stderr,"\n");
#endif
}
private:
CorrelationFunction* which_cf(CorrelationFunction all_cf[], int Ia, int Ib){
if((Ia==1)&(Ib==1)) return &all_cf[0];
else if ((Ia==2)&&(Ib==2)) return &all_cf[1];
else return &all_cf[2];
}
RandomDraws* which_rd(RandomDraws all_rd[], int Ia, int Ib){
if((Ia==1)&(Ib==1)) return &all_rd[0];
else if ((Ia==2)&&(Ib==2)) return &all_rd[1];
else return &all_rd[2];
}
Grid* which_grid(Grid all_grid[], int Ia){
if(Ia==1) return &all_grid[0];
else return &all_grid[1];
}

public:
compute_integral(){};

compute_integral(Grid all_grid[], Parameters *par, CorrelationFunction all_cf[], RandomDraws all_rd[], int I1, int I2, int I3, int I4, int iter_no){

int tot_iter=1; 
if(par->multi_tracers==true) tot_iter=7;

Grid *grid1 = which_grid(all_grid,I1);
Grid *grid2 = which_grid(all_grid,I2);
Grid *grid3 = which_grid(all_grid,I3);
Grid *grid4 = which_grid(all_grid,I4);

CorrelationFunction *cf12 = which_cf(all_cf,I1,I2);
CorrelationFunction *cf13 = which_cf(all_cf,I1,I3);
CorrelationFunction *cf24 = which_cf(all_cf,I2,I4);

RandomDraws *rd13 = which_rd(all_rd,I1,I3);
RandomDraws *rd24 = which_rd(all_rd,I2,I4);

nbin = par->nbin_short; 
mbin = par->mbin; 

STimer initial, TotalTime; 
initial.Start();

int convergence_counter=0, printtime=0;

std::random_device urandom("/dev/urandom");
std::uniform_int_distribution<unsigned int> dist(1, std::numeric_limits<unsigned int>::max());
unsigned long int steps = dist(urandom);

gsl_rng_env_setup(); 

Integrals sumint(par, cf12, cf13, cf24, I1, I2, I3, I4); 

uint64 tot_quads=0; 
uint64 cell_attempt4=0; 
uint64 used_cell4=0; 

check_threads(par,1); 

initial.Stop();
fprintf(stderr, "Init time: %g s\n",initial.Elapsed());
printf("# 1st grid filled cells: %d\n",grid1->nf);
printf("# All 1st grid points in use: %d\n",grid1->np);
printf("# Max points in one cell in grid 1%d\n",grid1->maxnp);
fflush(NULL);

TotalTime.Start(); 

#ifdef OPENMP

#pragma omp parallel firstprivate(steps,par,printtime,grid1,grid2,grid3,grid4,cf12,cf13,cf24) shared(sumint,TotalTime,gsl_rng_default,rd13,rd24) reduction(+:convergence_counter,cell_attempt4,used_cell4,tot_quads)
{ 
int thread = omp_get_thread_num();
assert(omp_get_num_threads()<=par->nthread);
if (thread==0) printf("# Starting integral computation %d of %d on %d threads.\n", iter_no, tot_iter, omp_get_num_threads());
#else
int thread = 0;
printf("# Starting integral computation %d of %d single threaded.\n",iter_no,tot_iter);
{ 
#endif

Particle *prim_list; 
int pln,sln,tln,fln,sln1,sln2; 
int pid_j, pid_k, pid_l; 
Particle particle_j, particle_k, particle_l; 
int* prim_ids; 
double p2,p3,p4; 

int *bin_ij; 
int mnp = grid1->maxnp; 
Float *xi_ik, *w_ijk, *w_ij; 
Float percent_counter;
int x, prim_id_1D;
integer3 delta2, delta3, delta4, prim_id, sec_id, thi_id;
Float3 cell_sep2,cell_sep3;

Integrals locint(par, cf12, cf13, cf24, I1, I2, I3, I4); 

gsl_rng* locrng = gsl_rng_alloc(gsl_rng_default); 
gsl_rng_set(locrng, steps*(thread+1));

int ec=0;
ec+=posix_memalign((void **) &prim_list, PAGE, sizeof(Particle)*mnp);
ec+=posix_memalign((void **) &prim_ids, PAGE, sizeof(int)*mnp);
ec+=posix_memalign((void **) &bin_ij, PAGE, sizeof(int)*mnp);
ec+=posix_memalign((void **) &w_ij, PAGE, sizeof(Float)*mnp);
ec+=posix_memalign((void **) &xi_ik, PAGE, sizeof(Float)*mnp);
ec+=posix_memalign((void **) &w_ijk, PAGE, sizeof(Float)*mnp);
assert(ec==0);

uint64 loc_used_quads; 
#ifdef OPENMP
#pragma omp for schedule(dynamic)
#endif
for (int n_loops = 0; n_loops<par->max_loops; n_loops++){
percent_counter=0.;
loc_used_quads=0;

if (convergence_counter==10){
if (printtime==0) printf("0.01%% convergence achieved in every bin 10 times, exiting.\n");
printtime++;
continue;
}
for (int n1=0; n1<grid1->nf;n1++){

if((float(n1)/float(grid1->nf)*100)>=percent_counter){
printf("Integral %d of %d, run %d of %d on thread %d: Using cell %d of %d - %.0f percent complete\n",iter_no,tot_iter,1+n_loops/par->nthread, int(ceil(float(par->max_loops)/(float)par->nthread)),thread, n1+1,grid1->nf,percent_counter);
percent_counter+=5.;
}

prim_id_1D = grid1-> filled[n1]; 
prim_id = grid1->cell_id_from_1d(prim_id_1D); 
pln = particle_list(prim_id_1D, prim_list, prim_ids, grid1); 

if(pln==0) continue; 

loc_used_quads+=pln*par->N2*par->N3*par->N4;

for (int n2=0; n2<par->N2; n2++){
delta2 = rd13->random_cubedraw_long(locrng, &p2); 
sec_id = prim_id + delta2;
cell_sep2 = grid2->cell_sep(delta2);
x = draw_particle(sec_id, particle_j, pid_j, cell_sep2, grid2, sln, locrng, sln1, sln2);
if(x==1) continue; 

p2*=1./(grid1->np*(double)sln); 

for (int n3=0; n3<par->N3; n3++){
delta3 = rd13->random_cubedraw(locrng, &p3); 
thi_id = prim_id + delta3;
cell_sep3 = grid3->cell_sep(delta3);
x = draw_particle_without_class(thi_id,particle_k,pid_k,cell_sep3,grid3,tln,locrng); 
if(x==1) continue;
if(pid_j==pid_k) continue;

p3*=p2/(double)tln; 

for (int n4=0; n4<par->N4; n4++){
cell_attempt4+=1; 

delta4 = rd24->random_cubedraw(locrng,&p4);
x = draw_particle_without_class(sec_id+delta4,particle_l,pid_l,cell_sep2+grid4->cell_sep(delta4),grid4,fln,locrng); 
if(x==1) continue;
if((pid_l==pid_j)||(pid_l==pid_k)) continue;

used_cell4+=1; 

p4*=p3/(double)fln;


locint.fourth(prim_list, prim_ids, pln, particle_j, particle_k, particle_l, pid_j, pid_k, pid_l, p4);

}
}
}
}

tot_quads+=loc_used_quads;

#ifdef OPENMP
#pragma omp critical 
#endif
{
if ((n_loops+1)%par->nthread==0){ 
TotalTime.Stop(); 
int current_runtime = TotalTime.Elapsed();
int remaining_time = current_runtime/((n_loops+1)/par->nthread)*(par->max_loops/par->nthread-(n_loops+1)/par->nthread);  
fprintf(stderr,"\nFinished integral loop %d of %d after %d s. Estimated time left:  %2.2d:%2.2d:%2.2d hms, i.e. %d s.\n",n_loops+1,par->max_loops, current_runtime,remaining_time/3600,remaining_time/60%60, remaining_time%60,remaining_time);

TotalTime.Start(); 
Float rmsrd_C4, maxrd_C4;

sumint.rel_difference(&locint, rmsrd_C4, maxrd_C4);
if (maxrd_C4 < 1e-4) convergence_counter++;
if (n_loops!=0) {
fprintf(stderr, "RMS relative difference after loop %d is %.3f%%\n", n_loops, rmsrd_C4*100);
fprintf(stderr, "max relative difference after loop %d is %.3f%%\n", n_loops, maxrd_C4*100);
}
}

sumint.sum_ints(&locint);

char output_string[50];
sprintf(output_string,"%d", n_loops);

locint.normalize();

locint.save_integrals(output_string,1);

locint.sum_total_counts(cnt4);
locint.reset();
}

} 

free(prim_list);
free(xi_ik);
free(bin_ij);
free(w_ij);
free(w_ijk);
} 

TotalTime.Stop();

sumint.normalize();

int runtime = TotalTime.Elapsed();
printf("\n\nINTEGRAL %d OF %d COMPLETE\n",iter_no,tot_iter);
fprintf(stderr, "\nTotal process time for %.2e sets of cells and %.2e quads of particles: %d s, i.e. %2.2d:%2.2d:%2.2d hms\n", double(used_cell4),double(tot_quads),runtime, runtime/3600,runtime/60%60,runtime%60);
printf("We tried %.2e quads of cells.\n",double(cell_attempt4));
printf("Of these, we accepted %.2e quads of cells.\n",double(used_cell4));
printf("We sampled %.2e quads of particles.\n",double(tot_quads));
printf("Of these, we have integral contributions from %.2e quads of particles.\n",double(cnt4));
printf("Cell acceptance ratio is %.3f for quads.\n",(double)used_cell4/cell_attempt4);

printf("Acceptance ratio is %.3f for quads.\n",(double)cnt4/tot_quads);

printf("\nTrial speed: %.2e quads per core per second\n",double(tot_quads)/(runtime*double(par->nthread)));
printf("Acceptance speed: %.2e quads per core per second\n",double(cnt4)/(runtime*double(par->nthread)));

char out_string[5];
sprintf(out_string,"full");
sumint.save_integrals(out_string,1); 
sumint.save_counts(tot_quads); 

fflush(NULL);
return;
}

};

#endif
