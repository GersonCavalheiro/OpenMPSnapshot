
#ifndef RESCALE_CORRELATION_H
#define RESCALE_CORRELATION_H

class correlation_integral{
public:
Float *cf_estimate; 
Float *rr_estimate; 
CorrelationFunction *old_cf; 
Integrals *integral; 
private:
int nbin,mbin;
bool rad;
Float dmu,mumin,mumax,*r_high,*r_low; 

public:
correlation_integral(Parameters *par, CorrelationFunction *_cf){

old_cf = new CorrelationFunction(_cf);


integral = new Integrals(); 

nbin = par->nbin_cf;
mbin = par->mbin_cf;
mumin = par->mumin;
mumax = par->mumax;
r_high = par->radial_bins_high_cf;
r_low = par->radial_bins_low_cf;
dmu = (mumax-mumin)/mbin; 
rad=mbin==1&&dmu==1.;

int ec=0;
ec+=posix_memalign((void **) &cf_estimate, PAGE, sizeof(double)*nbin*mbin);
ec+=posix_memalign((void **) &rr_estimate, PAGE, sizeof(double)*nbin*mbin);
assert(ec==0);

reset();

}

~correlation_integral(){
free(cf_estimate);
free(rr_estimate);
}
private:
inline int getbin(Float r, Float mu){

int which_bin = -1; 
for(int i=0;i<nbin;i++){
if((r>r_low[i])&&(r<r_high[i])){
which_bin=i;
break;
}
if((i==nbin-1)&&(r>r_high[i])){
which_bin=nbin; 
}
}
return which_bin*mbin + floor((mu-mumin)/dmu);
}

public:
void compile_integral(const Particle* pi_list, const int* prim_ids, int pln, const Particle pj, const int pj_id, const double prob){

Float tmp_weight, tmp_xi, rij_mag, rij_mu,xi_contrib,rr_contrib;
Particle pi;
int tmp_bin;

for(int i=0;i<pln;i++){ 
if(prim_ids[i]==pj_id){
continue; 
}
pi = pi_list[i]; 
integral->cleanup_l(pi.pos,pj.pos,rij_mag,rij_mu); 
tmp_bin = getbin(rij_mag, rij_mu); 

if ((tmp_bin<0)||(tmp_bin>=nbin*mbin)){
continue; 
}

tmp_weight = pi.w*pj.w; 
tmp_xi = old_cf->xi(rij_mag, rij_mu); 

rr_contrib = tmp_weight / prob; 
xi_contrib = rr_contrib*tmp_xi; 

cf_estimate[tmp_bin]+=xi_contrib;
rr_estimate[tmp_bin]+=rr_contrib;
}
}

void normalize(Float norm1, Float norm2, Float n_pairs){
double corrf2 = norm1*norm2; 

for(int i = 0; i<nbin*mbin;i++){
rr_estimate[i]/=(n_pairs*corrf2);
cf_estimate[i]/=(n_pairs*corrf2*rr_estimate[i]);
}
}

void sum(correlation_integral* corr){
for(int i=0;i<nbin*mbin;i++){
cf_estimate[i]+=corr->cf_estimate[i];
rr_estimate[i]+=corr->rr_estimate[i];
}
}

void reset(){
for(int i=0;i<nbin*mbin;i++){
cf_estimate[i]=0;
rr_estimate[i]=0;
}
}

};

class rescale_correlation{

private:
int nbin,mbin;
Integrals *integral;
compute_integral *compute;
Float *r_centers, *mu_centers, *new_xi_array;
public:
CorrelationFunction *new_cf;

public:
rescale_correlation(){};

~rescale_correlation(){
free(r_centers);
free(mu_centers);
free(new_xi_array);
}

rescale_correlation(Parameters *par){
nbin = par->nbin_cf;
mbin = par->mbin_cf;

integral = new Integrals();
compute = new compute_integral();

r_centers = (Float *)malloc(sizeof(Float)*nbin);
mu_centers = (Float *)malloc(sizeof(Float)*mbin);
new_xi_array = (Float *)malloc(sizeof(Float)*nbin*mbin);
for(int j=0;j<mbin;j++) mu_centers[j]=(0.5+j)/mbin;
for(int i=0;i<nbin;i++) r_centers[i]= 0.5*(par->radial_bins_low_cf[i]+par->radial_bins_high_cf[i]);
}


public:
void refine_wrapper(Parameters *par, Grid all_grid[], CorrelationFunction all_cf[], RandomDraws all_rd[], int number_xi){

CorrelationFunction true_cf;
int grid1_index[3] = {0,1,0}, grid2_index[3] = {0,1,1};

for(int index=0;index<number_xi;index++){ 
if (par->cf_loops>0){
printf("\nRefining correlation function %d of %d.\n",index+1,number_xi);
}
true_cf.copy_function(&all_cf[index]); 
for(int n_refine=0;n_refine<par->cf_loops;n_refine++){ 
CorrelationFunction output = rescale_xi(par, &all_grid[grid1_index[index]], &all_grid[grid2_index[index]], &all_cf[index], &true_cf, &all_rd[index],n_refine);
all_cf[index].copy_function(&output);
}
RandomDraws tmp_rd(&all_cf[index],par,NULL,0);
all_rd[index].copy(&tmp_rd);
}
}

CorrelationFunction rescale_xi(Parameters *par, Grid *grid1, Grid *grid2, CorrelationFunction *old_cf, CorrelationFunction *true_cf, RandomDraws *rd, int index){

std::random_device urandom("/dev/urandom");
std::uniform_int_distribution<unsigned int> dist(1, std::numeric_limits<unsigned int>::max());
unsigned long int steps = dist(urandom);
gsl_rng_env_setup(); 
correlation_integral full_xi_function(par,old_cf); 
uint64 used_pairs=0;

compute->check_threads(par,0); 

#ifdef OPENMP
#pragma omp parallel firstprivate(steps,par,grid1, grid2, old_cf) shared(gsl_rng_default,rd) reduction(+:used_pairs)
{ 
int thread = omp_get_thread_num();
assert(omp_get_num_threads()<=par->nthread);
if (thread==0) printf("# Computing correlation function iteration %d of %d on %d threads.\n", index+1,par->cf_loops, omp_get_num_threads());
#else
{ 
int thread = 0;
printf("# Computing correlation function iteration %d of %d single threaded.\n",index+1,par->cf_loops);
#endif

Particle *prim_list,particle_j;
int *prim_ids,pln,sln,pid_j,prim_id_1D,x;
integer3 delta2,prim_id,sec_id;
double p2;
Float3 cell_sep2;
gsl_rng* locrng = gsl_rng_alloc(gsl_rng_default); 
gsl_rng_set(locrng, steps*(thread+1));

correlation_integral thread_xi_function(par, old_cf);

int ec=0;
ec+=posix_memalign((void **) &prim_list, PAGE, sizeof(Particle)*grid1->maxnp);
ec+=posix_memalign((void **) &prim_ids, PAGE, sizeof(int)*grid1->maxnp);
assert(ec==0);

#ifdef OPENMP
#pragma omp for schedule(dynamic)
#endif
for(int n_loops = 0; n_loops<par->max_loops; n_loops++){
for(int n1=0;n1<grid1->nf;n1++){
prim_id_1D = grid1-> filled[n1]; 
prim_id = grid1->cell_id_from_1d(prim_id_1D); 
pln = compute->particle_list(prim_id_1D, prim_list, prim_ids, grid1); 

if(pln==0) continue; 

used_pairs+=pln*par->N2;

for (int n2=0;n2<par->N2;n2++){
delta2 = rd->random_xidraw(locrng,&p2);
sec_id = prim_id+delta2;
cell_sep2 = grid2->cell_sep(delta2);

x = compute->draw_particle_without_class(sec_id,particle_j,pid_j,cell_sep2,grid2,sln,locrng);

if(x==1) continue; 
p2*=1./(grid1->np*(double)sln);
thread_xi_function.compile_integral(prim_list, prim_ids, pln, particle_j, pid_j, p2);
}
}
#ifdef OPENMP
#pragma omp critical
#endif
{
full_xi_function.sum(&thread_xi_function);
thread_xi_function.reset();
}
} 

free(prim_list);
free(prim_ids);

} 

full_xi_function.normalize(grid1->norm,grid2->norm,(Float)used_pairs);

Float true_xi,old_xi;
for(int i=0;i<nbin;i++){
for(int j=0;j<mbin;j++){
old_xi = old_cf->xi(r_centers[i],mu_centers[j]);
true_xi = true_cf->xi(r_centers[i],mu_centers[j]);
new_xi_array[i*mbin+j]=true_xi/full_xi_function.cf_estimate[i*mbin+j]*old_xi;
}
}

new_cf = new CorrelationFunction(new_xi_array, r_centers, mu_centers, nbin, mbin);
return new_cf;
}

};

#endif
