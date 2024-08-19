#include "correlation_function.h"
#include "parameters.h"
#include <gsl/gsl_sf_dawson.h>

#ifndef RANDOM_DRAWS_H
#define RANDOM_DRAWS_H


class RandomDraws{


public:
static CorrelationFunction *corr;
int nside;     
int nsidecube; 
int nsidecube_long; 
double boxside;
double *x; 
double *xcube; 
double *xcube_long; 

private:
ransampl_ws* ws;

ransampl_ws* cube;

ransampl_ws* cube_long;

public:
void copy(RandomDraws *rd){

nside=rd->nside;
nsidecube=rd->nsidecube;
nsidecube_long=rd->nsidecube_long;
boxside=rd->boxside;

int x_size=pow(nside,3);
int xcube_size = pow(nsidecube,3);
int xcube_long_size = pow(nsidecube_long,3);
x = (double *)malloc(sizeof(double)*x_size);
xcube = (double *)malloc(sizeof(double)*xcube_size);
xcube_long = (double *)malloc(sizeof(double)*xcube_long_size);

for(int i=0;i<x_size;i++) x[i]=rd->x[i];
for(int i=0;i<xcube_size;i++) xcube[i]=rd->xcube[i];
for(int i=0;i<xcube_long_size;i++) xcube_long[i]=rd->xcube_long[i];

ws = ransampl_alloc(x_size);
ransampl_set(ws, x);

cube = ransampl_alloc(xcube_size);
ransampl_set(cube, xcube);

cube_long = ransampl_alloc(xcube_long_size);
ransampl_set(cube_long, xcube_long);
}



public:
RandomDraws(){
}

RandomDraws(CorrelationFunction *fun,Parameters *par,const double *xin, long np){

long n=np;
corr=fun;

Float box_max = fmax(fmax(par->rect_boxsize.x,par->rect_boxsize.y),par->rect_boxsize.z);
boxside=box_max/par->nside;
nside=2*ceil(par->xicutoff/boxside)+1;

if (par->loadname!=NULL&&(xin==NULL||np==0)){
int len = strlen(par->loadname);
if(len>4&&!strcmp(&par->loadname[len-4],".bin")){
printf("\n# Loading pre-computed binary-format probability grid\n");
readDataBin(&x,&n,par->loadname);
}
else{
printf("\n# Loading pre-computed ascii-format probability grid\n");
readData(&x,&n,par->loadname);
}
if(n==0){
printf("\n# Computing the probability grid\n");
integData2(&x,&n,xi_integrand,nside,boxside);
printf("# Probability grid computation complete\n");
}
}
else
if (xin==NULL||np==0){
printf("\n# Computing the probability grid");
integData2(&x,&n,xi_integrand,nside,boxside);
printf("# Probability grid computation complete\n");
}
else
copyData(&x,&n,xin);

if (par->savename!=NULL){
int len = strlen(par->savename);
if(len>4&&!strcmp(&par->savename[len-4],".bin"))
writeDataBin(&x,&n,par->savename);
else
fprintf(stderr,"Save file %s does not end in \".bin\". No output written.\n",par->savename);
}

ws = ransampl_alloc( n );
ransampl_set( ws, x );

double sum=0.;
for(int i=0;i<n;i++){
sum+=x[i];
}
for(int i=0;i<n;i++){
x[i]/=sum;
}



int maxsep = ceil(1.1*par->rmax_short/boxside);
nsidecube = 2 * maxsep + 1;
long nn=0;

compute_r2_prob(&xcube,&nn,nsidecube,boxside);

cube = ransampl_alloc( nn );
ransampl_set( cube, xcube );

sum=0.;
for(int i=0;i<nn;i++){
sum+=xcube[i];
}
for(int i=0;i<nn;i++){
xcube[i]/=sum;
}



int maxsep_long = ceil(1.1*par->rmax_long/boxside);
nsidecube_long = 2 * maxsep_long + 1;
long nn_long=0;

compute_r2_prob(&xcube_long,&nn_long,nsidecube_long,boxside);

cube_long = ransampl_alloc( nn_long );
ransampl_set( cube_long, xcube_long );

sum=0.;
for(int i=0;i<nn;i++){
sum+=xcube_long[i];
}
for(int i=0;i<nn;i++){
xcube_long[i]/=sum;
}

}


~RandomDraws() {
ransampl_free( ws );
ransampl_free( cube );
ransampl_free( cube_long );
free(x);
free(xcube);
free(xcube_long);
}

integer3 random_xidraw(gsl_rng* rng, double* p){
int n=ransampl_draw( ws, gsl_rng_uniform(rng), gsl_rng_uniform(rng) );
*p=x[n];
return cubifyindex(nside,n);
}

integer3 random_cubedraw(gsl_rng* rng, double* p){
int n=ransampl_draw( cube, gsl_rng_uniform(rng), gsl_rng_uniform(rng) );
*p=xcube[n];
return cubifyindex(nsidecube,n);
}

integer3 random_cubedraw_long(gsl_rng* rng, double* p){
int n=ransampl_draw( cube_long, gsl_rng_uniform(rng), gsl_rng_uniform(rng) );
*p=xcube_long[n];
return cubifyindex(nsidecube_long,n);
}

integer3 cubifyindex(int nside,int n){

assert(n>=0&&n<pow(nside,3));

integer3 cid;
cid.z = n%nside-((nside-1)/2);
n = n/nside;
cid.y = n%nside-((nside-1)/2);
cid.x = n/nside-((nside-1)/2);

return cid;
}

private:
void readData(double **x,long *np,const char *filename){

char line[10000];
int  n=0;
FILE *fp;
fp = fopen(filename, "r");
if (fp==NULL) {
fprintf(stderr,"File %s not found\n", filename); abort();
}

while (fgets(line,10000,fp)!=NULL) {
if (line[0]=='#') continue;
if (line[0]=='\n') continue;
n++;
}
rewind(fp);
*np = n;

if(n!=pow(nside,3)){
fprintf(stderr,"File %s does not contain the correct probability grid.\n", filename); abort();
}


*x = (double *)malloc(sizeof(double)*n);
printf("# Found %d lines in %s\n", n, filename);


int ct=0;
while (fgets(line,10000,fp)!=NULL) {
if (line[0]=='#') continue;
if (line[0]=='\n') continue;
sscanf(line, "%lf",  &((*x)[ct++]) );
}

assert(ct==n);

fclose(fp);

}

void integData2(double **x, long *np, integrand xi_fun, int nside, double boxside){

(*np)=(int)pow(nside,3);

*x = (double *)malloc(sizeof(double)*(*np));

printf("\nNumber of Boxes in Probability Grid: %ld\n",(*np));
fflush(NULL);
#ifdef OPENMP
#pragma omp parallel
#endif
{
int len=(nside-1)/2; 
Float n, R = boxside/2;

double xmin[1]={0}, xmax[1]={2*boxside*len}, val, err, param[2]={R,0};

#ifdef OPENMP
#pragma omp for schedule(dynamic,32)
#endif
for(int i = 0; i<=len;i++){
for(int k = 0; k<=len;k++){
for(int l = 0;l<=len;l++){
n = sqrt(pow(i,2)+pow(k,2)+pow(l,2))*boxside; 
param[1]=n;

hcubature(1, xi_fun, &param[0], 1, xmin, xmax, 0, 0, 1e-5, ERROR_INDIVIDUAL, &val, &err);


for(int is=-1;is<=1;is+=2){
for(int ks=-1;ks<=1;ks+=2){
for(int ls=-1;ls<=1;ls+=2){
int ic=is*i+len;
int kc=ks*k+len;
int lc=ls*l+len;
if(val<=0) val=0.;
(*x)[nside*nside*ic+nside*kc+lc]=val;
}
}
}
}
}
}
}
}

void compute_r2_prob(double **x, long *np, int nside, double boxside){

(*np)=(int)pow(nside,3);

*x = (double *)malloc(sizeof(double)*(*np));

printf("\nNumber of Boxes in Probability Grid: %ld\n",(*np));
fflush(NULL);

int len=(nside-1)/2; 
Float n, prob;

for(int i = 0; i<=len;i++){
for(int k = 0; k<=len;k++){
for(int l = 0;l<=len;l++){
n = sqrt(pow(i,2)+pow(k,2)+pow(l,2))*boxside; 
prob = r2prob(n,boxside);

for(int is=-1;is<=1;is+=2){
for(int ks=-1;ks<=1;ks+=2){
for(int ls=-1;ls<=1;ls+=2){
int ic=is*i+len;
int kc=ks*k+len;
int lc=ls*l+len;
(*x)[nside*nside*ic+nside*kc+lc]=prob;
}
}
}
}
}
}
}

void copyData(double **x, long *n,const double *xin){
(*x) = (double *)malloc(sizeof(double)*(*n));
for(int i=0;i<(*n);i++){
(*x)[i]=xin[i];
}
}

void readDataBin(double **x, long *n, const char *filename){

FILE *fp;
int _nside;
double _boxside;
int stat=0;
fp = fopen(filename, "rb");
if (fp==NULL) {
fprintf(stderr,"# File %s not found\n", filename); return;
}
stat+=fread(&_nside, sizeof(int), 1, fp);
stat+=fread(&_boxside, sizeof(double), 1, fp);
if(!(_nside==nside&&_boxside==boxside)){
fprintf(stderr,"# Size has changed. Recalculating probability grid.\n");
fflush(NULL);
*n=0;
fclose(fp);
return;
}
stat+=fread(n, sizeof(long), 1, fp);
*x = (double *)malloc(sizeof(double)*(*n));
if(*x==NULL){
fprintf(stderr,"Allocation error.\n");
fflush(NULL);
abort();
}

if(stat!=3||*n!=(long)fread(*x, sizeof(double), *n, fp)){
fprintf(stderr,"# Error reading file %s. Recalculating probability grid.\n", filename);
fflush(NULL);
*n=0;
free(*x);
fclose(fp);
return;
}

assert(*n==pow(nside,3));

fclose(fp);

}

void writeDataBin(double **x, long *n, const char *filename){

FILE *fp;
fp = fopen(filename, "wb");
if (fp==NULL) {
fprintf(stderr,"# File %s not found\n", filename);
return;
}
fwrite(&nside, 1, sizeof(int), fp);
fwrite(&boxside, 1, sizeof(double), fp);
fwrite(n, 1, sizeof(long), fp);
if(*n*sizeof(double)!=fwrite(*x, 1, *n*sizeof(double), fp)){
fprintf(stderr,"# Error writing to file %s.\n", filename);
return;
}
printf("# Probability grid written to file %s.\n", filename);

fclose(fp);

}

public:

static int xi_integrand(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval){
double* param = (double *) fdata;

const double R = param[0];
const double n = param[1];

Float factor_1 = pow((x[0]+n)/(2*R),2);
Float factor_2 = pow((x[0]-n)/(2*R),2);

Float tmp_xi = corr->xi(x[0]);

if(tmp_xi<1e-10){ 
tmp_xi=10./pow(x[0],2.);
}
if(n<=0){
fval[0]= (pow(x[0],2)*tmp_xi)/ (pow(R,3))*exp(-pow(x[0]/(2*R),2));
} else{
fval[0] = x[0]/ (R*n) * (exp(-factor_2)-exp(-factor_1))*tmp_xi;
}

return 0;
}

Float r2prob(const Float n, const Float a){

Float output;
if(n<=0){
output=pow(a,-2);
} else {
output = gsl_sf_dawson(n/a)/(a*n);
}
return output;
}

};

#endif
