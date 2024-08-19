#define _XOPEN_SOURCE
#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/stat.h>   
#include <unistd.h>    
#include <sys/types.h> 
#include <pwd.h>       
#include <mpi.h>
#include <omp.h>
double my_gettimeofday(){
struct timeval tmp_time;
gettimeofday(&tmp_time, NULL);
return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}
enum Refl_t {DIFF, SPEC, REFR};   
struct Sphere { 
double radius; 
double position[3];
double emission[3];     
double color[3];        
enum Refl_t refl;       
double max_reflexivity;
};
static const int KILL_DEPTH = 7;
static const int SPLIT_DEPTH = 4;
struct Sphere spheres[] = { 
{1e5,  { 1e5+1,  40.8,       81.6},      {},           {.75,  .25,  .25},  DIFF, -1}, 
{1e5,  {-1e5+99, 40.8,       81.6},      {},           {.25,  .25,  .75},  DIFF, -1}, 
{1e5,  {50,      40.8,       1e5},       {},           {.75,  .75,  .75},  DIFF, -1}, 
{1e5,  {50,      40.8,      -1e5 + 170}, {},           {},                 DIFF, -1}, 
{1e5,  {50,      1e5,        81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, 
{1e5,  {50,     -1e5 + 81.6, 81.6},      {},           {0.75, .75,  .75},  DIFF, -1}, 
{16.5, {40,      16.5,       47},        {},           {.999, .999, .999}, SPEC, -1}, 
{16.5, {73,      46.5,       88},        {},           {.999, .999, .999}, REFR, -1}, 
{10,   {15,      45,         112},       {},           {.999, .999, .999}, DIFF, -1}, 
{15,   {16,      16,         130},       {},           {.999, .999, 0},    REFR, -1}, 
{7.5,  {40,      8,          120},        {},           {.999, .999, 0   }, REFR, -1}, 
{8.5,  {60,      9,          110},        {},           {.999, .999, 0   }, REFR, -1}, 
{10,   {80,      12,         92},        {},           {0, .999, 0},       DIFF, -1}, 
{600,  {50,      681.33,     81.6},      {12, 12, 12}, {},                 DIFF, -1},  
{5,    {50,      75,         81.6},      {},           {0, .682, .999}, DIFF, -1}, 
}; 
static inline void copy(const double *x, double *y)
{
for (int i = 0; i < 3; i++)
y[i] = x[i];
} 
static inline void zero(double *x)
{
for (int i = 0; i < 3; i++)
x[i] = 0;
} 
static inline void axpy(double alpha, const double *x, double *y)
{
for (int i = 0; i < 3; i++)
y[i] += alpha * x[i];
} 
static inline void scal(double alpha, double *x)
{
for (int i = 0; i < 3; i++)
x[i] *= alpha;
} 
static inline double dot(const double *a, const double *b)
{ 
return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
} 
static inline double nrm2(const double *a)
{
return sqrt(dot(a, a));
}
static inline void mul(const double *x, const double *y, double *z)
{
for (int i = 0; i < 3; i++)
z[i] = x[i] * y[i];
} 
static inline void normalize(double *x)
{
scal(1 / nrm2(x), x);
}
static inline void cross(const double *a, const double *b, double *c)
{
c[0] = a[1] * b[2] - a[2] * b[1];
c[1] = a[2] * b[0] - a[0] * b[2];
c[2] = a[0] * b[1] - a[1] * b[0];
}
static inline void clamp(double *x)
{
for (int i = 0; i < 3; i++) {
if (x[i] < 0)
x[i] = 0;
if (x[i] > 1)
x[i] = 1;
}
} 
double sphere_intersect(const struct Sphere *s, const double *ray_origin, const double *ray_direction)
{ 
double op[3];
copy(s->position, op);
axpy(-1, ray_origin, op);
double eps = 1e-4;
double b = dot(op, ray_direction);
double discriminant = b * b - dot(op, op) + s->radius * s->radius; 
if (discriminant < 0)
return 0;   
else 
discriminant = sqrt(discriminant);
double t = b - discriminant;
if (t > eps) {
return t;
} else {
t = b + discriminant;
if (t > eps)
return t;
else
return 0;  
}
}
bool intersect(const double *ray_origin, const double *ray_direction, double *t, int *id)
{ 
int n = sizeof(spheres) / sizeof(struct Sphere);
double inf = 1e20; 
*t = inf;
for (int i = 0; i < n; i++) {
double d = sphere_intersect(&spheres[i], ray_origin, ray_direction);
if ((d > 0) && (d < *t)) {
*t = d;
*id = i;
} 
}
return *t < inf;
} 
void radiance(const double *ray_origin, const double *ray_direction, int depth, unsigned short *PRNG_state, double *out)
{ 
int id = 0;                             
double t;                               
if (!intersect(ray_origin, ray_direction, &t, &id)) {
zero(out);    
return; 
}
const struct Sphere *obj = &spheres[id];
double x[3];
copy(ray_origin, x);
axpy(t, ray_direction, x);
double n[3];  
copy(x, n);
axpy(-1, obj->position, n);
normalize(n);
double nl[3];
copy(n, nl);
if (dot(n, ray_direction) > 0)
scal(-1, nl);
double f[3];
copy(obj->color, f);
double p = obj->max_reflexivity;
depth++;
if (depth > KILL_DEPTH) {
if (erand48(PRNG_state) < p) {
scal(1 / p, f); 
} else {
copy(obj->emission, out);
return;
}
}
if (obj->refl == DIFF) {
double r1 = 2 * M_PI * erand48(PRNG_state);  
double r2 = erand48(PRNG_state);             
double r2s = sqrt(r2); 
double w[3];   
copy(nl, w);
double u[3];   
double uw[3] = {0, 0, 0};
if (fabs(w[0]) > .1)
uw[1] = 1;
else
uw[0] = 1;
cross(uw, w, u);
normalize(u);
double v[3];   
cross(w, u, v);
double d[3];   
zero(d);
axpy(cos(r1) * r2s, u, d);
axpy(sin(r1) * r2s, v, d);
axpy(sqrt(1 - r2), w, d);
normalize(d);
double rec[3];
radiance(x, d, depth, PRNG_state, rec);
mul(f, rec, out);
axpy(1, obj->emission, out);
return;
}
double reflected_dir[3];
copy(ray_direction, reflected_dir);
axpy(-2 * dot(n, ray_direction), n, reflected_dir);
if (obj->refl == SPEC) { 
double rec[3];
radiance(x, reflected_dir, depth, PRNG_state, rec);
mul(f, rec, out);
axpy(1, obj->emission, out);
return;
}
bool into = dot(n, nl) > 0;      
double nc = 1;                   
double nt = 1.5;                 
double nnt = into ? (nc / nt) : (nt / nc);
double ddn = dot(ray_direction, nl);
double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
if (cos2t < 0) {
double rec[3];
radiance(x, reflected_dir, depth, PRNG_state, rec);
mul(f, rec, out);
axpy(1, obj->emission, out);
return;
}
double tdir[3];
zero(tdir);
axpy(nnt, ray_direction, tdir);
axpy(-(into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)), n, tdir);
double a = nt - nc;
double b = nt + nc;
double R0 = a * a / (b * b);
double c = 1 - (into ? -ddn : dot(tdir, n));
double Re = R0 + (1 - R0) * c * c * c * c * c;   
double Tr = 1 - Re;                              
double rec[3];
if (depth > SPLIT_DEPTH) {
double P = .25 + .5 * Re;             
if (erand48(PRNG_state) < P) {
radiance(x, reflected_dir, depth, PRNG_state, rec);
double RP = Re / P;
scal(RP, rec);
} else {
radiance(x, tdir, depth, PRNG_state, rec);
double TP = Tr / (1 - P); 
scal(TP, rec);
}
} else {
double rec_re[3], rec_tr[3];
radiance(x, reflected_dir, depth, PRNG_state, rec_re);
radiance(x, tdir, depth, PRNG_state, rec_tr);
zero(rec);
axpy(Re, rec_re, rec);
axpy(Tr, rec_tr, rec);
}
mul(f, rec, out);
axpy(1, obj->emission, out);
return;
}
double wtime()
{
struct timeval ts;
gettimeofday(&ts, NULL);
return (double)ts.tv_sec + ts.tv_usec / 1E6;
}
int toInt(double x)
{
return pow(x, 1 / 2.2) * 255 + .5;   
} 
int main(int argc, char **argv)
{ 
int w = 720;
int h = 480;
int samples = 200;
if (argc == 2) 
samples = atoi(argv[1]) / 4;
static const double CST = 0.5135;  
double camera_position[3] = {50, 52, 295.6};
double camera_direction[3] = {0, -0.042612, -1};
normalize(camera_direction);
double cx[3] = {w * CST / h, 0, 0};    
double cy[3];
cross(cx, camera_direction, cy);  
normalize(cy);
scal(CST, cy);
int n = sizeof(spheres) / sizeof(struct Sphere);
for (int i = 0; i < n; i++) {
double *f = spheres[i].color;
if ((f[0] > f[1]) && (f[0] > f[2]))
spheres[i].max_reflexivity = f[0]; 
else {
if (f[1] > f[2])
spheres[i].max_reflexivity = f[1];
else
spheres[i].max_reflexivity = f[2]; 
}
}
double debut, fin;
int rank, p;
enum TYPE_MESSAGE{FINISH, ASK_FOR_WORK, SEND_WORK};
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
printf("niveau autorisé par MPI : %d \n", provided);
MPI_Comm_size(MPI_COMM_WORLD, &p);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Status status;
double *image = malloc(3 * w * h  * sizeof(*image));  
if (image == NULL) {
perror("Impossible d'allouer l'image");
exit(1);
}
for (int i = 0; i < 3 * w * h ; i++)
{
image[i] = 0;
}
double *image_save = NULL;
if(rank == 0) {
image_save = malloc(3 * w * h  * sizeof(*image_save)); 
if (image == NULL) {
perror("Impossible d'allouer l'image");
exit(1);
}
for (int i = 0; i < 3 * w * h ; i++) {
image_save[i] = 0;
}
}
debut = my_gettimeofday();
int fp = (p-1-rank)*h*w/p;			
int lp = ((p-rank)*h*w/p)-1;		
printf("rank = %d fp = %d lp = %d\n", rank, lp, fp);
int stop = 0, demande = 0;
int var = 1;
#pragma omp parallel num_threads(5)
{
if(omp_get_thread_num() == 0) {
int initiator = 0, msg, flag;
int epsilon=20, I1[2];
while(1) {
MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG , MPI_COMM_WORLD, &flag, &status);
if(flag) { 
if(status.MPI_TAG == FINISH) {
MPI_Recv(&msg, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
if(!initiator) {
MPI_Bsend(&msg, 1, MPI_INT, (rank+1)%p, FINISH, MPI_COMM_WORLD);	
}
stop = 1;
break;
}
if(status.MPI_TAG == ASK_FOR_WORK) {
MPI_Recv(&msg, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
if(msg == rank) {
initiator = 1;
MPI_Bsend(NULL, 0, MPI_INT, (rank+1)%p, FINISH, MPI_COMM_WORLD);	
}
else {
if((lp-fp)>epsilon) {			
#pragma omp critical 
{
int midwork = fp+(lp-fp)/2;		
int I[] = {midwork,lp};	
MPI_Send(I, 2, MPI_INT, msg, SEND_WORK, MPI_COMM_WORLD); 
lp = midwork;			
}
}
else {
MPI_Bsend(&msg, 1, MPI_INT, (rank+1)%p, ASK_FOR_WORK, MPI_COMM_WORLD);	
}
}
}
if(status.MPI_TAG == SEND_WORK) {
MPI_Recv(I1, 2, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp critical 
{
var = 1;
fp=I1[0];
lp=I1[1];
}
}
}
if(demande) {
MPI_Send(&rank, 1, MPI_INT, (rank+1)%p, ASK_FOR_WORK, MPI_COMM_WORLD);
#pragma omp critical 
{
demande = 0;
var = 0;
}
}
}
}
else { 
while(1) {
int current_p = fp, ok = 1;
#pragma omp critical 
{	
current_p = fp++;
ok = (current_p<lp);
if(!ok && var) demande = 1;
}
if(!ok) {
if(stop) {
break;
}
}
else
{
int i = current_p / w;
int j = current_p % w;
unsigned short PRNG_state[3] = {0, 0, i*i*i};
double pixel_radiance[3] = {0, 0, 0};
for (int sub_i = 0; sub_i < 2; sub_i++) {
for (int sub_j = 0; sub_j < 2; sub_j++) {
double subpixel_radiance[3] = {0, 0, 0};
for (int s = 0; s < samples; s++) { 
double r1 = 2 * erand48(PRNG_state);
double dx = (r1 < 1) ? sqrt(r1) - 1 : 1 - sqrt(2 - r1); 
double r2 = 2 * erand48(PRNG_state);
double dy = (r2 < 1) ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
double ray_direction[3];
copy(camera_direction, ray_direction);
axpy(((sub_i + .5 + dy) / 2 + i) / h - .5, cy, ray_direction);
axpy(((sub_j + .5 + dx) / 2 + j) / w - .5, cx, ray_direction);
normalize(ray_direction);
double ray_origin[3];
copy(camera_position, ray_origin);
axpy(140, ray_direction, ray_origin);
double sample_radiance[3];
radiance(ray_origin, ray_direction, 0, PRNG_state, sample_radiance);
axpy(1. / samples, sample_radiance, subpixel_radiance);
}
clamp(subpixel_radiance);
axpy(0.25, subpixel_radiance, pixel_radiance);
}
}
copy(pixel_radiance, image + 3 * ((h - 1 - i) * w + j)); 
}
}
}
}
fprintf( stdout, "\n");
fin = my_gettimeofday();
fprintf( stdout, "Temps total de calcul pour le processus %d : %g sec\n", rank, fin - debut);
if (rank == 0)
{
debut = my_gettimeofday();		
}
MPI_Reduce(image, image_save, 3*w*h, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);		
if( rank == 0 ) {
{
struct passwd *pass; 
char nom_sortie[100] = "";
char nom_rep[30] = "";
pass = getpwuid(getuid()); 
sprintf(nom_rep, "/tmp/%s", pass->pw_name);
mkdir(nom_rep, S_IRWXU);
sprintf(nom_sortie, "%s/image.ppm", nom_rep);
printf("\nLancer l'image: display %s\n", nom_sortie);
FILE *f = fopen(nom_sortie, "w");
fprintf(f, "P3\n%d %d\n%d\n", w, h, 255); 
for (int i = 0; i < w * h; i++) 
fprintf(f,"%d %d %d ", toInt(image_save[3 * i]), toInt(image_save[3 * i + 1]), toInt(image_save[3 * i + 2])); 
fclose(f); 
fin = my_gettimeofday();
fprintf( stdout, "0 a fini, il attend : %g sec pour rassembler et stocker les données\n", fin - debut);
}
}    
free(image);
if(rank == 0)
free(image_save);		
MPI_Finalize();
}
