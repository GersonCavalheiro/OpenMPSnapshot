#define _POSIX_C_SOURCE  1  
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>
#include <math.h>
#include <mmio.h>
typedef uint64_t u64;
typedef uint32_t u32;
long n = 1;
u64 prime;
char *matrix_filename;
char *kernel_filename;
bool right_kernel = false;
bool checkpoint = false;
int stop_after = -1;
int n_iterations;      
double start;
double checkpoint_elapsed = 0;
double elapsed;
double last_print;
bool ETA_flag;
int expected_iterations;
struct sparsematrix_t {
int nrows;        
int ncols;
int np;           
long int nnz;     
int *i;           
int *j;           
u32 *x;           
};
u64 rng_state[4] = {0x1415926535, 0x8979323846, 0x2643383279, 0x5028841971};
u64 rotl(u64 x, int k)
{
u64 foo = x << k;
u64 bar = x >> (64 - k);
return foo ^ bar;
}
u64 random64()
{
u64 result = rotl(rng_state[0] + rng_state[3], 23) + rng_state[0];
u64 t = rng_state[1] << 17;
rng_state[2] ^= rng_state[0];
rng_state[3] ^= rng_state[1];
rng_state[1] ^= rng_state[2];
rng_state[0] ^= rng_state[3];
rng_state[2] ^= t;
rng_state[3] = rotl(rng_state[3], 45);
return result;
}
double wtime()
{
struct timeval ts;
gettimeofday(&ts, NULL);
return (double) ts.tv_sec + ts.tv_usec / 1e6;
}
void human_format(char * target, long n) {
if (n < 1000) {
sprintf(target, "%" PRId64, n);
return;
}
if (n < 1000000) {
sprintf(target, "%.1fK", n / 1e3);
return;
}
if (n < 1000000000) {
sprintf(target, "%.1fM", n / 1e6);
return;
}
if (n < 1000000000000ll) {
sprintf(target, "%.1fG", n / 1e9);
return;
}
if (n < 1000000000000000ll) {
sprintf(target, "%.1fT", n / 1e12);
return;
}
}
void usage(char ** argv)
{
printf("%s [OPTIONS]\n\n", argv[0]);
printf("Options:\n");
printf("--matrix FILENAME           MatrixMarket file containing the spasre matrix\n");
printf("--prime P                   compute modulo P\n");
printf("--n N                       blocking factor [default 1]\n");
printf("--output-file FILENAME      store the block of kernel vectors\n");
printf("--right                     compute right kernel vectors\n");
printf("--left                      compute left kernel vectors [default]\n");
printf("--checkpoint                continue from the checkpoint\n");
printf("--stop-after N              stop the algorithm after N iterations\n");
printf("\n");
printf("The --matrix and --prime arguments are required\n");
printf("The --stop-after and --output-file arguments mutually exclusive\n");
exit(0);
}
void process_command_line_options(int argc, char ** argv)
{
struct option longopts[9] = {
{"matrix", required_argument, NULL, 'm'},
{"prime", required_argument, NULL, 'p'},
{"n", required_argument, NULL, 'n'},
{"output-file", required_argument, NULL, 'o'},
{"right", no_argument, NULL, 'r'},
{"left", no_argument, NULL, 'l'},
{"checkpoint", no_argument, NULL, 'c'},
{"stop-after", required_argument, NULL, 's'},
{NULL, 0, NULL, 0}
};
char ch;
while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
switch (ch) {
case 'm':
matrix_filename = optarg;
break;
case 'n':
n = atoi(optarg);
break;
case 'p':
prime = atoll(optarg);
break;
case 'o':
kernel_filename = optarg;
break;
case 'r':
right_kernel = true;
break;
case 'l':
right_kernel = false;
break;
case 'c':
checkpoint = true;
break;    
case 's':
stop_after = atoll(optarg);
break;
default:
errx(1, "Unknown option\n");
}
}
if (matrix_filename == NULL || prime == 0)
usage(argv);
if (kernel_filename != NULL && stop_after > 0)
usage(argv);
if (prime > 0x3fffffdd) {
errx(1, "p is capped at 2**30 - 35.  Slighly larger values could work, with the\n");
printf("suitable code modifications.\n");
exit(1);
}
}
void save_vector_block(char const * filename, int nrows, int ncols, u32 const * v)
{
printf("Saving result in %s\n", filename);
FILE * f = fopen(filename, "w");
if (f == NULL)
err(1, "cannot open %s", filename);
fprintf(f, "%%%%MatrixMarket matrix array integer general\n");
fprintf(f, "%%block of left-kernel vector computed by lanczos_modp\n");
fprintf(f, "%d %d\n", nrows, ncols);
for (long j = 0; j < ncols; j++)
for (long i = 0; i < nrows; i++)
fprintf(f, "%d\n", v[i*n + j]);
fclose(f);
}
void set_checkpoint(u32 *v, u32 *p, long block_size_pad)
{
char const *filename = "checkpoint.tmp";
FILE * f = fopen(filename, "w");
if (f == NULL)
err(1, "cannot open %s", filename);
fprintf(f, "%d\n", n_iterations);
fprintf(f, "%lf\n", start);
fprintf(f, "%lf\n", elapsed);
for (long i = 0; i < block_size_pad; i++)
fprintf(f, "%u\n", v[i]);
for (long i = 0; i < block_size_pad; i++)
fprintf(f, "%u\n", p[i]);
rename("checkpoint.tmp", "checkpoint.mtx");
fclose(f);
}
void get_checkpoint(u32 *v, u32 *p, long block_size_pad)
{
char const *filename = "checkpoint.mtx";
FILE *f = fopen(filename, "r");
if (f == NULL)
err(1, "impossible d'ouvrir %s", filename);
u32 i;
int j;
if (fscanf(f, "%d\n", &j)){
n_iterations = j;
}
double t;
if (fscanf(f, "%lf\n", &t)){
start = t;
}
if (fscanf(f, "%lf\n", &t)){
checkpoint_elapsed = t;
}
for (long u = 0; u < block_size_pad; u++) {
if (1 != fscanf(f, "%u\n", &i))
errx(1, "parse error entry %ld\n", u);
v[u] = i;
}
for (long u = 0; u < block_size_pad; u++) {
if (1 != fscanf(f, "%u\n", &i))
errx(1, "parse error entry %ld\n", u);
p[u] = i;
}
fclose(f);
}
void sum_mod(u32 *out, u32 *in){
*out = ((*out) + (*in)) % prime;
}
#pragma omp declare reduction(sum_mod:u32:sum_mod(&omp_out, &omp_in)) initializer(omp_priv = 0)
struct sparsematrix_t *coo_to_csr(struct sparsematrix_t const *Matrix) 
{
long int nnz = Matrix->nnz;
u32 *Mx = (u32*) malloc(sizeof(u32) * nnz); 
int *Mi = (int*) malloc(sizeof(int) * nnz);
int *Mj = (int*) malloc(sizeof(int) * nnz);
memcpy(Mi, Matrix->i,   sizeof(int) * nnz);
memcpy(Mj, Matrix->j,   sizeof(int) * nnz);
memcpy(Mx, Matrix->x,   sizeof(u32) * nnz);
struct sparsematrix_t *newM = (struct sparsematrix_t *) malloc(sizeof(struct sparsematrix_t));
u32 *B = (u32*) malloc(sizeof(u32) * nnz);
int *newMj = (int*) malloc(sizeof(int) * nnz);
int M = Mi[0];
for(long int i = 1; i < nnz; i++)
if(M < Mi[i])
M = Mi[i];
int *C = (int*) calloc(M+1, sizeof(int));
int *P = (int*) calloc(M+2, sizeof(int));
for(int i = 0; i < nnz; i++){
int bucket = Mi[i];
C[bucket]++;
}
P[0] = 0;
int s = 0;
for(int i = 0; i < M+1; i++){
P[i+1] = s;
s += C[i];
}
for(int i = 0; i < nnz; i++){
int bucket = Mi[i] + 1;
B[P[bucket]] = Mx[i];
newMj[P[bucket]] = Mj[i];
P[bucket]++;
}
newM->x = B;
newM->i = P;
newM->j = newMj;
newM->np = M+2;
newM->nrows = Matrix->nrows;
newM->ncols = Matrix->ncols;
newM->nnz = Matrix->nnz;
return newM;
}
struct sparsematrix_t *csr_to_coo(struct sparsematrix_t const *Matrix)
{
int *Mi = (int*) calloc(Matrix->nnz, sizeof(int));
int nb = 0;
for(int k = 1 ; k < Matrix->np +1 ; k++){
nb = Matrix->i[k];
for(int j = nb ; j <= nb + nb - Matrix->i[k-1] ; j++)
Mi[j] = k;
}
struct sparsematrix_t *coo = (struct sparsematrix_t*) malloc(sizeof(struct sparsematrix_t));
memcpy(coo, Matrix, sizeof(struct sparsematrix_t));
memcpy(coo->i, Mi, sizeof(int) * Matrix->nnz);
return coo;
}
void sparsematrix_mm_load(struct sparsematrix_t * M, char const * filename)
{
int nrows = 0;
int ncols = 0;
long nnz = 0;
printf("Loading matrix from %s\n", filename);
fflush(stdout);
FILE *f = fopen(filename, "r");
if (f == NULL)
err(1, "impossible d'ouvrir %s", filename);
MM_typecode matcode;
if (mm_read_banner(f, &matcode) != 0)
errx(1, "Could not process Matrix Market banner.\n");
if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", 
mm_typecode_to_str(matcode));
if (!mm_is_general(matcode) || !mm_is_integer(matcode))
errx(1, "Matrix type [%s] not supported (only integer general are OK)", 
mm_typecode_to_str(matcode));
if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0)
errx(1, "Cannot read matrix size");
fprintf(stderr, "  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, nnz);
fprintf(stderr, "  - Allocating %.1f MByte\n", 1e-6 * (12.0 * nnz));
int *Mi = malloc(nnz * sizeof(*Mi));
int *Mj = malloc(nnz * sizeof(*Mj));
u32 *Mx = malloc(nnz * sizeof(*Mx));
if (Mi == NULL || Mj == NULL || Mx == NULL)
err(1, "Cannot allocate sparse matrix");
double start = wtime();
for (long u = 0; u < nnz; u++) {
int i, j;
u32 x;
if (3 != fscanf(f, "%d %d %d\n", &i, &j, &x))
errx(1, "parse error entry %ld\n", u);
Mi[u] = i - 1;  
Mj[u] = j - 1;
Mx[u] = x % prime;
if ((u & 0xffff) == 0xffff) {
elapsed = wtime() - start + checkpoint_elapsed;
double percent = (100. * u) / nnz;
double rate = ftell(f) / 1048576. / elapsed;
printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
}
}
fclose(f);
printf("\n");
M->nrows = nrows;
M->ncols = ncols;
M->nnz = nnz;
M->i = Mi;
M->j = Mj;
M->x = Mx;
}
void sparse_matrix_vector_product_csr(u32 * y, struct sparsematrix_t const * M_csr, u32 const * x)
{
int nrows = M_csr->nrows;
int const * Mi = M_csr->i;
int const * Mj = M_csr->j;
u32 const * Mx = M_csr->x;
memset(y, 0, nrows * n * sizeof(u32));
#pragma omp parallel for 
for (long k = 0; k < M_csr->np; ++k) {
for(int i = Mi[k] ; i < Mi[k+1]; ++i){
int j = Mj[i];
u64 v = Mx[i];
for (int l = 0; l < n; l++) {
u64 a = y[k * n + l];
u64 b = x[j * n + l];
y[k * n + l] = (a + v * b) % prime;
}
}
}
}
void sparse_matrix_vector_product(u32 * y, struct sparsematrix_t const * M, u32 const * x, bool transpose)
{
long nnz = M->nnz;
int nrows = transpose ? M->ncols : M->nrows;
int const * Mi = transpose ? M->j : M->i;
int const * Mj = transpose ? M->i : M->j;
u32 const * Mx = M->x;
memset(y, 0, nrows * n * sizeof(u32));
#pragma omp parallel for reduction(sum_mod:y[0:nrows*n])
for (long k = 0; k < nnz; k++) {
int i = Mi[k];
int j = Mj[k];
u64 v = Mx[k];
for (int l = 0; l < n; l++) {
u64 a = y[i * n + l];
u64 b = x[j * n + l];
y[i * n + l] = (a + v * b) % prime;
}
}
}
void free_sparse_matrix(struct sparsematrix_t * M)
{
free(M->i);
free(M->j);
free(M->x);
free(M);
}
void matmul_CpAB(u32 * C, u32 const * A, u32 const * B)
{
for (int i = 0; i < n; i++)
for (int k = 0; k < n; k++) {
for (int j = 0; j < n; j++)
{
u64 x = C[i * n + j];
u64 y = A[i * n + k];
u64 z = B[k * n + j];
C[i * n + j] = (x + y * z) % prime;
}
}
}
void matmul_CpAtB(u32 * C, u32 const * A, u32 const * B)
{
for (int i = 0; i < n; i++)
for (int k = 0; k < n; k++) 
for (int j = 0; j < n; j++)
{
u64 x = C[i * n + j];
u64 y = A[k * n + i];
u64 z = B[k * n + j];
C[i * n + j] = (x + y * z) % prime;
}
}
u32 invmod(u32 a, u32 b)
{
long int t = 0;  
long int nt = 1;  
long int r = b;  
long int nr = a % b;
while (nr != 0) {
long int q = r / nr;
long int tmp = nt;  
nt = t - q*nt;  
t = tmp;
tmp = nr;  
nr = r - q*nr;  
r = tmp;
}
if (t < 0)
t += b;
return (u32) t;
}
int semi_inverse(u32 const * M_, u32 * winv, u32 * d)
{
u32 M[n * n];
int npiv = 0;
memcpy(M,M_,n*n*sizeof(u32)); 
memset(d, 0, n * sizeof(u32)); 
for (int j = 0; j < n; j++) {     
int pivot = n;
for (int i = j; i < n; i++)
if (M[i*n + j] != 0) {
pivot = i;
break;
}
if (pivot >= n)
continue;         
d[j] = 1;
npiv += 1;
u64 pinv = invmod(M[pivot*n + j], prime);  
for (int k = 0; k < n; k++) {
u64 x = M[pivot*n + k];
M[pivot*n + k] = (x * pinv) % prime;
}
for (int k = 0; k < n; k++) {   
u32 tmp = M[j*n + k];
M[j*n + k] = M[pivot*n + k];
M[pivot*n + k] = tmp;
}
for (int i = 0; i < n; i++) {  
if (i == j)
continue;
u64 multiplier = M[i*n+j];
for (int k = 0; k < n; k++) {
u64 x = M[i * n + k];
u64 y = M[j * n + k];
M[i * n + k] = (x + (prime - multiplier) * y) % prime;  
}
}
}
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++) {
M[i*n + j] = (d[i] && d[j]) ? M_[i*n + j] : 0;
winv[i*n + j] = ((i == j) && d[i]) ? 1 : 0;
}
npiv = 0;
memset(d, 0, n * sizeof(u32));
for (int j = 0; j < n; j++) { 
int pivot = n;
for (int i = j; i < n; i++)
if (M[i*n + j] != 0) {
pivot = i;
break;
}
if (pivot >= n)
continue;
d[j] = 1;
npiv += 1;
u64 pinv = invmod(M[pivot*n + j], prime);
for (int k = 0; k < n; k++) {
u64 x = M[pivot*n + k];
M[pivot*n + k] = (x * pinv) % prime;
}
for (int k = 0; k < n; k++) {
u32 tmp = M[j*n + k];
M[j*n + k] = M[pivot*n + k];
M[pivot*n + k] = tmp;
}
for (int k = 0; k < n; k++) {
u64 x = winv[pivot * n + k];
winv[pivot * n + k] = (x * pinv) % prime;
}
for (int k = 0; k < n; k++) {
u32 tmp = winv[j * n + k];
winv[j * n + k] = winv[pivot * n + k];
winv[pivot * n + k] = tmp;
}
for (int i = 0; i < n; i++) {
if (i == j)
continue;
u64 multiplier = M[i * n + j];
for (int k = 0; k < n; k++) {
u64 x = M[i * n + k];
u64 y = M[j * n + k];
M[i * n + k] = (x + (prime - multiplier) * y) % prime;
u64 w = winv[i * n + k];
u64 z = winv[j * n + k];
winv[i * n + k] = (w + (prime - multiplier) * z) % prime;  
}
}
}
return npiv;
}
void block_dot_products(u32 * vtAv, u32 * vtAAv, int N, u32 const * Av, u32 const * v)
{
memset(vtAv,  0, n * n * sizeof(u32));
memset(vtAAv, 0, n * n * sizeof(u32));
#pragma omp parallel for reduction(sum_mod:vtAv[0:n*n])
for (int i = 0; i < N; i += n)
matmul_CpAtB(vtAv, &v[i*n], &Av[i*n]);
#pragma omp parallel for reduction(sum_mod:vtAAv[0:n*n])
for (int i = 0; i < N; i += n)
matmul_CpAtB(vtAAv, &Av[i*n], &Av[i*n]);
}
void orthogonalize(u32 * v, u32 * tmp, u32 * p, u32 * d, u32 const * vtAv, const u32 *vtAAv, 
u32 const * winv, int N, u32 const * Av)
{
u32 spliced[n * n];
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++) 
spliced[i * n + j] = d[j] ? vtAAv[i * n + j] : vtAv[i * n + j];
u32 c[n * n];
memset(c, 0, n * n * sizeof(u32));
matmul_CpAB(c, winv, spliced);
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
c[i * n + j] = prime - c[i * n + j];
u32 vtAvd[n * n];
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
vtAvd[i * n + j] = d[j] ? prime - vtAv[i * n + j] : 0;
for (long i = 0; i < N; i++)
for (long j = 0; j < n; j++)
tmp[i * n + j] = d[j] ? Av[i * n + j] : v[i * n + j];
#pragma omp parallel for
for (long i = 0; i < N; i += n) 
matmul_CpAB(&tmp[i*n], &v[i*n], c);
#pragma omp parallel for
for (long i = 0; i < N; i += n)
matmul_CpAB(&tmp[i*n], &p[i*n], vtAvd);
for (long i = 0; i < N; i++)
for (long j = 0; j < n; j++)
p[i * n + j] = d[j] ? 0 : p[i * n + j];
#pragma omp parallel for
for (long i = 0; i < N; i += n) 
matmul_CpAB(&p[i*n], &v[i*n], winv);
}
void verbosity()
{
n_iterations += 1;
elapsed = wtime() - start + checkpoint_elapsed;
if (elapsed - last_print < 1)
return;
last_print = elapsed;
double per_iteration = elapsed / n_iterations;
double estimated_length = expected_iterations * per_iteration;
time_t end = start + estimated_length;
if (!ETA_flag) {
int d = estimated_length / 86400;
estimated_length -= d * 86400;
int h = estimated_length / 3600;
estimated_length -= h * 3600;
int m = estimated_length / 60;
estimated_length -= m * 60;
int s = estimated_length;
printf("    - Expected duration : ");
if (d > 0)
printf("%d j ", d);
if (h > 0)
printf("%d h ", h);
if (m > 0)
printf("%d min ", m);
printf("%d s\n", s);
ETA_flag = true;
}
char ETA[30];
ctime_r(&end, ETA);
ETA[strlen(ETA) - 1] = 0;  
printf("\r    - iteration %d / %d. %.3fs per iteration. ETA: %s", 
n_iterations, expected_iterations, per_iteration, ETA);
fflush(stdout);
}
void correctness_tests(u32 const * vtAv, u32 const * vtAAv, u32 const * winv, u32 const * d)
{
for (int i = 0; i < n; i++) 
for (int j = 0; j < n; j++) {
assert(vtAv[i*n + j] == vtAv[j*n + i]);
assert(vtAAv[i*n + j] == vtAAv[j*n + i]);
assert(winv[i*n + j] == winv[j*n + i]);
assert((winv[i*n + j] == 0) || d[i] || d[j]);
}
u32 vtAvd[n * n];
u32 check[n * n];
for (int i = 0; i < n; i++) 
for (int j = 0; j < n; j++) {
vtAvd[i*n + j] = d[j] ? vtAv[i*n + j] : 0;
check[i*n + j] = 0;
}
matmul_CpAB(check, winv, vtAvd);
for (int i = 0; i < n; i++) 
for (int j = 0; j < n; j++)
if (i == j)
assert(check[j*n + j] == d[i]);
else
assert(check[i*n + j] == 0);
}
void final_check(int nrows, int ncols, u32 const * v, u32 const * vtM)
{
printf("Final check:\n");
bool good = false;
for (long i = 0; i < nrows; i++)
for (long j = 0; j < n; j++)
good |= (v[i*n + j] != 0);
if (good)
printf("  - OK:    v != 0\n");
else
printf("  - KO:    v == 0\n");
good = true;
for (long i = 0; i < ncols; i++)
for (long j = 0; j < n; j++)
good &= (vtM[i*n + j] == 0);
if (good)
printf("  - OK: vt*M == 0\n");
else
printf("  - KO: vt*M != 0\n");                
}
u32 * block_lanczos(struct sparsematrix_t const * M, int n, bool transpose, bool csr)
{
printf("Block Lanczos\n");
int nrows = transpose ? M->ncols : M->nrows;
int ncols = transpose ? M->nrows : M->ncols;
long block_size = nrows * n;
long Npad = ((nrows + n - 1) / n) * n;
long Mpad = ((ncols + n - 1) / n) * n;
long block_size_pad = (Npad > Mpad ? Npad : Mpad) * n;
char human_size[8];
human_format(human_size, 4 * sizeof(int) * block_size_pad);
printf("  - Extra storage needed: %sB\n", human_size);
u32 *v   = calloc(block_size_pad, sizeof(*v));
u32 *tmp = calloc(block_size_pad, sizeof(*tmp));
u32 *Av  = calloc(block_size_pad, sizeof(*Av));
u32 *p   = calloc(block_size_pad, sizeof(*p));
if (v == NULL || tmp == NULL || Av == NULL || p == NULL)
errx(1, "impossible d'allouer les blocs de vecteur");
struct sparsematrix_t *M_csr1;
struct sparsematrix_t *Mt_csr1;
if (csr) {
struct sparsematrix_t *M_csr = coo_to_csr(M);
struct sparsematrix_t *tmp1 = (struct sparsematrix_t*) malloc(sizeof(struct sparsematrix_t));
memcpy(tmp1,M, sizeof(struct sparsematrix_t));
tmp1->i = transpose ? M->j : M->i;
tmp1->j = transpose ? M->i : M->j;
tmp1->nrows = transpose ? M->ncols : M->nrows;
tmp1->ncols = transpose ? M->nrows : M->ncols;
struct sparsematrix_t *Mt_csr = coo_to_csr(tmp1);
free(tmp1);
M_csr1 = transpose ? Mt_csr : M_csr;
Mt_csr1 = transpose ? M_csr : Mt_csr;
}
expected_iterations = 1 + ncols / n;
char human_its[8];
human_format(human_its, expected_iterations);
printf("  - Expecting %s iterations\n", human_its);
if (checkpoint)
get_checkpoint(v,p,block_size_pad);
else
for (long i = 0; i < block_size; i++)
v[i] = random64() % prime;
printf("  - Main loop\n");
start = wtime();
bool stop = false;
int cpt = 0;
while (true) {
if (stop_after > 0 && n_iterations == stop_after)
break;
if (csr) {
sparse_matrix_vector_product_csr(tmp, Mt_csr1, v);
sparse_matrix_vector_product_csr(Av, M_csr1, tmp);
} else {
sparse_matrix_vector_product(tmp, M, v, !transpose);
sparse_matrix_vector_product(Av, M, tmp, transpose);
}
u32 vtAv [n * n];
u32 vtAAv[n * n];
block_dot_products(vtAv, vtAAv, nrows, Av, v);
u32 winv[n * n];
u32 d[n];
stop = (semi_inverse(vtAv, winv, d) == 0);
correctness_tests(vtAv, vtAAv, winv, d);
if (stop)
break;
orthogonalize(v, tmp, p, d, vtAv, vtAAv, winv, nrows, Av);
memcpy(v,tmp,block_size*sizeof(u32));
if(++cpt == (expected_iterations/10)){ 
set_checkpoint(v,p,block_size_pad);
cpt = 0;
}
verbosity();
}
printf("\n");
if (stop_after < 0)
final_check(nrows, ncols, v, tmp);
printf("  - Terminated in %.1fs after %d iterations\n", wtime() - start, n_iterations);
set_checkpoint(v,p,block_size_pad);
free(tmp);
free(Av);
free(p);
if (csr) {
free_sparse_matrix(M_csr1);
free_sparse_matrix(Mt_csr1);
}
return v;
}
int main(int argc, char ** argv)
{
process_command_line_options(argc, argv);
struct sparsematrix_t M;
sparsematrix_mm_load(&M, matrix_filename);
u32 *kernel = block_lanczos(&M, n, right_kernel, true);
if (kernel_filename)
save_vector_block(kernel_filename, right_kernel ? M.ncols : M.nrows, n, kernel);
else
printf("Not saving result (no --output given)\n");
free(kernel);
exit(EXIT_SUCCESS);
}