#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <time.h>
#include <complex.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include "timing.h"
#include "calc_spec.h"
#include "dyn_array.h"
#include "scttr_io.h"
#include "sci_const.h"
#include "std_num_ops.h"
#include "std_char_ops.h"
#include "spectrum_s.h"
#include "spectrum.h"
#include "inp_node_s.h"
#include "cache_opt.h"
#include "cpu_opt.h"
#include "glob_time.h"
double para_t;
int
intf_0( struct inp_node *inp, struct spectrum *spec, struct metadata *md)
{
int x, y;
int j, k, l; 
int nth = 0;
env2int("OMP_NUM_THREADS", &nth);
int x_in, y_in; 
int j_in, k_in;
int lx_i, ly_i; 
lx_i = ly_i = - 1;
int n_st = spec -> n_st;
int rchunk,rem;
int * rchunks = malloc(nth * sizeof(int));
memset(rchunks, 0, nth*sizeof(nth));
char *ltime = malloc(20);
double tmp_int; 
double omega_x, omega_y;
double omega_x_in, omega_y_in;
double lx_t, ly_t;
double eu_lx, eu_ly;
double emin_x = spec -> emin_x;
double emin_y = spec -> emin_y;
eu_lx = -fabs(emin_x * 2);
eu_ly = -fabs(emin_y * 2);
double ediff_x, ediff_y;
double lx_mfac, lx_afac;
double ly_mfac, ly_afac;
double ** sm_th0; 
double ** tr = spec -> trs_red;
double de_x = md -> res[0] / AUTOEV;
double de_y = md -> res[1] / AUTOEV;
double wt; 
spec -> prsz = cache_cfg -> l_sz / sizeof(double);
spec -> npr = (int)floorf(spec -> n_ely / spec -> prsz) + (spec -> n_ely % spec -> prsz  > 1 ? 1 : 0);
spec -> npr_tot = spec -> n_elx * spec -> npr;
if ( spec-> npr * spec->prsz < spec -> n_ely) {
fprintf(stderr, "calc_spec.c, function intf_0: matrix incorrectly partitioned (spec-> npr * spec->prsz < spec -> n_ely)\n");
printf( "program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
rchunk = get_row_chunk(4, sizeof(double), cache_cfg);
goto threading;
threading:
memset(rchunks, 0, nth*sizeof(nth));
if (rchunk > n_st/nth) {
rchunk = n_st/nth;
}
else {
if (n_st/rchunk > nth) {
rchunk = n_st / nth;
}
}
rem = n_st % rchunk;
if (rem > 0) {
if (rem % nth > 0) {
for (j = 0; rem > 0; rem--, j++) {
rchunks[j] += 1;
}
}
else {
for (j = 0; j < nth; j++) {
rchunks[j] += rem / nth;
}
}
}
k = 0;
for (j = 0; j < nth; j++) {
rchunks[j] += rchunk;
k += rchunks[j];
}
j = k = l = 0;
while (l != nth) {
if (j >= n_st){
rchunks[l] = k;
l++;
k = 0;
break;
}
else if (k >= rchunks[l]) {
rchunks[l] = k;
l++;
k = 0;
}
for (k++; (++j < n_st) && ((int)tr[j][1] == 0); k++){};
}
rem = n_st - j - 1;
if (rem > 0) {
rchunks[l-1] += rem;
}
if (l < nth) {
printf("========Does not scale beyond %d (nth = %d)\n", l ,nth );
nth = l;
goto threading;
}
if((spec -> s_mat = malloc(spec -> npr_tot * sizeof(double *))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"spec -> s_mat\"\n");
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
for (j = 0; j < spec -> npr_tot; j++) {
if((spec -> s_mat[j] = malloc(spec -> prsz * sizeof(double))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"spec -> s_mat[%d]\"\n",j);
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
memset(spec -> s_mat[j], 0, spec -> prsz * sizeof(double));
}
wt = omp_get_wtime();
#pragma omp parallel num_threads(nth)
{
int j; 
int k; 
int l; 
int x, y; 
int gx_i, gy_i; 
gx_i = gy_i = -1;
int ith = omp_get_thread_num();
int l_st = 0;
for (j = 0; j != ith; j++) {
l_st += rchunks[j];
}
int l_fn = l_st + rchunks[ith];
double ediff_x, tmom_gi; 
double ediff_y, tmom_if; 
double de_gi, de_if; 
double bw; 
double omega_x, omega_y;
double gx_t, gy_t; 
double eu_gx, eu_gy; 
eu_gx = -fabs(emin_x * 2);
eu_gy = -fabs(emin_y * 2);
double gy_stddev, gx_stddev;
double gy_var, gx_var;
double gy_mfac, gx_mfac;
double complex tmp;  
double ** sm;
if((sm = malloc(spec -> npr_tot * sizeof(double *))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"sm\"\n");
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
for (j = 0; j < spec -> npr_tot; j++) {
if((sm[j] = malloc(spec -> prsz * sizeof(double))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"sm[%d]\"\n",j);
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
memset(sm[j], 0, spec -> prsz * sizeof(double));
}
if (ith == 0) {
sm_th0 = sm;
}
omega_x = emin_x;
for (j = 0, x = 0, y = 0; x < spec -> n_elx; j++) {
for (k = 0; (k < spec -> prsz) && (x < spec -> n_elx); k++, y++) {
omega_y = emin_y + (y * de_y);
if (omega_x > eu_gx) {
gx_i += 2;
gx_t = md -> gx[gx_i] / AUTOEV;
gx_stddev = gx_t / (2 * sqrt(2 * log(2)));
gx_var = 2.0 * powerl(gx_stddev, 2);
gx_mfac = de_x / gx_stddev * sqrt(2.0 * PI);
eu_gx = (md -> gx)[gx_i+1] / AUTOEV;
}
if (omega_y > eu_gy) {
gy_i += 2;
gy_t = md -> gy[gy_i] / AUTOEV;
gy_stddev = gy_t / (2 * sqrt(2 * log(2)));
gy_var = 2.0 * powerl(gy_stddev, 2);
gy_mfac = de_y / gy_stddev * sqrt(2.0 * PI);
eu_gy = (md -> gy)[gy_i+1] / AUTOEV;
}
for (l = l_st; l < l_fn;) {
de_if = tr[l][2];
tmom_if = tr[l][3];
tmp = 0 + 0*I;
while((int)tr[++l][1] == 0) { 
bw = tr[l][0];
de_gi = tr[l][2];
tmom_gi = tr[l][3];
ediff_x = omega_x - de_gi;
ediff_y = omega_y - de_gi - de_if;
tmp += tmom_gi * tmom_if * bw / (-de_gi + omega_x
- (gx_t / 2)*I);
tmp *= (exp(-(powerl(ediff_x, 2)) / gx_var) * gx_mfac)
* (exp(-(powerl(ediff_y, 2)) / gy_var) * gy_mfac);
}
tmp = fabsc(tmp);
tmp *= tmp;
sm[j][k] += creal(tmp);
}
if (y == spec -> n_ely-1) {
x++;
omega_x = emin_x + (x * de_x);
y = 0;
}
}
}
#pragma omp barrier
if (ith == 0) {
printf("\n      summing up the thread-local spectrum layers.. (%s)",get_loctime(ltime));
fflush(stdout);
}
#pragma omp critical
{
if (ith != 0) {
for (j = 0, l = 0; j < spec -> npr_tot; j++) {
for (k = 0; k < spec -> prsz; k++, l++) {
sm_th0[j][k] += sm[j][k];
}
}
}
} 
}
printf(" done (%s).", get_loctime(ltime));
fflush(stdout);
para_t = omp_get_wtime() - wt;
printf("\n      applying lorentzian boadening.. (%s)",get_loctime(ltime));
fflush(stdout);
for (j = 0, x = 0, y = 0; x < spec -> n_elx; j++) {
for (k = 0; (k < spec -> prsz) && (x < spec -> n_elx); k++, y++) {
omega_x = emin_x + (x * de_x);
omega_y = emin_y + (y * de_y);
if (omega_x > eu_lx) {
lx_i += 2;
lx_t = md -> lx[lx_i] / AUTOEV;
lx_mfac = 0.5 * lx_t / PI;
lx_afac = (0.25 * lx_t * lx_t);
eu_lx = (md -> lx)[lx_i+1] / AUTOEV;
}
if (omega_y > eu_ly) {
ly_i += 2;
ly_t = md -> ly[ly_i] / AUTOEV;
ly_mfac = 0.5 * ly_t / PI;
ly_afac = (0.25 * ly_t * ly_t);
eu_ly = (md -> ly)[ly_i+1] / AUTOEV;
}
tmp_int = 0;
for (j_in = 0, x_in = 0, y_in = 0; x_in < spec -> n_elx; j_in++) {
for (k_in = 0; (k_in < spec -> prsz) && (x_in < spec -> n_elx)
;k_in++, y_in++) {
omega_x_in = emin_x + (x_in * de_x);
omega_y_in = emin_y + (y_in * de_y);
ediff_x = omega_x - omega_x_in;
ediff_y = omega_y - omega_y_in;
tmp_int += sm_th0[j_in][k_in]
* (lx_mfac / ((ediff_x * ediff_x) + lx_afac))
* (ly_mfac / ((ediff_y * ediff_y) + ly_afac));
if (y_in == spec -> n_ely-1) {
x_in++;
y_in = 0;
}
}
}
spec -> s_mat[j][k] += tmp_int;
if (y == spec -> n_ely-1) {
x++;
y = 0;
}
}
}
printf(" done (%s).", get_loctime(ltime));
free(ltime);
free(rchunks);
return 0;
}
int
intf_0_old( struct inp_node *inp, struct spectrum *spec, struct metadata *md)
{
int x, y;
int j, k, l; 
int nth = 0;
env2int("OMP_NUM_THREADS", &nth);
int x_in, y_in; 
int j_in, k_in;
int lx_i, ly_i; 
lx_i = ly_i = - 1;
int n_st = spec -> n_st;
int rchunk,rem;
int * rchunks = malloc(nth * sizeof(int));
memset(rchunks, 0, nth*sizeof(nth));
char *ltime = malloc(20);
double tmp_int; 
double omega_x, omega_y;
double omega_x_in, omega_y_in;
double lx_t, ly_t;
double eu_lx, eu_ly;
double emin_x = spec -> emin_x;
double emin_y = spec -> emin_y;
eu_lx = -fabs(emin_x * 2);
eu_ly = -fabs(emin_y * 2);
double ediff_x, ediff_y;
double lx_mfac, lx_afac;
double ly_mfac, ly_afac;
double ** sm_th0; 
double ** tr = spec -> trs_red;
double de_x = md -> res[0] / AUTOEV;
double de_y = md -> res[1] / AUTOEV;
double wt; 
spec -> prsz = cache_cfg -> l_sz / sizeof(double);
spec -> npr = (int)floorf(spec -> n_ely / spec -> prsz) + (spec -> n_ely % spec -> prsz  > 1 ? 1 : 0);
spec -> npr_tot = spec -> n_elx * spec -> npr;
if (spec-> npr * spec->prsz < spec -> n_ely) {
fprintf(stderr, "calc_spec.c, function intf_0_old: matrix incorrectly partitioned (spec-> npr * spec->prsz (%d) < spec -> n_ely (%d), OMP_NUM_THREADS = %d)\n",spec-> npr * spec->prsz, spec -> n_ely, nth);
printf( "program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
else {
printf("\n\n MATRIX CORRECT \n\n");
}
rchunk = get_row_chunk(4, sizeof(double), cache_cfg);
goto threading;
threading:
memset(rchunks, 0, nth*sizeof(nth));
if (rchunk > n_st/nth) {
rchunk = n_st/nth;
}
else {
if (n_st/rchunk > nth) {
rchunk = n_st / nth;
}
}
rem = n_st % rchunk;
if (rem > 0) {
if (rem % nth > 0) {
for (j = 0; rem > 0; rem--, j++) {
rchunks[j] += 1;
}
}
else {
for (j = 0; j < nth; j++) {
rchunks[j] += rem / nth;
}
}
}
k = 0;
for (j = 0; j < nth; j++) {
rchunks[j] += rchunk;
k += rchunks[j];
}
j = k = l = 0;
while (l != nth) {
if (j >= n_st){
rchunks[l] = k;
l++;
k = 0;
break;
}
else if (k >= rchunks[l]) {
rchunks[l] = k;
l++;
k = 0;
}
for (k++; (++j < n_st) && ((int)tr[j][1] == 0); k++){};
}
rem = n_st - j - 1;
if (rem > 0) {
rchunks[l-1] += rem;
}
if (l < nth) {
printf("\n    - the calculation does not scale beyond %d threads (current nth = %d). Reducing the number of threads used to %d.\n", l , nth, l );
fflush(stdout);
nth = l;
goto threading;
}
if((spec -> s_mat = malloc(spec -> npr_tot * sizeof(double *))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"spec -> s_mat\"\n");
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
for (j = 0; j < spec -> npr_tot; j++) {
if((spec -> s_mat[j] = malloc(spec -> prsz * sizeof(double))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"spec -> s_mat[%d]\"\n",j);
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
memset(spec -> s_mat[j], 0, spec -> prsz * sizeof(double));
}
wt = omp_get_wtime();
#pragma omp parallel num_threads(nth)
{
int j; 
int k; 
int l; 
int x, y; 
int gx_i, gy_i; 
gx_i = gy_i = -1;
int ith = omp_get_thread_num();
int l_st = 0;
for (j = 0; j != ith; j++) {
l_st += rchunks[j];
}
int l_fn = l_st + rchunks[ith];
double ediff_x, tmom_gi; 
double ediff_y, tmom_if; 
double de_gi, de_if; 
double bw; 
double omega_x, omega_y;
double gx_t, gy_t; 
double eu_gx, eu_gy; 
eu_gx = -fabs(emin_x * 2);
eu_gy = -fabs(emin_y * 2);
double gy_stddev, gx_stddev;
double gy_var, gx_var;
double gy_mfac, gx_mfac;
double complex tmp;  
double ** sm;
if((sm = malloc(spec -> npr_tot * sizeof(double *))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"sm\"\n");
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
for (j = 0; j < spec -> npr_tot; j++) {
if((sm[j] = malloc(spec -> prsz * sizeof(double))) == NULL ) {
fprintf(stderr, "calc_spec.c, function intf_0: failed to allocate memory for \"sm[%d]\"\n",j);
printf("program terminating due to the previous error.\n");
exit(EXIT_FAILURE);
}
memset(sm[j], 0, spec -> prsz * sizeof(double));
}
if (ith == 0) {
sm_th0 = sm;
}
omega_x = emin_x;
for (j = 0, x = 0, y = 0; x < spec -> n_elx; j++) {
for (k = 0; (k < spec -> prsz) && (x < spec -> n_elx); k++, y++) {
omega_y = emin_y + (y * de_y);
if (omega_x > eu_gx) {
gx_i += 2;
gx_t = md -> gx[gx_i] / AUTOEV;
gx_stddev = gx_t / (2 * sqrt(2 * log(2)));
gx_var = 2.0 * powerl(gx_stddev, 2);
gx_mfac = de_x / gx_stddev * sqrt(2.0 * PI);
eu_gx = (md -> gx)[gx_i+1] / AUTOEV;
}
if (omega_y > eu_gy) {
gy_i += 2;
gy_t = md -> gy[gy_i] / AUTOEV;
gy_stddev = gy_t / (2 * sqrt(2 * log(2)));
gy_var = 2.0 * powerl(gy_stddev, 2);
gy_mfac = de_y / gy_stddev * sqrt(2.0 * PI);
eu_gy = (md -> gy)[gy_i+1] / AUTOEV;
}
for (l = l_st; l < l_fn;) {
de_if = tr[l][2];
tmom_if = tr[l][3];
tmp = 0 + 0*I;
while((int)tr[++l][1] == 0) { 
bw = tr[l][0];
de_gi = tr[l][2];
tmom_gi = tr[l][3];
ediff_x = omega_x - de_gi;
ediff_y = omega_y - de_gi - de_if;
tmp += tmom_gi * tmom_if * bw / (ediff_x
- (gx_t / 2)*I);
tmp *= (exp(-(powerl(ediff_x, 2)) / gx_var) * gx_mfac)
* (exp(-(powerl(ediff_y, 2)) / gy_var) * gy_mfac);
tmp = fabsc(tmp);
sm[j][k] += creal(tmp);
}
}
if (y == spec -> n_ely-1) {
x++;
omega_x = emin_x + (x * de_x);
y = 0;
}
}
}
#pragma omp barrier
#pragma omp critical
{
if (ith != 0) {
for (j = 0, l = 0; j < spec -> npr_tot; j++) {
for (k = 0; k < spec -> prsz; k++, l++) {
sm_th0[j][k] += sm[j][k];
}
}
}
} 
}
fflush(stdout);
para_t = omp_get_wtime() - wt;
if (inp -> md -> lorz) {
for (j = 0, x = 0, y = 0; x < spec -> n_elx; j++) {
for (k = 0; (k < spec -> prsz) && (x < spec -> n_elx); k++, y++) {
omega_x = emin_x + (x * de_x);
omega_y = emin_y + (y * de_y);
if (omega_x > eu_lx) {
lx_i += 2;
lx_t = md -> lx[lx_i] / AUTOEV;
lx_mfac = 0.5 * lx_t / PI;
lx_afac = (0.25 * lx_t * lx_t);
eu_lx = (md -> lx)[lx_i+1] / AUTOEV;
}
if (omega_y > eu_ly) {
ly_i += 2;
ly_t = md -> ly[ly_i] / AUTOEV;
ly_mfac = 0.5 * ly_t / PI;
ly_afac = (0.25 * ly_t * ly_t);
eu_ly = (md -> ly)[ly_i+1] / AUTOEV;
}
tmp_int = 0;
for (j_in = 0, x_in = 0, y_in = 0; x_in < spec -> n_elx; j_in++) {
for (k_in = 0; (k_in < spec -> prsz) && (x_in < spec -> n_elx)
;k_in++, y_in++) {
omega_x_in = emin_x + (x_in * de_x);
omega_y_in = emin_y + (y_in * de_y);
ediff_x = omega_x - omega_x_in;
ediff_y = omega_y - omega_y_in;
tmp_int += sm_th0[j_in][k_in]
* (lx_mfac / ((ediff_x * ediff_x) + lx_afac))
* (ly_mfac / ((ediff_y * ediff_y) + ly_afac));
if (y_in == spec -> n_ely-1) {
x_in++;
y_in = 0;
}
}
}
spec -> s_mat[j][k] += tmp_int;
if (y == spec -> n_ely-1) {
x++;
y = 0;
}
}
}
}
else {
spec -> s_mat = sm_th0;
}
free(ltime);
free(rchunks);
return 0;
}
int
calc_spec (struct inp_node *inp, int spec_idx)
{
char *ltime = malloc(20);
struct metadata *md = inp -> md;
struct spectrum *spec = get_spec(inp, spec_idx);
int j, k;
double emin_x = spec -> emin_x;
double emax_x = spec -> emax_x;
double emin_y = spec -> emin_y;
double emax_y = spec -> emax_y;
spec -> n_elx = (int)((emax_x - emin_x) / (md -> res[0] / AUTOEV));
spec -> n_ely = (int)((emax_y - emin_y) / (md -> res[1] / AUTOEV));
if((spec -> omega_x = malloc(spec -> n_elx * sizeof(double *))) == NULL ) {
fprintf(stderr, "calc_spec.c, function calc_spec: failed to allocate memory for \"spec -> omega_x\"\n");
printf("program terminating due to the previous error.\n");
exit(1);
}
if((spec -> omega_y = malloc(spec -> n_elx * sizeof(double *))) == NULL ) {
fprintf(stderr, "calc_spec.c, function calc_spec: failed to allocate memory for \"spec -> omega_y\"\n");
printf("program terminating due to the previous error.\n");
exit(1);
}
if((spec -> s_mat = malloc(spec -> n_elx * sizeof(double *))) == NULL ) {
fprintf(stderr, "calc_spec.c, function calc_spec: failed to allocate memory for \"s_mat\"\n");
printf("program terminating due to the previous error.\n");
exit(1);
}
for (j=0; j<spec -> n_elx; j++) {
if((spec -> s_mat[j] = malloc(spec -> n_ely * sizeof(double))) == NULL ) {
fprintf(stderr, "calc_spec.c, function calc_spec: failed to allocate memory for \"spec -> s_mat[%d]\"\n"
, j);
printf("program terminating due to the previous error.\n");
exit(1);
}
if((spec -> omega_x[j] = malloc(spec -> n_ely * sizeof(double))) == NULL ) {
fprintf(stderr, "calc_spec.c, function calc_spec: failed to allocate memory for \"spec -> omega_x[%d]\"\n"
, j);
printf("program terminating due to the previous error.\n");
exit(1);
}
if((spec -> omega_y[j] = malloc(spec -> n_ely * sizeof(double))) == NULL ) {
fprintf(stderr, "calc_spec.c, function calc_spec: failed to allocate memory for \"spec -> omega_y[%d]\"\n"
, j);
printf("program terminating due to the previous error.\n");
exit(1);
}
}
printf("  - calculating the RIXS map, using");
switch(inp -> md -> intf_mode)
{
case 1:
printf(" a constructive interference model (%s) ..", get_loctime(ltime));
fflush(stdout);
intf_0(inp, spec, md);
break;
default:
printf(" no interference model (%s) ..", get_loctime(ltime));
fflush(stdout);
intf_0_old(inp, spec, md);
}
printf("    done (%s).\n", get_loctime(ltime));
for (j = 0; j < spec -> npr_tot; j++) {
for (k = 0; k < spec -> prsz; k++) {
if (spec -> s_mat[j][k] > spec -> sfac) {
spec -> sfac = spec -> s_mat[j][k];
}
}
}
free(ltime);
return 0;
}
