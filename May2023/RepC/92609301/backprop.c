#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <omp.h>
#include "backprop.h"
#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))
#define fastcopy(to,from,len)\
{\
register char *_to,*_from;\
register int _i,_l;\
_to = (char *)(to);\
_from = (char *)(from);\
_l = (len);\
for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}
void bpnn_initialize(int seed)
{
printf("Random number generator seed: %d\n", seed);
srandom(seed);
}
double drnd()
{
return ((double) random() / (double) BIGRND);
}
double dpn1()
{
return ((drnd() * 2.0) - 1.0);
}
double squash(double x)
{
return (1.0 / (1.0 + exp(-x)));
}
double *alloc_1d_dbl(int n)
{
double *new;
new = (double *) malloc ((unsigned) (n * sizeof (double)));
if (new == NULL) {
printf("ALLOC_1D_DBL: Couldn't allocate array of doubles\n");
return (NULL);
}
return (new);
}
double **alloc_2d_dbl(int m,int n)
{
int i;
double **new;
new = (double **) malloc ((unsigned) (m * sizeof (double *)));
if (new == NULL) {
printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
return (NULL);
}
#pragma omp parallel for
for (i = 0; i < m; i++) {
new[i] = alloc_1d_dbl(n);
}
return (new);
}
void bpnn_randomize_weights(double **w,int m,int n)
{
int i, j;
#pragma omp parallel for
for (i = 0; i <= m; i++) {
for (j = 0; j <= n; j++) {
w[i][j] = dpn1();
}
}
}
void bpnn_zero_weights(double **w,int m,int n)
{
int i, j;
#pragma omp parallel for
for (i = 0; i <= m; i++) {
for (j = 0; j <= n; j++) {
w[i][j] = 0.0;
}
}
}
BPNN *bpnn_internal_create(int n_in,int n_hidden,int n_out)
{
BPNN *newnet;
newnet = (BPNN *) malloc (sizeof (BPNN));
if (newnet == NULL) {
printf("BPNN_CREATE: Couldn't allocate neural network\n");
return (NULL);
}
newnet->input_n = n_in;
newnet->hidden_n = n_hidden;
newnet->output_n = n_out;
newnet->input_units = alloc_1d_dbl(n_in + 1);
newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
newnet->output_units = alloc_1d_dbl(n_out + 1);
newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
newnet->output_delta = alloc_1d_dbl(n_out + 1);
newnet->target = alloc_1d_dbl(n_out + 1);
newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
return (newnet);
}
void bpnn_free(BPNN *net)
{
int n1, n2, i;
n1 = net->input_n;
n2 = net->hidden_n;
free((char *) net->input_units);
free((char *) net->hidden_units);
free((char *) net->output_units);
free((char *) net->hidden_delta);
free((char *) net->output_delta);
free((char *) net->target);
#pragma omp parallel for
for (i = 0; i <= n1; i++) {
free((char *) net->input_weights[i]);
free((char *) net->input_prev_weights[i]);
}
free((char *) net->input_weights);
free((char *) net->input_prev_weights);
#pragma omp parallel for
for (i = 0; i <= n2; i++) {
free((char *) net->hidden_weights[i]);
free((char *) net->hidden_prev_weights[i]);
}
free((char *) net->hidden_weights);
free((char *) net->hidden_prev_weights);
free((char *) net);
}
BPNN *bpnn_create(int n_in,int n_hidden,int n_out)
{
BPNN *newnet;
newnet = bpnn_internal_create(n_in, n_hidden, n_out);
#ifdef INITZERO
bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
return (newnet);
}
void bpnn_layerforward(double *l1,double *l2,double **conn,int n1,int n2)
{
double sum;
int j, k;
l1[0] = 1.0;
for (j = 1; j <= n2; j++)
{
sum = 0.0;
for (k = 0; k <= n1; k++)
{
sum += conn[k][j] * l1[k];  
}
l2[j] = squash(sum);
}
}
void bpnn_output_error(double *delta,double *target,double *output, int nj,double *err)
{
int j;
double o, t, errsum;
errsum = 0.0;
for (j = 1; j <= nj; j++)
{
o = output[j];
t = target[j];
delta[j] = o * (1.0 - o) * (t - o);
errsum += ABS(delta[j]);
}
*err = errsum;
}
void bpnn_hidden_error(double *delta_h,int nh,double *delta_o,int no,double **who,double *hidden,double *err)
{
int j, k;
double h, sum, errsum;
errsum = 0.0;
for (j = 1; j <= nh; j++)
{
h = hidden[j];
sum = 0.0;
for (k = 1; k <= no; k++)
{
sum += delta_o[k] * who[j][k];
}
delta_h[j] = h * (1.0 - h) * sum;
errsum += ABS(delta_h[j]);
}
*err = errsum;
}
void bpnn_adjust_weights(double *delta,int ndelta,double *ly,int nly,double **w,double **oldw,double learning_rate,double momentum)
{
double new_dw;
int k, j;
ly[0] = 1.0;
for (j = 1; j <= ndelta; j++) 
{
for (k = 0; k <= nly; k++)  
{
new_dw = ((learning_rate * delta[j] * ly[k]) + (momentum * oldw[k][j]));
w[k][j] += new_dw;
oldw[k][j] = new_dw;  
}
}
}
void bpnn_feedforward(BPNN *net)
{
int in, hid, out;
in = net->input_n;
hid = net->hidden_n;
out = net->output_n;
bpnn_layerforward(net->input_units, net->hidden_units,
net->input_weights, in, hid);
bpnn_layerforward(net->hidden_units, net->output_units,
net->hidden_weights, hid, out);
}
void bpnn_train(BPNN *net,double learning_rate,double momentum,double *erro,double *errh)
{
int in, hid, out;
double out_err, hid_err;
in = net->input_n;
hid = net->hidden_n;
out = net->output_n;
bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
*erro = out_err;
*errh = hid_err;
bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights, learning_rate, momentum);
bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights, learning_rate, momentum);
}
void bpnn_save(BPNN *net, char *filename)
{
int fd, n1, n2, n3, i, j, memcnt;
double dvalue, **w;
char *mem;
if ((fd = creat(filename, 0644)) == -1) {
printf("BPNN_SAVE: Cannot create '%s'\n", filename);
return;
}
n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
fflush(stdout);
write(fd, (char *) &n1, sizeof(int));
write(fd, (char *) &n2, sizeof(int));
write(fd, (char *) &n3, sizeof(int));
memcnt = 0;
w = net->input_weights;
mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
for (i = 0; i <= n1; i++) {
for (j = 0; j <= n2; j++) {
dvalue = w[i][j];
fastcopy(&mem[memcnt], &dvalue, sizeof(double));
memcnt += sizeof(double);
}
}
write(fd, mem, (n1+1) * (n2+1) * sizeof(double));
free(mem);
memcnt = 0;
w = net->hidden_weights;
mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
for (i = 0; i <= n2; i++) {
for (j = 0; j <= n3; j++) {
dvalue = w[i][j];
fastcopy(&mem[memcnt], &dvalue, sizeof(double));
memcnt += sizeof(double);
}
}
write(fd, mem, (n2+1) * (n3+1) * sizeof(double));
free(mem);
close(fd);
return;
}
BPNN *bpnn_read(char *filename)
{
char *mem;
BPNN *new;
int fd, n1, n2, n3, i, j, memcnt;
if ((fd = open(filename, 0, 0644)) == -1) {
return (NULL);
}
printf("Reading '%s'\n", filename);  fflush(stdout);
read(fd, (char *) &n1, sizeof(int));
read(fd, (char *) &n2, sizeof(int));
read(fd, (char *) &n3, sizeof(int));
new = bpnn_internal_create(n1, n2, n3);
printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
printf("Reading input weights...");  fflush(stdout);
memcnt = 0;
mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
read(fd, mem, (n1+1) * (n2+1) * sizeof(double));
for (i = 0; i <= n1; i++) {
for (j = 0; j <= n2; j++) {
fastcopy(&(new->input_weights[i][j]), &mem[memcnt], sizeof(double));
memcnt += sizeof(double);
}
}
free(mem);
printf("Done\nReading hidden weights...");  fflush(stdout);
memcnt = 0;
mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
read(fd, mem, (n2+1) * (n3+1) * sizeof(double));
for (i = 0; i <= n2; i++) {
for (j = 0; j <= n3; j++) {
fastcopy(&(new->hidden_weights[i][j]), &mem[memcnt], sizeof(double));
memcnt += sizeof(double);
}
}
free(mem);
close(fd);
printf("Done\n");  fflush(stdout);
bpnn_zero_weights(new->input_prev_weights, n1, n2);
bpnn_zero_weights(new->hidden_prev_weights, n2, n3);
return (new);
}
