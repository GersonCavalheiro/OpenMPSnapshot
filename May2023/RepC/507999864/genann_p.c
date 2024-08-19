#include "genann.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#ifndef _OPENMP
printf("openMP is not activated!\n");
return 0;
#endif 
#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif
#define LOOKUP_SIZE 4096
#define NUM_THREADS 4
double genann_act_hidden_indirect(const struct genann *ann, double a) {
return ann->activation_hidden(ann, a);
}
double genann_act_output_indirect(const struct genann *ann, double a) {
return ann->activation_output(ann, a);
}
const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;
double interval;
double lookup[LOOKUP_SIZE];
#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) 
#endif
double genann_act_sigmoid(const genann *ann unused, double a) {
if (a < -45.0) return 0;
if (a > 45.0) return 1;
return 1.0 / (1 + exp(-a));
}
void genann_init_sigmoid_lookup(const genann *ann) {
const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
int i;
interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
for (i = 0; i < LOOKUP_SIZE; ++i) {
lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
}
}
double genann_act_sigmoid_cached(const genann *ann unused, double a) {
assert(!isnan(a));
if (a < sigmoid_dom_min) return lookup[0];
if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];
size_t j = (size_t)((a - sigmoid_dom_min)*interval + 0.5);
if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];
return lookup[j];
}
double genann_act_linear(const struct genann *ann unused, double a) {
return a;
}
double genann_act_threshold(const struct genann *ann unused, double a) {
return a > 0;
}
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
if (hidden_layers < 0) return 0;
if (inputs < 1) return 0;
if (outputs < 1) return 0;
if (hidden_layers > 0 && hidden < 1) return 0;
const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
const int total_weights = (hidden_weights + output_weights);
const int total_neurons = (inputs + hidden * hidden_layers + outputs);
const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
genann *ret = malloc(size);
if (!ret) return 0;
ret->inputs = inputs;
ret->hidden_layers = hidden_layers;
ret->hidden = hidden;
ret->outputs = outputs;
ret->total_weights = total_weights;
ret->total_neurons = total_neurons;
ret->weight = (double*)((char*)ret + sizeof(genann));
ret->output = ret->weight + ret->total_weights;
ret->delta = ret->output + ret->total_neurons;
genann_randomize(ret);
ret->activation_hidden = genann_act_sigmoid_cached;
ret->activation_output = genann_act_sigmoid_cached;
genann_init_sigmoid_lookup(ret);
return ret;
}
genann *genann_read(FILE *in) {
int inputs, hidden_layers, hidden, outputs;
int rc;
errno = 0;
rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
if (rc < 4 || errno != 0) {
perror("fscanf");
return NULL;
}
genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);
int i;
for (i = 0; i < ann->total_weights; ++i) {
errno = 0;
rc = fscanf(in, " %le", ann->weight + i);
if (rc < 1 || errno != 0) {
perror("fscanf");
genann_free(ann);
return NULL;
}
}
return ann;
}
genann *genann_copy(genann const *ann) {
const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
genann *ret = malloc(size);
if (!ret) return 0;
memcpy(ret, ann, size);
ret->weight = (double*)((char*)ret + sizeof(genann));
ret->output = ret->weight + ret->total_weights;
ret->delta = ret->output + ret->total_neurons;
return ret;
}
void genann_randomize(genann *ann) {
int i;
for (i = 0; i < ann->total_weights; ++i) {
double r = GENANN_RANDOM();
ann->weight[i] = r - 0.5;
}
}
void genann_free(genann *ann) {
free(ann);
}
double const *genann_run(genann const *ann, double const *inputs) {
omp_set_num_threads(NUM_THREADS);
double const *w = ann->weight;
double *o = ann->output + ann->inputs;
double const *i = ann->output;
memcpy(ann->output, inputs, sizeof(double) * ann->inputs);
int h, j, k;
double *mw;
double sum;
if (!ann->hidden_layers) {
double *ret = o;
#pragma omp parallel for private(k, j, mw, sum) firstprivate(o, w)
for (j = 0; j < ann->outputs; ++j) {
mw = w + ((ann->inputs + 1) * j);
sum = *mw++ * -1.0;
for (k = 0; k < ann->inputs; ++k) {
sum += *mw++ * i[k];
}
double output = genann_act_output(ann, sum);
*(o + j) = output;
}
return ret;
}
#pragma omp parallel for private(k, j, mw, sum) firstprivate(o, w)
for (j = 0; j < ann->hidden; ++j) {
mw = w + ((ann->inputs + 1) * j);
sum = *mw++ * -1.0;
for (k = 0; k < ann->inputs; ++k) {
sum += *mw++ * i[k];
}
double output = genann_act_output(ann, sum);
*(o + j) = output;
}
w += ((ann->inputs + 1) * ann->hidden);
i += ann->inputs;
o += ann->hidden;
for (h = 1; h < ann->hidden_layers; ++h) {
#pragma omp parallel for private(k, j, mw, sum) firstprivate(o, w)
for (j = 0; j < ann->hidden; ++j) {
mw = w + ((ann->hidden + 1) * j);
sum = *mw++ * -1.0;
for (k = 0; k < ann->hidden; ++k) {
sum += *mw++ * i[k];
}
double output = genann_act_output(ann, sum);
*(o + j) = output;
}
w += ((ann->hidden + 1) * ann->hidden);
i += ann->hidden;
o += ann->hidden;
}
double const *ret = o;
#pragma omp parallel for private(k, j, mw, sum) firstprivate(o, w)
for (j = 0; j < ann->outputs; ++j) {
mw = w + ((ann->hidden + 1) * j);
sum = *mw++ * -1.0;
for (k = 0; k < ann->hidden; ++k) {
sum += *mw++ * i[k];
}
double output = genann_act_output(ann, sum);
*(o + j) = output;
}
w += ((ann->hidden + 1) * ann->outputs);
o += ann->outputs;
assert(w - ann->weight == ann->total_weights);
assert(o - ann->output == ann->total_neurons);
return ret;
}
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
genann_run(ann, inputs);
omp_set_num_threads(NUM_THREADS);
int h, j, k;
{
double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; 
double *d = ann->delta + ann->hidden * ann->hidden_layers; 
double const *t = desired_outputs; 
if (genann_act_output == genann_act_linear ||
ann->activation_output == genann_act_linear) {
#pragma omp parallel for firstprivate(d, t, o) private(j)
for (j = 0; j < ann->outputs; ++j) {
*(d + j) = *(t + j) - *(o + j);
}
}
else {
#pragma omp parallel for firstprivate(d, t, o) private(j)
for (j = 0; j < ann->outputs; ++j) {
*(d + j) = (*(t + j) - *(o + j)) * *(o + j) * (1.0 - *(o + j));
}
}
}
for (h = ann->hidden_layers - 1; h >= 0; --h) {
double const *o = ann->output + ann->inputs + (h * ann->hidden);
double *d = ann->delta + (h * ann->hidden);
double const * const dd = ann->delta + ((h + 1) * ann->hidden);
double const * const ww = ann->weight + ((ann->inputs + 1) * ann->hidden) + ((ann->hidden + 1) * ann->hidden * (h));
#pragma omp parallel for private(j, k) firstprivate(o, d)
for (j = 0; j < ann->hidden; ++j) {
double delta = 0;
for (k = 0; k < (h == ann->hidden_layers - 1 ? ann->outputs : ann->hidden); ++k) {
const double forward_delta = dd[k];
const int windex = k * (ann->hidden + 1) + (j + 1);
const double forward_weight = ww[windex];
delta += forward_delta * forward_weight;
}
*(d + j) = *(o + j) * (1.0 - *(o + j)) * delta;
}
}
{
double const *d = ann->delta + ann->hidden * ann->hidden_layers; 
double *w = ann->weight + (ann->hidden_layers
? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1))
: (0));
double const * const i = ann->output + (ann->hidden_layers
? (ann->inputs + (ann->hidden) * (ann->hidden_layers - 1))
: 0);
#pragma omp parallel for private(j, k) firstprivate(d, w)
for (j = 0; j < ann->outputs; ++j) {
double *mw = w + ((ann->hidden_layers ? ann->hidden : ann->inputs) + 1) * j;
*mw++ += *(d + j) * learning_rate * -1.0;
for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
*mw++ += *(d + j) * learning_rate * i[k - 1];
}
}
w += ((ann->hidden_layers ? ann->hidden : ann->inputs) + 1) * ann->outputs;
assert(w - ann->weight == ann->total_weights);
}
for (h = ann->hidden_layers - 1; h >= 0; --h) {
double const *d = ann->delta + (h * ann->hidden);
double const *i = ann->output + (h
? (ann->inputs + ann->hidden * (h - 1))
: 0);
double *w = ann->weight + (h
? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * (ann->hidden) * (h - 1))
: 0);
#pragma omp parallel for private(j, k) firstprivate(d, w)
for (j = 0; j < ann->hidden; ++j) {
double *mw = w + ((h == 0 ? ann->inputs : ann->hidden) + 1) * j;
*mw++ += *(d + j) * learning_rate * -1.0;
for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
*mw++ += *(d + j) * learning_rate * i[k - 1];
}
}
}
}
void genann_write(genann const *ann, FILE *out) {
fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);
int i;
for (i = 0; i < ann->total_weights; ++i) {
fprintf(out, " %.20e", ann->weight[i]);
}
}
