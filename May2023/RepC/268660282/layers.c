#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif
#include <omp.h>
#include "layers.h"
#include "volume.h"
conv_layer_t* make_conv_layer(int input_width, int input_height, int input_depth, int filter_width, int num_filters,
int stride, int pad) {
conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));
l->output_depth = num_filters;
l->filter_width = filter_width;
l->input_depth  = input_depth;
l->input_width  = input_width;
l->input_height = input_height;
l->filter_height = l->filter_width;
l->stride        = stride;
l->pad = pad;
l->output_width = (l->input_width + l->pad * 2 - l->filter_width) /
l->stride + 1;
l->output_height = (l->input_height + l->pad * 2 - l->filter_height) /
l->stride + 1;
l->filters = malloc(sizeof(volume_t*) * num_filters);
for (int i = 0; i < num_filters; i++) {
l->filters[i] = make_volume(l->filter_width, l->filter_height,
l->input_depth, 0.0);
}
l->bias   = 0.0;
l->biases = make_volume(1, 1, l->output_depth, l->bias);
return l;
}
void conv_forward(conv_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
volume_t* in  = inputs[0];
volume_t* out = outputs[0];
int stride = l->stride;
for (int f = 0; f < l->output_depth; f++) {
volume_t* filter = l->filters[f];
int y = -l->pad;
for (int out_y = 0; out_y < l->output_height; y += stride, out_y++) {
int x = -l->pad;
for (int out_x = 0; out_x < l->output_width; x += stride, out_x++) {
double sum = 0.0;
for (int fy = 0; fy < filter->height; fy++) {
int in_y = y + fy;
if (in_y >= 0 && in_y < in->height) {
for (int fx = 0; fx < filter->width; fx++) {
int in_x = x + fx;
if (in_x >= 0 && in_x < in->width) {
double res[4];
__m256d sum_v = _mm256_set1_pd(0.0);
if (filter->depth == 3) {
sum += filter->weights[((filter->width * fy) + fx) * filter->depth ] * in->weights[((in->width * in_y) + in_x) * in->depth];
sum += filter->weights[((filter->width * fy) + fx) * filter->depth + 1] * in->weights[((in->width * in_y) + in_x) * in->depth + 1];
sum += filter->weights[((filter->width * fy) + fx) * filter->depth + 2] * in->weights[((in->width * in_y) + in_x) * in->depth + 2];
} else if (filter->depth == 16) {
__m256d vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth );
__m256d vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth );
__m256d temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth + 4 );
vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth + 4 );
temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth + 8 );
vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth + 8 );
temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth + 12 );
vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth + 12 );
temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
_mm256_store_pd(res, sum_v); 
sum += res[0] + res[1] + res[2] + res[3];
} else if (filter->depth == 20) {
__m256d vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth );
__m256d vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth );
__m256d temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth + 4 );
vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth + 4 );
temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth + 8 );
vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth + 8 );
temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth + 12 );
vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth + 12 );
temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
vector1 = _mm256_loadu_pd( filter->weights + ((filter->width * fy) + fx) * filter->depth + 16 );
vector2 = _mm256_loadu_pd( in->weights + ((in->width * in_y) + in_x) * in->depth + 16 );
temp_product = _mm256_mul_pd(vector1, vector2);
sum_v = _mm256_add_pd(sum_v, temp_product);
_mm256_store_pd(res, sum_v); 
sum += res[0] + res[1] + res[2] + res[3];
}
}
}
}
}
sum = sum + l->biases->weights[f];
out->weights[((out->width * out_y) + out_x) * out->depth + f] = sum;
}
}
}
}
void conv_load(conv_layer_t* l, const char* file_name) {
int filter_width;
int filter_height;
int depth;
int filters;
FILE* fin = fopen(file_name, "r");
fscanf(fin, "%d %d %d %d", &filter_width, &filter_height, &depth, &filters);
assert(filter_width == l->filter_width);
assert(filter_height == l->filter_height);
assert(depth == l->input_depth);
assert(filters == l->output_depth);
for (int f = 0; f < filters; f++) {
for (int x = 0; x < filter_width; x++) {
for (int y = 0; y < filter_height; y++) {
for (int d = 0; d < depth; d++) {
double val;
fscanf(fin, "%lf", &val);
volume_set(l->filters[f], x, y, d, val);
}
}
}
}
for (int d = 0; d < l->output_depth; d++) {
double val;
fscanf(fin, "%lf", &val);
volume_set(l->biases, 0, 0, d, val);
}
fclose(fin);
}
relu_layer_t* make_relu_layer(int input_width, int input_height, int input_depth) {
relu_layer_t* l = (relu_layer_t*)malloc(sizeof(relu_layer_t));
l->input_depth  = input_depth;
l->input_width  = input_width;
l->input_height = input_height;
l->output_width  = l->input_width;
l->output_height = l->input_height;
l->output_depth  = l->input_depth;
return l;
}
void relu_forward(relu_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
for (int x = 0; x < l->input_width; x++) {
for (int y = 0; y < l->input_height; y++) {
for (int d = 0; d < l->input_depth; d++) { 
double perspective = inputs[0]->weights[((inputs[0]->width * y) + x) * inputs[0]->depth + d];
double value = (perspective < 0.0) ? 0.0 : perspective;
volume_set(outputs[0], x, y, d, value); 
}
}
}
}
pool_layer_t* make_pool_layer(int input_width, int input_height, int input_depth, int pool_width, int stride) {
pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));
l->pool_width   = pool_width;
l->input_depth  = input_depth;
l->input_width  = input_width;
l->input_height = input_height;
l->pool_height = l->pool_width;
l->stride      = stride;
l->pad         = 0;
l->output_depth  = input_depth;
l->output_width  = floor((l->input_width + l->pad * 2 - l->pool_width) / l->stride + 1);
l->output_height = floor((l->input_height + l->pad * 2 - l->pool_height) / l->stride + 1);
return l;
}
void pool_forward(pool_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
volume_t* in  = inputs[0];
volume_t* out = outputs[0];
int n = 0;
for (int d = 0; d < l->output_depth; d++) {
int x = -l->pad;
for (int out_x = 0; out_x < l->output_width; x += l->stride, out_x++) {
int y = -l->pad;
for (int out_y = 0; out_y < l->output_height; y += l->stride, out_y++) {
double max = -INFINITY;
for (int fx = 0; fx < l->pool_width; fx++) {
for (int fy = 0; fy < l->pool_height; fy++) {
int in_y = y + fy;
int in_x = x + fx;
if (in_x >= 0 && in_x < in->width && in_y >= 0 && in_y < in->height) {
double v = in->weights[((in->width * in_y) + in_x) * in->depth + d];;
if (v > max) {
max = v;
}
}
}
}
n++;
volume_set(out, out_x, out_y, d, max);
}
}
}
}
fc_layer_t* make_fc_layer(int input_width, int input_height, int input_depth, int num_neurons) {
fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));
l->output_depth = num_neurons;
l->input_depth  = input_depth;
l->input_width  = input_width;
l->input_height = input_height;
l->num_inputs    = l->input_width * l->input_height * l->input_depth;
l->output_width  = 1;
l->output_height = 1;
l->filters = (volume_t**)malloc(sizeof(volume_t*) * num_neurons);
for (int i = 0; i < l->output_depth; i++) {
l->filters[i] = make_volume(1, 1, l->num_inputs, 0.0);
}
l->bias   = 0.0;
l->biases = make_volume(1, 1, l->output_depth, l->bias);
return l;
}
void fc_forward(fc_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
volume_t* in  = inputs[0];
volume_t* out = outputs[0];
for (int i = 0; i < l->output_depth; i++) {
double dot = 0.0;
for (int d = 0; d < l->num_inputs; d++) {
dot += in->weights[d] * l->filters[i]->weights[d];
}
dot += l->biases->weights[i];
out->weights[i] = dot;
}
}
void fc_load(fc_layer_t* l, const char* filename) {
FILE* fin = fopen(filename, "r");
int num_inputs;
int output_depth;
fscanf(fin, "%d %d", &num_inputs, &output_depth);
assert(output_depth == l->output_depth);
assert(num_inputs == l->num_inputs);
for (int i = 0; i < l->output_depth; i++) {
for (int j = 0; j < l->num_inputs; j++) {
fscanf(fin, "%lf", &(l->filters[i]->weights[j]));
}
}
for (int i = 0; i < l->output_depth; i++) {
fscanf(fin, "%lf", &(l->biases->weights[i]));
}
fclose(fin);
}
softmax_layer_t* make_softmax_layer(int input_width, int input_height, int input_depth) {
softmax_layer_t* l = (softmax_layer_t*)malloc(sizeof(softmax_layer_t));
l->input_depth  = input_depth;
l->input_width  = input_width;
l->input_height = input_height;
l->output_width  = 1;
l->output_height = 1;
l->output_depth  = l->input_width * l->input_height * l->input_depth;
l->likelihoods = (double*)malloc(sizeof(double) * l->output_depth);
return l;
}
void softmax_forward(softmax_layer_t* l, volume_t** inputs, volume_t** outputs, int start, int end) {
double likelihoods[l->output_depth];
volume_t* in  = inputs[0];
volume_t* out = outputs[0];
double amax = in->weights[0];
for (int i = 1; i < l->output_depth; i++) {
if (in->weights[i] > amax) {
amax = in->weights[i];
}
}
double e = 0.0;
double total = 0.0;
for (int i = 0; i < l->output_depth / 4 * 4; i+=4) {
e = exp(in->weights[i] - amax);
total += e;
likelihoods[i] = e;
e = exp(in->weights[i+1] - amax);
total += e;
likelihoods[i+1] = e;
e = exp(in->weights[i+2] - amax);
total += e;
likelihoods[i+2] = e;
e = exp(in->weights[i+3]- amax);
total += e;
likelihoods[i+3] = e;
}
for (int i=l->output_depth / 4 * 4; i<l->output_depth; i++) {
e = exp(in->weights[i] - amax);
total += e;
likelihoods[i] = e;
}
for (int i = 0; i < l->output_depth / 4 * 4; i+=4) {
out->weights[i] = likelihoods[i] / total;
out->weights[i+1] = likelihoods[i+1] / total;
out->weights[i+2] = likelihoods[i+2] / total;
out->weights[i+3] = likelihoods[i+3] / total;
}
for (int i = l->output_depth / 4 * 4; i < l->output_depth; i++) {
out->weights[i] = likelihoods[i] / total;
}
}
