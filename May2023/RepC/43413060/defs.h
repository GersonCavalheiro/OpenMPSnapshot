#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>
#include "macros.h"
#include "cpu_features.h"
#ifdef __cplusplus
extern "C" {
#endif
#define API_VERSION          STR("2.5.0")
#define BINNING_REF_MASK         0x0000000F 
#define BINNING_ORD_MASK         0x000000F0 
#define BINNING_DFL   0x0
#define BINNING_CUST  0x1
struct api_cell_timings
{
int64_t N1;
int64_t N2;
int64_t time_in_ns;
int first_cellindex;
int second_cellindex;
int tid;
};
#define MAX_FAST_DIVIDE_NR_STEPS  3
#define OPTIONS_HEADER_SIZE     1024
#define BOXSIZE_NOTGIVEN (-2.)
struct config_options
{
union {
double boxsize;
double boxsize_x;
};
double boxsize_y;
double boxsize_z;
struct{
double OMEGA_M;
double OMEGA_B;
double OMEGA_L;
double HUBBLE;
double LITTLE_H;
double SIGMA_8;
double NS;
};
double c_api_time;
struct api_cell_timings *cell_timings;
int64_t totncells_timings;
size_t float_type; 
int32_t instruction_set; 
char version[32];
uint8_t verbose; 
uint8_t c_api_timer; 
uint8_t c_cell_timer;
uint8_t need_avg_sep; 
uint8_t autocorr;
uint8_t periodic; 
uint8_t sort_on_z;
uint8_t is_comoving_dist;
uint8_t link_in_dec;
uint8_t link_in_ra; 
uint8_t fast_divide_and_NR_steps;
uint8_t fast_acos;
uint8_t enable_min_sep_opt;
int8_t bin_refine_factors[3];
uint16_t max_cells_per_dim;
uint8_t copy_particles;
uint8_t use_heap_sort;
union{
uint32_t binning_flags;
uint8_t bin_masks[4];
};
uint8_t reserved[OPTIONS_HEADER_SIZE - 33*sizeof(char) - sizeof(size_t) - 11*sizeof(double) - 3*sizeof(int)
- sizeof(uint16_t) - 16*sizeof(uint8_t) - sizeof(struct api_cell_timings *) - sizeof(int64_t)
];
};
static inline void set_bin_refine_scheme(struct config_options *options, const int8_t flag)
{
options->binning_flags = (options->binning_flags & ~BINNING_REF_MASK) | (flag & BINNING_REF_MASK);
}
static inline void reset_bin_refine_scheme(struct config_options *options)
{
set_bin_refine_scheme(options, BINNING_DFL);
}
static inline int8_t get_bin_refine_scheme(struct config_options *options)
{
return (int8_t) (options->binning_flags & BINNING_REF_MASK);
}
static inline void set_bin_refine_factors(struct config_options *options, const int bin_refine_factors[3])
{
for(int i=0;i<3;i++) {
int8_t bin_refine = bin_refine_factors[i];
if(bin_refine_factors[i] > INT8_MAX) {
fprintf(stderr,"Warning: bin refine factor[%d] can be at most %d. Found %d instead\n", i,
INT8_MAX, bin_refine_factors[i]);
bin_refine = 1;
}
options->bin_refine_factors[i] = bin_refine;
}
reset_bin_refine_scheme(options);
}
static inline void set_custom_bin_refine_factors(struct config_options *options, const int bin_refine_factors[3])
{
set_bin_refine_factors(options, bin_refine_factors);
set_bin_refine_scheme(options, BINNING_CUST);
}
static inline void reset_bin_refine_factors(struct config_options *options)
{
options->bin_refine_factors[0] = 2;
options->bin_refine_factors[1] = 2;
options->bin_refine_factors[2] = 1;
reset_bin_refine_scheme(options);
}
static inline void set_max_cells(struct config_options *options, const int max)
{
if(max <=0) {
fprintf(stderr,"Warning: Max. cells per dimension was requested to be set to "
"a negative number = %d...returning\n", max);
return;
}
if(max > INT16_MAX) {
fprintf(stderr,"Warning: Max cells per dimension is a 2-byte integer and can not "
"hold supplied value of %d. Max. allowed value for max_cells_per_dim is %d\n",
max, INT16_MAX);
}
options->max_cells_per_dim = max;
}
static inline void reset_max_cells(struct config_options *options)
{
options->max_cells_per_dim = NLATMAX;
}
static inline struct config_options get_config_options(void)
{
ENSURE_STRUCT_SIZE(struct config_options, OPTIONS_HEADER_SIZE);
if(strncmp(API_VERSION, STR(VERSION), 32) != 0) {
fprintf(stderr,"Error: Version mismatch between header and Makefile. Header claims version = `%s' while Makefile claims version = `%s'\n"
"Library header probably needs to be updated\n", API_VERSION, STR(VERSION));
exit(EXIT_FAILURE);
}
struct config_options options;
BUILD_BUG_OR_ZERO(sizeof(options.max_cells_per_dim) == sizeof(int16_t), max_cells_per_dim_must_be_16_bits);
BUILD_BUG_OR_ZERO(sizeof(options.binning_flags) == sizeof(uint32_t), binning_flags_must_be_32_bits);
BUILD_BUG_OR_ZERO(sizeof(options.bin_refine_factors[0]) == sizeof(int8_t), bin_refine_factors_must_be_8_bits);
memset(&options, 0, OPTIONS_HEADER_SIZE);
snprintf(options.version, sizeof(options.version)/sizeof(char)-1, "%s", API_VERSION);
options.boxsize_x = BOXSIZE_NOTGIVEN;
options.boxsize_y = BOXSIZE_NOTGIVEN;
options.boxsize_z = BOXSIZE_NOTGIVEN;
#ifdef DOUBLE_PREC
options.float_type = sizeof(double);
#else
options.float_type = sizeof(float);
#endif
#ifndef SILENT
options.verbose = 1;
#endif
#ifdef OUTPUT_RPAVG
options.need_avg_sep = 1;
#endif
#ifdef PERIODIC
options.periodic = 1;
#endif
#ifdef __AVX512F__
options.instruction_set = AVX512F;
#elif defined(__AVX2__)
options.instruction_set = AVX2;
#elif defined(__AVX__)
options.instruction_set = AVX;
#elif defined(__SSE4_2__)
options.instruction_set = SSE42;
#else
options.instruction_set = FALLBACK;
#endif
#if defined(FAST_DIVIDE)
#if FAST_DIVIDE > MAX_FAST_DIVIDE_NR_STEPS
options.fast_divide_and_NR_steps = MAX_FAST_DIVIDE_NR_STEPS;
#else
options.fast_divide_and_NR_steps = FAST_DIVIDE;
#endif
#endif
#ifdef OUTPUT_THETAAVG
options.need_avg_sep = 1;
#endif
#ifdef LINK_IN_DEC
options.link_in_dec = 1;
#endif
#ifdef LINK_IN_RA
options.link_in_ra=1;
options.link_in_dec=1;
#endif
#ifdef ENABLE_MIN_SEP_OPT
options.enable_min_sep_opt=1;
#endif
#ifdef FAST_ACOS
options.fast_acos=1;
#endif
#ifdef COMOVING_DIST
options.is_comoving_dist=1;
#endif
#ifdef COPY_PARTICLES
options.copy_particles = 1;
#else
options.copy_particles = 0;
#endif 
options.totncells_timings = 0;
options.cell_timings = NULL;
reset_max_cells(&options);
reset_bin_refine_factors(&options);
return options;
}
#define EXTRA_OPTIONS_HEADER_SIZE     (1024)
#define MAX_NUM_WEIGHTS 10
typedef struct
{
void *weights[MAX_NUM_WEIGHTS];  
int64_t num_weights;
} weight_struct;
typedef enum {
NONE=-42, 
PAIR_PRODUCT=0,
NUM_WEIGHT_TYPE
} weight_method_t; 
static inline int get_num_weights_by_method(const weight_method_t method){
switch(method){
case PAIR_PRODUCT:
return 1;
default:
case NONE:
return 0;
}
}
static inline int get_weight_method_by_name(const char *name, weight_method_t *method){
if(name == NULL || strcmp(name, "") == 0){
*method = NONE;
return EXIT_SUCCESS;
}
if(strcmp(name, "pair_product") == 0 || strcmp(name, "p") == 0){
*method = PAIR_PRODUCT;
return EXIT_SUCCESS;
}
return EXIT_FAILURE;
}
struct extra_options
{
weight_struct weights0;
weight_struct weights1;
weight_method_t weight_method; 
uint8_t reserved[EXTRA_OPTIONS_HEADER_SIZE - 2*sizeof(weight_struct) - sizeof(weight_method_t)];
};
static inline struct extra_options get_extra_options(const weight_method_t weight_method)
{
struct extra_options extra;
ENSURE_STRUCT_SIZE(struct extra_options, EXTRA_OPTIONS_HEADER_SIZE);
memset(&extra, 0, EXTRA_OPTIONS_HEADER_SIZE);
extra.weight_method = weight_method;
weight_struct *w0 = &(extra.weights0);
weight_struct *w1 = &(extra.weights1);
w0->num_weights = get_num_weights_by_method(extra.weight_method);
w1->num_weights = w0->num_weights;
return extra;
}
static inline void print_cell_timings(struct config_options *options)
{
fprintf(stderr,"#########################################################################\n");
fprintf(stderr,"#  Cell_1    Cell_2          N1          N2        Time_ns     ThreadID  \n");
fprintf(stderr,"#########################################################################\n");
for(int64_t i=0;i<options->totncells_timings;i++) {
fprintf(stderr,"%8d %8d %12"PRId64" %12"PRId64" %12"PRId64" %12d\n",
options->cell_timings[i].first_cellindex,
options->cell_timings[i].second_cellindex,
options->cell_timings[i].N1,
options->cell_timings[i].N2,
options->cell_timings[i].time_in_ns,
options->cell_timings[i].tid);
}
}
static inline void free_cell_timings(struct config_options *options)
{
if(options->totncells_timings > 0 && options->cell_timings != NULL) {
free(options->cell_timings);
}
options->totncells_timings = 0;
return;
}
static inline void allocate_cell_timer(struct config_options *options, const int64_t num_cell_pairs)
{
if(options->totncells_timings >= num_cell_pairs) return;
free_cell_timings(options);
options->cell_timings = calloc(num_cell_pairs, sizeof(*(options->cell_timings)));
if(options->cell_timings == NULL) {
fprintf(stderr,"Warning: In %s> Could not allocate memory to store the API timings per cell. \n",
__FUNCTION__);
} else {
options->totncells_timings = num_cell_pairs;
}
return;
}
static inline void assign_cell_timer(struct api_cell_timings *cell_timings, const int64_t num_cell_pairs, struct config_options *options)
{
allocate_cell_timer(options, num_cell_pairs);
if(options->totncells_timings >= num_cell_pairs) {
memmove(options->cell_timings, cell_timings, sizeof(struct api_cell_timings) * num_cell_pairs);
}
}
#include "macros.h"
#ifdef __cplusplus
}
#endif
