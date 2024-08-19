

















































#ifdef USE_COMPOUND_WD
#ifndef FORCE_MANDATORY_CREATION
#define FORCE_MANDATORY_CREATION
#endif
#endif

#include <nanos.h>
#include "omp.h"

#define MODEL NANOX-TASKS
#define BOTS_MODEL_DESC "Nanos++"

#ifdef SERIAL_VERSION
#define BOTS_APP_NAME "Sort (serial version)"
#else
#ifdef MANDATORY_CREATION
#ifdef USE_COMPOUND_WD
#define BOTS_APP_NAME "Sort (parallel: compound wd's slicer)"
#else
#define BOTS_APP_NAME "Sort (parallel: mandatory wd creation)"
#endif
#else
#define BOTS_APP_NAME "Sort (parallel version)"
#endif
#endif

#define BOTS_APP_PARAMETERS_DESC "N=%d:Q=%d:I=%d:M=%d"
#define BOTS_APP_PARAMETERS_LIST ,bots_arg_size,bots_app_cutoff_value_1,bots_app_cutoff_value_2,bots_app_cutoff_value

#define BOTS_APP_USES_ARG_SIZE
#define BOTS_APP_DEF_ARG_SIZE (1*1024*1024)
#define BOTS_APP_DESC_ARG_SIZE "Array size"

#define BOTS_APP_USES_ARG_CUTOFF
#define BOTS_APP_DEF_ARG_CUTOFF (2*1024)
#define BOTS_APP_DESC_ARG_CUTOFF "Sequential Merge cutoff value"

#define BOTS_APP_USES_ARG_CUTOFF_1
#define BOTS_APP_DEF_ARG_CUTOFF_1 (2*1024)
#define BOTS_APP_DESC_ARG_CUTOFF_1 "Sequential Quicksort cutoff value"

#define BOTS_APP_USES_ARG_CUTOFF_2
#define BOTS_APP_DEF_ARG_CUTOFF_2 (20)
#define BOTS_APP_DESC_ARG_CUTOFF_2 "Sequential Insertion cutoff value"

#define BOTS_PARAM_TYPE_NONE 0
#define BOTS_PARAM_TYPE_INT 1
#define BOTS_PARAM_TYPE_BOOL 2
#define BOTS_PARAM_TYPE_STR 3

#ifdef _OPENMP
# include <omp.h>
#else
# define omp_get_max_threads()  1
# define omp_get_thread_num()   0
# define omp_set_num_threads(x)
#endif

void sort_par (void);
void sort_init (void);
int sort_verify (void);

#define BOTS_APP_INIT sort_init()

#define KERNEL_INIT
#define KERNEL_CALL sort_par()
#define KERNEL_CHECK sort_verify()


#ifndef _COMMON_H
#define _COMMON_H

#ifndef CC
#define CC ""
#endif
#ifndef CFLAGS
#define CFLAGS ""
#endif
#ifndef LD
#define LD ""
#endif
#ifndef LDFLAGS
#define LDFLAGS ""
#endif
#ifndef CDATE
#define CDATE ""
#endif
#ifndef CMESSAGE
#define CMESSAGE ""
#endif

#define BOTS_ERROR                         0
#define BOTS_ERROR_NOT_ENOUGH_MEMORY       1
#define BOTS_ERROR_UNRECOGNIZED_PARAMETER  2

#define BOTS_WARNING                       0

void bots_get_date(char *str);
void bots_get_architecture(char *str);
void bots_get_load_average(char *str);
void bots_print_results(void);
#endif

void bots_print_usage(void);
void bots_print_usage_option(char opt, int type, char* description, char *val, int subc, char **subv);


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _BOTS_H_


extern int bots_sequential_flag;
extern int bots_benchmark_flag;
extern int bots_check_flag;
extern int bots_result;
extern int bots_output_format;
extern int bots_print_header;

extern char bots_name[];
extern char bots_parameters[];
extern char bots_model[];
extern char bots_resources[];

extern char bots_exec_date[];
extern char bots_exec_message[];
extern char bots_comp_date[];
extern char bots_comp_message[];
extern char bots_cc[];
extern char bots_cflags[];
extern char bots_ld[];
extern char bots_ldflags[];

extern double bots_time_program;
extern double bots_time_sequential;
extern int    bots_number_of_tasks;

extern char bots_cutoff[];
extern int  bots_cutoff_value;

extern int  bots_app_cutoff_value;
extern int  bots_app_cutoff_value_1;
extern int  bots_app_cutoff_value_2;

extern int bots_arg_size;
extern int bots_arg_size_1;
extern int bots_arg_size_2;


long bots_usecs();
void bots_error(int error, char *message);
void bots_warning(int warning, char *message);

#define BOTS_RESULT_NA 0
#define BOTS_RESULT_SUCCESSFUL 1
#define BOTS_RESULT_UNSUCCESSFUL 2
#define BOTS_RESULT_NOT_REQUESTED 3


typedef enum { BOTS_VERBOSE_NONE=0,
BOTS_VERBOSE_DEFAULT,
BOTS_VERBOSE_DEBUG } bots_verbose_mode_t;

extern bots_verbose_mode_t bots_verbose_mode;

#define bots_message(msg, ...) \
{\
if ( bots_verbose_mode >= BOTS_VERBOSE_DEFAULT ) {\
fprintf(stdout, msg , ##__VA_ARGS__);\
}\
}

#ifdef BOTS_DEBUG
#define bots_debug(msg, ...) \
{\
if ( bots_verbose_mode >= BOTS_VERBOSE_DEBUG ) {\
fprintf(stdout, msg , ##__VA_ARGS__);\
}\
}
#define bots_debug_with_location_info(msg, ...) \
{\
if ( bots_verbose_mode >= BOTS_VERBOSE_DEBUG ) {\
fprintf(stdout, "%s:%d:%s:" msg ,__FILE__, __LINE__,__func__,##__VA_ARGS__);\
}\
}
#else
#define bots_debug(msg, ...)
#define bots_debug_with_location_info(msg, ...)
#endif

#define FALSE 0
#define TRUE 1

#endif


typedef long ELM;

ELM *array, *tmp;

static unsigned long rand_nxt = 0;

static inline unsigned long my_rand(void)
{
rand_nxt = rand_nxt * 1103515245 + 12345;
return rand_nxt;
}

static inline void my_srand(unsigned long seed)
{
rand_nxt = seed;
}

static inline ELM med3(ELM a, ELM b, ELM c)
{
if (a < b) {
if (b < c) {
return b;
} else {
if (a < c)
return c;
else
return a;
}
} else {
if (b > c) {
return b;
} else {
if (a > c)
return c;
else
return a;
}
}
}


static inline ELM choose_pivot(ELM *low, ELM *high)
{
return med3(*low, *high, low[(high - low) / 2]);
}

static ELM *seqpart(ELM *low, ELM *high)
{
ELM pivot;
ELM h, l;
ELM *curr_low = low;
ELM *curr_high = high;

pivot = choose_pivot(low, high);

while (1) {
while ((h = *curr_high) > pivot)
curr_high--;

while ((l = *curr_low) < pivot)
curr_low++;

if (curr_low >= curr_high)
break;

*curr_high-- = l;
*curr_low++ = h;
}


if (curr_high < high)
return curr_high;
else
return curr_high - 1;
}

#define swap(a, b) \
{ \
ELM tmp;\
tmp = a;\
a = b;\
b = tmp;\
}

static void insertion_sort(ELM *low, ELM *high)
{
ELM *p, *q;
ELM a, b;

for (q = low + 1; q <= high; ++q) {
a = q[0];
for (p = q - 1; p >= low && (b = p[0]) > a; p--)
p[1] = b;
p[1] = a;
}
}


void seqquick(ELM *low, ELM *high)
{
ELM *p;

while (high - low >= bots_app_cutoff_value_2) {
p = seqpart(low, high);
seqquick(low, p);
low = p + 1;
}

insertion_sort(low, high);
}

void seqmerge(ELM *low1, ELM *high1, ELM *low2, ELM *high2,
ELM *lowdest)
{
ELM a1, a2;



if (low1 < high1 && low2 < high2) {
a1 = *low1;
a2 = *low2;
for (;;) {
if (a1 < a2) {
*lowdest++ = a1;
a1 = *++low1;
if (low1 >= high1)
break;
} else {
*lowdest++ = a2;
a2 = *++low2;
if (low2 >= high2)
break;
}
}
}
if (low1 <= high1 && low2 <= high2) {
a1 = *low1;
a2 = *low2;
for (;;) {
if (a1 < a2) {
*lowdest++ = a1;
++low1;
if (low1 > high1)
break;
a1 = *low1;
} else {
*lowdest++ = a2;
++low2;
if (low2 > high2)
break;
a2 = *low2;
}
}
}
if (low1 > high1) {
memcpy(lowdest, low2, sizeof(ELM) * (high2 - low2 + 1));
} else {
memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1 + 1));
}
}

#define swap_indices(a, b) \
{ \
ELM *tmp;\
tmp = a;\
a = b;\
b = tmp;\
}

ELM *binsplit(ELM val, ELM *low, ELM *high)
{

ELM *mid;

while (low != high) {
mid = low + ((high - low + 1) >> 1);
if (val <= *mid)
high = mid - 1;
else
low = mid;
}

if (*low > val)
return low - 1;
else
return low;
}

void cilkmerge_par(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest);

typedef struct {
ELM *low1;
ELM *high1;
ELM *low2;
ELM *high2;
ELM *lowdest;
} cilkmerge_par_1_args;
void cilkmerge_par_1( void *ptr )
{
nanos_event_t event;

nanos_instrument_get_key ("user-funct-name", &(event.key));
nanos_instrument_register_value ( &(event.value), "user-funct-name", "cilk-merge", "Merge function", false );

event.type = NANOS_BURST_START;
nanos_instrument_events(1, &event);

cilkmerge_par_1_args * args = ( cilkmerge_par_1_args * ) ptr;
cilkmerge_par(args->low1,args->high1,args->low2,args->high2,args->lowdest);

event.type = NANOS_BURST_END;
nanos_instrument_events(1, &event);
}
nanos_smp_args_t cilkmerge_par_1_device_arg = { cilkmerge_par_1 };



struct nanos_const_wd_definition_1
{
nanos_const_wd_definition_t base;
nanos_device_t devices[1];
};

struct nanos_const_wd_definition_1 const_data1 = 
{
{{
#ifdef FORCE_MANDATORY_CREATION
.mandatory_creation = true,
#else
.mandatory_creation = false,
#endif
.tied = false},
__alignof__(cilkmerge_par_1_args),
0,
1,0,NULL},
{
{
nanos_smp_factory,
&cilkmerge_par_1_device_arg
}
}
};

void cilkmerge_par(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest)
{


ELM *split1, *split2;	
long int lowsize;		



if (high2 - low2 > high1 - low1) {
swap_indices(low1, low2);
swap_indices(high1, high2);
}
if (high2 < low2) {

memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1));
return;
}
if (high2 - low2 < bots_app_cutoff_value ) {
seqmerge(low1, high1, low2, high2, lowdest);
return;
}


split1 = ((high1 - low1 + 1) / 2) + low1;
split2 = binsplit(*split1, low2, high2);
lowsize = split1 - low1 + split2 - low2;


*(lowdest + lowsize + 1) = *split1;
#ifdef SERIAL_VERSION
cilkmerge_par(low1, split1 - 1, low2, split2, lowdest );
cilkmerge_par(split1 + 1, high1, split2 + 1, high2, lowdest + lowsize + 2 );

#else
{

nanos_wd_t wd[2] = {NULL,NULL};

nanos_wd_dyn_props_t dyn_props = {0};

cilkmerge_par_1_args *args_w1 = NULL;
cilkmerge_par_1_args *args_w2 = NULL;

NANOS_SAFE( nanos_create_wd_compact ( &wd[0], &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
( void ** )&args_w1, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[0] != (nanos_wd_t) 0)
{       
#endif
args_w1->low1 = low1;
args_w1->high1 = split1 - 1;
args_w1->low2 = low2;
args_w1->high2 = split2;
args_w1->lowdest = lowdest;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[0],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilkmerge_par_1_args imm_args;

imm_args.low1 = low1;
imm_args.high1 = split1 - 1;
imm_args.low2 = low2;
imm_args.high2 = split2;
imm_args.lowdest = lowdest;

NANOS_SAFE( nanos_create_wd_and_run_compact( &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

NANOS_SAFE( nanos_create_wd_compact ( &wd[1], &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
( void ** )&args_w2, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[1] != (nanos_wd_t) 0)
{       
#endif
args_w2->low1 = split1 + 1;
args_w2->high1 = high1;
args_w2->low2 = split2 + 1;
args_w2->high2 = high2;
args_w2->lowdest = lowdest + lowsize + 2;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[1],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilkmerge_par_1_args imm_args;

imm_args.low1 = split1 + 1;
imm_args.high1 = high1;
imm_args.low2 = split2 + 1;
imm_args.high2 = high2;
imm_args.lowdest = lowdest + lowsize + 2;

NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

#ifdef USE_COMPOUND_WD
nanos_slicer_t slicer = nanos_find_slicer("compound_wd");
nanos_wd_t cwd = NULL;
void * compound_f;
nanos_slicer_get_specific_data ( slicer, &compound_f );
nanos_smp_args_t main__sections_device_args = { compound_f };
nanos_device_t main__sections_device[1] = { NANOS_SMP_DESC( main__sections_device_args ) };
nanos_compound_wd_data_t *list_of_wds = NULL;


NANOS_SAFE( nanos_create_sliced_wd ( &cwd, 1, main__sections_device,
sizeof(nanos_compound_wd_data_t) + (2) * sizeof(nanos_wd_t), __alignof__(nanos_compound_wd_data_t),
(void **) &list_of_wds, nanos_current_wd(), slicer, &const_data1.base.props , &dyn_props, 0, NULL, 0, NULL ) );


list_of_wds->nsect = 2;
list_of_wds->lwd[0] = wd[0];
list_of_wds->lwd[1] = wd[1];
NANOS_SAFE( nanos_submit( cwd,0,0,0 ) );
#endif

}

NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
#endif 

return;
}

void cilksort_par(ELM *low, ELM *tmp, long size);

typedef struct {
ELM *low;
ELM *tmp;
long size;
} cilksort_par_1_args;
void cilksort_par_1( void *ptr )
{
nanos_event_t event;

nanos_instrument_get_key ("user-funct-name", &(event.key));
nanos_instrument_register_value ( &(event.value), "user-funct-name", "cilk-sort", "Sort function", false );

event.type = NANOS_BURST_START;
nanos_instrument_events(1, &event);

cilksort_par_1_args * args = ( cilksort_par_1_args * ) ptr;
cilksort_par(args->low, args->tmp, args->size);

event.type = NANOS_BURST_END;
nanos_instrument_events(1, &event);
}
nanos_smp_args_t cilksort_par_1_device_arg = { cilksort_par_1 };


struct nanos_const_wd_definition_1 const_data2 = 
{
{{
#ifdef FORCE_MANDATORY_CREATION
.mandatory_creation = true,
#else
.mandatory_creation = false,
#endif
.tied = false},
__alignof__(cilksort_par_1_args),
0,
1,0,NULL},
{
{
nanos_smp_factory,
&cilksort_par_1_device_arg
}
}
};

void cilksort_par(ELM *low, ELM *tmp, long size)
{

long quarter = size / 4;
ELM *A, *B, *C, *D, *tmpA, *tmpB, *tmpC, *tmpD;

if (size < bots_app_cutoff_value_1 ) {

seqquick(low, low + size - 1);
return;
}
A = low;
tmpA = tmp;
B = A + quarter;
tmpB = tmpA + quarter;
C = B + quarter;
tmpC = tmpB + quarter;
D = C + quarter;
tmpD = tmpC + quarter;

#ifdef SERIAL_VERSION
cilksort_par (A,tmpA,quarter);
cilksort_par (B,tmpB,quarter);
cilksort_par (C,tmpC,quarter);
cilksort_par (D,tmpD,size - 3*quarter);
#else
{
nanos_wd_t wd[4] = {NULL,NULL,NULL,NULL};

nanos_wd_dyn_props_t dyn_props = {0};

cilksort_par_1_args *args_w0 = NULL;
cilksort_par_1_args *args_w1 = NULL;
cilksort_par_1_args *args_w2 = NULL;
cilksort_par_1_args *args_w3 = NULL;

NANOS_SAFE( nanos_create_wd_compact ( &wd[0], &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
( void ** )&args_w0, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[0] != (nanos_wd_t) 0)
{       
#endif
args_w0->low = A;
args_w0->tmp = tmpA;
args_w0->size = quarter;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[0],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilksort_par_1_args imm_args;

imm_args.low = A;
imm_args.tmp = tmpA;
imm_args.size = quarter;

NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

NANOS_SAFE( nanos_create_wd_compact ( &wd[1], &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
( void ** )&args_w1, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[1] != (nanos_wd_t) 0)
{       
#endif
args_w1->low = B;
args_w1->tmp = tmpB;
args_w1->size = quarter;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[1],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilksort_par_1_args imm_args;

imm_args.low = B;
imm_args.tmp = tmpB;
imm_args.size = quarter;

NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

NANOS_SAFE( nanos_create_wd_compact ( &wd[2], &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
( void ** )&args_w2, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[2] != (nanos_wd_t) 0)
{       
#endif
args_w2->low = C;
args_w2->tmp = tmpC;
args_w2->size = quarter;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[2],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilksort_par_1_args imm_args;

imm_args.low = C;
imm_args.tmp = tmpC;
imm_args.size = quarter;

NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

NANOS_SAFE( nanos_create_wd_compact ( &wd[3], &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
( void ** )&args_w3, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[3] != (nanos_wd_t) 0)
{       
#endif
args_w3->low = D;
args_w3->tmp = tmpD;
args_w3->size = size - 3 * quarter;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[3],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilksort_par_1_args imm_args;

imm_args.low = D;
imm_args.tmp = tmpD;
imm_args.size = size - 3 * quarter;

NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data2.base, &dyn_props, sizeof( cilksort_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

#ifdef USE_COMPOUND_WD
nanos_slicer_t slicer = nanos_find_slicer("compound_wd");

nanos_wd_t cwd = NULL;

void * compound_f;
nanos_slicer_get_specific_data ( slicer, &compound_f );

nanos_smp_args_t main__sections_device_args = { compound_f };
nanos_device_t main__sections_device[1] = { NANOS_SMP_DESC( main__sections_device_args ) };
nanos_compound_wd_data_t *list_of_wds = NULL;


NANOS_SAFE( nanos_create_sliced_wd ( &cwd, 1, main__sections_device,
sizeof(nanos_compound_wd_data_t) + (4) * sizeof(nanos_wd_t), __alignof__( nanos_compound_wd_data_t),
(void **) &list_of_wds,
nanos_current_wd(), slicer, &const_data2.base.props , &dyn_props, 0, NULL, 0, NULL ) );


list_of_wds->nsect = 4;
list_of_wds->lwd[0] = wd[0];
list_of_wds->lwd[1] = wd[1];
list_of_wds->lwd[2] = wd[2];
list_of_wds->lwd[3] = wd[3];

NANOS_SAFE( nanos_submit( cwd,0,0,0 ) );
#endif

}

NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
#endif 

#ifdef SERIAL_VERSION
cilkmerge_par(A, A + quarter - 1, B, B + quarter - 1, tmpA );
cilkmerge_par(C, C + quarter - 1, D, low + size - 1, tmpC);
#else
{

nanos_wd_t wd[2] = {NULL,NULL};

nanos_wd_dyn_props_t dyn_props = {0};

cilkmerge_par_1_args *args_w0 = NULL;
cilkmerge_par_1_args *args_w1 = NULL;

NANOS_SAFE( nanos_create_wd_compact ( &wd[0], &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
( void ** )&args_w0, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[0] != (nanos_wd_t) 0)
{       
#endif
args_w0->low1 = A;
args_w0->high1 = A + quarter - 1;
args_w0->low2 = B;
args_w0->high2 = B + quarter - 1;
args_w0->lowdest = tmpA;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[0],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilkmerge_par_1_args imm_args;

imm_args.low1 = A;
imm_args.high1 = A + quarter - 1;
imm_args.low2 = B;
imm_args.high2 = B + quarter - 1;
imm_args.lowdest = tmpA;

NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

NANOS_SAFE( nanos_create_wd_compact ( &wd[1], &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
( void ** )&args_w1, nanos_current_wd(), NULL, NULL ) );

#ifndef FORCE_MANDATORY_CREATION
if (wd[1] != (nanos_wd_t) 0)
{       
#endif
args_w1->low1 = C;
args_w1->high1 = C + quarter - 1;
args_w1->low2 = D;
args_w1->high2 = low + size - 1;
args_w1->lowdest = tmpC;

#ifndef USE_COMPOUND_WD
NANOS_SAFE( nanos_submit( wd[1],0,0,0 ) );
#endif

#ifndef FORCE_MANDATORY_CREATION
} else {       
cilkmerge_par_1_args imm_args;

imm_args.low1 = C;
imm_args.high1 = C + quarter - 1;
imm_args.low2 = D;
imm_args.high2 = low + size - 1;
imm_args.lowdest = tmpC;

NANOS_SAFE( nanos_create_wd_and_run_compact ( &const_data1.base, &dyn_props, sizeof( cilkmerge_par_1_args ),
&imm_args, 0, (nanos_data_access_t *) 0, NULL, NULL, NULL ) );
}
#endif

#ifdef USE_COMPOUND_WD
nanos_slicer_t slicer = nanos_find_slicer("compound_wd");
nanos_wd_t cwd = NULL;
void * compound_f;
nanos_slicer_get_specific_data ( slicer, &compound_f );
nanos_smp_args_t main__sections_device_args = { compound_f };
nanos_device_t main__sections_device[1] = { NANOS_SMP_DESC( main__sections_device_args ) };
nanos_compound_wd_data_t *list_of_wds = NULL;


NANOS_SAFE( nanos_create_sliced_wd ( &cwd, 1, main__sections_device,
sizeof(nanos_compound_wd_data_t) + (2) * sizeof(nanos_wd_t), __alignof__(nanos_compound_wd_data_t),
(void **) &list_of_wds,
nanos_current_wd(), slicer, &const_data1.base.props , &dyn_props, 0, NULL, 0, NULL ) );


list_of_wds->nsect = 2;
list_of_wds->lwd[0] = wd[0];
list_of_wds->lwd[1] = wd[1];
NANOS_SAFE( nanos_submit( cwd,0,0,0 ) );
#endif

}

NANOS_SAFE( nanos_wg_wait_completion( nanos_current_wd(), false ) );
#endif 

cilkmerge_par(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A);
}

void scramble_array( ELM *array )
{
unsigned long i;
unsigned long j;

for (i = 0; i < bots_arg_size; ++i) {
j = my_rand();
j = j % bots_arg_size;
swap(array[i], array[j]);
}
}

void fill_array( ELM *array )
{
unsigned long i;

my_srand(1);

for (i = 0; i < bots_arg_size; ++i) {
array[i] = i;
}
}

void sort_init ( void )
{

if (bots_arg_size < 4) {
bots_message("%s can not be less than 4, using 4 as a parameter.\n", BOTS_APP_DESC_ARG_SIZE );
bots_arg_size = 4;
}

if (bots_app_cutoff_value < 2) {
bots_message("%s can not be less than 2, using 2 as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF);
bots_app_cutoff_value = 2;
}
else if (bots_app_cutoff_value > bots_arg_size ) {
bots_message("%s can not be greather than vector size, using %d as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF, bots_arg_size);
bots_app_cutoff_value = bots_arg_size;
}

if (bots_app_cutoff_value_1 > bots_arg_size ) {
bots_message("%s can not be greather than vector size, using %d as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF_1, bots_arg_size);
bots_app_cutoff_value_1 = bots_arg_size;
}
if (bots_app_cutoff_value_2 > bots_arg_size ) {
bots_message("%s can not be greather than vector size, using %d as a parameter.\n", BOTS_APP_DESC_ARG_CUTOFF_2, bots_arg_size);
bots_app_cutoff_value_2 = bots_arg_size;
}

if (bots_app_cutoff_value_2 > bots_app_cutoff_value_1) {
bots_message("%s can not be greather than %s, using %d as a parameter.\n",
BOTS_APP_DESC_ARG_CUTOFF_2,
BOTS_APP_DESC_ARG_CUTOFF_1,
bots_app_cutoff_value_1
);
bots_app_cutoff_value_2 = bots_app_cutoff_value_1;
}

array = (ELM *) malloc(bots_arg_size * sizeof(ELM));
tmp = (ELM *) malloc(bots_arg_size * sizeof(ELM));
fill_array(array);
scramble_array(array);
}

void sort_par ( void )
{
bots_message("Computing multisort algorithm (n=%d) ", bots_arg_size);
cilksort_par(array, tmp, bots_arg_size);
bots_message(" completed!\n");
}

int sort_verify ( void )
{
int i, success = 1;
for (i = 0; i < bots_arg_size; ++i)
if (array[i] != i)
success = 0;

return success ? BOTS_RESULT_SUCCESSFUL : BOTS_RESULT_UNSUCCESSFUL;
}

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <sys/resource.h>

void
bots_error(int error, char *message)
{
if (message == NULL)
{
switch(error)
{
case BOTS_ERROR:
fprintf(stderr, "Error (%d): %s\n",error,"Unspecified error.");
break;
case BOTS_ERROR_NOT_ENOUGH_MEMORY:
fprintf(stderr, "Error (%d): %s\n",error,"Not enough memory.");
break;
case BOTS_ERROR_UNRECOGNIZED_PARAMETER:
fprintf(stderr, "Error (%d): %s\n",error,"Unrecognized parameter.");
bots_print_usage();
break;
default:
fprintf(stderr, "Error (%d): %s\n",error,"Invalid error code.");
break;
}
}
else fprintf(stderr, "Error (%d): %s\n",error,message);
exit(100+error);
}

void
bots_warning(int warning, char *message)
{
if (message == NULL)
{
switch(warning)
{
case BOTS_WARNING:
fprintf(stderr, "Warning (%d): %s\n",warning,"Unspecified warning.");
break;
default:
fprintf(stderr, "Warning (%d): %s\n",warning,"Invalid warning code.");
break;
}
}
else fprintf(stderr, "Warning (%d): %s\n",warning,message);
}

long bots_usecs (void)
{
struct timeval t;
gettimeofday(&t,NULL);
return t.tv_sec*1000000+t.tv_usec;
}

void
bots_get_date(char *str)
{
time_t now;
time(&now);
strftime(str, 32, "%Y/%m/%d;%H:%M", gmtime(&now));
}

#if defined (__linux)

void bots_get_architecture(char *str)
{
int ncpus = sysconf(_SC_NPROCESSORS_CONF);
struct utsname architecture;

uname(&architecture);
sprintf(str, "%s-%s;%d" ,architecture.sysname, architecture.machine, ncpus);
}
void bots_get_load_average(char *str)
{
double loadavg[3];
getloadavg (loadavg, 3);
sprintf(str, "%.2f;%.2f;%.2f",loadavg[0],loadavg[1],loadavg[2]);
}
#else

int bots_get_max_cpus(void) { return 0; }
void bots_get_architecture(char *str) { sprintf(str,";"); } 
void bots_get_load_average(char *str) { sprintf(str,";;"); }
#endif

void bots_print_results()
{
char str_name[128];
char str_parameters[128];
char str_model[128];
char str_resources[128];
char str_result[15];
char str_time_program[15];
char str_time_sequential[15];
char str_speed_up[15];
char str_number_of_tasks[15];
char str_number_of_tasks_per_second[15];
char str_exec_date[128];
char str_exec_message[128];
char str_architecture[128];
char str_load_avg[128];
char str_comp_date[128];
char str_comp_message[128];
char str_cc[128];
char str_cflags[128];
char str_ld[128];
char str_ldflags[128];
char str_cutoff[128];


sprintf(str_name, "%s", bots_name);
sprintf(str_parameters, "%s", bots_parameters);
sprintf(str_model, "%s", bots_model);
sprintf(str_cutoff, "%s", bots_cutoff);
sprintf(str_resources, "%s", bots_resources);
switch(bots_result)
{
case BOTS_RESULT_NA: 
sprintf(str_result, "n/a");
break;
case BOTS_RESULT_SUCCESSFUL: 
sprintf(str_result, "successful");
break;
case BOTS_RESULT_UNSUCCESSFUL: 
sprintf(str_result, "UNSUCCESSFUL");
break;
case BOTS_RESULT_NOT_REQUESTED:
sprintf(str_result, "Not requested");
break;
default: 
sprintf(str_result, "error");
break;
}
sprintf(str_time_program, "%f", bots_time_program);
if (bots_sequential_flag) sprintf(str_time_sequential, "%f", bots_time_sequential);
else sprintf(str_time_sequential, "n/a");
if (bots_sequential_flag)
sprintf(str_speed_up, "%3.2f", bots_time_sequential/bots_time_program);
else sprintf(str_speed_up, "n/a");

sprintf(str_number_of_tasks, "%3.2f", (float) bots_number_of_tasks);
sprintf(str_number_of_tasks_per_second, "%3.2f", (float) bots_number_of_tasks/bots_time_program);

sprintf(str_exec_date, "%s", bots_exec_date);
sprintf(str_exec_message, "%s", bots_exec_message);
bots_get_architecture(str_architecture);
bots_get_load_average(str_load_avg);
sprintf(str_comp_date, "%s", bots_comp_date);
sprintf(str_comp_message, "%s", bots_comp_message);
sprintf(str_cc, "%s", bots_cc);
sprintf(str_cflags, "%s", bots_cflags);
sprintf(str_ld, "%s", bots_ld);
sprintf(str_ldflags, "%s", bots_ldflags);

if(bots_print_header)
{
switch(bots_output_format)
{
case 0:
break;
case 1:
break;
case 2:
fprintf(stdout,
"Benchmark;Parameters;Model;Cutoff;Resources;Result;\
Time;Sequential;Speed-up;\
Nodes;Nodes/Sec;\
Exec Date;Exec Time;Exec Message;\
Architecture;Processors;Load Avg-1;Load Avg-5;Load Avg-15;\
Comp Date;Comp Time;Comp Message;CC;CFLAGS;LD;LDFLAGS\n");
break;
case 3:
break;
case 4:
fprintf(stdout,
"Benchmark;Parameters;Model;Cutoff;Resources;Result;\
Time;Sequential;Speed-up;\
Nodes;Nodes/Sec;\n");
break;
default:
break;
}
}


switch(bots_output_format)
{
case 0:
break;
case 1:
fprintf(stdout, "\n");
fprintf(stdout, "Program             = %s\n", str_name); 
fprintf(stdout, "Parameters          = %s\n", str_parameters); 
fprintf(stdout, "Model               = %s\n", str_model); 
fprintf(stdout, "Embedded cut-off    = %s\n", str_cutoff); 
fprintf(stdout, "# of Threads        = %s\n", str_resources);
fprintf(stdout, "Verification        = %s\n", str_result);

fprintf(stdout, "Time Program        = %s seconds\n", str_time_program);
if (bots_sequential_flag) {
fprintf(stdout, "Time Sequential     = %s seconds\n", str_time_sequential);
fprintf(stdout, "Speed-up            = %s\n", str_speed_up);
}

if ( bots_number_of_tasks > 0 ) {
fprintf(stdout, "Nodes               = %s\n", str_number_of_tasks);
fprintf(stdout, "Nodes/Sec           = %s\n", str_number_of_tasks_per_second);
}

fprintf(stdout, "Execution Date      = %s\n", str_exec_date);
fprintf(stdout, "Execution Message   = %s\n", str_exec_message);

fprintf(stdout, "Architecture        = %s\n", str_architecture);
fprintf(stdout, "Load Avg [1:5:15]   = %s\n", str_load_avg);

fprintf(stdout, "Compilation Date    = %s\n", str_comp_date);
fprintf(stdout, "Compilation Message = %s\n", str_comp_message);

fprintf(stdout, "Compiler            = %s\n", str_cc);
fprintf(stdout, "Compiler Flags      = %s\n", str_cflags);
fprintf(stdout, "Linker              = %s\n", str_ld);
fprintf(stdout, "Linker Flags        = %s\n", str_ldflags);
fflush(stdout);
break;
case 2:
fprintf(stdout,"%s;%s;%s;%s;%s;%s;", 
str_name, 
str_parameters, 
str_model, 
str_cutoff, 
str_resources, 
str_result
);
fprintf(stdout,"%s;%s;%s;", 
str_time_program, 
str_time_sequential, 
str_speed_up 
);
fprintf(stdout,"%s;%s;", 
str_number_of_tasks, 
str_number_of_tasks_per_second
);
fprintf(stdout,"%s;%s;", 
str_exec_date,
str_exec_message
);
fprintf(stdout,"%s;%s;", 
str_architecture,
str_load_avg
);
fprintf(stdout,"%s;%s;", 
str_comp_date,
str_comp_message
);
fprintf(stdout,"%s;%s;%s;%s;",
str_cc,
str_cflags,
str_ld,
str_ldflags
);
fprintf(stdout,"\n");
break;
case 3:
fprintf(stdout, "\n");
fprintf(stdout, "Program             = %s\n", str_name); 
fprintf(stdout, "Parameters          = %s\n", str_parameters); 
fprintf(stdout, "Model               = %s\n", str_model); 
fprintf(stdout, "Embedded cut-off    = %s\n", str_cutoff); 
fprintf(stdout, "# of Threads        = %s\n", str_resources);
fprintf(stdout, "Verification        = %s\n", str_result);

fprintf(stdout, "Time Program        = %s seconds\n", str_time_program);
if (bots_sequential_flag) {
fprintf(stdout, "Time Sequential     = %s seconds\n", str_time_sequential);
fprintf(stdout, "Speed-up            = %s\n", str_speed_up);
}

if ( bots_number_of_tasks > 0 ) {
fprintf(stdout, "Nodes               = %s\n", str_number_of_tasks);
fprintf(stdout, "Nodes/Sec           = %s\n", str_number_of_tasks_per_second);
}
break;
case 4:
fprintf(stdout,"%s;%s;%s;%s;%s;%s;", 
str_name, 
str_parameters, 
str_model, 
str_cutoff, 
str_resources, 
str_result
);
fprintf(stdout,"%s;%s;%s;", 
str_time_program, 
str_time_sequential, 
str_speed_up 
);
fprintf(stdout,"%s;%s;", 
str_number_of_tasks, 
str_number_of_tasks_per_second
);
fprintf(stdout,"\n");
break;
default:
bots_error(BOTS_ERROR,"No valid output format\n");
break;
}
}


#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <memory.h>
#include <sys/time.h>
#include <libgen.h>


void bots_initialize();
void bots_finalize();
void bots_sequential_ini();
long bots_sequential();
void bots_sequential_fini();
int bots_check_result();
void bots_print_usage_specific();
void bots_get_params_specific(int argc, char **argv);
void bots_set_info();



int bots_sequential_flag = FALSE;
int bots_check_flag = TRUE;
bots_verbose_mode_t bots_verbose_mode = BOTS_VERBOSE_DEFAULT;
int bots_result = BOTS_RESULT_NOT_REQUESTED;
int bots_output_format = 1;
int bots_print_header = FALSE;

char bots_name[128];
char bots_execname[128];
char bots_parameters[128];
char bots_model[128];
char bots_resources[128];

char bots_exec_date[128];
char bots_exec_message[128];
char bots_comp_date[128];
char bots_comp_message[128];
char bots_cc[128];
char bots_cflags[128];
char bots_ld[128];
char bots_ldflags[128];
char bots_cutoff[128];


double bots_time_program = 0.0;
double bots_time_sequential = 0.0;
int    bots_number_of_tasks = 0;



#ifndef BOTS_APP_NAME
#error "Application name must be defined (#define BOTS_APP_NAME)"
#endif

#ifndef BOTS_APP_PARAMETERS_DESC
#define BOTS_APP_PARAMETERS_DESC ""
#endif

#ifndef BOTS_APP_PARAMETERS_LIST
#define BOTS_APP_PARAMETERS_LIST
#endif

#ifndef BOTS_APP_INIT
#define BOTS_APP_INIT
#endif

#ifndef BOTS_APP_FINI
#define BOTS_APP_FINI
#endif

#ifndef KERNEL_CALL
#error "Initial kernell call must be specified (#define KERNEL_CALL)"
#endif

#ifndef KERNEL_INIT
#define KERNEL_INIT
#endif

#ifndef KERNEL_FINI
#define KERNEL_FINI
#endif

#ifndef KERNEL_SEQ_INIT
#define KERNEL_SEQ_INIT
#endif

#ifndef KERNEL_SEQ_FINI
#define KERNEL_SEQ_FINI
#endif

#ifndef BOTS_MODEL_DESC
#define BOTS_MODEL_DESC "Unknown"
#endif

#ifdef BOTS_APP_USES_ARG_SIZE
#ifndef BOTS_APP_DEF_ARG_SIZE
#error "Default vaule for argument size must be specified (#define BOTS_APP_DEF_ARG_SIZE)"
#endif
#ifndef BOTS_APP_DESC_ARG_SIZE
#error "Help description for argument size must be specified (#define BOTS_APP_DESC_ARG_SIZE)"
#endif
int bots_arg_size = BOTS_APP_DEF_ARG_SIZE;
#endif

#ifdef BOTS_APP_USES_ARG_SIZE_1
#ifndef BOTS_APP_DEF_ARG_SIZE_1
#error "Default vaule for argument size must be specified (#define BOTS_APP_DEF_ARG_SIZE_1)"
#endif
#ifndef BOTS_APP_DESC_ARG_SIZE_1
#error "Help description for argument size must be specified (#define BOTS_APP_DESC_ARG_SIZE_1)"
#endif
int bots_arg_size_1 = BOTS_APP_DEF_ARG_SIZE_1;
#endif

#ifdef BOTS_APP_USES_ARG_SIZE_2
#ifndef BOTS_APP_DEF_ARG_SIZE_2
#error "Default vaule for argument size must be specified (#define BOTS_APP_DEF_ARG_SIZE_2)"
#endif
#ifndef BOTS_APP_DESC_ARG_SIZE_2
#error "Help description for argument size must be specified (#define BOTS_APP_DESC_ARG_SIZE_2)"
#endif
int bots_arg_size_2 = BOTS_APP_DEF_ARG_SIZE_2;
#endif

#ifdef BOTS_APP_USES_ARG_REPETITIONS
#ifndef BOTS_APP_DEF_ARG_REPETITIONS
#error "Default vaule for argument repetitions must be specified (#define BOTS_APP_DEF_ARG_REPETITIONS)"
#endif
#ifndef BOTS_APP_DESC_ARG_REPETITIONS
#error "Help description for argument repetitions must be specified (#define BOTS_APP_DESC_ARG_REPETITIONS)"
#endif
int bots_arg_repetitions = BOTS_APP_DEF_ARG_REPETITIONS;
#endif

#ifdef BOTS_APP_USES_ARG_FILE
#ifndef BOTS_APP_DESC_ARG_FILE
#error "Help description for argument file must be specified (#define BOTS_APP_DESC_ARG_FILE)"
#endif
char bots_arg_file[255]="";
#endif

#ifdef BOTS_APP_USES_ARG_BLOCK
#ifndef BOTS_APP_DEF_ARG_BLOCK
#error "Default value for argument block must be specified (#define BOTS_APP_DEF_ARG_BLOCK)"
#endif
#ifndef BOTS_APP_DESC_ARG_BLOCK
#error "Help description for argument block must be specified (#define BOTS_APP_DESC_ARG_BLOCK)"
#endif
int bots_arg_block = BOTS_APP_DEF_ARG_BLOCK;
#endif

#ifdef BOTS_APP_USES_ARG_CUTOFF
#ifndef BOTS_APP_DEF_ARG_CUTOFF
#error "Default value for argument cutoff  must be specified (#define BOTS_APP_DEF_ARG_CUTOFF)"
#endif
#ifndef BOTS_APP_DESC_ARG_CUTOFF
#error "Help description for argument cutoff must be specified (#define BOTS_APP_DESC_ARG_CUTOFF)"
#endif
int bots_app_cutoff_value = BOTS_APP_DEF_ARG_CUTOFF;
#endif

#ifdef BOTS_APP_USES_ARG_CUTOFF_1
#ifndef BOTS_APP_DEF_ARG_CUTOFF_1
#error "Default value for argument cutoff  must be specified (#define BOTS_APP_DEF_ARG_CUTOFF_1)"
#endif
#ifndef BOTS_APP_DESC_ARG_CUTOFF_1
#error "Help description for argument cutoff must be specified (#define BOTS_APP_DESC_ARG_CUTOFF_1)"
#endif
int bots_app_cutoff_value_1 = BOTS_APP_DEF_ARG_CUTOFF_1;
#endif

#ifdef BOTS_APP_USES_ARG_CUTOFF_2
#ifndef BOTS_APP_DEF_ARG_CUTOFF_2
#error "Default value for argument cutoff  must be specified (#define BOTS_APP_DEF_ARG_CUTOFF_2)"
#endif
#ifndef BOTS_APP_DESC_ARG_CUTOFF_2
#error "Help description for argument cutoff must be specified (#define BOTS_APP_DESC_ARG_CUTOFF_2)"
#endif
int bots_app_cutoff_value_2 = BOTS_APP_DEF_ARG_CUTOFF_2;
#endif

#if defined(MANUAL_CUTOFF) || defined(IF_CUTOFF) || defined(FINAL_CUTOFF)
int  bots_cutoff_value = BOTS_CUTOFF_DEF_VALUE;
#endif


void bots_print_usage()
{
fprintf(stderr, "\n");
fprintf(stderr, "Usage: %s -[options]\n", bots_execname);
fprintf(stderr, "\n");
fprintf(stderr, "Where options are:\n");
#ifdef BOTS_APP_USES_REPETITIONS
fprintf(stderr, "  -r <value> : Set the number of repetitions (default = 1).\n");
#endif
#ifdef BOTS_APP_USES_ARG_SIZE
fprintf(stderr, "  -n <size>  : "BOTS_APP_DESC_ARG_SIZE"\n");
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_1
fprintf(stderr, "  -m <size>  : "BOTS_APP_DESC_ARG_SIZE_1"\n");
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_2
fprintf(stderr, "  -l <size>  : "BOTS_APP_DESC_ARG_SIZE_2"\n");
#endif
#ifdef BOTS_APP_USES_ARG_FILE
fprintf(stderr, "  -f <file>  : "BOTS_APP_DESC_ARG_FILE"\n");
#endif
#if defined(MANUAL_CUTOFF) || defined(IF_CUTOFF) || defined(FINAL_CUTOFF)
fprintf(stderr, "  -x <value> : OpenMP tasks cut-off value (default=%d)\n",BOTS_CUTOFF_DEF_VALUE);
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF
fprintf(stderr, "  -y <value> : "BOTS_APP_DESC_ARG_CUTOFF"(default=%d)\n", BOTS_APP_DEF_ARG_CUTOFF);
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_1
fprintf(stderr, "  -a <value> : "BOTS_APP_DESC_ARG_CUTOFF_1"(default=%d)\n", BOTS_APP_DEF_ARG_CUTOFF_1);
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_2
fprintf(stderr, "  -b <value> : "BOTS_APP_DESC_ARG_CUTOFF_2"(default=%d)\n", BOTS_APP_DEF_ARG_CUTOFF_2);
#endif

fprintf(stderr, "\n");
fprintf(stderr, "  -e <str>   : Include 'str' execution message.\n");
fprintf(stderr, "  -v <level> : Set verbose level (default = 1).\n");
fprintf(stderr, "               0 - none.\n");
fprintf(stderr, "               1 - default.\n");
fprintf(stderr, "               2 - debug.\n");
fprintf(stderr, "  -o <value> : Set output format mode (default = 1).\n");
fprintf(stderr, "               0 - no benchmark output.\n");
fprintf(stderr, "               1 - detailed list format.\n");
fprintf(stderr, "               2 - detailed row format.\n");
fprintf(stderr, "               3 - abridged list format.\n");
fprintf(stderr, "               4 - abridged row format.\n");
fprintf(stderr, "  -z         : Print row header (if output format is a row variant).\n");
fprintf(stderr, "\n");
#ifdef KERNEL_SEQ_CALL
fprintf(stderr, "  -s         : Run sequential version.\n");
#endif
#ifdef BOTS_APP_CHECK_USES_SEQ_RESULT
fprintf(stderr, "  -c         : Check mode ON (implies running sequential version).\n");
#else
fprintf(stderr, "  -c         : Check mode ON.\n");
#endif
fprintf(stderr, "\n");
fprintf(stderr, "  -h         : Print program's usage (this help).\n");
fprintf(stderr, "\n");
}

void
bots_get_params_common(int argc, char **argv)
{
int i;
strcpy(bots_execname, basename(argv[0]));
bots_get_date(bots_exec_date);
strcpy(bots_exec_message,"");
for (i=1; i<argc; i++) 
{
if (argv[i][0] == '-')
{
switch (argv[i][1])
{
#ifdef BOTS_APP_USES_ARG_CUTOFF_1
case 'a':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_app_cutoff_value_1 = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF_2
case 'b':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_app_cutoff_value_2 = atoi(argv[i]);
break;
#endif
case 'c': 
argv[i][1] = '*';
bots_check_flag = TRUE;
break;
case 'e': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
strcpy(bots_exec_message, argv[i]);
break;
#ifdef BOTS_APP_USES_ARG_FILE
case 'f': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
strcpy(bots_arg_file,argv[i]);
break;
#endif
case 'h': 
argv[i][1] = '*';
bots_print_usage();
exit (100);
#ifdef BOTS_APP_USES_ARG_SIZE_2
case 'l': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_size_2 = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_SIZE_1
case 'm': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_size_1 = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_SIZE
case 'n': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_size = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_BLOCK

#endif
case 'o': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_output_format = atoi(argv[i]);
break;
#ifdef BOTS_APP_USES_REPETITIONS
case 'r': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_arg_repetition = atoi(argv[i]);
break;
#endif
#ifdef KERNEL_SEQ_CALL
case 's': 
argv[i][1] = '*';
bots_sequential_flag = TRUE;
break;
#endif
case 'v': 
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_verbose_mode = (bots_verbose_mode_t) atoi(argv[i]);
#ifndef BOTS_DEBUG
if ( bots_verbose_mode > 1 ) {
fprintf(stderr, "Error: Configure the suite using '--debug' option in order to use a verbose level greather than 1.\n");
exit(100);
}
#endif
break;
#if defined(MANUAL_CUTOFF) || defined(IF_CUTOFF) || defined(FINAL_CUTOFF)
case 'x':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_cutoff_value = atoi(argv[i]);
break;
#endif
#ifdef BOTS_APP_USES_ARG_CUTOFF
case 'y':
argv[i][1] = '*';
i++;
if (argc == i) { bots_print_usage(); exit(100); }
bots_app_cutoff_value = atoi(argv[i]);
break;
#endif
case 'z':
argv[i][1] = '*';
bots_print_header = TRUE;
break;
default:
fprintf(stderr, "Error: Unrecognized parameter.\n");
bots_print_usage();
exit (100);
}
}
else
{
fprintf(stderr, "Error: Unrecognized parameter.\n");
bots_print_usage();
exit (100);
}
}
}

void
bots_get_params(int argc, char **argv)
{
bots_get_params_common(argc, argv);
}



void bots_set_info ()
{

sprintf(bots_name,BOTS_APP_NAME);
sprintf(bots_parameters,BOTS_APP_PARAMETERS_DESC BOTS_APP_PARAMETERS_LIST);
sprintf(bots_model,BOTS_MODEL_DESC);
sprintf(bots_resources,"%d", omp_get_num_procs());


strcpy(bots_comp_date,CDATE);
strcpy(bots_comp_message,CMESSAGE);
strcpy(bots_cc,CC);
strcpy(bots_cflags,CFLAGS);
strcpy(bots_ld,LD);
strcpy(bots_ldflags,LDFLAGS);

#if defined(MANUAL_CUTOFF) 
sprintf(bots_cutoff,"manual (%d)",bots_cutoff_value);
#elif defined(IF_CUTOFF) 
sprintf(bots_cutoff,"pragma-if (%d)",bots_cutoff_value);
#elif defined(FINAL_CUTOFF)
sprintf(bots_cutoff,"final (%d)",bots_cutoff_value);
#else
strcpy(bots_cutoff,"none");
#endif
}


int
main(int argc, char* argv[])
{
#ifndef BOTS_APP_SELF_TIMING
long bots_t_start;
long bots_t_end;
#endif

bots_get_params(argc,argv);
BOTS_APP_INIT;
bots_set_info();

#ifdef KERNEL_SEQ_CALL
#ifdef BOTS_APP_CHECK_USES_SEQ_RESULT
if (bots_sequential_flag || bots_check_flag)
#else
if (bots_sequential_flag)
#endif
{
bots_sequential_flag = 1;
KERNEL_SEQ_INIT;
#ifdef BOTS_APP_SELF_TIMING
bots_time_sequential = KERNEL_SEQ_CALL;
#else
bots_t_start = bots_usecs();
KERNEL_SEQ_CALL;
bots_t_end = bots_usecs();
bots_time_sequential = ((double)(bots_t_end-bots_t_start))/1000000;
#endif
KERNEL_SEQ_FINI;
}
#endif

KERNEL_INIT;
#ifdef BOTS_APP_SELF_TIMING
bots_time_program = KERNEL_CALL;
#else
bots_t_start = bots_usecs();
KERNEL_CALL;
bots_t_end = bots_usecs();
bots_time_program = ((double)(bots_t_end-bots_t_start))/1000000;
#endif
KERNEL_FINI;

#ifdef KERNEL_CHECK
if (bots_check_flag) {
bots_result = KERNEL_CHECK;
}
#endif

BOTS_APP_FINI;

bots_print_results();
if (bots_result == BOTS_RESULT_SUCCESSFUL) { return 0; } else { return -1; }
}

