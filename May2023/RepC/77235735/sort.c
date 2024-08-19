#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bots.h"
#include "app-desc.h"
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
#pragma omp task untied
cilkmerge_par(low1, split1 - 1, low2, split2, lowdest);
#pragma omp task untied
cilkmerge_par(split1 + 1, high1, split2 + 1, high2,
lowdest + lowsize + 2);
#pragma omp taskwait
return;
}
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
#pragma omp task untied
cilksort_par(A, tmpA, quarter);
#pragma omp task untied
cilksort_par(B, tmpB, quarter);
#pragma omp task untied
cilksort_par(C, tmpC, quarter);
#pragma omp task untied
cilksort_par(D, tmpD, size - 3 * quarter);
#pragma omp taskwait
#pragma omp task untied
cilkmerge_par(A, A + quarter - 1, B, B + quarter - 1, tmpA);
#pragma omp task untied
cilkmerge_par(C, C + quarter - 1, D, low + size - 1, tmpC);
#pragma omp taskwait
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
#pragma omp parallel
#pragma omp single nowait
#pragma omp task untied
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
