#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <argp.h>
#include <stdbool.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include "fixedPoint.h"
#include "timer.h"
#include "myMalloc.h"
#include "graphConfig.h"
#include "graphCSR.h"
#include "pageRank.h"
#include "graphStats.h"
static int min3(int a, int b, int c)
{
if (a < b && a < c)
{
return a;
}
if (b < a && b < c)
{
return b;
}
return c;
}
static uint32_t levenshtein_matrix_calculate(edit **mat, const uint32_t *array1, uint32_t len1,
const uint32_t *array2, uint32_t len2)
{
uint32_t i, j;
for (j = 1; j <= len2; j++)
{
for (i = 1; i <= len1; i++)
{
uint32_t substitution_cost;
uint32_t del = 0, ins = 0, subst = 0;
uint32_t best;
if (array1[i - 1] == array2[j - 1])
{
substitution_cost = 0;
}
else
{
substitution_cost = 1;
}
del = mat[i - 1][j].score + 1; 
ins = mat[i][j - 1].score + 1; 
subst = mat[i - 1][j - 1].score + substitution_cost; 
best = min3(del, ins, subst);
mat[i][j].score = best;
mat[i][j].arg1 = array1[i - 1];
mat[i][j].arg2 = array2[j - 1];
mat[i][j].pos = i - 1;
if (best == del)
{
mat[i][j].type = DELETION;
mat[i][j].prev = &mat[i - 1][j];
}
else if (best == ins)
{
mat[i][j].type = INSERTION;
mat[i][j].prev = &mat[i][j - 1];
}
else
{
if (substitution_cost > 0)
{
mat[i][j].type = SUBSTITUTION;
}
else
{
mat[i][j].type = NONE;
}
mat[i][j].prev = &mat[i - 1][j - 1];
}
}
}
return mat[len1][len2].score;
}
static edit **levenshtein_matrix_create(const uint32_t *array1, uint32_t len1, const uint32_t *array2,
uint32_t len2)
{
uint32_t i, j;
edit **mat = malloc((len1 + 1) * sizeof(edit *));
if (mat == NULL)
{
return NULL;
}
for (i = 0; i <= len1; i++)
{
mat[i] = malloc((len2 + 1) * sizeof(edit));
if (mat[i] == NULL)
{
for (j = 0; j < i; j++)
{
free(mat[j]);
}
free(mat);
return NULL;
}
}
for (i = 0; i <= len1; i++)
{
mat[i][0].score = i;
mat[i][0].prev = NULL;
mat[i][0].arg1 = 0;
mat[i][0].arg2 = 0;
}
for (j = 0; j <= len2; j++)
{
mat[0][j].score = j;
mat[0][j].prev = NULL;
mat[0][j].arg1 = 0;
mat[0][j].arg2 = 0;
}
return mat;
}
uint32_t levenshtein_distance(const uint32_t *array1, const uint32_t len1, const uint32_t *array2, const uint32_t len2, edit **script)
{
uint32_t i, distance;
edit **mat, *head;
if (len1 == 0)
{
return len2;
}
if (len2 == 0)
{
return len1;
}
mat = levenshtein_matrix_create(array1, len1, array2, len2);
if (!mat)
{
*script = NULL;
return 0;
}
distance = levenshtein_matrix_calculate(mat, array1, len1, array2, len2);
*script = malloc(distance * sizeof(edit));
if (*script)
{
i = distance - 1;
for (head = &mat[len1][len2];
head->prev != NULL;
head = head->prev)
{
if (head->type != NONE)
{
memcpy(*script + i, head, sizeof(edit));
i--;
}
}
}
else
{
distance = 0;
}
for (i = 0; i <= len1; i++)
{
free(mat[i]);
}
free(mat);
return distance;
}
void print(const edit *e)
{
if (e->type == INSERTION)
{
printf("Insert %u", e->arg2);
}
else if (e->type == DELETION)
{
printf("Delete %u", e->arg1);
}
else
{
printf("Substitute %u for %u", e->arg2, e->arg1);
}
printf(" at %u\n", e->pos);
}
static int insertionSort(float *arr, int len)
{
int maxJ, i, j, swapCount = 0;
if(len < 2)
{
return 0;
}
maxJ = len - 1;
for(i = len - 2; i >= 0; --i)
{
float  val = arr[i];
for(j = i; j < maxJ && arr[j + 1] < val; ++j)
{
arr[j] = arr[j + 1];
}
arr[j] = val;
swapCount += (j - i);
}
return swapCount;
}
static int merge(float *from, float *to, int middle, int len)
{
int bufIndex, leftLen, rightLen, swaps ;
float *left, *right;
bufIndex = 0;
swaps = 0;
left = from;
right = from + middle;
rightLen = len - middle;
leftLen = middle;
while(leftLen && rightLen)
{
if(right[0] < left[0])
{
to[bufIndex] = right[0];
swaps += leftLen;
rightLen--;
right++;
}
else
{
to[bufIndex] = left[0];
leftLen--;
left++;
}
bufIndex++;
}
if(leftLen)
{
#pragma omp critical (MEMCPY)
memcpy(to + bufIndex, left, leftLen * sizeof(float));
}
else if(rightLen)
{
#pragma omp critical (MEMCPY)
memcpy(to + bufIndex, right, rightLen * sizeof(float));
}
return swaps;
}
static int mergeSort(float *x, float *buf, int len)
{
int swaps, half ;
if(len < 10)
{
return insertionSort(x, len);
}
swaps = 0;
if(len < 2)
{
return 0;
}
half = len / 2;
swaps += mergeSort(x, buf, half);
swaps += mergeSort(x + half, buf + half, len - half);
swaps += merge(x, buf, half, len);
#pragma omp critical (MEMCPY)
memcpy(x, buf, len * sizeof(float));
return swaps;
}
static int getMs(float *data, int len)  
{
int Ms = 0, tieCount = 0, i ;
for(i = 1; i < len; i++)
{
if(data[i] == data[i - 1])
{
tieCount++;
}
else if(tieCount)
{
Ms += (tieCount * (tieCount + 1)) / 2;
tieCount = 0;
}
}
if(tieCount)
{
Ms += (tieCount * (tieCount + 1)) / 2;
}
return Ms;
}
float kendallNlogN( float *arr1, float *arr2, int len )
{
int m1 = 0, m2 = 0, tieCount, swapCount, nPair, s, i ;
float cor ;
if( len < 2 ) return (float)0 ;
nPair = len * (len - 1) / 2;
s = nPair;
tieCount = 0;
for(i = 1; i < len; i++)
{
if(arr1[i - 1] == arr1[i])
{
tieCount++;
}
else if(tieCount > 0)
{
insertionSort(arr2 + i - tieCount - 1, tieCount + 1);
m1 += tieCount * (tieCount + 1) / 2;
s += getMs(arr2 + i - tieCount - 1, tieCount + 1);
tieCount = 0;
}
}
if(tieCount > 0)
{
insertionSort(arr2 + i - tieCount - 1, tieCount + 1);
m1 += tieCount * (tieCount + 1) / 2;
s += getMs(arr2 + i - tieCount - 1, tieCount + 1);
}
swapCount = mergeSort(arr2, arr1, len);
m2 = getMs(arr2, len);
s -= (m1 + m2) + 2 * swapCount;
if( m1 < nPair && m2 < nPair )
cor = s / ( sqrtf((float)(nPair - m1)) * sqrtf((float)(nPair - m2)) ) ;
else
cor = 0.0f ;
return cor ;
}
float kendallSmallN( float *arr1, float *arr2, int len )
{
int m1 = 0, m2 = 0, s = 0, nPair, i, j ;
float cor ;
for(i = 0; i < len; i++)
{
for(j = i + 1; j < len; j++)
{
if(arr2[i] > arr2[j])
{
if (arr1[i] > arr1[j])
{
s++;
}
else if(arr1[i] < arr1[j])
{
s--;
}
else
{
m1++;
}
}
else if(arr2[i] < arr2[j])
{
if (arr1[i] > arr1[j])
{
s--;
}
else if(arr1[i] < arr1[j])
{
s++;
}
else
{
m1++;
}
}
else
{
m2++;
if(arr1[i] == arr1[j])
{
m1++;
}
}
}
}
nPair = len * (len - 1) / 2;
if( m1 < nPair && m2 < nPair )
cor = s / ( sqrtf((float)(nPair - m1)) * sqrtf((float)(nPair - m2)) ) ;
else
cor = 0.0f ;
return cor ;
}
void rvereseArray(uint32_t *arr, uint32_t start, uint32_t end)
{
while (start < end)
{
int temp = arr[start];
arr[start] = arr[end];
arr[end] = temp;
start++;
end--;
}
}
uint32_t levenshtein_distance_topK(uint32_t *array1, uint32_t *array2, uint32_t size_k)
{
edit *script;
uint32_t distance;
distance = levenshtein_distance(array1, size_k, array2, size_k, &script);
free(script);
return distance;
}
uint32_t avg_mismatch_ranks_real_topK(uint32_t *array1, uint32_t *array2, uint32_t *array3, uint32_t size_k, uint32_t topk)
{
uint32_t v;
uint32_t mismatch = 0;
if(topk > size_k)
topk = size_k;
for(v = size_k - topk; v < size_k; v++)
{
if(array2[array3[v]] != array1[array3[v]])
mismatch++;
}
return mismatch ;
}
double avg_error_ranks_real_topK(uint32_t *array1, uint32_t *array2, uint32_t *array3, uint32_t size_k, uint32_t topk)
{
uint32_t v;
double error = 0.0f;
if(topk > size_k)
topk = size_k;
for(v = size_k - topk; v < size_k; v++)
{
error += abs(array2[array3[v]] - array1[array3[v]])/((double)array1[array3[v]]+1);
}
return error / topk;
}
double avg_error_ranks_float_topK(float *array1, float *array2, uint32_t *array3, uint32_t size_k, uint32_t topk)
{
uint32_t v;
double error = 0.0f;
if(topk > size_k)
topk = size_k;
for(v = size_k - topk; v < size_k; v++)
{
if((double)array1[array3[v]] > 0.0f)
error += fabs((double)array2[array3[v]] - (double)array1[array3[v]])/(double)array1[array3[v]];
}
return error / topk;
}
uint32_t intersection_topK(uint32_t *array1, uint32_t *array2, uint32_t size_k, uint32_t topk)
{
uint32_t v;
uint32_t intersection = 0;
if(topk > size_k)
topk = size_k;
for(v = size_k - topk; v < size_k; v++)
{
if(array2[array1[v]] >= size_k - topk)
intersection++;
}
return intersection;
}
struct PageRankCorrelationStats collectStatsPageRank_topK(struct PageRankStats *ref_stats, struct PageRankStats *stats, uint32_t *ref_rankedVertices_total, uint32_t *ref_rankedVertices_inverse, uint32_t *rankedVertices_inverse,  uint32_t topk, uint32_t num_vertices, FILE *fptr, uint32_t verbose)
{
uint32_t v;
uint32_t u;
struct PageRankCorrelationStats pageRankCorrelationStats;
if(topk > num_vertices)
topk = num_vertices;
uint32_t *rankedVertices = (uint32_t *) my_malloc(topk * sizeof(uint32_t));
uint32_t *ref_rankedVertices = (uint32_t *) my_malloc(topk * sizeof(uint32_t));
float *rankedVerticesfloat = (float *) my_malloc(topk * sizeof(float));
float *ref_rankedVerticesfloat = (float *) my_malloc(topk * sizeof(float));
float *rankedVerticesReal = (float *) my_malloc(topk * sizeof(float));
float *ref_rankedVerticesReal = (float *) my_malloc(topk * sizeof(float));
uint32_t levenshtein_distance = 0;
float float_Kendall   = 0.0f;
float real_Kendall    = 0.0f;
uint32_t intersection = 0;
uint32_t mismatch = 0;
double avg_error_float = 0.0f;
double avg_error_relative = 0.0f;
for(u = 0, v = (num_vertices - topk); v < num_vertices; v++, u++)
{
rankedVertices[u] =  stats->realRanks[v];
rankedVerticesfloat[u] =  stats->pageRanks[stats->realRanks[v]];
rankedVerticesReal[u]  =  (float)stats->realRanks[v] / (float)1.0;
}
for(u = 0, v = (num_vertices - topk); v < num_vertices; v++, u++)
{
ref_rankedVertices[u] = ref_stats->realRanks[v];
ref_rankedVerticesfloat[u] =  ref_stats->pageRanks[stats->realRanks[v]];
ref_rankedVerticesReal[u]  =   (float)ref_stats->realRanks[v] / (float)1.0;
}
levenshtein_distance = levenshtein_distance_topK(ref_rankedVertices, rankedVertices, topk);
float_Kendall = kendallSmallN(ref_rankedVerticesfloat, rankedVerticesfloat, topk);
real_Kendall = kendallSmallN(ref_rankedVerticesReal, rankedVerticesReal, topk);
intersection = intersection_topK(ref_rankedVertices_total, rankedVertices_inverse, num_vertices, topk);
avg_error_float = avg_error_ranks_float_topK(ref_stats->pageRanks, stats->pageRanks, ref_stats->realRanks, num_vertices, topk);
avg_error_relative = avg_error_ranks_real_topK(ref_rankedVertices_inverse, rankedVertices_inverse, ref_stats->realRanks, num_vertices, topk);
mismatch = avg_mismatch_ranks_real_topK(ref_rankedVertices_inverse, rankedVertices_inverse, ref_stats->realRanks, num_vertices, topk);
if(verbose > 0)
{
fprintf(stdout, "\n-----------------------------------------------------\n");
fprintf(stdout, "topk:         %u \n", topk);
fprintf(stdout, "-----------------------------------------------------\n");
fprintf(stdout, "levenshtein_distance: %u \n", levenshtein_distance);
fprintf(stdout, "Rank float Kendall:   %lf\n", float_Kendall);
fprintf(stdout, "Rank real  Kendall:   %lf\n", real_Kendall);
fprintf(stdout, "intersection:         %u \n", intersection);
fprintf(stdout, "mismatch:             %u \n", mismatch);
fprintf(stdout, "avg_error_float:      %.22lf\n", avg_error_float);
fprintf(stdout, "avg_error_relative:   %.22lf\n", avg_error_relative);
fprintf(stdout, "-----------------------------------------------------\n");
if(fptr)
{
fprintf(fptr, "\n-----------------------------------------------------\n");
fprintf(fptr, "topk:         %u \n", topk);
fprintf(fptr, "-----------------------------------------------------\n");
fprintf(fptr, "levenshtein_distance: %u \n", levenshtein_distance);
fprintf(fptr, "Rank float Kendall:   %lf\n", float_Kendall);
fprintf(fptr, "Rank real  Kendall:   %lf\n", real_Kendall);
fprintf(fptr, "intersection:         %u \n", intersection);
fprintf(fptr, "mismatch:             %u \n", mismatch);
fprintf(fptr, "avg_error_float:      %lf\n", avg_error_float);
fprintf(fptr, "avg_error_relative:   %lf\n", avg_error_relative);
fprintf(fptr, "-----------------------------------------------------\n");
}
}
pageRankCorrelationStats.levenshtein_distance = levenshtein_distance;
pageRankCorrelationStats.float_Kendall = float_Kendall;
pageRankCorrelationStats.real_Kendall = real_Kendall;
pageRankCorrelationStats.intersection = intersection;
pageRankCorrelationStats.mismatch = mismatch;
pageRankCorrelationStats.avg_error_float = avg_error_float;
pageRankCorrelationStats.avg_error_relative = avg_error_relative;
free(rankedVertices);
free(ref_rankedVertices);
free(rankedVerticesfloat);
free(ref_rankedVerticesfloat);
free(rankedVerticesReal);
free(ref_rankedVerticesReal);
return pageRankCorrelationStats;
}
void collectStatsPageRank( struct Arguments *arguments,   struct PageRankStats *ref_stats, struct PageRankStats *stats, uint32_t trial)
{
uint32_t v;
uint32_t topk;
uint32_t x;
uint32_t chunk_x;
uint32_t chunk_num;
uint32_t *rankedVertices_inverse = (uint32_t *) my_malloc(ref_stats->num_vertices * sizeof(uint32_t));
uint32_t *ref_rankedVertices_inverse = (uint32_t *) my_malloc(ref_stats->num_vertices * sizeof(uint32_t));
uint32_t *ref_rankedVertices_total = (uint32_t *) my_malloc(ref_stats->num_vertices * sizeof(uint32_t));
uint32_t topK_array_size = 6;
uint32_t topK_array[] = {30, 100, 300, 1000, 5000, 10000} ;
struct PageRankCorrelationStats pageRankCorrelationStats_array[6];
struct PageRankCorrelationStats pageRankCorrelationStats;
struct PageRankCorrelationStats pageRankCorrelationStatsAvg;
struct PageRankCorrelationStats pageRankCorrelationStatsSum;
pageRankCorrelationStats.levenshtein_distance = 0;
pageRankCorrelationStats.float_Kendall = 0.0f;
pageRankCorrelationStats.real_Kendall = 0.0f;
pageRankCorrelationStats.intersection = 0;
pageRankCorrelationStats.mismatch = 0;
pageRankCorrelationStats.avg_error_float = 0.0f;
pageRankCorrelationStats.avg_error_relative = 0.0f;
pageRankCorrelationStatsSum.levenshtein_distance = 0;
pageRankCorrelationStatsSum.float_Kendall = 0.0f;
pageRankCorrelationStatsSum.real_Kendall = 0.0f;
pageRankCorrelationStatsSum.intersection = 0;
pageRankCorrelationStatsSum.mismatch = 0;
pageRankCorrelationStatsSum.avg_error_float = 0.0f;
pageRankCorrelationStatsSum.avg_error_relative = 0.0f;
pageRankCorrelationStatsAvg.levenshtein_distance = 0;
pageRankCorrelationStatsAvg.float_Kendall = 0.0f;
pageRankCorrelationStatsAvg.real_Kendall = 0.0f;
pageRankCorrelationStatsAvg.intersection = 0;
pageRankCorrelationStatsAvg.mismatch = 0;
pageRankCorrelationStatsAvg.avg_error_float = 0.0f;
pageRankCorrelationStatsAvg.avg_error_relative = 0.0f;
char *fname_txt = (char *) malloc((strlen(arguments->fnameb) + 50) * sizeof(char));
sprintf(fname_txt, "%s_%d_%d_%d_%d.%s", arguments->fnameb, arguments->algorithm, arguments->datastructure, trial, arguments->pushpull, "stats");
FILE *fptr;
fptr = fopen(fname_txt, "a+");
topk = arguments->binSize;
if(topk > ref_stats->num_vertices)
topk = ref_stats->num_vertices;
for(v = 0; v < stats->num_vertices; v++)
{
rankedVertices_inverse[stats->realRanks[v]] = v;
ref_rankedVertices_inverse[ref_stats->realRanks[v]] = v;
ref_rankedVertices_total[v] = ref_stats->realRanks[v];
}
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
pageRankCorrelationStats = collectStatsPageRank_topK(ref_stats, stats, ref_rankedVertices_total, ref_rankedVertices_inverse, rankedVertices_inverse, topK_array[x], ref_stats->num_vertices, fptr, 0);
pageRankCorrelationStats_array[x] = pageRankCorrelationStats;
}
if(arguments->verbosity > 0)
{
fprintf(stdout, "----------------------------------------------------------------------------------------------------------\n");
fprintf(stdout, "Top K                 ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14u ",  topK_array[x]);
}
fprintf(stdout, "\n");
fprintf(stdout, "----------------------------------------------------------------------------------------------------------\n");
fprintf(stdout, "levenshtein_distance  ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14u ",  pageRankCorrelationStats_array[x].levenshtein_distance);
}
fprintf(stdout, "\n");
fprintf(stdout, "Rank float Kendall    ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14lf ",  pageRankCorrelationStats_array[x].float_Kendall);
}
fprintf(stdout, "\n");
fprintf(stdout, "Rank real  Kendall    ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14lf ",  pageRankCorrelationStats_array[x].real_Kendall);
}
fprintf(stdout, "\n");
fprintf(stdout, "intersection          ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14u ",  pageRankCorrelationStats_array[x].intersection);
}
fprintf(stdout, "\n");
fprintf(stdout, "mismatch              ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14u ",  pageRankCorrelationStats_array[x].mismatch);
}
fprintf(stdout, "\n");
fprintf(stdout, "avg_error_float       ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14lf ",  pageRankCorrelationStats_array[x].avg_error_float);
}
fprintf(stdout, "\n");
fprintf(stdout, "avg_error_relative    ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(stdout, "%-14lf ",  pageRankCorrelationStats_array[x].avg_error_relative);
}
fprintf(stdout, "\n");
fprintf(stdout, "----------------------------------------------------------------------------------------------------------\n");
fprintf(fptr, "----------------------------------------------------------------------------------------------------------\n");
fprintf(fptr, "Top K                 ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14u ",  topK_array[x]);
}
fprintf(fptr, "\n");
fprintf(fptr, "----------------------------------------------------------------------------------------------------------\n");
fprintf(fptr, "levenshtein_distance  ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14u ",  pageRankCorrelationStats_array[x].levenshtein_distance);
}
fprintf(fptr, "\n");
fprintf(fptr, "Rank float Kendall    ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14lf ",  pageRankCorrelationStats_array[x].float_Kendall);
}
fprintf(fptr, "\n");
fprintf(fptr, "Rank real  Kendall    ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14lf ",  pageRankCorrelationStats_array[x].real_Kendall);
}
fprintf(fptr, "\n");
fprintf(fptr, "intersection          ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14u ",  pageRankCorrelationStats_array[x].intersection);
}
fprintf(fptr, "\n");
fprintf(fptr, "mismatch              ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14u ",  pageRankCorrelationStats_array[x].mismatch);
}
fprintf(fptr, "\n");
fprintf(fptr, "avg_error_float       ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14lf ",  pageRankCorrelationStats_array[x].avg_error_float);
}
fprintf(fptr, "\n");
fprintf(fptr, "avg_error_relative    ");
for (x = 0; x < topK_array_size; ++x)
{
if(ref_stats->num_vertices < topK_array[x])
break;
fprintf(fptr, "%-14lf ",  pageRankCorrelationStats_array[x].avg_error_relative);
}
fprintf(fptr, "\n");
fprintf(fptr, "----------------------------------------------------------------------------------------------------------\n");
}
chunk_x   = 1000;
chunk_num = (ref_stats->num_vertices + chunk_x - 1) / chunk_x;
if(arguments->verbosity > 1)
{
if(chunk_num == 1)
{
chunk_num = 1;
chunk_x = ref_stats->num_vertices;
pageRankCorrelationStats = collectStatsPageRank_topK(ref_stats, stats, ref_rankedVertices_total, ref_rankedVertices_inverse, rankedVertices_inverse, chunk_x, ref_stats->num_vertices, fptr, 1);
pageRankCorrelationStatsSum.levenshtein_distance += pageRankCorrelationStats.levenshtein_distance;
pageRankCorrelationStatsSum.float_Kendall += pageRankCorrelationStats.float_Kendall;
pageRankCorrelationStatsSum.real_Kendall += pageRankCorrelationStats.real_Kendall;
pageRankCorrelationStatsSum.intersection += pageRankCorrelationStats.intersection;
pageRankCorrelationStatsSum.mismatch += pageRankCorrelationStats.mismatch;
pageRankCorrelationStatsSum.avg_error_float += pageRankCorrelationStats.avg_error_float;
pageRankCorrelationStatsSum.avg_error_relative += pageRankCorrelationStats.avg_error_relative;
}
else
{
for(x = 0; x < chunk_num; x++)
{
pageRankCorrelationStats = collectStatsPageRank_topK(ref_stats, stats, ref_rankedVertices_total, ref_rankedVertices_inverse, rankedVertices_inverse, chunk_x, (ref_stats->num_vertices - (chunk_x * x)), fptr, 0);
pageRankCorrelationStatsSum.levenshtein_distance += pageRankCorrelationStats.levenshtein_distance;
pageRankCorrelationStatsSum.float_Kendall += pageRankCorrelationStats.float_Kendall;
pageRankCorrelationStatsSum.real_Kendall += pageRankCorrelationStats.real_Kendall;
pageRankCorrelationStatsSum.intersection += pageRankCorrelationStats.intersection;
pageRankCorrelationStatsSum.mismatch += pageRankCorrelationStats.mismatch;
pageRankCorrelationStatsSum.avg_error_float += pageRankCorrelationStats.avg_error_float;
pageRankCorrelationStatsSum.avg_error_relative += pageRankCorrelationStats.avg_error_relative;
}
}
}
pageRankCorrelationStatsAvg.levenshtein_distance = pageRankCorrelationStatsSum.levenshtein_distance / chunk_num;
pageRankCorrelationStatsAvg.float_Kendall = pageRankCorrelationStatsSum.float_Kendall / chunk_num;
pageRankCorrelationStatsAvg.real_Kendall = pageRankCorrelationStatsSum.real_Kendall / chunk_num;
pageRankCorrelationStatsAvg.intersection = pageRankCorrelationStatsSum.intersection / chunk_num;
pageRankCorrelationStatsAvg.mismatch = pageRankCorrelationStatsSum.mismatch / chunk_num;
pageRankCorrelationStatsAvg.avg_error_float = pageRankCorrelationStatsSum.avg_error_float / chunk_num;
pageRankCorrelationStatsAvg.avg_error_relative = pageRankCorrelationStatsSum.avg_error_relative / chunk_num;
pageRankCorrelationStats = collectStatsPageRank_topK(ref_stats, stats, ref_rankedVertices_total, ref_rankedVertices_inverse, rankedVertices_inverse, topk, ref_stats->num_vertices, fptr, 1);
fprintf(stdout, "-----------------------------------------------------\n");
fprintf(stdout, "Avg (Sum(bin)*n)/n:    (Sum(%u)*%u)/%u \n", chunk_x, chunk_num, chunk_num);
fprintf(stdout, "-----------------------------------------------------\n");
fprintf(stdout, "levenshtein_distance: %u \n", pageRankCorrelationStatsAvg.levenshtein_distance);
fprintf(stdout, "Rank float Kendall:   %lf\n", pageRankCorrelationStatsAvg.float_Kendall);
fprintf(stdout, "Rank real  Kendall:   %lf\n", pageRankCorrelationStatsAvg.real_Kendall);
fprintf(stdout, "intersection:         %u \n", pageRankCorrelationStatsAvg.intersection);
fprintf(stdout, "mismatch:             %u \n", pageRankCorrelationStatsAvg.mismatch);
fprintf(stdout, "avg_error_float:      %lf\n", pageRankCorrelationStatsAvg.avg_error_float);
fprintf(stdout, "avg_error_relative:   %lf\n", pageRankCorrelationStatsAvg.avg_error_relative);
fprintf(stdout, "-----------------------------------------------------\n");
fprintf(stdout, "-----------------------------------------------------\n");
fprintf(stdout, "numThreads:           %u \n", arguments->pre_numThreads);
fprintf(stdout, "Time (S):             %lf\n", stats->time_total);
fprintf(stdout, "Iterations:           %u \n", stats->iterations);
fprintf(stdout, "-----------------------------------------------------\n");
if(arguments->verbosity > 0)
{
fprintf(fptr, "-----------------------------------------------------\n");
fprintf(fptr, "Avg (Sum_n(bins))/n:    (Sum_%u(%u)/%u \n", chunk_num, chunk_x, chunk_num);
fprintf(fptr, "-----------------------------------------------------\n");
fprintf(fptr, "levenshtein_distance: %u \n", pageRankCorrelationStatsAvg.levenshtein_distance);
fprintf(fptr, "Rank float Kendall:   %lf\n", pageRankCorrelationStatsAvg.float_Kendall);
fprintf(fptr, "Rank real  Kendall:   %lf\n", pageRankCorrelationStatsAvg.real_Kendall);
fprintf(fptr, "intersection:         %u \n", pageRankCorrelationStatsAvg.intersection);
fprintf(fptr, "mismatch:             %u \n", pageRankCorrelationStatsAvg.mismatch);
fprintf(fptr, "avg_error_float:      %lf\n", pageRankCorrelationStatsAvg.avg_error_float);
fprintf(fptr, "avg_error_relative:   %lf\n", pageRankCorrelationStatsAvg.avg_error_relative);
fprintf(fptr, "-----------------------------------------------------\n");
fprintf(fptr, "-----------------------------------------------------\n");
fprintf(fptr, "numThreads:           %u \n", arguments->pre_numThreads);
fprintf(fptr, "Time (S):             %lf\n", stats->time_total);
fprintf(fptr, "Iterations:           %u \n", stats->iterations);
fprintf(fptr, "-----------------------------------------------------\n");
if(arguments->verbosity > 2)
{
fprintf(fptr, " ----------------------------------------------------- ");
fprintf(fptr, " -----------------------------------------------------\n");
fprintf(fptr, "| %-14s | %-14s | %-17s | ", "Ref Rank", "Vertex", "PageRank");
fprintf(fptr, "| %-14s | %-14s | %-17s | \n", "Rank", "Vertex", "PageRank");
fprintf(fptr, " ----------------------------------------------------- ");
fprintf(fptr, " -----------------------------------------------------\n");
for(v = (ref_stats->num_vertices - topk); v < ref_stats->num_vertices; v++)
{
fprintf(fptr, "| %-14u | %-14u | %-10.15lf | ", v, ref_stats->realRanks[v], ref_stats->pageRanks[ref_stats->realRanks[v]]);
fprintf(fptr, "| %-14u | %-14u | %-10.15lf | \n", v, stats->realRanks[v], stats->pageRanks[stats->realRanks[v]]);
}
fprintf(fptr, " ----------------------------------------------------- ");
fprintf(fptr, " -----------------------------------------------------\n");
}
}
fclose(fptr);
free(fname_txt);
free(ref_rankedVertices_total);
free(rankedVertices_inverse);
free(ref_rankedVertices_inverse);
}
