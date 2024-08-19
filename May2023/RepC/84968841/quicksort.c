#include "quicksort.h"
#include "event_generator.h"
#include "event_list.h"
#if defined(__clang__) || defined (__GNUC__)
# define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
# define ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif
void swap(event * array[], int first_index, int second_index);
void swap_valued_events(valued_event * array[], int first_index, int second_index);
void swap_valued_events_with_array(valued_event * array, int first_index, int second_index);
void sort_events(event * array[], int begin, int end)
{
int pivot, l, r;
if (end > begin) {
pivot = array[begin]->timestamp;
l = begin + 1;
r = end+1;
while(l < r)
if (array[l]->timestamp < pivot)
l++;
else {
r--;
swap(array,l,r);
}
l--;
swap(array,begin,l);
#pragma omp parallel sections shared(array,pivot,l,r,begin,end)
{
#pragma omp section
sort_events(array, begin, l);
#pragma omp section
sort_events(array, r, end);
}
}
}
void sort_valued_events(valued_event * array[], int begin, int end)
{
int pivot, l, r;
if (end > begin) {
pivot = array[begin]->valued_event_ts;
l = begin + 1;
r = end+1;
while(l < r)
if (array[l]->valued_event_ts < pivot)
l++;
else {
r--;
swap_valued_events(array,l,r);
}
l--;
swap_valued_events(array,begin,l);
#pragma omp parallel sections shared(array,begin,end,pivot,l,r)
{
#pragma omp section
sort_valued_events(array, begin, l);
#pragma omp section
sort_valued_events(array, r, end);
}
}
}
void sort_valued_events_on_score_with_array(valued_event * array, int begin, int end)
{
int pivot_score, pivot_lc_ts, l, r, pivot_ve_ts;
if (end > begin) {
pivot_score = array[begin].score;
pivot_ve_ts = array[begin].valued_event_ts;
pivot_lc_ts = array[begin].last_comment_ts;
l = begin + 1;
r = end+1;
while(l < r)
if (array[l].score > pivot_score ||
(array[l].score==pivot_score && array[l].valued_event_ts < pivot_ve_ts) ||
(array[l].score==pivot_score && array[l].valued_event_ts == pivot_ve_ts && array[l].last_comment_ts < pivot_lc_ts) )
l++;
else {
r--;
swap_valued_events_with_array(array,l,r);
}
l--;
swap_valued_events_with_array(array,begin,l);
sort_valued_events_on_score_with_array(array, begin, l);
sort_valued_events_on_score_with_array(array, r, end);
}
}
void swap(event * array[], int first_index, int second_index)
{
event * temp = array[first_index];
array[first_index] = array[second_index];
array[second_index] = temp;
}
void swap_valued_events(valued_event * array[], int first_index, int second_index)
{
valued_event * temp = array[first_index];
array[first_index] = array[second_index];
array[second_index] = temp;
}
void swap_valued_events_with_array(valued_event * array, int first_index, int second_index)
{
valued_event temp = array[first_index];
array[first_index] = array[second_index];
array[second_index] = temp;
}
