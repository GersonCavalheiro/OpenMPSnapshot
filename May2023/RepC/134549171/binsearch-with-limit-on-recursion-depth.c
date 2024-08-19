#include<stdlib.h>
#include<stdio.h>
#include <string.h>
#include <omp.h>
#define LIMIT  3 
void check_solution(int *state);
void bin_search (int pos, int n, int *state)
{
if ( pos == n ) {
return;
}
#pragma omp task final( pos > LIMIT ) mergeable
{
int new_state[n];
if (!omp_in_final() ) {
memcpy(new_state, state, pos );
state = new_state;
}
state[pos] = 0;
bin_search(pos+1, n, state );
}
#pragma omp task final( pos > LIMIT ) mergeable
{
int new_state[n];
if (! omp_in_final() ) {
memcpy(new_state, state, pos );
state = new_state;
}
state[pos] = 1;
bin_search(pos+1, n, state );
}
#pragma omp taskwait
}
int main()
{
int arr[] = {2, 3, 4, 10, 40}; 
int n = sizeof(arr)/ sizeof(arr[0]); 
int x = 10; 
bin_search(0, n, arr);
return 0;
}
