

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>

#include "omp.h"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000


namespace utils {
int N; 
int *mat; 

void abort_with_error_message(string msg) {
std::cerr << msg << endl;
abort();
}

int convert_dimension_2D_1D(int x, int y, int n) {
return x * n + y;
}

int read_file(string filename) {
std::ifstream inputf(filename, std::ifstream::in);
if (!inputf.good()) {
abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
}
inputf >> N;
assert(N < (1024 * 1024 * 20));
mat = (int *) malloc(N * N * sizeof(int));
for (int i = 0; i < N; i++)
for (int j = 0; j < N; j++) {
inputf >> mat[convert_dimension_2D_1D(i, j, N)];
}
return 0;
}

int print_result(bool has_negative_cycle, int *dist) {
std::ofstream outputf("output.txt", std::ofstream::out);
if (!has_negative_cycle) {
for (int i = 0; i < N; i++) {
if (dist[i] > INF)
dist[i] = INF;
outputf << dist[i] << '\n';
}
outputf.flush();
} else {
outputf << "FOUND NEGATIVE CYCLE!" << endl;
}
outputf.close();
return 0;
}
}



void bellman_ford(int p, int n, int *mat, int *dist, bool *has_negative_cycle) {

int local_start[p], local_end[p];
*has_negative_cycle = false;

omp_set_num_threads(p);

int ave = n / p;
#pragma omp parallel for
for (int i = 0; i < p; i++) {
local_start[i] = ave * i;
local_end[i] = ave * (i + 1);
if (i == p - 1) {
local_end[i] = n;
}
}

#pragma omp parallel for
for (int i = 0; i < n; i++) {
dist[i] = INF;
}
dist[0] = 0;

int iter_num = 0;
bool has_change;
bool local_has_change[p];
#pragma omp parallel
{
int my_rank = omp_get_thread_num();
for (int iter = 0; iter < n - 1; iter++) {
local_has_change[my_rank] = false;
for (int u = 0; u < n; u++) {
for (int v = local_start[my_rank]; v < local_end[my_rank]; v++) {
int weight = mat[utils::convert_dimension_2D_1D(u, v, n)];
if (weight < INF) {
int new_dis = dist[u] + weight;
if (new_dis < dist[v]) {
local_has_change[my_rank] = true;
dist[v] = new_dis;
}
}
}
}
#pragma omp barrier
#pragma omp single
{
iter_num++;
has_change = false;
for (int rank = 0; rank < p; rank++) {
has_change |= local_has_change[rank];
}
}
if (!has_change) {
break;
}
}
}

if (iter_num == n - 1) {
has_change = false;
for (int u = 0; u < n; u++) {
#pragma omp parallel for reduction(|:has_change)
for (int v = 0; v < n; v++) {
int weight = mat[u * n + v];
if (weight < INF) {
if (dist[u] + weight < dist[v]) { 
has_change = true;;
}
}
}
}
*has_negative_cycle = has_change;
}

}

int main(int argc, char **argv) {
if (argc <= 1) {
utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
}
if (argc <= 2) {
utils::abort_with_error_message("NUMBER OF THREADS WAS NOT FOUND!");
}
string filename = argv[1];
int p = atoi(argv[2]);

int *dist;
bool has_negative_cycle = false;


assert(utils::read_file(filename) == 0);
dist = (int *) malloc(sizeof(int) * utils::N);

timeval start_wall_time_t, end_wall_time_t;
float ms_wall;

gettimeofday(&start_wall_time_t, nullptr);

bellman_ford(p, utils::N, utils::mat, dist, &has_negative_cycle);

gettimeofday(&end_wall_time_t, nullptr);
ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
+ end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

std::cerr.setf(std::ios::fixed);
std::cerr << std::setprecision(6) << "Time(s): " << (ms_wall / 1000.0) << endl;
utils::print_result(has_negative_cycle, dist);
free(dist);
free(utils::mat);

return 0;
}
