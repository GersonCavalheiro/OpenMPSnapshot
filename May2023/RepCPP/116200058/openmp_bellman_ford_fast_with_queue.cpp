

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <queue>
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

*has_negative_cycle = false;

omp_set_num_threads(p);

for (int i = 0; i < n; i++) {
dist[i] = INF;
}
dist[0] = 0;

int current_queue = 0;
int *enqueue_counter;
std::queue<int> queue;
bool *in_queue;
in_queue = (bool *) calloc(sizeof(bool), n);
enqueue_counter = (int *) calloc(n, sizeof(int));
in_queue[0] = true;
queue.push(0);
enqueue_counter[0] = 1;

while(!queue.empty() && !*has_negative_cycle){
int u = queue.front(); queue.pop();
in_queue[u] = false;
#pragma omp parallel for
for (int v = 0; v < n; v++) {
int weight = mat[u * n + v];
if (weight < INF) {
int new_dist = weight + dist[u];
if (new_dist < dist[v]) {
dist[v] = new_dist;
enqueue_counter[v]++;
if (in_queue[v] == false) {
in_queue[v] = true;
if (enqueue_counter[v] >= n) {
*has_negative_cycle = true;
}
#pragma omp critical
{
queue.push(v);
}
}
}
}
}
}


free(enqueue_counter);
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
