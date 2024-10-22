

#include "trinity/partition.h"

namespace trinity {

Partit::Partit(int max_graph_size, int max_part_size) {

max.capa = max_graph_size;
max.part = std::max(6, max_part_size);
nb.cores = omp_get_max_threads();
sync.offset = new int[nb.cores];
task.cardin = new int[max.part];
task.subset = new int* [max.part];

for (int i = 0; i < max.part; ++i) {
task.subset[i] = new int[max.capa];
}

task.lists   = task.subset;    
task.mapping = task.subset[2];

#pragma omp parallel
reset();
}

Partit::Partit(Mesh const* mesh, int max_part_size)
: Partit(mesh->getCapaNode(), max_part_size)
{}

Partit::~Partit() {

for (int i = 0; i < max.part; ++i) {
delete[] task.subset[i];
}

delete[] task.subset;
delete[] task.cardin;
delete[] sync.offset;
}


void Partit::reset() {

#pragma omp master
{
nb.parts = nb.defect = nb.rounds = 0;
std::memset(nb.remain  , 0, sizeof(int) * 2);
std::memset(task.cardin, 0, sizeof(int) * max.part);
}

for (int i = 0; i < 3; ++i)
#pragma omp for
for (int j = 0; j < max.capa; ++j)
task.subset[i][j] = 0;
}


void Partit::extractColoring(const Mesh* mesh) {

reset();

std::vector<int> forbidden;     
std::vector<int> conflicts;

forbidden.resize((size_t) max.part, std::numeric_limits<int>::max());
conflicts.reserve((size_t) mesh->nb.nodes);

int* color = task.mapping;
const auto& primal = mesh->topo.vicin;

#pragma omp for
for (int i = 0; i < mesh->nb.nodes; ++i) {
if (primal[i].empty()) {
continue;
}

for (const int& j : primal[i]) {
forbidden[color[j]] = i;
}

for (int c = 1; c < max.part; ++c) {
if (forbidden[c] != i) {
color[i] = c;
break;
}
}
assert(color[i]);
}

#pragma omp for schedule(guided) nowait
for (int i = 0; i < mesh->nb.nodes; ++i) {
for (const int& j : primal[i]) {
if (i < j and color[i] == color[j]) {
conflicts.push_back(i);   
break;
}
}
}
sync::reduceTasks(task.lists[0], &conflicts, nb.remain, sync.offset);

int k = 0;  

while (nb.remain[k]) {
#pragma omp single
{
nb.rounds++;
nb.defect += nb.remain[k];
}

#pragma omp for nowait
for (int i = 0; i < nb.remain[k]; ++i) {
const int& v = task.lists[k][i];
if (primal[v].empty()) {
continue;
}
for (const int& w : primal[v])
forbidden[color[w]] = v;

for (int c = 1; c < max.part; ++c) {
if (forbidden[c] != v) {
color[v] = c;
break;
}
}
assert(color[v]);
}

#pragma omp single
nb.remain[k ^ 1] = 0;

#pragma omp for schedule(guided) nowait
for (int i = 0; i < nb.remain[k]; ++i) {
const int& v = task.lists[k][i];
for (const int& w : primal[v]) {
if (v < w and color[v] == color[w]) {
conflicts.push_back(v);   
break;
}
}
}
k ^= 1;
sync::reduceTasks(task.lists[k], &conflicts, nb.remain + k, sync.offset);
}

auto* list = new std::vector<int>[max.part];  

int nb_col = 0;

#pragma omp for nowait
for (int i = 0; i < mesh->nb.nodes; ++i) {
auto j = task.mapping[i];
if (j) {
list[j - 1].push_back(i);
nb_col = std::max(j, nb_col);
}
}

#pragma omp critical
if (nb.parts < nb_col) {
nb.parts = nb_col;
}
#pragma omp barrier

for (int j = 0; j < nb.parts; ++j) {
sync::reduceTasks(task.subset[j], list + j, task.cardin + j, sync.offset);
}

delete[] list;
}


void Partit::extractIndepSet(const Graph& graph, int nb_nodes) {

if (__builtin_expect(1 == nb_nodes, 0)) {
task.subset[0][0] = graph[0][0];
task.cardin[0] = 1;
return;
}

reset();

int* color = task.mapping;

std::vector<int> forbidden;     
std::vector<int> heap;

forbidden.resize((size_t) max.part, std::numeric_limits<int>::max());
heap.reserve((size_t) nb_nodes / nb.cores);

#pragma omp for
for (int i = 0; i < nb_nodes; ++i) {
const int& v = graph[i][0];
for (auto w = graph[i].begin() + 1; w < graph[i].end(); ++w)
forbidden[color[*w]] = v;

for (int c = 1; c < max.part; ++c) {
if (forbidden[c] != v) {
color[v] = c;
break;
}
}
assert(color[v]);
}

#pragma omp for schedule(guided) nowait
for (int i = 0; i < nb_nodes; ++i) {
const int& v = graph[i][0];
if (color[v] == 1) {
for (auto w = graph[i].begin() + 1; w < graph[i].end(); ++w) {
if (v < *w and color[*w] == 1) {
heap.push_back(i);   
break;
}
}
}
}
sync::reduceTasks(task.lists[0], &heap, nb.remain, sync.offset);

int k = 0;  

while (nb.remain[k]) {
#pragma omp single
{
nb.rounds++;
nb.defect += nb.remain[k];
}

#pragma omp for nowait
for (int i = 0; i < nb.remain[k]; ++i) {
const int& j = task.lists[k][i];
const int& v = graph[j][0];
for (auto w = graph[j].begin() + 1; w < graph[j].end(); ++w)
forbidden[color[*w]] = v;

for (int c = 1; c < max.part; ++c) {
if (forbidden[c] != v) {
color[v] = c;
break;
}
}
assert(color[v]);
}

#pragma omp single
nb.remain[k ^ 1] = 0;

#pragma omp for schedule(guided) nowait
for (int i = 0; i < nb.remain[k]; ++i) {
const int& j = task.lists[k][i];
const int& v = graph[j][0];
if (color[v] == 1) {
for (auto w = graph[j].begin() + 1; w < graph[j].end(); ++w) {
if (v < *w and color[*w] == 1) {
heap.push_back(j);   
break;
}
}
}
}
k ^= 1;
sync::reduceTasks(task.lists[k], &heap, nb.remain + k, sync.offset);
}


#pragma omp for schedule(guided) nowait
for (int i = 0; i < nb_nodes; ++i) {
const int& v = graph[i][0];
if (color[v] == 1) {
heap.push_back(v);
}
}
sync::reduceTasks(task.subset[0], &heap, task.cardin, sync.offset);
}

} 
