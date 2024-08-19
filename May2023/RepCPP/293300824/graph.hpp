#pragma once

#include <omp.h>

#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <functional>

struct Graph {
using Node = int;

int task_threshold = 60;
int max_depth_rdfs = 10'000;

std::vector<std::vector<int>> adj_matrix;

bool edge_exists(Node n1, Node n2)
{
return adj_matrix[n1][n2] > 0;
}

int n_nodes() {
return adj_matrix.size();
}

int size()
{
return n_nodes();
}

void dfs(Node src, std::vector<int>& visited)
{
std::vector<Node> queue { src };

while (!queue.empty()) {
Node node = queue.back();
queue.pop_back();

if (!visited[node]) {
visited[node] = true;

for (int next_node = 0; next_node < n_nodes(); next_node++)
if (edge_exists(node, next_node) && !visited[next_node])
queue.push_back(next_node);
}
}
}

void rdfs(Node src, std::vector<int>& visited, int depth = 0)
{
visited[src] = true;

for (int node = 0; node < n_nodes(); node++) {
if (edge_exists(src, node) && !visited[node]) {
if (depth <= max_depth_rdfs)
rdfs(node, visited, depth + 1);
else
dfs(node, visited);
}
}
}

void p_dfs(Node src, std::vector<int>& visited)
{
std::vector<Node> queue { src };

while (!queue.empty()) {
Node node = queue.back();
queue.pop_back();

if (!visited[node]) {
visited[node] = true;

#pragma omp parallel shared(queue, visited)
{
std::vector<Node> private_queue;

#pragma omp for nowait schedule(static)
for (int next_node = 0; next_node < n_nodes(); next_node++)
if (edge_exists(node, next_node) && !visited[next_node])
private_queue.push_back(next_node);

#pragma omp critical(queue_update)
queue.insert(queue.end(), private_queue.begin(), private_queue.end());
}
}
}
}

void p_dfs_with_locks(Node src, std::vector<int>& visited, std::vector<omp_lock_t>& node_locks)
{

std::vector<Node> queue { src };

while (!queue.empty()) {
Node node = queue.back();
queue.pop_back();

bool already_visited = atomic_test_visited(node, visited, &node_locks[node]);

if (!already_visited) {
atomic_set_visited(node, visited, &node_locks[node]);

#pragma omp parallel shared(queue, visited)
{
std::vector<Node> private_queue;

#pragma omp for nowait
for (int next_node = 0; next_node < n_nodes(); next_node++) {
if (edge_exists(node, next_node)) {
if (atomic_test_visited(next_node, visited, &node_locks[next_node])) {
private_queue.push_back(next_node);
}
}
}

#pragma omp critical(queue_update)
queue.insert(queue.end(), private_queue.begin(), private_queue.end());
}
}
}
}

void p_rdfs(Node src, std::vector<int>& visited)
{
std::vector<omp_lock_t> node_locks;
node_locks.reserve(size());

for (int node = 0; node < n_nodes(); node++) {
omp_lock_t lock;
node_locks[node] = lock;
omp_init_lock(&(node_locks[node]));
}

#pragma omp parallel shared(src, visited, node_locks)
#pragma omp single
p_rdfs(src, visited, node_locks);

for (int node = 0; node < n_nodes(); node++)
omp_destroy_lock(&(node_locks[node]));
}

void p_rdfs(Node src, std::vector<int>& visited, std::vector<omp_lock_t>& node_locks, int depth = 0)
{
atomic_set_visited(src, visited, &node_locks[src]);

int task_count = 0;

for (int node = 0; node < n_nodes(); node++) {
if (edge_exists(src, node) && !atomic_test_visited(node, visited, &node_locks[node])) {
if (depth <= max_depth_rdfs && task_count <= task_threshold) {
task_count++;

#pragma omp task untied default(shared) firstprivate(node)
{
p_rdfs(node, visited, node_locks, depth + 1);
task_count--;
}

} else {
p_dfs_with_locks(node, visited, node_locks);
}
}
}

#pragma omp taskwait
}

std::pair<std::vector<Node>, std::vector<Node>> dijkstra(Node src)
{
std::vector<Node> queue;
queue.push_back(src);

std::vector<Node> came_from(size(), -1);
std::vector<Node> cost_so_far(size(), -1);

came_from[src] = src;
cost_so_far[src] = 0;

while (!queue.empty()) {
Node current = queue.back();
queue.pop_back();

for (int next = 0; next < n_nodes(); next++) {
if (edge_exists(current, next)) {
int new_cost = cost_so_far[current] + adj_matrix[current][next];

if (cost_so_far[next] == -1 || new_cost < cost_so_far[next]) {
cost_so_far[next] = new_cost;
queue.push_back(next);
came_from[next] = current;
}
}
}
}

return std::make_pair(came_from, cost_so_far);
}

inline std::vector<omp_lock_t> initialize_locks() {
std::vector<omp_lock_t> node_locks;
node_locks.reserve(n_nodes());

for (int node = 0; node < n_nodes(); node++) {
omp_lock_t lock;
node_locks[node] = lock;
omp_init_lock(&(node_locks[node]));
}

return node_locks;
}

std::pair<std::vector<Node>, std::vector<Node>> p_dijkstra(Node src)
{
std::vector<Node> queue;
queue.push_back(src);

std::vector<Node> came_from(size(), -1);
std::vector<Node> cost_so_far(size(), -1);

came_from[src] = src;
cost_so_far[src] = 0;

auto node_locks = initialize_locks();

while (!queue.empty()) {
Node current = queue.back();
queue.pop_back();

#pragma omp parallel shared(queue, node_locks)
#pragma omp for
for (int next = 0; next < n_nodes(); next++) {
if (edge_exists(current, next)) {
omp_set_lock(&node_locks[current]);
auto cost_so_far_current = cost_so_far[current];
omp_unset_lock(&node_locks[current]);

int new_cost = cost_so_far_current + adj_matrix[current][next];

omp_set_lock(&node_locks[next]);
auto cost_so_far_next = cost_so_far[next];
omp_unset_lock(&node_locks[next]);

if (cost_so_far_next == -1 || new_cost < cost_so_far_next) {
omp_set_lock(&node_locks[next]);
cost_so_far[next] = new_cost;
came_from[next] = current;
omp_unset_lock(&node_locks[next]);

#pragma omp critical(queue_update)
queue.push_back(next);
}
}
}
}

for (int node = 0; node < n_nodes(); node++)
omp_destroy_lock(&(node_locks[node]));

return std::make_pair(came_from, cost_so_far);
}

std::vector<Node> reconstruct_path(Node src, Node dst, std::vector<Node> origins)
{
auto current_node = dst;
std::vector<Node> path;

while (current_node != src) {
path.push_back(current_node);
current_node = origins.at(current_node);
}

path.push_back(src);
reverse(path.begin(), path.end());

return path;
}

private:
inline bool atomic_test_visited(Node node, const std::vector<int>& visited, omp_lock_t* lock)
{
omp_set_lock(lock);
bool already_visited = visited.at(node);
omp_unset_lock(lock);

return already_visited;
}

inline void atomic_set_visited(Node node, std::vector<int>& visited, omp_lock_t* lock)
{
omp_set_lock(lock);
visited[node] = true;
omp_unset_lock(lock);
}
};

Graph import_graph(std::string &path)
{
Graph graph;

std::ifstream file(path);
if (!file.is_open()) {
throw std::invalid_argument("Input file does not exist or is not readable.");
}

std::string line;

while (getline(file, line)) {
std::vector<int> lineData;
std::stringstream lineStream(line);

int value;
while (lineStream >> value)
lineData.push_back(value);

lineData.shrink_to_fit(); 
graph.adj_matrix.push_back(lineData);
}

graph.adj_matrix.shrink_to_fit();

return graph;
}
