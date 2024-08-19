




#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <omp.h>
#include "nqueue/queue.hpp"

using namespace std;



typedef struct city { 
int id;                
double min1;           
double min2;           
} * City;

typedef struct node { 
int * tour;            
double cost;           
double lower_bound;    
int length;            
} * Node;


struct cmp_op { bool operator()(const Node& n1, const Node& n2) {
return (n1->lower_bound > n2->lower_bound) 
|| (n1->lower_bound == n2->lower_bound && n1->tour[n1->length - 1] > n2->tour[n2->length - 1]);
}};




double initialLowerBound(int n_cities, City * cities) {
double LB = 0;
for (int i = 0; i < n_cities; i++) {
LB += cities[i]->min1 + cities[i]->min2;
}
return LB / 2;
}


double newBound(City source, City target, double lower_bound, int n_cities, double * matrix){
double cost = matrix[source->id * n_cities + target->id];
double cf = cost >= source->min2 ? source->min2 : source->min1;
double ct = cost >= target->min2 ? target->min2 : target->min1;

return lower_bound + cost - (cf + ct) / 2;
}


void update_mins(int coord, double dist, City ** cities) {
if ((*cities)[coord]->min1 == -1) { (*cities)[coord]->min1 = dist; } 
else if ((*cities)[coord]->min2 == -1) {  
if (dist < (*cities)[coord]->min1) { 
(*cities)[coord]->min2 = (*cities)[coord]->min1;
(*cities)[coord]->min1 = dist;
}
else { 
(*cities)[coord]->min2 = dist;
}
}
else { 
if (dist < (*cities)[coord]->min1) { 
(*cities)[coord]->min2 = (*cities)[coord]->min1;
(*cities)[coord]->min1 = dist;
}
else if (dist < (*cities)[coord]->min2) { 
(*cities)[coord]->min2 = dist;
}
}
}



void merge_best_tour(double * array_best_tour_cost, int ** array_best_tour, int thread_id, int n_threads, int n_cities) {
double new_best = array_best_tour_cost[thread_id];
for (int i = 0; i < n_threads; i++) {
if (array_best_tour_cost[i] > new_best) {
array_best_tour_cost[i] = new_best;
memcpy(array_best_tour[i], array_best_tour[thread_id], (n_cities + 1) * sizeof(int));
}
}
}

void tsp(double * best_tour_cost, int max_value, int n_cities, int ** best_tour, double * matrix, City * cities){

int n_threads = omp_get_max_threads();


Node root = (Node) calloc(1, sizeof(struct node));
root->tour = (int *) calloc(1, sizeof(int));
root->tour[0] = 0;
root->cost = 0;
root->lower_bound = initialLowerBound(n_cities, cities);
root->length = 1;


(*best_tour_cost) = max_value;
(*best_tour) = (int *) calloc(n_cities + 1, sizeof(int));


bool * tour_nodes_init = (bool *) calloc(n_cities, sizeof(bool));
PriorityQueue<Node, cmp_op> initial_queue;
initial_queue.push(root);
while (initial_queue.size() < n_threads * n_threads || initial_queue.empty()) {
Node node = initial_queue.pop();
int id = node->tour[node->length - 1];

if (node->lower_bound >= (*best_tour_cost)) {
free(node->tour);
free(node);
while (!initial_queue.empty()) {
Node n = initial_queue.pop();
free(n->tour);
free(n);
}
free(tour_nodes_init);
return;
}

if (node->length == n_cities) {
if (node->cost + matrix[id * n_cities + 0] < (*best_tour_cost) && matrix[id * n_cities + 0] >= 0.0001) {
int i;
for (i = 0; i < n_cities; i++) {
(*best_tour)[i] = node->tour[i];
}
(*best_tour)[i] = 0;

(*best_tour_cost) = node->cost + matrix[id * n_cities + 0];
}
} else {
for (int i = 0; i < node->length; i++) {
tour_nodes_init[node->tour[i]] = true;
}
for (int i = 0; i < n_cities; i++) {
if (matrix[id * n_cities + i] != 0 && !tour_nodes_init[i]) {
double new_bound_value = newBound(cities[id], cities[i], node->lower_bound, n_cities, matrix);
if(new_bound_value > (*best_tour_cost)) {
continue;
}
Node newNode = (Node) calloc(1, sizeof(struct node));
newNode->tour = (int *) calloc(node->length + 1, sizeof(int));
for (int j = 0; j < node->length; j++) {
newNode->tour[j] = node->tour[j];
}
newNode->tour[node->length] = i;
newNode->cost = node->cost + matrix[id * n_cities + i];
newNode->lower_bound = new_bound_value;
newNode->length = node->length + 1;
initial_queue.push(newNode);
}
}
memset(tour_nodes_init, false, n_cities * sizeof(bool));

}
free(node->tour);
free(node);
}
free(tour_nodes_init);


PriorityQueue<Node, cmp_op> * queue_array = (PriorityQueue<Node, cmp_op> *) calloc(n_threads, sizeof(PriorityQueue<Node, cmp_op>));
double * array_best_tour_cost = (double *) calloc(n_threads, sizeof(double));
int ** array_best_tour = (int **) calloc(n_threads, sizeof(int *));
for (int i = 0; i < n_threads; i++) {
array_best_tour_cost[i] = max_value;
array_best_tour[i] = (int *) calloc(n_cities + 1, sizeof(int));
}


for (int i = 0; i < initial_queue.size(); i++) {
Node n = initial_queue.pop();
queue_array[i % n_threads].push(n);
}

Node * shared_nodes = (Node *) calloc(n_threads * (n_threads - 1), sizeof(Node));
bool * free_queue = (bool *) calloc(n_threads, sizeof(bool));
int shared_nodes_size = 0;
#pragma omp parallel
{
bool * tour_nodes = (bool *) calloc(n_cities, sizeof(bool));
const int thread_id = omp_get_thread_num();
bool thereAreQueueNodes = true;
bool thereAreNodes = true;
int iterations = 0;

while (thereAreNodes) {

iterations++;
if (iterations % 500000 == 0) {
merge_best_tour(array_best_tour_cost, array_best_tour, thread_id, n_threads, n_cities);
}

if (!queue_array[thread_id].empty()) {
Node node = queue_array[thread_id].pop();

if (free_queue[thread_id]) {
free_queue[thread_id] = false;
}
for (int i = 0; i < n_threads; i++) {
if (i != thread_id) {
if (free_queue[thread_id]) {
int n_added = 0;
while (n_added < n_threads && !queue_array[thread_id].empty()) {
if (shared_nodes_size < n_threads * (n_threads - 1)) {
shared_nodes[shared_nodes_size++] = queue_array[thread_id].pop();
n_added++;
}
}
}
}
}

int id = node->tour[node->length - 1];

bool worse_than_best = (node->lower_bound >= array_best_tour_cost[thread_id]);

if (worse_than_best) {
free(node->tour);
free(node);

while (!queue_array[thread_id].empty()) {
Node n = queue_array[thread_id].pop();
free(n->tour);
free(n);
}
}
else {
if (node->length == n_cities) {
if (node->cost + matrix[id * n_cities + 0] < array_best_tour_cost[thread_id] && matrix[id * n_cities + 0] >= 0.0001) {
int i;
for (i = 0; i < n_cities; i++) {
array_best_tour[thread_id][i] = node->tour[i];
}                        
array_best_tour[thread_id][i] = 0;

array_best_tour_cost[thread_id] = node->cost + matrix[id * n_cities + 0];
}
} 
else {
for (int i = 0; i < node->length; i++) {
tour_nodes[node->tour[i]] = true;
}

int id = node->tour[node->length - 1];
for (int i = 0; i < n_cities; i++) {
if (matrix[id * n_cities + i] != 0 && !tour_nodes[i]) {
double new_bound_value = newBound(cities[id], cities[i], node->lower_bound, n_cities, matrix);

if (new_bound_value <= array_best_tour_cost[thread_id]) {
Node newNode = (Node) calloc(1, sizeof(struct node));
newNode->tour = (int *) calloc(node->length + 1, sizeof(int));
for (int j = 0; j < node->length; j++) {
newNode->tour[j] = node->tour[j];
}
newNode->tour[node->length] = i;
newNode->cost = node->cost + matrix[id * n_cities + i];
newNode->lower_bound = new_bound_value;
newNode->length = node->length + 1;
queue_array[thread_id].push(newNode);
}
}
}
memset(tour_nodes, false, n_cities * sizeof(bool));
}
free(node->tour);
free(node);   
}
}
else {
free_queue[thread_id] = true;

int n_removed = 0;
while (n_removed < n_threads && shared_nodes_size > 0) {
queue_array[thread_id].push(shared_nodes[shared_nodes_size - 1]);
shared_nodes[--shared_nodes_size] = NULL;
n_removed++;
}


}

thereAreQueueNodes = false;
for (int i = 0; i < n_threads; i++) {
if (!queue_array[i].empty()) thereAreQueueNodes = true;
}
thereAreNodes = ((shared_nodes_size != 0) || thereAreQueueNodes);
}

free(tour_nodes);
}

for (int i = 0; i < n_threads; i++) {
if (array_best_tour_cost[i] < (*best_tour_cost)) {
(*best_tour_cost) = array_best_tour_cost[i];
for (int j = 0; j < n_cities + 1; j++) {
(*best_tour)[j] = array_best_tour[i][j];
}
}
free(array_best_tour[i]);
}

free(shared_nodes);
free(free_queue);
free(array_best_tour);
free(array_best_tour_cost);
free(queue_array);
return;
}

void readInputFile(string inputFile, int * n_cities, int * n_roads, double ** matrix, City ** cities) {
ifstream myFile;
myFile.open(inputFile, ios::in);
if(!myFile) {
cerr << "error opening file";
exit(1);
}
int coord1, coord2;
double dist;

myFile >> *n_cities >> *n_roads;

*cities = (City *) calloc((*n_cities), sizeof(City));

for (int i = 0; i < (*n_cities); i++) {
City city = (City) calloc(1, sizeof(struct city));
city->id = i;
city->min1 = -1;
city->min2 = -1;
(*cities)[i] = city;
}

*matrix = (double *) calloc((*n_cities) * (*n_cities), sizeof(double));

while (myFile >> coord1 >> coord2 >> dist) {
(*matrix)[coord1 * (*n_cities) + coord2] = dist;
(*matrix)[coord2 * (*n_cities) + coord1] = dist;

update_mins(coord1, dist, cities);
update_mins(coord2, dist, cities);
}

myFile.close();
}

void print_result(double best_tour_cost, double max_value, int n_cities, int * best_tour) {
if (best_tour_cost > max_value || best_tour[1] == 0) {
cout << "NO SOLUTION" << endl;
} else {
fprintf(stdout, "%.1f\n", best_tour_cost);
for (int i = 0; i < n_cities + 1; i++) {
if (i == n_cities) 
cout << best_tour[i];
else
cout << best_tour[i] << " ";
}
cout << endl;
}
free(best_tour);
}

int main(int argc, char *argv[]) {
double max_value;       
double best_tour_cost;  
double * matrix = NULL; 
double exec_time;       

int * best_tour;        
int n_cities;           
int n_roads;            

City * cities = NULL;   



if (argc < 3) {
cerr << "Usage: tsp <input_file> <max-value>\n";
return 1;
}

srand (time(NULL)); 

readInputFile(argv[1], &n_cities, &n_roads, &matrix, &cities);

max_value = stod(argv[2]);

exec_time = -omp_get_wtime();

tsp(&best_tour_cost, max_value, n_cities, &best_tour, matrix, cities);

exec_time += omp_get_wtime();
fprintf(stderr, "%.1fs\n", exec_time);

print_result(best_tour_cost, max_value, n_cities, best_tour);

free(matrix);
for (int i = 0; i < n_cities; i++) {
free(cities[i]);
}
free(cities);
return 0;
}
