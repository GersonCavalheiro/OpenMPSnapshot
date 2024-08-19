

#include <omp.h>
#include <mpi.h>
#include <immintrin.h>

#include "model/Model.hpp"
#include "model/Node.hpp"


#define NUM_THREADS 4 
#define DYNAMIC_SCHEDULING_CHUNK_SIZE 32





#define TAG_DATA_COMM_X_VECTOR 102

#define CHUNK_SIZE(num_nodes, size) std::ceil((float) num_nodes / size)
#define END(start, chunk_size, num_nodes) std::min(start + chunk_size, num_nodes)






Model load_model(const int rank, std::string dataset, int init_no, int seed) {
Model model(dataset, init_no, seed);

if (rank == 0) { 
model.load_model();
}

int num_nodes, num_edges, dim_features, dim_hidden, num_classes;

if (rank == 0) { 
num_nodes = model.num_nodes;
num_edges = model.num_edges;
dim_features = model.dim_features;
dim_hidden = model.dim_hidden;
num_classes = model.num_classes;
}

MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&dim_features, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&dim_hidden, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);

float* weight_1 = (float*) calloc(dim_features * dim_hidden, sizeof(float));
float* weight_2 = (float*) calloc(dim_hidden * num_classes, sizeof(float));
float* bias_1 = (float*) calloc(dim_hidden, sizeof(float));
float* bias_2 = (float*) calloc(num_classes, sizeof(float));
int* labels = (int*) calloc(num_nodes, sizeof(int));
int* edges = (int*) calloc(num_edges * 2, sizeof(int));

if (rank == 0) { 
weight_1 = model.weight_1;
weight_2 = model.weight_2;
bias_1 = model.bias_1;
bias_2 = model.bias_2;
labels = model.labels;
edges = model.edges;
}

MPI_Bcast(weight_1, dim_features * dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Bcast(weight_2, dim_hidden * num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Bcast(bias_1, dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Bcast(bias_2, num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Bcast(labels, num_nodes, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(edges, num_edges * 2, MPI_INT, 0, MPI_COMM_WORLD);

if (rank != 0) {
model.num_nodes = num_nodes;
model.num_edges = num_edges;
model.dim_features = dim_features;
model.dim_hidden = dim_hidden;
model.num_classes = num_classes;
model.weight_1 = weight_1;
model.weight_2 = weight_2;
model.bias_1 = bias_1;
model.bias_2 = bias_2;
model.labels = labels;
model.edges = edges;
}

return model;
}

Model create_model(const int rank) {
std::string dataset("");
int init_no = -1;
int seed = -1;

if (rank == 0) { 
Model::specify_problem(dataset, &init_no, &seed);
}

return load_model(rank, dataset, init_no, seed);
}






Node** create_nodes(const int rank, Model& model) {
Node** nodes = (Node**) malloc(model.num_nodes * sizeof(Node*));

if (nodes == nullptr) {
MPI_Finalize();
exit(1);
}

Node* node;

#pragma omp parallel for private(node)
for (int n = 0; n < model.num_nodes; ++n) {
node = new Node(n, model, rank == 0); 
nodes[n] = node;

if (rank != 0) { 
node->tmp_hidden = (float*) calloc(node->dim_hidden, sizeof(float));
node->hidden = (float*) calloc(node->dim_hidden, sizeof(float));
node->tmp_logits = (float*) calloc(node->num_classes, sizeof(float));
node->logits = (float*) calloc(node->num_classes, sizeof(float));
node->x = (float*) calloc(model.dim_features, sizeof(float));
}
}

return nodes;
}

void create_graph(Node** nodes, Model &model) {
int source, target;

for (int e = 0; e < model.num_edges; ++e) {
source = model.edges[e];
target = model.edges[model.num_edges + e];

if (source != target) {
nodes[source]->neighbors.push_back(target);
}
}

Node* node;

#pragma omp parallel for private(node)
for (int n = 0; n < model.num_nodes; ++n) {
node = nodes[n];

node->neighbors.push_back(node->ID);
node->degree = node->neighbors.size();
}
}






void inner_first_layer_transform_simd(const int n, const int start, Node* node, const int c_in, float* weight_1, float* chunk_tmp_hidden) {
const int last_chunk_idx = node->dim_hidden - (node->dim_hidden % 8) - 1;
__m256 x_in = _mm256_set1_ps(node->x[c_in]); 

for (int c_out = 0; c_out <= last_chunk_idx; c_out += 8) {
__m256 partial_sum = _mm256_loadu_ps(node->tmp_hidden + c_out);

__m256 w_out = _mm256_loadu_ps(weight_1 + c_in * node->dim_hidden + c_out); 
__m256 hidden_mult = _mm256_mul_ps(x_in, w_out); 

partial_sum = _mm256_add_ps(partial_sum, hidden_mult); 

_mm256_storeu_ps(node->tmp_hidden + c_out, partial_sum);
_mm256_storeu_ps(chunk_tmp_hidden + (n - start) * node->dim_hidden + c_out, partial_sum);
}

for (int c_out = last_chunk_idx + 1; c_out < node->dim_hidden; ++c_out) {
float tmp_hidden_item = node->x[c_in] * weight_1[c_in * node->dim_hidden + c_out];

node->tmp_hidden[c_out] += tmp_hidden_item;
chunk_tmp_hidden[(n - start) * node->dim_hidden + c_out] += tmp_hidden_item;
}
}

float* first_layer_transform(const int chunk_size, const int start, const int end, Node** nodes, Model& model) {
Node* node;

float* chunk_tmp_hidden = (float*) calloc(chunk_size * model.dim_hidden, sizeof(float));

#pragma omp parallel for private(node) schedule(dynamic, DYNAMIC_SCHEDULING_CHUNK_SIZE)
for (int n = start; n < end; ++n) {
node = nodes[n];

for (int c_in = 0; c_in < node->dim_features; ++c_in) {
if (node->x[c_in] == 0) {
continue;
}

inner_first_layer_transform_simd(n, start, node, c_in, model.weight_1, chunk_tmp_hidden);
}
}

return chunk_tmp_hidden;
}

void inner_first_layer_aggregate_simd(Node* node, float* message, float norm, float* bias_1) {
const int last_chunk_idx = node->dim_hidden - (node->dim_hidden % 8) - 1;

__m256 norm_ps = _mm256_set1_ps(norm);
__m256 degree_ps = _mm256_set1_ps(node->degree);

for (int c = 0; c <= last_chunk_idx; c += 8) {
__m256 partial_sum = _mm256_loadu_ps(node->hidden + c);

__m256 message_ps = _mm256_loadu_ps(message + c);
__m256 message_div_ps = _mm256_div_ps(message_ps, norm_ps);

__m256 bias_ps = _mm256_loadu_ps(bias_1 + c);
__m256 bias_div_ps = _mm256_div_ps(bias_ps, degree_ps);

__m256 inner_sum = _mm256_add_ps(message_div_ps, bias_div_ps);
partial_sum = _mm256_add_ps(partial_sum, inner_sum);

_mm256_storeu_ps(node->hidden + c, partial_sum);
}

for (int c = last_chunk_idx + 1; c < node->dim_hidden; ++c) {
node->hidden[c] += message[c] / norm + bias_1[c] / node->degree;
}
}

void first_layer_aggregate(const int start, const int end, Node** nodes, Model &model, float* big_tmp_hidden) {
Node* node;

float* message;
float norm;

#pragma omp parallel for private(node, message, norm) schedule(dynamic, DYNAMIC_SCHEDULING_CHUNK_SIZE)
for (int n = start; n < end; ++n) {
node = nodes[n];

for (int neighbor : node->neighbors) {
message = &big_tmp_hidden[neighbor * model.dim_hidden];

norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

inner_first_layer_aggregate_simd(node, message, norm, model.bias_1);
}

for (int c = 0; c < node->dim_hidden; ++c) {
node->hidden[c] = (node->hidden[c] >= 0.0) * node->hidden[c];
}
}
}






void inner_second_layer_transform_simd(const int n, const int start, Node* node, const int c_in, float* weight_2, float* chunk_tmp_logits) {
const int last_chunk_idx = node->num_classes - (node->num_classes % 8) - 1;
__m256 h_in = _mm256_set1_ps(node->hidden[c_in]); 

for (int c_out = 0; c_out <= last_chunk_idx; c_out += 8) {
__m256 partial_sum = _mm256_loadu_ps(node->tmp_logits + c_out);

__m256 w_out = _mm256_loadu_ps(weight_2 + c_in * node->num_classes + c_out); 
__m256 hidden_mult = _mm256_mul_ps(h_in, w_out); 

partial_sum = _mm256_add_ps(partial_sum, hidden_mult); 

_mm256_storeu_ps(node->tmp_logits + c_out, partial_sum);
_mm256_storeu_ps(chunk_tmp_logits + (n - start) * node->num_classes + c_out, partial_sum);
}

for (int c_out = last_chunk_idx + 1; c_out < node->num_classes; ++c_out) {
float tmp_logit = node->hidden[c_in] * weight_2[c_in * node->num_classes + c_out];

node->tmp_logits[c_out] += tmp_logit;
chunk_tmp_logits[(n - start) * node->num_classes + c_out] += tmp_logit;
}
}

float* second_layer_transform(const int chunk_size, const int start, const int end, Node** nodes, Model& model) {
Node* node;

float* chunk_tmp_logits = (float*) calloc(chunk_size * model.num_classes, sizeof(float));

#pragma omp parallel for private(node) schedule(dynamic, DYNAMIC_SCHEDULING_CHUNK_SIZE)
for (int n = start; n < end; ++n) {
node = nodes[n];

for (int c_in = 0; c_in < node->dim_hidden; ++c_in) {
if (node->hidden[c_in] == 0) {
continue;
}

inner_second_layer_transform_simd(n, start, node, c_in, model.weight_2, chunk_tmp_logits);
}
}

return chunk_tmp_logits;
}

void inner_second_layer_aggregate_simd(Node* node, float* message, float norm, float* bias_2) {
const int last_chunk_idx = node->num_classes - (node->num_classes % 8) - 1;

__m256 norm_ps = _mm256_set1_ps(norm);
__m256 degree_ps = _mm256_set1_ps(node->degree);

for (int c = 0; c <= last_chunk_idx; c += 8) {
__m256 partial_sum = _mm256_loadu_ps(node->logits + c);

__m256 message_ps = _mm256_loadu_ps(message + c);
__m256 message_div_ps = _mm256_div_ps(message_ps, norm_ps);

__m256 bias_ps = _mm256_loadu_ps(bias_2 + c);
__m256 bias_div_ps = _mm256_div_ps(bias_ps, degree_ps);

__m256 inner_sum = _mm256_add_ps(message_div_ps, bias_div_ps);
partial_sum = _mm256_add_ps(partial_sum, inner_sum);

_mm256_storeu_ps(node->logits + c, partial_sum);
}

for (int c = last_chunk_idx + 1; c < node->num_classes; ++c) {
node->logits[c] += message[c] / norm + bias_2[c] / node->degree;
}
}

void second_layer_aggregate(const int start, const int end, Node** nodes, Model &model, float* big_tmp_logits) {
Node* node;

float* message;
float norm;

#pragma omp parallel for private(node, message, norm) schedule(dynamic, DYNAMIC_SCHEDULING_CHUNK_SIZE)
for (int n = start; n < end; ++n) {
node = nodes[n];

for (int neighbor : node->neighbors) {
message = &big_tmp_logits[neighbor * model.num_classes];

norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

inner_second_layer_aggregate_simd(node, message, norm, model.bias_2);
}
}        
}






float get_num_correct_preds(const int start, const int end, Node** nodes, Model& model) {
int correct = 0.0;

#pragma omp parallel for reduction(+: correct)
for (int n = start; n < end; ++n) {
correct += nodes[n]->get_prediction() == model.labels[n];
}

return correct;
}




int main(int argc, char** argv) {
MPI_Init(&argc, &argv);
int size, rank;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

omp_set_num_threads(NUM_THREADS);

Model model = create_model(rank);

Node** nodes = create_nodes(rank, model);
create_graph(nodes, model);

const int chunk_size = CHUNK_SIZE(model.num_nodes, size);
const int start = rank * chunk_size;
const int end = END(start, chunk_size, model.num_nodes);



float* big_tmp_hidden = (float*) calloc(model.num_nodes * model.dim_hidden, sizeof(float));

if (rank == 0) { 
omp_set_num_threads(omp_get_num_threads());

big_tmp_hidden = first_layer_transform(model.num_nodes, 0, model.num_nodes, nodes, model);

omp_set_num_threads(NUM_THREADS);
}

MPI_Bcast(big_tmp_hidden, model.num_nodes * model.dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);

first_layer_aggregate(start, end, nodes, model, big_tmp_hidden);



float* tmp_logits = second_layer_transform(chunk_size, start, end, nodes, model);
float* tmp_logits_gathered = (float*) calloc(chunk_size * size * model.num_classes, sizeof(float));

MPI_Gather(tmp_logits, chunk_size * model.num_classes, MPI_FLOAT,
tmp_logits_gathered, chunk_size * model.num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Bcast(tmp_logits_gathered, chunk_size * size * model.num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);

second_layer_aggregate(start, end, nodes, model, tmp_logits_gathered);



int num_correct_preds = get_num_correct_preds(start, end, nodes, model);

int total_num_correct_preds;
MPI_Reduce(&num_correct_preds, &total_num_correct_preds, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) { 
float acc = (float) total_num_correct_preds / model.num_nodes;

std::cout << "accuracy " << acc << std::endl;
std::cout << "DONE" << std::endl;
}

for (int n = 0; n < model.num_nodes; ++n) {
nodes[n]->free_node();
delete nodes[n];
}

free(nodes);
model.free_model();

(void) argc;
(void) argv;

MPI_Finalize();

return 0;
}


