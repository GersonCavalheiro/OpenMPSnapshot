

#include <omp.h>

#include "model/Model.hpp"
#include "model/Node.hpp"


#define NUM_THREADS 16



void create_graph(Node** nodes, Model& model) {
#pragma omp parallel for
for (int e = 0; e < model.num_edges; ++e) {
int source = model.edges[e];
int target = model.edges[model.num_edges + e];

if (source != target) {
nodes[source]->neighbors.push_back(target);
}
}

#pragma omp parallel for
for (int n = 0; n < model.num_nodes; ++n) {
Node *node = nodes[n];

node->neighbors.push_back(node->ID);
node->degree = node->neighbors.size();
}
}




void first_layer_transform(Node** nodes, int num_nodes, Model& model) {
for (int n = 0; n < num_nodes; ++n) {
#pragma omp task default(none) firstprivate(nodes, n, model)
{
Node* node = nodes[n];

for (int c_in = 0; c_in < node->dim_features; ++c_in) {
float x_in = node->x[c_in];
float* weight_1_start_idx = model.weight_1 + (c_in * node->dim_hidden);

if (x_in == 0) {
continue;
}

for (int c_out = 0; c_out < node->dim_hidden; ++c_out) {
node->tmp_hidden[c_out] += x_in * *(weight_1_start_idx + c_out);
}
}
}
}

#pragma omp taskwait
}




void first_layer_aggregate(Node** nodes, int num_nodes, Model& model) {
#pragma omp parallel for
for (int n = 0; n < num_nodes; ++n) {
Node* node = nodes[n];

for (int neighbor : node->neighbors) {
float* message = nodes[neighbor]->tmp_hidden;

float norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

for (int c = 0; c < node->dim_hidden; ++c) {
node->hidden[c] += message[c] / norm + model.bias_1[c] / node->degree;
}
}

for (int c = 0; c < node->dim_hidden; ++c) {
node->hidden[c] = (node->hidden[c] >= 0.0) * node->hidden[c];
}
}
}




void second_layer_transform(Node** nodes, int num_nodes, Model& model) {
#pragma omp parallel for
for (int n = 0; n < num_nodes; ++n) {
Node* node = nodes[n];

for (int c_in = 0; c_in < node->dim_hidden; ++c_in) {
float h_in = node->hidden[c_in];
float* weight_2_start_idx = model.weight_2 + (c_in * node->num_classes);

if (h_in == 0) {
continue;
}

for (int c_out = 0; c_out < node->num_classes; ++c_out) {
node->tmp_logits[c_out] += h_in * *(weight_2_start_idx + c_out);
}
}
}
}




void second_layer_aggregate(Node** nodes, int num_nodes, Model& model) {
#pragma omp parallel for
for (int n = 0; n < num_nodes; ++n) {
Node* node = nodes[n];

for (int neighbor : node->neighbors) {
float* message = nodes[neighbor]->tmp_logits;

float norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

for (int c = 0; c < node->num_classes; ++c) {
node->logits[c] += message[c] / norm + model.bias_2[c] / node->degree;
}
}
}
}




int main(int argc, char** argv) {
int seed = -1;
int init_no = -1;
std::string dataset("");

Model::specify_problem(dataset, &init_no, &seed);

Model model(dataset, init_no, seed);
model.load_model();

Node** nodes = (Node**) malloc(model.num_nodes * sizeof(Node*));

if (nodes == nullptr) {
exit(1);
}

omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for
for (int n = 0; n < model.num_nodes; ++n) {
nodes[n] = new Node(n, model, 1);
}

create_graph(nodes, model);

#pragma omp parallel
#pragma omp single
{
first_layer_transform(nodes, model.num_nodes, model);
}

first_layer_aggregate(nodes, model.num_nodes, model);
second_layer_transform(nodes, model.num_nodes, model);
second_layer_aggregate(nodes, model.num_nodes, model);

float acc = 0.0;

#pragma omp parallel for reduction(+: acc)
for (int n = 0; n < model.num_nodes; ++n) {
int pred = nodes[n]->get_prediction();
int correct = pred == model.labels[n];

acc += (float) correct;
}

acc /= model.num_nodes;

std::cout << "accuracy " << acc << std::endl;
std::cout << "DONE" << std::endl;

#pragma omp parallel for
for (int n = 0; n < model.num_nodes; ++n) {
nodes[n]->free_node();
delete nodes[n];
}

free(nodes);
model.free_model();

(void) argc;
(void) argv;

return 0;
}


