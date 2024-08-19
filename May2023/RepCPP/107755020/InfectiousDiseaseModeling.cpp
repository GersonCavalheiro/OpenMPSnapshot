#include <omp.h>
#include <ctime>
#include <iostream>
#include <sstream>
#include "Individual.h"
#include "GraphHandler.h"
#include "Settings.h"
#include <fstream>

using namespace std;
using namespace boost;

void simulate_serial(int individual_count, std::uint8_t total_epochs, const LocationUndirectedGraph& individual_graph,
vector<Individual>& individuals, vector<std::tuple<int, int, int>>& epoch_statistics) {

int index = 0;
int max_index = static_cast<int>(individuals.size());
int chunk = static_cast<int>(max_index / DEFAULT_NUMBER_OF_THREADS);

vector<vector<int>> neighborhood_lookup_vector = GraphHandler::get_node_neighborhood_lookup_vector(individual_graph);

for (std::uint8_t current_epoch = 0; current_epoch < (total_epochs + 1); ++current_epoch) {

for (index = 0; index < max_index; ++index) {

int current_location = individuals[index].get_location(); 
vector<int> neighborhood = neighborhood_lookup_vector[current_location]; 
individuals[index].move(neighborhood); 
}


for (index = 0; index < max_index; ++index) {
if (!individuals[index].is_infected()) { 
int affecting_index;
for (affecting_index = 0; affecting_index < individual_count; ++affecting_index) {

if (individuals[affecting_index].is_infected()) { 
if (individuals[index].get_location() == individuals[affecting_index].get_location()) { 
individuals[index].try_infect();
if (individuals[index].is_infected()) { 
break; 
}
}
}
}
}
}

int hit_count = 0;
int infected_count = 0;
int recovered_count = 0;

for (index = 0; index < max_index; ++index) {
individuals[index].advance_epoch();	
if (individuals[index].is_infected())
++infected_count;
if (individuals[index].is_hit())
++hit_count;
if (individuals[index].is_recovered())
++recovered_count;
}

epoch_statistics.push_back(std::make_tuple(hit_count, infected_count, recovered_count)); 
}

if (SAVE_CSV)
GraphHandler::save_epoch_statistics_to_csv("output.csv", epoch_statistics);
if (SAVE_GRAPHVIZ)
GraphHandler::save_undirected_graph_to_graphviz_file("individualGraph.dot", individual_graph);
if (SHOW_EPIDEMIC_RESULTS)
GraphHandler::show_epidemic_results(individual_count, epoch_statistics);
}

void simulate_parallel(int individual_count, std::uint8_t total_epochs, const LocationUndirectedGraph& individual_graph,
vector<Individual>& individuals, vector<std::tuple<int, int, int>>& epoch_statistics) {

int index = 0;
int max_index = static_cast<int>(individuals.size());
int chunk = static_cast<int>(max_index / DEFAULT_NUMBER_OF_THREADS);

boost::unordered_map<int, vector<int>> neighborhood_lookup_map = GraphHandler::get_node_neighborhood_lookup_map(individual_graph);

for (std::uint8_t current_epoch = 0; current_epoch < (total_epochs + 1); ++current_epoch) {

#pragma omp parallel private(index) shared(individuals, neighborhood_lookup_map) firstprivate(chunk, max_index)
{
#pragma omp for schedule(static, chunk) nowait
for (index = 0; index < max_index; ++index) {

Individual current_individual = individuals[index]; 
int current_location = current_individual.get_location(); 
vector<int> neighborhood = neighborhood_lookup_map[current_location]; 
current_individual.move(neighborhood); 

individuals[index] = current_individual; 
}
} 

#pragma omp parallel private(index) shared(individuals) firstprivate(chunk, max_index)
{
#pragma omp for schedule(auto) nowait
for (index = 0; index < max_index; ++index) {
if (!individuals[index].is_infected()) { 
Individual current_individual = individuals[index]; 
int affecting_index;
for (affecting_index = 0; affecting_index < individual_count; ++affecting_index) {

if (individuals[affecting_index].is_infected()) { 
Individual affecting_individual = individuals[affecting_index]; 
if (current_individual.get_location() == affecting_individual.get_location()) { 
current_individual.try_infect();
if (current_individual.is_infected()) { 
individuals[index] = current_individual; 
break; 
}
}
}
}
}
}

} 

int hit_count = 0;
int infected_count = 0;
int recovered_count = 0;
#pragma omp parallel private(index) shared(individuals) firstprivate(chunk, max_index) reduction(+:infected_count, hit_count, recovered_count)
{
#pragma omp for schedule(static, chunk) nowait
for (index = 0; index < max_index; ++index) {		
Individual current_individual = individuals[index]; 
current_individual.advance_epoch();	
individuals[index] = current_individual; 
if (current_individual.is_infected())
++infected_count;
if (current_individual.is_hit())
++hit_count;
if (current_individual.is_recovered())
++recovered_count;
}
} 

epoch_statistics.push_back(std::make_tuple(hit_count, infected_count, recovered_count)); 
}

if (SAVE_CSV)
GraphHandler::save_epoch_statistics_to_csv("output.csv", epoch_statistics);
if (SAVE_GRAPHVIZ)
GraphHandler::save_undirected_graph_to_graphviz_file("individualGraph.dot", individual_graph);
if (SHOW_EPIDEMIC_RESULTS)
GraphHandler::show_epidemic_results(individual_count, epoch_statistics);
}

void simulate_serial_naive(int individual_count, int total_epochs, const LocationUndirectedGraph& individual_graph, vector<Individual>& individuals) {

vector<std::tuple<int, int, int>> epoch_statistics;

boost::unordered_map<int, vector<int>> neighborhood_lookup_map = GraphHandler::get_node_neighborhood_lookup_map(individual_graph);

for (int current_epoch = 0; current_epoch < (total_epochs + 1); ++current_epoch) {

for (Individual& current_individual : individuals)
current_individual.move(neighborhood_lookup_map[current_individual.get_location()]); 

for (int individual_index = 0; individual_index != individuals.size(); ++individual_index) {			

if (individuals[individual_index].is_infected()) { 

for (int affecting_individual = 0; affecting_individual != individuals.size(); ++affecting_individual) {
if (individual_index != affecting_individual) {

if (individuals[individual_index].get_location() == individuals[affecting_individual].get_location()) 
individuals[affecting_individual].try_infect();
}
}
}
}

int hit_count = 0;
int infected_count = 0;
int recovered_count = 0;
for (Individual& current_individual : individuals) {
current_individual.advance_epoch();

if (current_individual.is_infected())
++infected_count;
if (current_individual.is_hit())
++hit_count;
if (current_individual.is_recovered())
++recovered_count;
}
epoch_statistics.push_back(std::make_tuple(hit_count, infected_count, recovered_count));
}

if (SAVE_CSV)
GraphHandler::save_epoch_statistics_to_csv("output.csv", epoch_statistics);
if (SAVE_GRAPHVIZ)
GraphHandler::save_undirected_graph_to_graphviz_file("individualGraph.dot", individual_graph);
if (SHOW_EPIDEMIC_RESULTS)
GraphHandler::show_epidemic_results(individual_count, epoch_statistics);
}

void reset_input(string filename, int individual_count, int& location_count, int& edge_count, LocationUndirectedGraph& individual_graph, vector<Individual>& individuals) {
individual_graph = GraphHandler::get_location_undirected_graph_from_file(filename); 

location_count = individual_graph.m_vertices.size();
edge_count = individual_graph.m_edges.size();

individuals = GraphHandler::get_random_individuals(individual_count, location_count); 

for (int i = 0; i < INITIAL_INFECTED_COUNT; ++i) {
individuals[i].infect();
}
}

void benchmark() {

int thread_count = DEFAULT_NUMBER_OF_THREADS;
int individual_count = DEFAULT_INDIVIDUAL_COUNT;
std::uint8_t total_epochs = DEFAULT_TOTAL_EPOCHS;
std::uint8_t repeat_count = DEFAULT_REPEAT_COUNT;
string input_graph_filename = "antwerp.edges";

int benchmark_init_thread_count = 1;
int benchmark_max_thread_count = 4;
size_t benchmark_repeat_count = 1; 
size_t benchmark_init_individual_count = 503138; 
size_t benchmark_individual_count_multiplier = 10;
size_t benchmark_max_individual_count = 503138; 
std::string execution_type = "serial";	
total_epochs = 30; 

omp_set_num_threads(benchmark_init_thread_count);

std::cout << "----- Benchmark Infectious Disease Modelling -----" << std::endl;

std::time_t timer = std::time(nullptr);
std::string benchmark_file_name = "benchmark_.csv";
std::stringstream benchmark_string_stream;
benchmark_string_stream << "execution_time,execution_type,thread_count,individual_count,node_count,edge_count,total_epochs,epoch_timestep,repeat_count" << std::endl;

LocationUndirectedGraph individual_graph; 
int location_count, edge_count;
vector<Individual> individuals; 
vector<std::tuple<int, int, int>> epoch_statistics;

reset_input(input_graph_filename, benchmark_init_individual_count, location_count, edge_count, individual_graph, individuals);
std::cout << "Location Count: " << location_count << std::endl; 
std::cout << "Edge Count: " << edge_count << std::endl; 

double time_start, time_end, total_time, average_execution_time;

cout << endl << "Running serial..." << std::flush;	
for (size_t benchmark_individual_count = benchmark_init_individual_count; benchmark_individual_count <= benchmark_max_individual_count;
benchmark_individual_count *= benchmark_individual_count_multiplier) {

total_time = 0.0;
average_execution_time = 0.0;
for (std::uint8_t current_repeat = 0; current_repeat != benchmark_repeat_count; ++current_repeat) {
reset_input(input_graph_filename, benchmark_individual_count, location_count, edge_count, individual_graph, individuals); 
time_start = omp_get_wtime();
simulate_serial(benchmark_individual_count, total_epochs, individual_graph, individuals, epoch_statistics);
time_end = omp_get_wtime() - time_start;
total_time += time_end;
cout << "." << flush;
}
average_execution_time = (total_time / benchmark_repeat_count) * 1000.0;

benchmark_string_stream << average_execution_time << "," << execution_type << ","  << 1 << "," << benchmark_individual_count << ","
<< location_count << "," << edge_count << "," << static_cast<int>(total_epochs)
<< "," << 1 << "," << benchmark_repeat_count << std::endl;
}

cout << endl << "Running with OpenMP...";
execution_type = "openmp";

for (size_t benchmark_individual_count = benchmark_init_individual_count; benchmark_individual_count <= benchmark_max_individual_count;
benchmark_individual_count *= benchmark_individual_count_multiplier) {

for (int current_thread_count = benchmark_init_thread_count; current_thread_count <= benchmark_max_thread_count; current_thread_count++) {

omp_set_num_threads(current_thread_count);

total_time = 0.0;
average_execution_time = 0.0;
for (std::uint8_t current_repeat = 0; current_repeat != benchmark_repeat_count; ++current_repeat) {
reset_input(input_graph_filename, benchmark_individual_count, location_count, edge_count, individual_graph, individuals); 
time_start = omp_get_wtime();
simulate_parallel(benchmark_individual_count, total_epochs, individual_graph, individuals, epoch_statistics);
time_end = omp_get_wtime() - time_start;
total_time += time_end;
if (!GraphHandler::assert_epidemic_results(benchmark_individual_count, epoch_statistics))
cout << "Error." << endl << std::flush;
cout << "." << flush;
}

average_execution_time = (total_time / benchmark_repeat_count) * 1000.0;

benchmark_string_stream << average_execution_time << "," << execution_type << "," << current_thread_count << "," << benchmark_individual_count << ","
<< location_count << "," << edge_count << "," << static_cast<int>(total_epochs)
<< "," << 1 << "," << benchmark_repeat_count << std::endl;
}
}

std::cout << std::endl << "Writing results to csv: " << benchmark_file_name << endl;

std::ofstream output_benchmark_csv;
output_benchmark_csv.open(std::string(benchmark_file_name));

output_benchmark_csv << benchmark_string_stream.str();
output_benchmark_csv.close();

std::cout << "Done!" << std::endl;

system("pause");
}

int main() {

bool do_benchmark = false;

if (do_benchmark) {
benchmark();
}
else {

int thread_count = DEFAULT_NUMBER_OF_THREADS;
int individual_count = DEFAULT_INDIVIDUAL_COUNT;
std::uint8_t total_epochs = DEFAULT_TOTAL_EPOCHS;
std::uint8_t repeat_count = DEFAULT_REPEAT_COUNT;
string input_graph_filename = "antwerp.edges";

individual_count = 1000; 
total_epochs = 30; 
thread_count = 4;
repeat_count = 1;

omp_set_num_threads(thread_count);

std::cout << "----- Infectious Disease Modelling -----" << std::endl;
std::cout << "Number of threads: " << thread_count << std::endl;
std::cout << "Individual Count: " << individual_count << std::endl;
std::cout << "Total Epochs: " << static_cast<int>(total_epochs) << std::endl;
std::cout << "Graph from file: " << input_graph_filename << std::endl;
std::cout << "Repeat count: " << static_cast<int>(repeat_count) << std::endl;

LocationUndirectedGraph individual_graph; 
int location_count, edge_count;
vector<Individual> individuals; 
vector<std::tuple<int, int, int>> epoch_statistics;

reset_input(input_graph_filename, individual_count, location_count, edge_count, individual_graph, individuals);
std::cout << "Location Count: " << location_count << std::endl; 
std::cout << "Edge Count: " << edge_count << std::endl; 

double time_start, time_end, total_time;

cout << endl << "Running serial...";
total_time = 0.0;
for (std::uint8_t current_repeat = 0; current_repeat != repeat_count; ++current_repeat) {
reset_input(input_graph_filename, individual_count, location_count, edge_count, individual_graph, individuals); 
time_start = omp_get_wtime();
simulate_serial_naive(individual_count, total_epochs, individual_graph, individuals);
time_end = omp_get_wtime() - time_start;
total_time += time_end;
cout << ".";
}
cout << (total_time / repeat_count) * 1000.0 << " ms" << endl;

cout << endl << "Running with OpenMP...";
total_time = 0.0;
for (std::uint8_t current_repeat = 0; current_repeat != repeat_count; ++current_repeat) {
reset_input(input_graph_filename, individual_count, location_count, edge_count, individual_graph, individuals); 
time_start = omp_get_wtime();
simulate_parallel(individual_count, total_epochs, individual_graph, individuals, epoch_statistics);
time_end = omp_get_wtime() - time_start;
total_time += time_end;
if (!GraphHandler::assert_epidemic_results(individual_count, epoch_statistics))
cout << "Error." << endl;
cout << ".";
}
cout << (total_time / repeat_count) * 1000.0 << " ms" << endl;

system("pause");
}
}