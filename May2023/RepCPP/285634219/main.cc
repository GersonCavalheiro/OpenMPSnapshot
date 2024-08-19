

#include <cassert>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "multians.h"

#define NUM_SYMBOLS 256
#define NUM_STATES 1024

#define SEED 5


#define SUBSEQUENCE_SIZE 4

#define THREADS_PER_BLOCK 128

void run(long int input_size) {

std::cout << "\u03BB | compressed size (bytes) | ";
std::cout << std::endl << std::endl;

auto start = std::chrono::steady_clock::now();

for(float lambda = 0.1f; lambda < 2.5f; lambda += 0.16) {

std::cout << std::left << std::setw(5) << lambda << std::setfill(' ');

auto dist = ANSTableGenerator::generate_distribution(
SEED, NUM_SYMBOLS, NUM_STATES,
[&](double x) {return lambda * exp(-lambda * x);});

auto random_data = ANSTableGenerator::generate_test_data(
dist.dist, input_size, NUM_STATES, SEED);

auto table = ANSTableGenerator::generate_table(
dist.prob, dist.dist, nullptr, NUM_SYMBOLS,
NUM_STATES);

auto encoder_table = ANSTableGenerator::generate_encoder_table(table);

auto decoder_table = ANSTableGenerator::get_decoder_table(encoder_table);

auto input_buffer = ANSEncoder::encode(
random_data->data(), input_size, encoder_table);

auto output_buffer = std::make_shared<CUHDOutputBuffer>(input_size);

size_t compressed_size = input_buffer->get_compressed_size();
size_t input_buffer_size = (compressed_size + 4);
UNIT_TYPE *d_input_buffer = input_buffer->get_compressed_data(); 

size_t decoder_table_size = decoder_table->get_size();
std::uint32_t *d_decoder_table = reinterpret_cast<std::uint32_t*>(decoder_table->get()) ;

size_t output_buffer_size = output_buffer->get_uncompressed_size();
SYMBOL_TYPE *d_output_buffer = output_buffer->get_decompressed_data().get();

size_t num_subseq = SDIV(compressed_size, SUBSEQUENCE_SIZE);
size_t num_blocks = SDIV(num_subseq, THREADS_PER_BLOCK);

uint4 *d_sync_info = (uint4*) calloc (num_subseq, sizeof(uint4));

std::uint32_t *d_output_sizes = (std::uint32_t*) malloc (num_subseq * sizeof(std::uint32_t));

std::uint8_t *d_sequence_synced = (std::uint8_t*) calloc (num_blocks, sizeof(std::uint8_t));

#pragma omp target data map(to: d_input_buffer[0:input_buffer_size],\
d_decoder_table[0:decoder_table_size],\
d_sync_info[0:num_subseq],\
d_sequence_synced[0:num_blocks]),\
map(alloc: d_output_sizes[0:num_subseq]),\
map(from: d_output_buffer[0:output_buffer_size])
{
cuhd::CUHDGPUDecoder::decode(
d_input_buffer, input_buffer->get_compressed_size(),
d_output_buffer, output_buffer->get_uncompressed_size(),
d_decoder_table,
d_sync_info,
d_output_sizes,
d_sequence_synced,
input_buffer->get_first_state(),
input_buffer->get_first_bit(), 
decoder_table->get_num_entries(),
11, 
SUBSEQUENCE_SIZE, 
THREADS_PER_BLOCK);
}

output_buffer->reverse();

if(cuhd::CUHDUtil::equals(random_data->data(),
output_buffer->get_decompressed_data().get(), input_size));
else std::cout << "********* MISMATCH ************" << std::endl;

std::cout << std::left << std::setw(10)
<< input_buffer->get_compressed_size() * sizeof(UNIT_TYPE)
<< std::setfill(' ') << std::endl;

free(d_sync_info);
free(d_output_sizes);
free(d_sequence_synced);
}

auto end = std::chrono::steady_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
std::cout << "Total elapsed time " << time * 1e-9f << " (s)\n";
}

int main(int argc, char **argv) {

const char* bin = argv[0];

auto print_help = [&]() {
std::cout << "USAGE: " << bin << "<size of input in megabytes> " << std::endl;
};

if(argc < 2) {print_help(); return 1;}

const long int size = atoi(argv[1]) * 1024 * 1024;

if(size < 1) {
print_help();
return 1;
}

assert(SUBSEQUENCE_SIZE % 4 == 0);

run(size);

return 0;
}

