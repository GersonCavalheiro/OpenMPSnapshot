#pragma once
#include <costa/grid2grid/block.hpp>
#include <costa/grid2grid/memory_utils.hpp>

#include <chrono>
#include <memory>
#include <vector>

namespace costa {
template <typename T>
class message {
public:
message() = default;

message(block<T> b, int rank, 
char ordering,
T alpha, T beta,
bool trans, bool conj
);

block<T> get_block() const;

int get_rank() const;

bool operator<(const message<T> &other) const;

std::string to_string() const;

T alpha = T{1};
T beta = T{0};
bool transpose = false;
bool conjugate = false;
bool col_major = true;

private:
block<T> b;
int rank = 0;
};

template <typename T>
class communication_data {
public:
std::vector<int> dspls;
std::vector<int> counts;
std::vector<message<T>> mpi_messages;
std::vector<message<T>> local_messages;
int n_ranks = 0;
int total_size = 0;
int my_rank;
int n_packed_messages = 0;

CommType type;

communication_data() = default;

communication_data(std::vector<message<T>> &msgs, 
int my_rank, int n_ranks, 
CommType type);

void copy_to_buffer();
void copy_from_buffer();

void copy_from_buffer(int idx);

T *data();

void partition_messages();

private:
std::vector<int> package_ticks;
std::vector<int> offset_per_message;
};

template <typename T>
void copy_local_blocks(std::vector<message<T>>& from, 
std::vector<message<T>>& to);
} 
