#include "block_chain.h"
#include "sha256.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <thread>

using namespace std;

block::block(uint32_t index, const string &data, thread_alloc option)
: _index(index),
_data(data),
_nonce(0),
_modified_hash(false),
_time(static_cast<long>(index)),
_thread_alloc(option) {}

void block::mine_block(uint32_t difficulty, uint32_t max_difficulty) noexcept {
const auto optimal_thread_count = [difficulty,
max_difficulty]() -> unsigned int {
constexpr auto potentially_used_thread_count = 1u;  
constexpr auto minimum_thread_count = 2u;
const auto calculate_optimal = [difficulty, max_difficulty] {
const auto hardware_thread_count =
thread::hardware_concurrency() - potentially_used_thread_count;
const auto dynamic_thread_count =
minimum_thread_count + max_difficulty -
abs(static_cast<int>(max_difficulty / difficulty));
if (difficulty < 2) {
return dynamic_thread_count;
} else if (difficulty == 2) {
return static_cast<unsigned int>(pow(minimum_thread_count, difficulty));
}
return max(hardware_thread_count, dynamic_thread_count);
};
return calculate_optimal();
};

if (_thread_alloc == thread_alloc::hardware) {
_thread_num = thread::hardware_concurrency();
} else {
_thread_num = optimal_thread_count();
}

auto concurrent_calculate_hash = [difficulty, this] {
const string str(difficulty, '0');
while (!_modified_hash) {
if (const auto local_hash = calculate_hash();
local_hash.substr(0, difficulty) == str && !_modified_hash) {
#pragma omp critical
{
_modified_hash = true;
}
_hash = local_hash;
}
}
};

const auto thread_num = _thread_num;
cout << "Running using " << thread_num << " threads" << endl;

#pragma omp parallel num_threads(thread_num) default(none) \
shared(concurrent_calculate_hash)
{ std::invoke(concurrent_calculate_hash); }

cout << "Block mined: " << _hash << endl;
}

std::string block::calculate_hash() const noexcept {
stringstream ss;
#pragma omp critical
{ ss << _index << _time << _data << ++_nonce << prev_hash; }
return sha256(ss.str());
}

block_chain::block_chain() : _difficulty(1), _max_difficulty(1) {
_chain.emplace_back(block(0, "Genesis Block"));
}

block_chain::block_chain(uint32_t difficulty, uint32_t max_difficulty)
: _difficulty(difficulty), _max_difficulty(max_difficulty) {
_chain.emplace_back(block(0, "Genesis Block"));
}

void block_chain::add_block(block &&new_block) noexcept {
new_block.prev_hash = get_last_block().get_hash();
new_block.mine_block(_difficulty, _max_difficulty);
_chain.push_back(new_block);
}
