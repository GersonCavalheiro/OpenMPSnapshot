#pragma once

#include "../include/tpool/tpool.hpp"

#include <string>
#include <vector>

enum class thread_alloc : int { dynamic = 0, hardware = 1 };

class block final {
public:
block(uint32_t index, const std::string &data,
thread_alloc option = thread_alloc::hardware);

void mine_block(uint32_t difficulty, uint32_t max_difficulty) noexcept;

inline const std::string &get_hash() const noexcept { return _hash; }

std::string prev_hash;

private:
uint32_t _index;

mutable std::shared_ptr<std::atomic<uint64_t>> _nonce;
mutable std::shared_ptr<std::atomic<bool>> _modified_hash;

std::string _data;
std::string _hash;
long _time;

unsigned int _thread_num;
thread_alloc _thread_alloc;

std::string calculate_hash() const noexcept;
};

class block_chain final {
private:
uint32_t _difficulty;
uint32_t _max_difficulty;
std::vector<block> _chain;

inline const block &get_last_block() const noexcept { return _chain.back(); }

public:
block_chain();
block_chain(uint32_t difficulty, uint32_t max_difficulty);

void add_block(block &&new_block) noexcept;
};