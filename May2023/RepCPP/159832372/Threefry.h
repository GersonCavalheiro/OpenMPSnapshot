#pragma once

#include "Random123/threefry.h"

#include <zlib.h>

#include <vector>
#include <utility>
#include <cfloat>

class Threefry
{
public:
typedef r123::Threefry2x64_R<20> Threefry123;

private:
std::vector<unsigned long long> counters_;
int X_, Y_, N_;
Threefry123::key_type seed_;

public:

enum Phase {REPROD = 0, MUTATION = 1, NPHASES};

Threefry(int X, int Y, unsigned int seed) : counters_(X*Y*NPHASES, 0), X_(X), Y_(Y), N_(X*Y) {
seed_[0] = 0;
seed_[1] = seed;
}
Threefry(int X, int Y, gzFile backup_file);

std::vector<unsigned long long>& counters() { return counters_; }

class Gen
{
Threefry123::ctr_type state_;
Threefry123 gen_;

Threefry* parent_;

Gen(Threefry* parent, size_t idx, Phase phase)
: parent_(parent)
{
state_[0] = idx + parent_->N_*phase;
state_[1] = parent_->counters_[state_[0]];
}
Gen(Threefry* parent, int x, int y, Phase phase)
: Gen(parent, y*parent_->X_ + x , phase)
{}

friend class Threefry;

public:

Gen(const Gen&) = delete;
Gen& operator=(const Gen&) = delete;

Gen(Gen&& other) {
parent_ = other.parent_;
other.parent_ = nullptr;
state_ = other.state_;
}
Gen& operator=(Gen&& other) {
parent_ = other.parent_;
other.parent_ = nullptr;
state_ = other.state_;
}

~Gen() {
if(parent_)
parent_->counters_[state_[0]] = state_[1];
}

Threefry123::ctr_type random_raw() {
++state_[1];
return gen_(state_, parent_->seed_);
}

double random() {
return (random_raw()[0]&((1llu<<48)-1))/double(1llu<<48);
}

unsigned int random(unsigned int max) {
return random()*max;
}

int32_t roulette_random(double* probs, int32_t nb_elts, bool verbose = false);
int32_t binomial_random(int32_t nb, double prob); 
};

Gen gen(unsigned int x, unsigned int y, Phase phase) {
return std::move(Gen(this, x, y, phase));
}
Gen gen(size_t idx, Phase phase) {
return std::move(Gen(this, idx, phase));
}

void save(gzFile backup_file) const;

class Device;
class DeviceCollectiveBlock;

inline void initDevice();
};
