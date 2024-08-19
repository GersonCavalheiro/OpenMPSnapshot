#pragma once

#include <random>    
#include <armadillo> 

namespace pass
{
class seed
{
public:

seed() = delete;
seed(const seed &) = delete;
seed &operator=(const seed &) = delete;


static std::mt19937_64 &get_generator();


static void set_seed(
const arma::arma_rng::seed_type seed);


static void set_random_seed();


static arma::arma_rng::seed_type get_seed();

protected:
static arma::arma_rng::seed_type seed_;
static std::mt19937_64 generator_;
};
} 
