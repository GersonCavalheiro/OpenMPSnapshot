#pragma once

#include "pass_bits/config.hpp"
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

#if defined(SUPPORT_MPI)
#include <mpi.h>
#endif
namespace pass
{

class problem
{
public:

const arma::vec lower_bounds;


const arma::vec upper_bounds;


const std::string name;


arma::vec bounds_range() const noexcept;


arma::uword dimension() const noexcept;


problem(const arma::uword dimension, const double lower_bound,
const double upper_bound, const std::string &name);


problem(const arma::vec &lower_bounds, const arma::vec &upper_bounds, const std::string &name);


virtual double evaluate(const arma::vec &agent) const = 0;


double evaluate_normalised(const arma::vec &normalised_agent) const;


arma::mat normalised_random_agents(const arma::uword count) const;


arma::mat normalised_hammersley_agents(const arma::uword count) const;


arma::mat initialise_normalised_agents(const arma::uword count) const;
};

} 
