#pragma once

#include "pass_bits/problem.hpp"
#include "pass_bits/helper/stopwatch.hpp"
#include "pass_bits/config.hpp"
#include <stdexcept> 
#include <chrono>    
#include <armadillo> 
#include <cassert>   
#include <string>    

#if defined(SUPPORT_OPENMP)
#include <omp.h>
#endif

#if defined(SUPPORT_MPI)
#include <mpi.h>
#endif

namespace pass
{

struct optimise_result
{

arma::vec normalised_agent;


double fitness_value;


const double acceptable_fitness_value;


const pass::problem &problem;


arma::uword iterations;


arma::uword evaluations;


std::chrono::nanoseconds duration;

optimise_result(const pass::problem &problem,
const double acceptable_fitness_value) noexcept;


bool solved() const;


arma::vec agent() const;
};


class optimiser
{
public:

double acceptable_fitness_value;


arma::uword maximal_iterations;


arma::uword maximal_evaluations;


std::chrono::nanoseconds maximal_duration;


const std::string name;


optimiser(const std::string &name);


virtual optimise_result optimise(const pass::problem &problem) = 0;
};

} 
