

#pragma once

#include <iostream>
#include <sstream>

#include "program_options.hpp"
#include "utility.hpp"

#include "rocalution_bench_solver_parameters.hpp"

struct rocalution_arguments_config : rocalution_bench_solver_parameters
{

char        precision;
char        indextype;
std::string function;
int         device_id;

rocalution_arguments_config();

void set_description(options_description& desc);

int parse(int& argc, char**& argv, options_description& desc);

int parse_no_default(int& argc, char**& argv, options_description& desc);
};
