

#pragma once

#include "rocalution_arguments_config.hpp"
#include "rocalution_enum_itsolver.hpp"

class rocalution_bench
{
public:
rocalution_bench();

rocalution_bench(int& argc, char**& argv);

rocalution_bench& operator()(int& argc, char**& argv);

bool run();

bool execute();

int get_device_id() const;

void info_devices(std::ostream& out_) const;

private:
void parse(int& argc, char**& argv, rocalution_arguments_config& config);

options_description desc;

rocalution_arguments_config config;
};

std::string rocalution_get_version();
