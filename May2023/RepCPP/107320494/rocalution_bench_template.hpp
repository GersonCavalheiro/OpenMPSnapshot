

#pragma once

#include "rocalution_bench_solver.hpp"

template <typename T>
bool rocalution_bench_template(const rocalution_arguments_config& config)
{
try
{
rocalution_bench_solver<T> bench_solver(&config);
bool                       success = bench_solver.Run();
return success;
}
catch(bool)
{
rocalution_bench_errmsg << "rocalution_bench_template failure" << std::endl;
;
return false;
}
catch(std::exception&)
{
rocalution_bench_errmsg << "rocalution_bench_template unknown failure" << std::endl;
;
return false;
}
}
