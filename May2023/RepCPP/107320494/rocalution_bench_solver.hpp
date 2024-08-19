
#pragma once

#include "rocalution_bench_itsolver_impl.hpp"

template <typename T>
struct rocalution_bench_solver
{

private:
rocalution_bench_itsolver<T>*             m_itsolver{};
rocalution_bench_solver_results           m_output_results{};
const rocalution_bench_solver_parameters* m_input_parameters{};

public:
~rocalution_bench_solver()
{
if(this->m_itsolver)
{
delete this->m_itsolver;
this->m_itsolver = nullptr;
}
}

rocalution_bench_solver(const rocalution_bench_solver_parameters* config)
: m_input_parameters(config)
{
const rocalution_enum_itsolver enum_itsolver
= this->m_input_parameters->GetEnumIterativeSolver();
if(enum_itsolver.is_invalid())
{
rocalution_bench_errmsg << "invalid iterative solver." << std::endl;
throw false;
}

this->m_itsolver = nullptr;
switch(enum_itsolver.value)
{

#define ROCALUTION_ENUM_ITSOLVER_TRANSFORM(x_)                                                  \
case rocalution_enum_itsolver::x_:                                                          \
{                                                                                           \
this->m_itsolver = new rocalution_bench_itsolver_impl<rocalution_enum_itsolver::x_, T>( \
this->m_input_parameters, &this->m_output_results);                                 \
break;                                                                                  \
}

ROCALUTION_ENUM_ITSOLVER_TRANSFORM_EACH;

#undef ROCALUTION_ENUM_ITSOLVER_TRANSFORM
}

if(this->m_itsolver == nullptr)
{
rocalution_bench_errmsg << "iterative solver instantiation failed." << std::endl;
throw false;
}
}

bool Run()
{
if(this->m_itsolver != nullptr)
{
return this->m_itsolver->Run(this->m_output_results);
}
return false;
}
};
