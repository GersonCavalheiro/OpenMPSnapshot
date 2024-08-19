#pragma once

#include <omp.h>

#include <chrono>
#include <memory>
#include <thread>
#include <random>

class shared_generator{
protected:
std::mt19937* seed(){
#ifdef _OPENMP
int thread_num =  omp_get_thread_num();
#endif
static thread_local std::unique_ptr<std::mt19937> generator;
if(!generator){
const auto cc = std::chrono::high_resolution_clock::now().time_since_epoch().count();
#ifdef _OPENMP
generator = std::make_unique<std::mt19937>(cc+thread_num);
#else
generator = std::make_unique<std::mt19937>(cc);
#endif
}
return generator.get();
}
};

template<typename T>
class uniform_dist : public shared_generator{
public:
uniform_dist(T inf, T sup) : _distr(inf, sup){}

T operator()(){ return _distr(*seed()); }

private:
template<typename dType>
using dist_uniform_t = std::conditional_t<std::is_integral<dType>::value, std::uniform_int_distribution<dType>, std::uniform_real_distribution<dType>>;
dist_uniform_t<T> _distr;
};

template<typename T>
class normal_dist : public shared_generator{
public:
normal_dist(T mean, float sd) : _distr(mean, sd){}

T operator()(){
return _distr(*seed());
}

private:
template<typename dType>
using dist_normal_t =  std::conditional_t<std::is_integral<dType>::value, std::binomial_distribution<dType>, std::normal_distribution<dType>>;
dist_normal_t<T> _distr;
};
