#pragma once

#include <cstdlib>
#include <cstdint>
#include <utility>
#include <chrono>
#include <curand_kernel.h>

struct MonteCarloResult
{
std::uint64_t pointsInCircle{};
std::uint64_t totalPoints{};

double calculateRatio() const
{
return static_cast<double>(pointsInCircle) / static_cast<double>(totalPoints);
}
};

template <typename Callable, typename... Args>
auto runWithTimeMeasurementCpu(Callable&& function, Args&&... params)
{
const auto start = std::chrono::high_resolution_clock::now();
const auto result = std::forward<Callable>(function)(std::forward<Args>(params)...);
const auto stop = std::chrono::high_resolution_clock::now();
const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
return std::make_pair(result, duration);
}

__global__ void monteCarlo_cuda(std::size_t*);
MonteCarloResult monteCarlo();