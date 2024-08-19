#pragma once

#include <vector>

namespace Filter
{
using KernelRow = std::vector<float>;
using Kernel = std::vector<KernelRow>;

auto blurKernel()
{
Kernel filter = {
KernelRow{ 0.0625, 0.125, 0.0625 },
KernelRow{ 0.125, 0.25, 0.125 },
KernelRow{ 0.0625, 0.125, 0.0625 } };
return filter;
}

auto sharpenKernel()
{
Kernel filter = {
KernelRow{ 0, -1, 0 },
KernelRow{ -1, 5, -1 },
KernelRow{ 0, -1, 0 } };
return filter;
}

auto edgeDetectionKernel()
{
Kernel filter = {
KernelRow{ 0, 1, 0 },
KernelRow{ 1, -4, 1 },
KernelRow{ 0, 1, 0 } };
return filter;
}

auto embossKernel()
{
Kernel filter = {
KernelRow{ -2, -1, 0 },
KernelRow{ -1, 1, 1 },
KernelRow{ 0, 1, 2 } };
return filter;
}

auto outlineKernel()
{
Kernel filter = {
KernelRow{ -1, -1, -1 },
KernelRow{ -1, +8, -1 },
KernelRow{ -1, -1, -1 } };
return filter;
}
}