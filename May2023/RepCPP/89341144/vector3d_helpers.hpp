#pragma once
#include <array>

namespace pass
{
namespace gtoc
{

std::array<double, 3> cross_product(std::array<double, 3> v1,
std::array<double, 3> v2);

double dot_product(std::array<double, 3> v1, std::array<double, 3> v2);

double norm(std::array<double, 3> v);

std::array<double, 3> unit_vector(const std::array<double, 3> &v);

std::array<double, 3> add(std::array<double, 3> v1, std::array<double, 3> v2);
std::array<double, 3> sub(std::array<double, 3> v1, std::array<double, 3> v2);

std::array<double, 3> mul(std::array<double, 3> v1, const double r);
std::array<double, 3> div(std::array<double, 3> v1, const double r);
} 
} 
