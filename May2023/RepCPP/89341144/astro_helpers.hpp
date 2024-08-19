#pragma once

#include <array>   
#include <utility> 

namespace pass
{

typedef struct lambert_solution
{
std::array<double, 3> departure_velocity;
std::array<double, 3> arrival_velocity;
} lambert_solution;


lambert_solution lambert(std::array<double, 3> r1_in,
std::array<double, 3> r2_in, double t, const double mu,
const int lw);


std::pair<double, double> pow_swing_by_inv(const double Vin, const double Vout,
const double alpha);


std::pair<std::array<double, 3>, std::array<double, 3>> conversion(
const std::array<double, 6> &E, const double mu);


double mean_to_eccentric(const double m, const double e);
} 
