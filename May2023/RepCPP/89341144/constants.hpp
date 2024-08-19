#pragma once

#include <array>
#include <utility>

namespace pass
{

struct celestial_body
{
public:
static const celestial_body SUN;
static const celestial_body MERCURY;
static const celestial_body VENUS;
static const celestial_body EARTH;
static const celestial_body MARS;
static const celestial_body JUPITER;
static const celestial_body SATURN;
static const celestial_body URANUS;
static const celestial_body NEPTUNE;

double mu;
double penalty;
double penalty_coefficient;


std::pair<std::array<double, 3>, std::array<double, 3>> ephemeris(
const double mjd2000) const;

private:
celestial_body(double mu, double penalty,
double penalty_coefficient) noexcept;
};

struct asteroid
{
public:
asteroid(std::array<double, 6> keplerian, double epoch, double mu) noexcept;

std::array<double, 6> keplerian;
double epoch;
double mu;


std::pair<std::array<double, 3>, std::array<double, 3>> ephemeris(
const double jd) const;
};
} 
