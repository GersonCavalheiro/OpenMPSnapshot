#pragma once

inline double Pi_Sequential(int step_count) noexcept
{
const double dx = 1.0 / step_count;

double integral = 0.0;
for (int i = 0; i < step_count; ++i)
{
const double a  = i * dx;
const double b  = a + dx;
const double fa = 1.0 / (1.0 + a*a);
const double fb = 1.0 / (1.0 + b*b);
integral += fa + fb;
}

return 2.0 * dx * integral;
}

inline double Pi_Parallel(int step_count) noexcept
{
const double dx = 1.0 / step_count;

double integral = 0.0;
#pragma omp parallel for schedule(static) default(none) firstprivate(step_count) reduction(+:integral)
for (int i = 0; i < step_count; ++i)
{
const double a  = i * dx;
const double b  = a + dx;
const double fa = 1.0 / (1.0 + a*a);
const double fb = 1.0 / (1.0 + b*b);
integral += fa + fb;
}

return 2.0 * dx * integral;
}

inline double Pi_ParallelWithoutReduction(int step_count) noexcept
{
const double dx = 1.0 / step_count;

double integral = 0.0;
#pragma omp parallel for schedule(static) default(none) firstprivate(step_count) shared(integral)
for (int i = 0; i < step_count; ++i)
{
const double a  = i * dx;
const double b  = a + dx;
const double fa = 1.0 / (1.0 + a*a);
const double fb = 1.0 / (1.0 + b*b);
#pragma omp critical
integral += fa + fb;
}

return 2.0 * dx * integral;
}
