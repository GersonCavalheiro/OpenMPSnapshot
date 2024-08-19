
#pragma once
#include "HelperAliasis.hpp"

namespace nVK {
void gBenchMandelbrot();
void gBenchMandelbrotOld();
void gBenchParallel(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);
}
