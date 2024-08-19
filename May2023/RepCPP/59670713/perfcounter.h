#pragma once

#include "octreebuilder_api.h"

#include <chrono>
#include <iosfwd>

namespace octreebuilder {

class OCTREEBUILDER_API PerfCounter {
public:
PerfCounter();

void start();
::std::chrono::high_resolution_clock::duration stop();

void resume();

::std::chrono::high_resolution_clock::duration elapsedTime() const;

private:
::std::chrono::high_resolution_clock::time_point m_start;
::std::chrono::high_resolution_clock::duration m_sumDurations;
bool m_stopped;
};

::std::ostream& operator<<(::std::ostream& os, const PerfCounter& pc);

::std::ostream& logPerf();

#ifdef PROFILING_ENABLED
#define LOG_PROF(msg) logPerf() << msg << '\n'
#else
#define LOG_PROF(msg)
#endif

}

