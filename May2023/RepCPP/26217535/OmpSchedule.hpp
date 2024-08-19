

#pragma once

#include <alpaka/core/Common.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <cstdint>


namespace alpaka::omp
{
struct Schedule
{
enum Kind
{
NoSchedule,
Static = 1u,
Dynamic = 2u,
Guided = 3u,
#if defined _OPENMP && _OPENMP >= 200805
Auto = 4u,
#endif
Runtime = 5u
};

Kind kind;

int chunkSize;

ALPAKA_FN_HOST constexpr Schedule(Kind myKind = NoSchedule, int myChunkSize = 0)
: kind(myKind)
, chunkSize(myChunkSize)
{
}
};

ALPAKA_FN_HOST inline auto getSchedule()
{
#if defined _OPENMP && _OPENMP >= 200805
omp_sched_t ompKind;
int chunkSize = 0;
omp_get_schedule(&ompKind, &chunkSize);
return Schedule{static_cast<Schedule::Kind>(ompKind), chunkSize};
#else
return Schedule{};
#endif
}

ALPAKA_FN_HOST inline void setSchedule(Schedule schedule)
{
if((schedule.kind != Schedule::NoSchedule) && (schedule.kind != Schedule::Runtime))
{
#if defined _OPENMP && _OPENMP >= 200805
omp_set_schedule(static_cast<omp_sched_t>(schedule.kind), schedule.chunkSize);
#endif
}
}
} 
