#ifndef SPLATT_TIMER_H
#define SPLATT_TIMER_H
#include <time.h>
#include <stddef.h>
#include <stdbool.h>
#ifdef __MACH__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif
typedef struct
{
bool running;
double seconds;
double start;
double stop;
} sp_timer_t;
typedef enum
{
TIMER_LVL0,   
TIMER_ALL,
TIMER_CPD,
TIMER_REORDER,
TIMER_CONVERT,
TIMER_LVL1,   
TIMER_MTTKRP,
TIMER_INV,
TIMER_FIT,
TIMER_MATMUL,
TIMER_ATA,
TIMER_MATNORM,
TIMER_IO,
TIMER_PART,
TIMER_LVL2,   
#ifdef SPLATT_USE_MPI
TIMER_MPI,
TIMER_MPI_IDLE,
TIMER_MPI_COMM,
TIMER_MPI_ATA,
TIMER_MPI_REDUCE,
TIMER_MPI_PARTIALS,
TIMER_MPI_NORM,
TIMER_MPI_UPDATE,
TIMER_MPI_FIT,
TIMER_MTTKRP_MAX,
TIMER_MPI_MAX,
TIMER_MPI_IDLE_MAX,
TIMER_MPI_COMM_MAX,
#endif
TIMER_SPLATT,
TIMER_GIGA,
TIMER_DFACTO,
TIMER_TTBOX,
TIMER_SORT,
TIMER_TILE,
TIMER_MISC,
TIMER_NTIMERS 
} timer_id;
extern int timer_lvl;
extern sp_timer_t timers[TIMER_NTIMERS];
#define init_timers splatt_init_timers
void init_timers(void);
#define report_times splatt_report_times
void report_times(void);
#define timer_inc_verbose splatt_timer_inc_verbose
void timer_inc_verbose(void);
static inline double monotonic_seconds()
{
#ifdef __MACH__
static mach_timebase_info_data_t info;
static double seconds_per_unit;
if(seconds_per_unit == 0) {
#pragma omp critical
{
mach_timebase_info(&info);
seconds_per_unit = (info.numer / info.denom) / 1e9;
}
}
return seconds_per_unit * mach_absolute_time();
#else
struct timespec ts;
clock_gettime(CLOCK_MONOTONIC, &ts);
return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}
static inline void timer_reset(sp_timer_t * const timer)
{
timer->running = false;
timer->seconds = 0;
timer->start   = 0;
timer->stop    = 0;
}
static inline void timer_start(sp_timer_t * const timer)
{
if(!timer->running) {
timer->running = true;
timer->start = monotonic_seconds();
}
}
static inline void timer_stop(sp_timer_t * const timer)
{
timer->running = false;
timer->stop = monotonic_seconds();
timer->seconds += timer->stop - timer->start;
}
static inline void timer_fstart(sp_timer_t * const timer)
{
timer_reset(timer);
timer_start(timer);
}
#endif
