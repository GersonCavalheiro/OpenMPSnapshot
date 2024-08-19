#pragma once
#include <windows.h>

struct Timer
{
void Start()
{
QueryPerformanceCounter(&m_start);
}

void Stop()
{
QueryPerformanceCounter(&m_stop);
}

double Elapsed()
{
return (m_stop.QuadPart - m_start.QuadPart - m_overhead) \
* 1000.0 / m_freq.QuadPart;
}

private:

static LONGLONG GetOverhead()
{
Timer t;
t.Start();
t.Stop();
return t.m_stop.QuadPart - t.m_start.QuadPart;
}

LARGE_INTEGER m_start;
LARGE_INTEGER m_stop;
static LARGE_INTEGER m_freq;
static LONGLONG m_overhead;
};