#include "windows.h"

#pragma once

class CStopWatch
{
public:
CStopWatch();
virtual ~CStopWatch();

public:
void Start();
void End();
const float GetDurationSecond() const { return m_fTimeforDuration; }
const float GetDurationMilliSecond() const { return m_fTimeforDuration*1000.f; }

protected:
LARGE_INTEGER		m_swFreq, m_swStart, m_swEnd;
float				m_fTimeforDuration;
};