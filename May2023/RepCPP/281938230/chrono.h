

#pragma once

#include "pybind11.h"
#include <cmath>
#include <ctime>
#include <chrono>
#include <datetime.h>

#ifndef PyDateTime_DELTA_GET_DAYS
#define PyDateTime_DELTA_GET_DAYS(o)         (((PyDateTime_Delta*)o)->days)
#endif
#ifndef PyDateTime_DELTA_GET_SECONDS
#define PyDateTime_DELTA_GET_SECONDS(o)      (((PyDateTime_Delta*)o)->seconds)
#endif
#ifndef PyDateTime_DELTA_GET_MICROSECONDS
#define PyDateTime_DELTA_GET_MICROSECONDS(o) (((PyDateTime_Delta*)o)->microseconds)
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <typename type> class duration_caster {
public:
typedef typename type::rep rep;
typedef typename type::period period;

typedef std::chrono::duration<uint_fast32_t, std::ratio<86400>> days;

bool load(handle src, bool) {
using namespace std::chrono;

if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

if (!src) return false;
if (PyDelta_Check(src.ptr())) {
value = type(duration_cast<duration<rep, period>>(
days(PyDateTime_DELTA_GET_DAYS(src.ptr()))
+ seconds(PyDateTime_DELTA_GET_SECONDS(src.ptr()))
+ microseconds(PyDateTime_DELTA_GET_MICROSECONDS(src.ptr()))));
return true;
}
else if (PyFloat_Check(src.ptr())) {
value = type(duration_cast<duration<rep, period>>(duration<double>(PyFloat_AsDouble(src.ptr()))));
return true;
}
else return false;
}

static const std::chrono::duration<rep, period>& get_duration(const std::chrono::duration<rep, period> &src) {
return src;
}

template <typename Clock> static std::chrono::duration<rep, period> get_duration(const std::chrono::time_point<Clock, std::chrono::duration<rep, period>> &src) {
return src.time_since_epoch();
}

static handle cast(const type &src, return_value_policy , handle ) {
using namespace std::chrono;

auto d = get_duration(src);

if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

using dd_t = duration<int, std::ratio<86400>>;
using ss_t = duration<int, std::ratio<1>>;
using us_t = duration<int, std::micro>;

auto dd = duration_cast<dd_t>(d);
auto subd = d - dd;
auto ss = duration_cast<ss_t>(subd);
auto us = duration_cast<us_t>(subd - ss);
return PyDelta_FromDSU(dd.count(), ss.count(), us.count());
}

PYBIND11_TYPE_CASTER(type, _("datetime.timedelta"));
};

template <typename Duration> class type_caster<std::chrono::time_point<std::chrono::system_clock, Duration>> {
public:
typedef std::chrono::time_point<std::chrono::system_clock, Duration> type;
bool load(handle src, bool) {
using namespace std::chrono;

if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

if (!src) return false;

std::tm cal;
microseconds msecs;

if (PyDateTime_Check(src.ptr())) {
cal.tm_sec   = PyDateTime_DATE_GET_SECOND(src.ptr());
cal.tm_min   = PyDateTime_DATE_GET_MINUTE(src.ptr());
cal.tm_hour  = PyDateTime_DATE_GET_HOUR(src.ptr());
cal.tm_mday  = PyDateTime_GET_DAY(src.ptr());
cal.tm_mon   = PyDateTime_GET_MONTH(src.ptr()) - 1;
cal.tm_year  = PyDateTime_GET_YEAR(src.ptr()) - 1900;
cal.tm_isdst = -1;
msecs        = microseconds(PyDateTime_DATE_GET_MICROSECOND(src.ptr()));
} else if (PyDate_Check(src.ptr())) {
cal.tm_sec   = 0;
cal.tm_min   = 0;
cal.tm_hour  = 0;
cal.tm_mday  = PyDateTime_GET_DAY(src.ptr());
cal.tm_mon   = PyDateTime_GET_MONTH(src.ptr()) - 1;
cal.tm_year  = PyDateTime_GET_YEAR(src.ptr()) - 1900;
cal.tm_isdst = -1;
msecs        = microseconds(0);
} else if (PyTime_Check(src.ptr())) {
cal.tm_sec   = PyDateTime_TIME_GET_SECOND(src.ptr());
cal.tm_min   = PyDateTime_TIME_GET_MINUTE(src.ptr());
cal.tm_hour  = PyDateTime_TIME_GET_HOUR(src.ptr());
cal.tm_mday  = 1;   
cal.tm_mon   = 0;   
cal.tm_year  = 70;  
cal.tm_isdst = -1;
msecs        = microseconds(PyDateTime_TIME_GET_MICROSECOND(src.ptr()));
}
else return false;

value = system_clock::from_time_t(std::mktime(&cal)) + msecs;
return true;
}

static handle cast(const std::chrono::time_point<std::chrono::system_clock, Duration> &src, return_value_policy , handle ) {
using namespace std::chrono;

if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

std::time_t tt = system_clock::to_time_t(time_point_cast<system_clock::duration>(src));
std::tm localtime = *std::localtime(&tt);

using us_t = duration<int, std::micro>;

return PyDateTime_FromDateAndTime(localtime.tm_year + 1900,
localtime.tm_mon + 1,
localtime.tm_mday,
localtime.tm_hour,
localtime.tm_min,
localtime.tm_sec,
(duration_cast<us_t>(src.time_since_epoch() % seconds(1))).count());
}
PYBIND11_TYPE_CASTER(type, _("datetime.datetime"));
};

template <typename Clock, typename Duration> class type_caster<std::chrono::time_point<Clock, Duration>>
: public duration_caster<std::chrono::time_point<Clock, Duration>> {
};

template <typename Rep, typename Period> class type_caster<std::chrono::duration<Rep, Period>>
: public duration_caster<std::chrono::duration<Rep, Period>> {
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
