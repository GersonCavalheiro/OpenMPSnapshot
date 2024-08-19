

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <chrono>

namespace Aws
{
namespace Utils
{
enum class DateFormat
{
RFC822, 
ISO_8601, 
AutoDetect
};

enum class Month
{
January = 0,
February,
March,
April,
May,
June,
July,
August,
September,
October,
November,
December
};

enum class DayOfWeek
{
Sunday = 0,
Monday,
Tuesday,
Wednesday,
Thursday,
Friday,
Saturday
};


class AWS_CORE_API DateTime
{
public:

DateTime();


DateTime(const std::chrono::system_clock::time_point& timepointToAssign);


DateTime(int64_t millisSinceEpoch);


DateTime(double epoch_millis);


DateTime(const Aws::String& timestamp, DateFormat format);


DateTime(const char* timestamp, DateFormat format);

bool operator == (const DateTime& other) const;
bool operator < (const DateTime& other) const;
bool operator > (const DateTime& other) const;
bool operator != (const DateTime& other) const;
bool operator <= (const DateTime& other) const;
bool operator >= (const DateTime& other) const;

DateTime operator+(const std::chrono::milliseconds& a) const;
DateTime operator-(const std::chrono::milliseconds& a) const;


DateTime& operator=(double secondsSinceEpoch);


DateTime& operator=(int64_t millisSinceEpoch);


DateTime& operator=(const std::chrono::system_clock::time_point& timepointToAssign);


inline bool WasParseSuccessful() { return m_valid; }


Aws::String ToLocalTimeString(DateFormat format) const;


Aws::String ToLocalTimeString(const char* formatStr) const;


Aws::String ToGmtString(DateFormat format) const;


Aws::String ToGmtString(const char* formatStr) const;


double SecondsWithMSPrecision() const;


int64_t Millis() const;


std::chrono::system_clock::time_point UnderlyingTimestamp() const;


int GetYear(bool localTime = false) const;


Month GetMonth(bool localTime = false) const;


int GetDay(bool localTime = false) const;


DayOfWeek GetDayOfWeek(bool localTime = false) const;


int GetHour(bool localTime = false) const;


int GetMinute(bool localTime = false) const;


int GetSecond(bool localTime = false) const;


bool IsDST(bool localTime = false) const;


static DateTime Now(); 


static int64_t CurrentTimeMillis();


static Aws::String CalculateLocalTimestampAsString(const char* formatStr);


static Aws::String CalculateGmtTimestampAsString(const char* formatStr);


static int CalculateCurrentHour();


static double ComputeCurrentTimestampInAmazonFormat();


static std::chrono::milliseconds Diff(const DateTime& a, const DateTime& b);

private:
std::chrono::system_clock::time_point m_time;
bool m_valid;

void ConvertTimestampStringToTimePoint(const char* timestamp, DateFormat format);
tm GetTimeStruct(bool localTime) const;
tm ConvertTimestampToLocalTimeStruct() const;
tm ConvertTimestampToGmtStruct() const;   
};

} 
} 
