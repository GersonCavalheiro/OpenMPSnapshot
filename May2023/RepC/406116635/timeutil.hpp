#pragma once
#include <sstream>
#include <time.h>
inline std::string formatDateTime( const tm& tm )
{
std::stringstream ss;
switch ( tm.tm_wday )
{
case 0: ss << "Sun "; break;
case 1: ss << "Mon "; break;
case 2: ss << "Tue "; break;
case 3: ss << "Wed "; break;
case 4: ss << "Thu "; break;
case 5: ss << "Fri "; break;
case 6: ss << "Sat "; break;
}
if ( tm.tm_mday < 10 ) ss << "0" << tm.tm_mday << ".";
else ss << tm.tm_mday << ".";
if ( tm.tm_mon+1 < 10 ) ss << "0" << tm.tm_mon+1 << ".";
else ss << tm.tm_mon+1 << ".";
ss << tm.tm_year+1900 << " ";
ss << tm.tm_hour << ":";
if ( tm.tm_min < 10 ) ss << "0" << tm.tm_min << ":";
else ss << tm.tm_min << ":";
if ( tm.tm_sec < 10 ) ss << "0" << tm.tm_sec;
else ss << tm.tm_sec;
return ss.str();
}
inline std::string formatDateTimeFilename( const tm& tm )
{
std::stringstream ss;
ss << tm.tm_year+1900 << "_";
if ( tm.tm_mon+1 < 10 ) ss << "0" << tm.tm_mon+1 << "_";
else ss << tm.tm_mon+1 << "_";
if ( tm.tm_mday < 10 ) ss << "0" << tm.tm_mday << "_";
else ss << tm.tm_mday << "_";
ss << tm.tm_hour << "_";
if ( tm.tm_min < 10 ) ss << "0" << tm.tm_min << "_";
else ss << tm.tm_min << "_";
if ( tm.tm_sec < 10 ) ss << "0" << tm.tm_sec;
else ss << tm.tm_sec;
return ss.str();
}
inline std::string getDateTimeStr()
{
time_t t = time( NULL );
#ifndef LINUX
tm now;
localtime_r( &t, &now);
return formatDateTime( now );
#else
tm* now = localtime( &t );
return formatDateTime( *now );
#endif
}
inline std::string getDateTimeFilenameStr()
{
time_t t = time( NULL );
#ifndef LINUX
tm now;
localtime_r( &t, &now);
return formatDateTimeFilename( now );
#else
tm* now = localtime( &t );
return formatDateTimeFilename( *now );
#endif
}