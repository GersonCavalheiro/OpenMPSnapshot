

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "tracetypes.h"
#include "prvgetline.h"
#ifdef USE_PARAVER_EXCEPTIONS
#include "traceheaderexception.h"
#else
#include <iostream>
#endif

template< class TraceStreamT,
typename TimeT,
class ResourceModelT,
class ProcessModelT >
void parseTraceHeader( TraceStreamT& traceStream,
std::string& traceDate,
TTimeUnit& traceTimeUnit,
TimeT& traceEndTime,
ResourceModelT& traceResourceModel,
ProcessModelT& traceProcessModel,
std::vector< std::string >& communicators )
{
std::string tmpstr;

prvGetLine( traceStream, tmpstr );
if ( tmpstr.compare( "new format" ) == 0 )
prvGetLine( traceStream, tmpstr );

std::istringstream header( tmpstr );

std::getline( header, traceDate, ')' );
traceDate = traceDate.substr( traceDate.find_first_of( '(' ) + 1 );

header.get();

std::getline( header, tmpstr, ':' );
size_t pos = tmpstr.find( '_' );
if ( pos == string::npos )
{
traceTimeUnit = US;
std::istringstream stringEndTime( tmpstr );
if ( !( stringEndTime >> traceEndTime ) )
{
#ifdef USE_PARAVER_EXCEPTIONS
throw TraceHeaderException( TTraceHeaderErrorCode::invalidTime,
tmpstr.c_str() );
#else
std::cerr << "[Error] Invalid header time: " << tmpstr << std::endl;
exit( 1 );
#endif
}
}
else
{
std::string strTimeUnit( tmpstr.substr( pos, tmpstr.length() ) );
if ( strTimeUnit == "_ns" )
traceTimeUnit = NS;
else if ( strTimeUnit == "_ms" )
traceTimeUnit = MS;
else 
traceTimeUnit = US;

std::istringstream stringEndTime( tmpstr.substr( 0, pos ) );
if ( !( stringEndTime >> traceEndTime ) )
{
#ifdef USE_PARAVER_EXCEPTIONS
throw TraceHeaderException( TTraceHeaderErrorCode::invalidTime,
tmpstr.c_str() );
#else
std::cerr << "[Error] Invalid header time: " << tmpstr << std::endl;
exit( 1 );
#endif
}
}

traceResourceModel = ResourceModelT( header );
traceProcessModel = ProcessModelT( header, traceResourceModel.isReady() );

uint32_t numberComm = 0;
if ( !header.eof() )
{
std::getline( header, tmpstr );
if ( tmpstr != "" )
{
std::istringstream streamComm( tmpstr );

if ( !( streamComm >> numberComm ) )
{
#ifdef USE_PARAVER_EXCEPTIONS
throw TraceHeaderException( TTraceHeaderErrorCode::invalidCommNumber,
tmpstr.c_str() );
#else
std::cerr << "[Error] Invalid communicator number: " << tmpstr << std::endl;
exit( 1 );
#endif
}
}
}

for ( uint32_t count = 0; count < numberComm; count++ )
{
prvGetLine( traceStream, tmpstr );
if ( tmpstr[0] != 'C' && tmpstr[0] != 'c' && tmpstr[0] != 'I' && tmpstr[0] != 'i' )
{
#ifdef USE_PARAVER_EXCEPTIONS
throw TraceHeaderException( TTraceHeaderErrorCode::unknownCommLine,
tmpstr.c_str() );
#else
std::cerr << "[Error] Invalid communicator record: " << tmpstr << std::endl;
exit( 1 );
#endif
}
communicators.push_back( tmpstr );
}
}

template< class TraceStreamT,
typename TimeT,
class ResourceModelT,
class ProcessModelT >
void dumpTraceHeader( TraceStreamT& file, 
const std::string& traceTime,
const TimeT& traceEndTime,
const TTimeUnit& traceTimeUnit,
const ResourceModelT& traceResourceModel,
const ProcessModelT& traceProcessModel,
const std::vector< std::string >& communicators )
{
std::ostringstream ostr;

ostr << fixed;
ostr << dec;
ostr.precision( 0 );

file << fixed;
file << dec;
file.precision( 0 );

file << "#Paraver (";
file << traceTime << "):";

ostr << traceEndTime;
file << ostr.str();
if ( traceTimeUnit != US )
file << "_ns";
file << ':';
dumpResourceModelToFile( traceResourceModel, file );
file << ':';
dumpProcessModelToFile( traceProcessModel, file, traceResourceModel.isReady() );
if ( communicators.begin() != communicators.end() )
{
file << ',' << communicators.size() << endl;
for ( vector<string>::const_iterator it = communicators.begin();
it != communicators.end(); ++it )
file << ( *it ) << endl;
}
else
file << endl;
}
