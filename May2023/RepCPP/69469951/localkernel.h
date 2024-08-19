


#pragma once


#include <map>

#include "kernelconnection.h"

class KTimeline;
class PreviousFiles;

constexpr size_t MAX_TRACES_HISTORY_LENGTH = 256;

class LocalKernel: public KernelConnection
{
public:
static void init();

LocalKernel( bool (*messageFunction)(UserMessageID) );
virtual ~LocalKernel();

virtual bool checkTraceSize( const std::string& filename, TTraceSize maxSize ) const override;
virtual TTraceSize getTraceSize( const std::string& filename ) const override;
virtual Trace *newTrace( const std::string& whichFile, bool noLoad, ProgressController *progress = nullptr, TTraceSize traceSize = 0 ) const override;
virtual std::string getPCFFileLocation( const std::string& traceFile ) const override;
virtual std::string getROWFileLocation( const std::string& traceFile ) const override;
virtual Timeline *newSingleWindow() const override;
virtual Timeline *newSingleWindow( Trace *whichTrace ) const override;
virtual Timeline *newDerivedWindow() const override;
virtual Timeline *newDerivedWindow( Timeline *window1, Timeline * window2 ) const override;
virtual Histogram *newHistogram() const override;
virtual ProgressController *newProgressController() const override;
virtual Filter *newFilter( Filter *concreteFilter ) const override;
virtual TraceEditSequence *newTraceEditSequence() const override;

virtual std::string getToolID( const std::string &toolName ) const override;
virtual std::string getToolName( const std::string &toolID ) const override;
virtual TraceOptions *newTraceOptions() const override;
virtual TraceCutter *newTraceCutter( TraceOptions *options,
const std::vector< TEventType > &whichTypesWithValuesZero ) const override;
virtual TraceFilter *newTraceFilter( char *trace_in,
char *trace_out,
TraceOptions *options,
const std::map< TTypeValuePair, TTypeValuePair >& whichTranslationTable,
ProgressController *progress = nullptr ) const override;
virtual TraceSoftwareCounters *newTraceSoftwareCounters( char *trace_in,
char *trace_out,
TraceOptions *options,
ProgressController *progress = nullptr ) const override;
virtual TraceShifter *newTraceShifter( std::string traceIn,
std::string traceOut,
std::string shiftTimesFile,
TWindowLevel shiftLevel,
ProgressController *progress ) const override;
virtual EventDrivenCutter *newEventDrivenCutter( std::string traceIn,
std::string traceOut,
TEventType whichEvent,
ProgressController *progress = nullptr ) const override;
virtual EventTranslator *newEventTranslator( std::string traceIn,
std::string traceOut,
std::string traceReference,
ProgressController *progress = nullptr ) const override;

virtual void getAllStatistics( std::vector<std::string>& onVector ) const override;
virtual void getAllFilterFunctions( std::vector<std::string>& onVector ) const override;
virtual void getAllSemanticFunctions( TSemanticGroup whichGroup,
std::vector<std::string>& onVector ) const override;

virtual bool userMessage( UserMessageID messageID ) const override;

virtual bool isTraceFile( const std::string &filename ) const override;
virtual void copyPCF( const std::string& name, const std::string& traceToLoad ) const override;
virtual void copyROW( const std::string& name, const std::string& traceToLoad ) const override;

virtual void getNewTraceName( char *name,
char *new_trace_name,
std::string action,
bool saveNewNameInfo = true ) override;

virtual std::string getNewTraceName( const std::string& fullPathTraceName,
const std::string& traceFilterID,
const bool commitName = false ) const override;

virtual std::string getNewTraceName( const std::string& fullPathTraceName,
const std::vector< std::string >& traceFilterID,
const bool commitName = false ) const override;

virtual std::string getNewTraceName( const std::string& fullPathTraceName,
const std::string& outputPath,
const std::vector< std::string >& traceFilterID,
const bool commitName = false ) const override;

static std::string composeName( const std::string& name, const std::string& newExtension );
bool isFileReadable( const std::string& filename,
const std::string& message,
const bool verbose = true,
const bool keepOpen = true,
const bool exitProgram = true ) const override;

virtual void commitNewTraceName( const std::string& newTraceName ) const override;

virtual std::string getPathSeparator() const override { return pathSeparator; }
virtual void setPathSeparator( const std::string& whichPath ) override { pathSeparator = whichPath; }

virtual std::string getDistributedCFGsPath() const override { return distributedCFGsPath; }

virtual std::string getParaverUserDir() const override { return paraverUserDir; }

protected:

private:
std::string pathSeparator;
std::string distributedCFGsPath;
std::string paraverUserDir;

bool (*myMessageFunction)(UserMessageID);

struct traces_table
{
char *name;
int num_chop;
int num_filter;
int num_sc;
};

struct traces_table trace_names_table[ MAX_TRACES_HISTORY_LENGTH ];
int trace_names_table_last; 

PreviousFiles *prevTraceNames;

void copyFile( const std::string& in, const std::string& out ) const;
void fileUnreadableError( const std::string& filename,
const std::string& message,
const bool verbose,
const bool exitProgram ) const;
};



