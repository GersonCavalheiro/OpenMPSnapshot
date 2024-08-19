


#pragma once


#include <string>

class TraceOptions;
class KernelConnection;
class ProgressController;

class TraceCutter
{
public:
static TraceCutter *create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
TraceOptions *options,
ProgressController *progress );

static std::string getID();
static std::string getName();
static std::string getExtension();

virtual ~TraceCutter()
{}


virtual void execute( std::string trace_in,
std::string trace_out,
ProgressController *progress = nullptr ) = 0;

virtual void set_by_time( bool byTime )
{}
virtual void set_min_cutting_time( unsigned long long minCutTime )
{}
virtual void set_max_cutting_time( unsigned long long maxCutTime )
{}
virtual void set_minimum_time_percentage( unsigned long long minimumPercentage )
{}
virtual void set_maximum_time_percentage( unsigned long long maximumPercentage )
{}
virtual void set_tasks_list( char *tasksList )
{}
virtual void set_original_time( bool originalTime )
{}
virtual void set_max_trace_size( int traceSize )
{}
virtual void set_break_states( bool breakStates )
{}
virtual void set_remFirstStates( bool remStates )
{}
virtual void set_remLastStates( bool remStates )
{}
virtual void set_keep_boundary_events( bool keepBoundaryEvents )
{}
virtual void set_keep_all_events( bool keepAllEvents )
{}
virtual void setCutterApplicationCaller( std::string caller ) = 0;


private:
static std::string traceToolID;
static std::string traceToolName;
static std::string traceToolExtension;
};

class TraceCutterProxy : public TraceCutter
{
public:
virtual ~TraceCutterProxy();

virtual void execute( std::string trace_in,
std::string trace_out,
ProgressController *progress = nullptr ) override;
virtual void set_by_time( bool byTime ) override;
virtual void set_min_cutting_time( unsigned long long minCutTime ) override;
virtual void set_max_cutting_time( unsigned long long maxCutTime ) override;
virtual void set_minimum_time_percentage( unsigned long long minimumPercentage ) override;
virtual void set_maximum_time_percentage( unsigned long long maximumPercentage ) override;
virtual void set_tasks_list( char *tasksList ) override;
virtual void set_original_time( bool originalTime ) override;
virtual void set_max_trace_size( int traceSize ) override;
virtual void set_break_states( bool breakStates ) override;
virtual void set_remFirstStates( bool remStates ) override;
virtual void set_remLastStates( bool remStates ) override;
virtual void set_keep_boundary_events( bool keepBoundaryEvents ) override;
virtual void set_keep_all_events( bool keepAllEvents ) override;
virtual void setCutterApplicationCaller( std::string caller ) override;

private:
TraceCutter *myTraceCutter;

TraceCutterProxy( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
TraceOptions *options,
ProgressController *progress );

friend TraceCutter *TraceCutter::create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
TraceOptions *options,
ProgressController *progress );
};



