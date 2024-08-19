


#pragma once


#include <map>

#include "ktraceoptions.h"
#include "zlib.h"
#include "tracefilter.h"

class KTraceFilter: public TraceFilter
{
public:
KTraceFilter( char *trace_in,
char *trace_out,
TraceOptions *options,
const std::map< TTypeValuePair, TTypeValuePair >& whichTranslationTable,
ProgressController *progress );
virtual ~KTraceFilter();
virtual void execute( char *trace_in, char *trace_out, ProgressController *progress );

private:

char line[MAX_LINE_SIZE];


FILE *infile;
FILE *outfile;
gzFile gzInfile;


unsigned long long total_trace_size;
unsigned long long current_read_size;
unsigned long total_iters;



KTraceOptions *exec_options;


bool show_states;
bool show_comms;
bool show_events;
bool filter_all_types;
bool all_states;
bool filter_by_call_time;
unsigned long long min_state_time;
int min_comm_size;
bool is_zip_filter;

struct states_info
{
int ids[20];
int last_id;
};

struct states_info states_info;

struct buffer_elem
{
char *record;
bool dump;
int appl;
int task;
int thread;
unsigned long long event_time;
struct buffer_elem *next;
};

struct buffer_elem *buffer_first;
struct buffer_elem *buffer_last;

struct buffer_elem *thread_call_info[MAX_APPL][MAX_TASK][MAX_THREAD];

std::map<TTypeValuePair, TTypeValuePair> translationTable;

void read_params();
void filter_process_header( char *header );
int filter_allowed_type(  int appl, int task, int thread,
unsigned long long time,
unsigned long long type,
unsigned long long value );
void ini_progress_bar( char *file_name, ProgressController *progress );
void show_progress_bar( ProgressController *progress );
void load_pcf( char *pcf_name );
void dump_buffer();
};




