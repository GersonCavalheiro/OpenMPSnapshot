

#include "tbb/pipeline.h"
#include "tbb/spin_mutex.h"
#include "tbb/atomic.h"
#include "tbb/tbb_thread.h"
#include <cstdlib>
#include <cstdio>
#include "harness.h"


tbb::tbb_thread::id thread_id;
tbb::tbb_thread::id id0;
bool is_serial_execution;
double sleeptime; 

struct Buffer {
static const unsigned long unused = ~0ul;
unsigned long id;
bool is_busy;
unsigned long sequence_number;
Buffer() : id(unused), is_busy(false), sequence_number(unused) {}
};

class waiting_probe {
size_t check_counter;
public:
waiting_probe() : check_counter(0) {}
bool required( ) {
++check_counter;
return !((check_counter+1)&size_t(0x7FFF));
}
void probe( ); 
};

static const unsigned MaxStreamSize = 8000;
static const unsigned MaxFilters = 4;
static unsigned StreamSize;
static const unsigned MaxBuffer = 8;
static bool Done[MaxFilters][MaxStreamSize];
static waiting_probe WaitTest;
static unsigned out_of_order_count;

#include "harness_concurrency_tracker.h"

template<typename T>
class BaseFilter: public T {
bool* const my_done;
const bool my_is_last;
bool my_is_running;
public:
tbb::atomic<tbb::internal::Token> current_token;
BaseFilter( tbb::filter::mode type, bool done[], bool is_last ) :
T(type),
my_done(done),
my_is_last(is_last),
my_is_running(false),
current_token()
{}
virtual Buffer* get_buffer( void* item ) {
current_token++;
return static_cast<Buffer*>(item);
}
void* operator()( void* item ) __TBB_override {
if( is_serial_execution && !this->is_bound() ) {
tbb::tbb_thread::id id = tbb::this_tbb_thread::get_id();
if( thread_id == id0 )
thread_id = id;
ASSERT( thread_id == id, "non-thread-bound stages executed on different threads when must be executed on a single one");
}
Harness::ConcurrencyTracker ct;
if( this->is_serial() )
ASSERT( !my_is_running, "premature entry to serial stage" );
my_is_running = true;
Buffer* b = get_buffer(item);
if( b ) {
if(!this->is_bound() && sleeptime > 0) {
if(this->is_serial()) {
Harness::Sleep((int)sleeptime);
}
else {
int i = (int)((5 - b->sequence_number) * sleeptime);
if(i < (int)sleeptime) i = (int)sleeptime;
Harness::Sleep(i);
}
}
if( this->is_ordered() ) {
if( b->sequence_number == Buffer::unused )
b->sequence_number = current_token-1;
else
ASSERT( b->sequence_number==current_token-1, "item arrived out of order" );
} else if( this->is_serial() ) {
if( b->sequence_number != current_token-1 && b->sequence_number != Buffer::unused )
out_of_order_count++;
}
ASSERT( b->id < StreamSize, NULL );
ASSERT( !my_done[b->id], "duplicate processing of token?" );
ASSERT( b->is_busy, NULL );
my_done[b->id] = true;
if( my_is_last ) {
b->id = Buffer::unused;
b->sequence_number = Buffer::unused;
__TBB_store_with_release(b->is_busy, false);
}
}
my_is_running = false;
return b;
}
};

template<typename T>
class InputFilter: public BaseFilter<T> {
tbb::spin_mutex input_lock;
Buffer buffer[MaxBuffer];
const tbb::internal::Token my_number_of_tokens;
public:
InputFilter( tbb::filter::mode type, tbb::internal::Token ntokens, bool done[], bool is_last ) :
BaseFilter<T>(type, done, is_last),
my_number_of_tokens(ntokens)
{}
Buffer* get_buffer( void* ) __TBB_override {
unsigned long next_input;
unsigned free_buffer = 0;
{ 
tbb::spin_mutex::scoped_lock lock(input_lock);
if( this->current_token>=StreamSize )
return NULL;
next_input = this->current_token++;
if( this->is_serial() && WaitTest.required() )
WaitTest.probe( );
while( free_buffer<MaxBuffer )
if( __TBB_load_with_acquire(buffer[free_buffer].is_busy) )
++free_buffer;
else {
buffer[free_buffer].is_busy = true;
break;
}
}
ASSERT( free_buffer<my_number_of_tokens, "premature reuse of buffer" );
Buffer* b = &buffer[free_buffer];
ASSERT( &buffer[0] <= b, NULL );
ASSERT( b <= &buffer[MaxBuffer-1], NULL );
ASSERT( b->id == Buffer::unused, NULL);
b->id = next_input;
ASSERT( b->sequence_number == Buffer::unused, NULL);
return b;
}
};

class process_loop {
public:
void operator()( tbb::thread_bound_filter* tbf ) {
tbb::thread_bound_filter::result_type flag;
do
flag = tbf->process_item();
while( flag != tbb::thread_bound_filter::end_of_stream );
}
};

struct hacked_pipeline {
tbb::filter* filter_list;
tbb::filter* filter_end;
tbb::empty_task* end_counter;
tbb::atomic<tbb::internal::Token> input_tokens;
tbb::atomic<tbb::internal::Token> global_token_counter;
bool end_of_input;
bool has_thread_bound_filters;

virtual ~hacked_pipeline();
};

struct hacked_ordered_buffer {
void* array; 
tbb::internal::Token array_size;
tbb::internal::Token low_token;
tbb::spin_mutex array_mutex;
tbb::internal::Token high_token;
bool is_ordered;
bool is_bound;
};

struct hacked_filter {
tbb::filter* next_filter_in_pipeline;
hacked_ordered_buffer* input_buffer;
unsigned char my_filter_mode;
tbb::filter* prev_filter_in_pipeline;
tbb::pipeline* my_pipeline;
tbb::filter* next_segment;

virtual ~hacked_filter();
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (disable: 4127)
#endif

void clear_global_state() {
Harness::ConcurrencyTracker::Reset();
memset( Done, 0, sizeof(Done) );
thread_id = id0;
is_serial_execution = false;
}


class PipelineTest {
static const tbb::filter::mode non_tb_filters_table[3]; 
static const tbb::filter::mode tb_filters_table[2]; 

static const unsigned number_of_non_tb_filter_types = sizeof(non_tb_filters_table)/sizeof(non_tb_filters_table[0]);
static const unsigned number_of_tb_filter_types = sizeof(tb_filters_table)/sizeof(tb_filters_table[0]);
static const unsigned number_of_filter_types = number_of_non_tb_filter_types + number_of_tb_filter_types;
public:
static double TestOneConfiguration( unsigned numeral, unsigned nthread, unsigned number_of_filters, tbb::internal::Token ntokens);
static void TestTrivialPipeline( unsigned nthread, unsigned number_of_filters );
static void TestIdleSpinning(unsigned nthread);

static void PrintConfiguration(unsigned numeral, unsigned nFilters) {
REMARK( "{ ");
for( unsigned i = 0; i < nFilters; ++i) {
switch( numeral % number_of_filter_types ) {
case 0: REMARK("s  "); break;
case 1: REMARK("B  "); break;
case 2: REMARK("o  "); break;
case 3: REMARK("Bo "); break;
case 4: REMARK("P  "); break;
default: REMARK(" ** ERROR** "); break;
}
numeral /= number_of_filter_types;
}
REMARK("}");
}
static bool ContainsBoundFilter(unsigned numeral) {
for( ;numeral != 0; numeral /= number_of_filter_types)
if(numeral & 0x1) return true;
return false;
}
};

const tbb::filter::mode PipelineTest::non_tb_filters_table[3] = {
tbb::filter::serial_in_order,       
tbb::filter::serial_out_of_order,   
tbb::filter::parallel               
};
const tbb::filter::mode PipelineTest::tb_filters_table[2] = {
tbb::filter::serial_in_order,       
tbb::filter::serial_out_of_order    
};

#include "harness_cpu.h"

double PipelineTest::TestOneConfiguration(unsigned numeral, unsigned nthread, unsigned number_of_filters, tbb::internal::Token ntokens)
{
tbb::pipeline pipeline;
tbb::filter* filter[MaxFilters];
unsigned temp = numeral;
unsigned parallelism_limit = 0;
unsigned number_of_tb_filters = 0;
unsigned array_of_tb_filter_numbers[MaxFilters];
if(!ContainsBoundFilter(numeral)) return 0.0;
for( unsigned i=0; i<number_of_filters; ++i, temp/=number_of_filter_types ) {
bool is_bound = temp%number_of_filter_types&0x1;
tbb::filter::mode filter_type;
if( is_bound ) {
filter_type = tb_filters_table[temp%number_of_filter_types/number_of_non_tb_filter_types];
} else
filter_type = non_tb_filters_table[temp%number_of_filter_types/number_of_tb_filter_types];
const bool is_last = i==number_of_filters-1;
if( is_bound ) {
if( i == 0 )
filter[i] = new InputFilter<tbb::thread_bound_filter>(filter_type,ntokens,Done[i],is_last);
else
filter[i] = new BaseFilter<tbb::thread_bound_filter>(filter_type,Done[i],is_last);
array_of_tb_filter_numbers[number_of_tb_filters] = i;
number_of_tb_filters++;
} else {
if( i == 0 )
filter[i] = new InputFilter<tbb::filter>(filter_type,ntokens,Done[i],is_last);
else
filter[i] = new BaseFilter<tbb::filter>(filter_type,Done[i],is_last);
}
pipeline.add_filter(*filter[i]);
if ( filter[i]->is_serial() ) {
parallelism_limit += 1;
} else {
parallelism_limit = nthread;
}
}
ASSERT(number_of_tb_filters,NULL);
clear_global_state();
if( parallelism_limit>nthread )
parallelism_limit = nthread;
if( parallelism_limit>ntokens )
parallelism_limit = (unsigned)ntokens;
StreamSize = nthread; 

for( unsigned i=0; i<number_of_filters; ++i ) {
static_cast<BaseFilter<tbb::filter>*>(filter[i])->current_token=0;
}
tbb::tbb_thread* t[MaxFilters];
for( unsigned j = 0; j<number_of_tb_filters; j++)
t[j] = new tbb::tbb_thread(process_loop(), static_cast<tbb::thread_bound_filter*>(filter[array_of_tb_filter_numbers[j]]));
if( ntokens == 1 || ( number_of_filters == 1 && number_of_tb_filters == 0 && filter[0]->is_serial() ))
is_serial_execution = true;
double strttime = GetCPUUserTime();
pipeline.run( ntokens );
double endtime = GetCPUUserTime();
for( unsigned j = 0; j<number_of_tb_filters; j++)
t[j]->join();
ASSERT( !Harness::ConcurrencyTracker::InstantParallelism(), "filter still running?" );
for( unsigned i=0; i<number_of_filters; ++i )
ASSERT( static_cast<BaseFilter<tbb::filter>*>(filter[i])->current_token==StreamSize, NULL );
for( unsigned i=0; i<MaxFilters; ++i )
for( unsigned j=0; j<StreamSize; ++j ) {
ASSERT( Done[i][j]==(i<number_of_filters), NULL );
}
if( Harness::ConcurrencyTracker::PeakParallelism() < parallelism_limit )
REMARK( "nthread=%lu ntokens=%lu MaxParallelism=%lu parallelism_limit=%lu\n",
nthread, ntokens, Harness::ConcurrencyTracker::PeakParallelism(), parallelism_limit );
for( unsigned i=0; i < number_of_filters; ++i ) {
delete filter[i];
filter[i] = NULL;
}
for( unsigned j = 0; j<number_of_tb_filters; j++)
delete t[j];
pipeline.clear();
return endtime - strttime;
} 

void PipelineTest::TestTrivialPipeline( unsigned nthread, unsigned number_of_filters ) {

REMARK( "testing with %lu threads and %lu filters\n", nthread, number_of_filters );
ASSERT( number_of_filters<=MaxFilters, "too many filters" );
tbb::internal::Token max_tokens = nthread < MaxBuffer ? nthread : MaxBuffer;
unsigned max_iteration = max_tokens > 1 ? 2 : 1;
tbb::internal::Token ntokens = 1;
for( unsigned iteration = 0; iteration < max_iteration; iteration++) {
if( iteration > 0 )
ntokens = max_tokens;
unsigned limit = 1;
for( unsigned i=0; i<number_of_filters; ++i)
limit *= number_of_filter_types;
for( unsigned numeral=0; numeral<limit; ++numeral ) {
REMARK( "testing configuration %lu of %lu\n", numeral, limit );
(void)TestOneConfiguration(numeral, nthread, number_of_filters, ntokens);
}
}
}

void PipelineTest::TestIdleSpinning( unsigned nthread)  {
unsigned sample_setups[] = {
1,   
5,   
25,  
125, 
6,   
26,  
126, 
30,  
130, 
150, 
31,  
131, 
155, 
495, 
71,  
355, 
95,  
475, 
};
const int nsetups = sizeof(sample_setups) / sizeof(unsigned);
const int ntests = 4;
const double bignum = 1000000000.0;
const double allowable_slowdown = 3.5;
unsigned zero_count = 0;

REMARK( "testing idle spinning with %lu threads\n", nthread );
tbb::internal::Token max_tokens = nthread < MaxBuffer ? nthread : MaxBuffer;
for( int i=0; i<nsetups; ++i ) {
unsigned numeral = sample_setups[i];
unsigned temp = numeral;
unsigned nbound = 0;
while(temp) {
if((temp%number_of_filter_types)&0x01) nbound++;
temp /= number_of_filter_types;
}
sleeptime = 20.0;
double s0 = bignum;
double s1 = bignum;
int v0cnt = 0;
int v1cnt = 0;
double s0sum = 0.0;
double s1sum = 0.0;
REMARK(" TestOneConfiguration, pipeline == ");
PrintConfiguration(numeral, MaxFilters);
REMARK(", max_tokens== %d\n", (int)max_tokens);
for(int j = 0; j < ntests; ++j) {
double s1a = TestOneConfiguration(numeral, nthread, MaxFilters, max_tokens);
double s0a = TestOneConfiguration((unsigned)0, nthread, MaxFilters, max_tokens);
s1sum += s1a;
s0sum += s0a;
if(s0a > 0.0) {
++v0cnt;
s0 = (s0a < s0) ? s0a : s0;
}
else {
++zero_count;
}
if(s1a > 0.0) {
++v1cnt;
s1 = (s1a < s1) ? s1a : s1;
}
else {
++zero_count;
}
}
if(s0 == bignum || s1 == bignum) continue;
s0sum /= (double)v0cnt;
s1sum /= (double)v1cnt;
double slowdown = (s1sum-s0sum)/s0sum;
if(slowdown > allowable_slowdown)
REMARK( "with %lu threads configuration %lu has slowdown > %g (%g)\n", nthread, numeral, allowable_slowdown, slowdown );
}
REMARK("Total of %lu zero times\n", zero_count);
}

static int nthread; 

void waiting_probe::probe( ) {
if( nthread==1 ) return;
REMARK("emulating wait for input\n");
TestCPUUserTime(nthread, 2);
}

#include "tbb/task_scheduler_init.h"

int TestMain () {
out_of_order_count = 0;
if( MinThread<1 ) {
REPORT("must have at least one thread");
exit(1);
}

for( nthread=MinThread; nthread<=MaxThread; ++nthread ) {
tbb::task_scheduler_init init(nthread);
sleeptime = 0.0;  

for( unsigned n=1; n<=MaxFilters; n*=MaxFilters ) {
PipelineTest::TestTrivialPipeline(nthread,n);
}

TestCPUUserTime(nthread);
if((unsigned)nthread >= MaxFilters)  
PipelineTest::TestIdleSpinning(nthread);
}
if( !out_of_order_count )
REPORT("Warning: out of order serial filter received tokens in order\n");
return Harness::Done;
}
