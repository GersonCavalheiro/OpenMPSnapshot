

#include "tbb/pipeline.h"
#include "tbb/spin_mutex.h"
#include "tbb/cache_aligned_allocator.h"
#include "itt_notify.h"
#include "semaphore.h"
#include "tls.h"  


namespace tbb {

namespace internal {

struct task_info {
void* my_object;
Token my_token;
bool my_token_ready;
bool is_valid;
void reset() {
my_object = NULL;
my_token = 0;
my_token_ready = false;
is_valid = false;
}
};

class input_buffer : no_copy {
friend class tbb::internal::pipeline_root_task;
friend class tbb::filter;
friend class tbb::thread_bound_filter;
friend class tbb::internal::stage_task;
friend class tbb::pipeline;

typedef  Token  size_type;

task_info* array;

semaphore* my_sem;


size_type array_size;


Token low_token;

spin_mutex array_mutex;


void grow( size_type minimum_size );


static const size_type initial_buffer_size = 4;

Token high_token;

bool is_ordered;

bool is_bound;

typedef basic_tls<intptr_t> end_of_input_tls_t;
end_of_input_tls_t end_of_input_tls;
bool end_of_input_tls_allocated; 

void create_sema(size_t initial_tokens) { __TBB_ASSERT(!my_sem,NULL); my_sem = new internal::semaphore(initial_tokens); }
void free_sema() { __TBB_ASSERT(my_sem,NULL); delete my_sem; }
void sema_P() { __TBB_ASSERT(my_sem,NULL); my_sem->P(); }
void sema_V() { __TBB_ASSERT(my_sem,NULL); my_sem->V(); }

public:
input_buffer( bool is_ordered_, bool is_bound_ ) :
array(NULL), my_sem(NULL), array_size(0),
low_token(0), high_token(0),
is_ordered(is_ordered_), is_bound(is_bound_),
end_of_input_tls_allocated(false) {
grow(initial_buffer_size);
__TBB_ASSERT( array, NULL );
if(is_bound) create_sema(0);
}

~input_buffer() {
__TBB_ASSERT( array, NULL );
cache_aligned_allocator<task_info>().deallocate(array,array_size);
poison_pointer( array );
if(my_sem) {
free_sema();
}
if(end_of_input_tls_allocated) {
destroy_my_tls();
}
}


bool put_token( task_info& info_, bool force_put = false ) {
{
info_.is_valid = true;
spin_mutex::scoped_lock lock( array_mutex );
Token token;
bool was_empty = !array[low_token&(array_size-1)].is_valid;
if( is_ordered ) {
if( !info_.my_token_ready ) {
info_.my_token = high_token++;
info_.my_token_ready = true;
}
token = info_.my_token;
} else
token = high_token++;
__TBB_ASSERT( (tokendiff_t)(token-low_token)>=0, NULL );
if( token!=low_token || is_bound || force_put ) {
if( token-low_token>=array_size )
grow( token-low_token+1 );
ITT_NOTIFY( sync_releasing, this );
array[token&(array_size-1)] = info_;
if(was_empty && is_bound) {
sema_V();
}
return true;
}
}
return false;
}


template<typename StageTask>
void note_done( Token token, StageTask& spawner ) {
task_info wakee;
wakee.reset();
{
spin_mutex::scoped_lock lock( array_mutex );
if( !is_ordered || token==low_token ) {
task_info& item = array[++low_token & (array_size-1)];
ITT_NOTIFY( sync_acquired, this );
wakee = item;
item.is_valid = false;
}
}
if( wakee.is_valid )
spawner.spawn_stage_task(wakee);
}

#if __TBB_TASK_GROUP_CONTEXT
void clear( filter* my_filter ) {
long t=low_token;
for( size_type i=0; i<array_size; ++i, ++t ){
task_info& temp = array[t&(array_size-1)];
if (temp.is_valid ) {
my_filter->finalize(temp.my_object);
temp.is_valid = false;
}
}
}
#endif

bool return_item(task_info& info, bool advance) {
spin_mutex::scoped_lock lock( array_mutex );
task_info& item = array[low_token&(array_size-1)];
ITT_NOTIFY( sync_acquired, this );
if( item.is_valid ) {
info = item;
item.is_valid = false;
if (advance) low_token++;
return true;
}
return false;
}

bool has_item() { spin_mutex::scoped_lock lock(array_mutex); return array[low_token&(array_size -1)].is_valid; }

void create_my_tls() { int status = end_of_input_tls.create(); if(status) handle_perror(status, "TLS not allocated for filter"); end_of_input_tls_allocated = true; }
void destroy_my_tls() { int status = end_of_input_tls.destroy(); if(status) handle_perror(status, "Failed to destroy filter TLS"); }
bool my_tls_end_of_input() { return end_of_input_tls.get() != 0; }
void set_my_tls_end_of_input() { end_of_input_tls.set(1); }
};

void input_buffer::grow( size_type minimum_size ) {
size_type old_size = array_size;
size_type new_size = old_size ? 2*old_size : initial_buffer_size;
while( new_size<minimum_size )
new_size*=2;
task_info* new_array = cache_aligned_allocator<task_info>().allocate(new_size);
task_info* old_array = array;
for( size_type i=0; i<new_size; ++i )
new_array[i].is_valid = false;
long t=low_token;
for( size_type i=0; i<old_size; ++i, ++t )
new_array[t&(new_size-1)] = old_array[t&(old_size-1)];
array = new_array;
array_size = new_size;
if( old_array )
cache_aligned_allocator<task_info>().deallocate(old_array,old_size);
}

class stage_task: public task, public task_info {
private:
friend class tbb::pipeline;
pipeline& my_pipeline;
filter* my_filter;
bool my_at_start;

public:

stage_task( pipeline& pipeline ) :
my_pipeline(pipeline),
my_filter(pipeline.filter_list),
my_at_start(true)
{
task_info::reset();
}
stage_task( pipeline& pipeline, filter* filter_, const task_info& info ) :
task_info(info),
my_pipeline(pipeline),
my_filter(filter_),
my_at_start(false)
{}
void reset() {
task_info::reset();
my_filter = my_pipeline.filter_list;
my_at_start = true;
}
task* execute() __TBB_override;
#if __TBB_TASK_GROUP_CONTEXT
~stage_task()
{
if (my_filter && my_object && (my_filter->my_filter_mode & filter::version_mask) >= __TBB_PIPELINE_VERSION(4)) {
__TBB_ASSERT(is_cancelled(), "Trying to finalize the task that wasn't cancelled");
my_filter->finalize(my_object);
my_object = NULL;
}
}
#endif 
void spawn_stage_task(const task_info& info)
{
stage_task* clone = new (allocate_additional_child_of(*parent()))
stage_task( my_pipeline, my_filter, info );
spawn(*clone);
}
};

task* stage_task::execute() {
__TBB_ASSERT( !my_at_start || !my_object, NULL );
__TBB_ASSERT( !my_filter->is_bound(), NULL );
if( my_at_start ) {
if( my_filter->is_serial() ) {
my_object = (*my_filter)(my_object);
if( my_object || ( my_filter->object_may_be_null() && !my_pipeline.end_of_input) )
{
if( my_filter->is_ordered() ) {
my_token = my_pipeline.token_counter++; 
my_token_ready = true;
} else if( (my_filter->my_filter_mode & my_filter->version_mask) >= __TBB_PIPELINE_VERSION(5) ) {
if( my_pipeline.has_thread_bound_filters )
my_pipeline.token_counter++; 
}
if( !my_filter->next_filter_in_pipeline ) { 
reset();
goto process_another_stage;
} else {
ITT_NOTIFY( sync_releasing, &my_pipeline.input_tokens );
if( --my_pipeline.input_tokens>0 )
spawn( *new( allocate_additional_child_of(*parent()) ) stage_task( my_pipeline ) );
}
} else {
my_pipeline.end_of_input = true;
return NULL;
}
} else  {
if( my_pipeline.end_of_input )
return NULL;
if( (my_filter->my_filter_mode & my_filter->version_mask) >= __TBB_PIPELINE_VERSION(5) ) {
if( my_pipeline.has_thread_bound_filters )
my_pipeline.token_counter++;
}
ITT_NOTIFY( sync_releasing, &my_pipeline.input_tokens );
if( --my_pipeline.input_tokens>0 )
spawn( *new( allocate_additional_child_of(*parent()) ) stage_task( my_pipeline ) );
my_object = (*my_filter)(my_object);
if( !my_object && (!my_filter->object_may_be_null() || my_filter->my_input_buffer->my_tls_end_of_input()) )
{
my_pipeline.end_of_input = true;
if( (my_filter->my_filter_mode & my_filter->version_mask) >= __TBB_PIPELINE_VERSION(5) ) {
if( my_pipeline.has_thread_bound_filters )
my_pipeline.token_counter--;  
}
return NULL;
}
}
my_at_start = false;
} else {
my_object = (*my_filter)(my_object);
if( my_filter->is_serial() )
my_filter->my_input_buffer->note_done(my_token, *this);
}
my_filter = my_filter->next_filter_in_pipeline;
if( my_filter ) {
if( my_filter->is_serial() ) {
if( my_filter->my_input_buffer->put_token(*this) ){
if( my_filter->is_bound() ) {
do {
my_filter = my_filter->next_filter_in_pipeline;
} while( my_filter && my_filter->is_bound() );
if( my_filter && my_filter->my_input_buffer->return_item(*this, !my_filter->is_serial()))
goto process_another_stage;
}
my_filter = NULL; 
return NULL;
}
}
} else {
size_t ntokens_avail = ++my_pipeline.input_tokens;
if(my_pipeline.filter_list->is_bound() ) {
if(ntokens_avail == 1) {
my_pipeline.filter_list->my_input_buffer->sema_V();
}
return NULL;
}
if( ntokens_avail>1  
|| my_pipeline.end_of_input ) {
return NULL; 
}
ITT_NOTIFY( sync_acquired, &my_pipeline.input_tokens );
reset();
}
process_another_stage:

recycle_as_continuation();
return this;
}

class pipeline_root_task: public task {
pipeline& my_pipeline;
bool do_segment_scanning;

task* execute() __TBB_override {
if( !my_pipeline.end_of_input )
if( !my_pipeline.filter_list->is_bound() )
if( my_pipeline.input_tokens > 0 ) {
recycle_as_continuation();
set_ref_count(1);
return new( allocate_child() ) stage_task( my_pipeline );
}
if( do_segment_scanning ) {
filter* current_filter = my_pipeline.filter_list->next_segment;

filter* first_suitable_filter = current_filter;
while( current_filter ) {
__TBB_ASSERT( !current_filter->is_bound(), "filter is thread-bound?" );
__TBB_ASSERT( current_filter->prev_filter_in_pipeline->is_bound(), "previous filter is not thread-bound?" );
if( !my_pipeline.end_of_input || current_filter->has_more_work())
{
task_info info;
info.reset();
task* bypass = NULL;
int refcnt = 0;
task_list list;
while( current_filter->my_input_buffer->return_item(info, !current_filter->is_serial()) ) {
task* t = new( allocate_child() ) stage_task( my_pipeline, current_filter, info );
if( ++refcnt == 1 )
bypass = t;
else 
list.push_back(*t);
__TBB_ASSERT( refcnt <= int(my_pipeline.token_counter), "token counting error" );
info.reset();
}
if( refcnt ) {
set_ref_count( refcnt );
if( refcnt > 1 )
spawn(list);
recycle_as_continuation();
return bypass;
}
current_filter = current_filter->next_segment;
if( !current_filter ) {
if( !my_pipeline.end_of_input ) {
recycle_as_continuation();
return this;
}
current_filter = first_suitable_filter;
__TBB_Yield();
}
} else {

first_suitable_filter = first_suitable_filter->next_segment;
current_filter = first_suitable_filter;
}
} 
return NULL;
} else {
if( !my_pipeline.end_of_input ) {
recycle_as_continuation();
return this;
}
return NULL;
}
}
public:
pipeline_root_task( pipeline& pipeline ): my_pipeline(pipeline), do_segment_scanning(false)
{
__TBB_ASSERT( my_pipeline.filter_list, NULL );
filter* first = my_pipeline.filter_list;
if( (first->my_filter_mode & first->version_mask) >= __TBB_PIPELINE_VERSION(5) ) {
filter* head_of_previous_segment = first;
for(  filter* subfilter=first->next_filter_in_pipeline;
subfilter!=NULL;
subfilter=subfilter->next_filter_in_pipeline )
{
if( subfilter->prev_filter_in_pipeline->is_bound() && !subfilter->is_bound() ) {
do_segment_scanning = true;
head_of_previous_segment->next_segment = subfilter;
head_of_previous_segment = subfilter;
}
}
}
}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (disable: 4127)
#endif

class pipeline_cleaner: internal::no_copy {
pipeline& my_pipeline;
public:
pipeline_cleaner(pipeline& _pipeline) :
my_pipeline(_pipeline)
{}
~pipeline_cleaner(){
#if __TBB_TASK_GROUP_CONTEXT
if (my_pipeline.end_counter->is_cancelled()) 
my_pipeline.clear_filters();
#endif
my_pipeline.end_counter = NULL;
}
};

} 

void pipeline::inject_token( task& ) {
__TBB_ASSERT(false,"illegal call to inject_token");
}

#if __TBB_TASK_GROUP_CONTEXT
void pipeline::clear_filters() {
for( filter* f = filter_list; f; f = f->next_filter_in_pipeline ) {
if ((f->my_filter_mode & filter::version_mask) >= __TBB_PIPELINE_VERSION(4))
if( internal::input_buffer* b = f->my_input_buffer )
b->clear(f);
}
}
#endif

pipeline::pipeline() :
filter_list(NULL),
filter_end(NULL),
end_counter(NULL),
end_of_input(false),
has_thread_bound_filters(false)
{
token_counter = 0;
input_tokens = 0;
}

pipeline::~pipeline() {
clear();
}

void pipeline::clear() {
filter* next;
for( filter* f = filter_list; f; f=next ) {
if( internal::input_buffer* b = f->my_input_buffer ) {
delete b;
f->my_input_buffer = NULL;
}
next=f->next_filter_in_pipeline;
f->next_filter_in_pipeline = filter::not_in_pipeline();
if ( (f->my_filter_mode & filter::version_mask) >= __TBB_PIPELINE_VERSION(3) ) {
f->prev_filter_in_pipeline = filter::not_in_pipeline();
f->my_pipeline = NULL;
}
if ( (f->my_filter_mode & filter::version_mask) >= __TBB_PIPELINE_VERSION(5) )
f->next_segment = NULL;
}
filter_list = filter_end = NULL;
}

void pipeline::add_filter( filter& filter_ ) {
#if TBB_USE_ASSERT
if ( (filter_.my_filter_mode & filter::version_mask) >= __TBB_PIPELINE_VERSION(3) )
__TBB_ASSERT( filter_.prev_filter_in_pipeline==filter::not_in_pipeline(), "filter already part of pipeline?" );
__TBB_ASSERT( filter_.next_filter_in_pipeline==filter::not_in_pipeline(), "filter already part of pipeline?" );
__TBB_ASSERT( !end_counter, "invocation of add_filter on running pipeline" );
#endif
if ( (filter_.my_filter_mode & filter::version_mask) >= __TBB_PIPELINE_VERSION(3) ) {
filter_.my_pipeline = this;
filter_.prev_filter_in_pipeline = filter_end;
if ( filter_list == NULL)
filter_list = &filter_;
else
filter_end->next_filter_in_pipeline = &filter_;
filter_.next_filter_in_pipeline = NULL;
filter_end = &filter_;
} else {
if( !filter_end )
filter_end = reinterpret_cast<filter*>(&filter_list);

*reinterpret_cast<filter**>(filter_end) = &filter_;
filter_end = reinterpret_cast<filter*>(&filter_.next_filter_in_pipeline);
*reinterpret_cast<filter**>(filter_end) = NULL;
}
if( (filter_.my_filter_mode & filter_.version_mask) >= __TBB_PIPELINE_VERSION(5) ) {
if( filter_.is_serial() ) {
if( filter_.is_bound() )
has_thread_bound_filters = true;
filter_.my_input_buffer = new internal::input_buffer( filter_.is_ordered(), filter_.is_bound() );
} else {
if(filter_.prev_filter_in_pipeline) {
if(filter_.prev_filter_in_pipeline->is_bound()) {
filter_.my_input_buffer = new internal::input_buffer( false, false );
}
} else {  
if(filter_.object_may_be_null() ) {
filter_.my_input_buffer = new internal::input_buffer( false, false );
filter_.my_input_buffer->create_my_tls();
}
}
}
} else {
if( filter_.is_serial() ) {
filter_.my_input_buffer = new internal::input_buffer( filter_.is_ordered(), false );
}
}

}

void pipeline::remove_filter( filter& filter_ ) {
__TBB_ASSERT( filter_.prev_filter_in_pipeline!=filter::not_in_pipeline(), "filter not part of pipeline" );
__TBB_ASSERT( filter_.next_filter_in_pipeline!=filter::not_in_pipeline(), "filter not part of pipeline" );
__TBB_ASSERT( !end_counter, "invocation of remove_filter on running pipeline" );
if (&filter_ == filter_list)
filter_list = filter_.next_filter_in_pipeline;
else {
__TBB_ASSERT( filter_.prev_filter_in_pipeline, "filter list broken?" );
filter_.prev_filter_in_pipeline->next_filter_in_pipeline = filter_.next_filter_in_pipeline;
}
if (&filter_ == filter_end)
filter_end = filter_.prev_filter_in_pipeline;
else {
__TBB_ASSERT( filter_.next_filter_in_pipeline, "filter list broken?" );
filter_.next_filter_in_pipeline->prev_filter_in_pipeline = filter_.prev_filter_in_pipeline;
}
if( internal::input_buffer* b = filter_.my_input_buffer ) {
delete b;
filter_.my_input_buffer = NULL;
}
filter_.next_filter_in_pipeline = filter_.prev_filter_in_pipeline = filter::not_in_pipeline();
if ( (filter_.my_filter_mode & filter::version_mask) >= __TBB_PIPELINE_VERSION(5) )
filter_.next_segment = NULL;
filter_.my_pipeline = NULL;
}

void pipeline::run( size_t max_number_of_live_tokens
#if __TBB_TASK_GROUP_CONTEXT
, tbb::task_group_context& context
#endif
) {
__TBB_ASSERT( max_number_of_live_tokens>0, "pipeline::run must have at least one token" );
__TBB_ASSERT( !end_counter, "pipeline already running?" );
if( filter_list ) {
internal::pipeline_cleaner my_pipeline_cleaner(*this);
end_of_input = false;
input_tokens = internal::Token(max_number_of_live_tokens);
if(has_thread_bound_filters) {
if(filter_list->is_bound()) {
filter_list->my_input_buffer->sema_V();
}
}
#if __TBB_TASK_GROUP_CONTEXT
end_counter = new( task::allocate_root(context) ) internal::pipeline_root_task( *this );
#else
end_counter = new( task::allocate_root() ) internal::pipeline_root_task( *this );
#endif
task::spawn_root_and_wait( *end_counter );

if(has_thread_bound_filters) {
for(filter* f = filter_list->next_filter_in_pipeline; f; f=f->next_filter_in_pipeline) {
if(f->is_bound()) {
f->my_input_buffer->sema_V(); 
}
}
}
}
}

#if __TBB_TASK_GROUP_CONTEXT
void pipeline::run( size_t max_number_of_live_tokens ) {
if( filter_list ) {
uintptr_t ctx_traits = filter_list->my_filter_mode & filter::exact_exception_propagation ?
task_group_context::default_traits :
task_group_context::default_traits & ~task_group_context::exact_exception;
task_group_context context(task_group_context::bound, ctx_traits);
run(max_number_of_live_tokens, context);
}
}
#endif 

bool filter::has_more_work() {
__TBB_ASSERT(my_pipeline, NULL);
__TBB_ASSERT(my_input_buffer, "has_more_work() called for filter with no input buffer");
return (internal::tokendiff_t)(my_pipeline->token_counter - my_input_buffer->low_token) != 0;
}

filter::~filter() {
if ( (my_filter_mode & version_mask) >= __TBB_PIPELINE_VERSION(3) ) {
if ( next_filter_in_pipeline != filter::not_in_pipeline() )
my_pipeline->remove_filter(*this);
else
__TBB_ASSERT( prev_filter_in_pipeline == filter::not_in_pipeline(), "probably filter list is broken" );
} else {
__TBB_ASSERT( next_filter_in_pipeline==filter::not_in_pipeline(), "cannot destroy filter that is part of pipeline" );
}
}

void filter::set_end_of_input() {
__TBB_ASSERT(my_input_buffer, NULL);
__TBB_ASSERT(object_may_be_null(), NULL);
if(is_serial()) {
my_pipeline->end_of_input = true;
} else {
__TBB_ASSERT(my_input_buffer->end_of_input_tls_allocated, NULL);
my_input_buffer->set_my_tls_end_of_input();
}
}

thread_bound_filter::result_type thread_bound_filter::process_item() {
return internal_process_item(true);
}

thread_bound_filter::result_type thread_bound_filter::try_process_item() {
return internal_process_item(false);
}

thread_bound_filter::result_type thread_bound_filter::internal_process_item(bool is_blocking) {
__TBB_ASSERT(my_pipeline != NULL,"It's not supposed that process_item is called for a filter that is not in a pipeline.");
internal::task_info info;
info.reset();

if( my_pipeline->end_of_input && !has_more_work() )
return end_of_stream;

if( !prev_filter_in_pipeline ) {
if( my_pipeline->end_of_input )
return end_of_stream;
while( my_pipeline->input_tokens == 0 ) {
if( !is_blocking )
return item_not_available;
my_input_buffer->sema_P();
}
info.my_object = (*this)(info.my_object);
if( info.my_object ) {
__TBB_ASSERT(my_pipeline->input_tokens > 0, "Token failed in thread-bound filter");
my_pipeline->input_tokens--;
if( is_ordered() ) {
info.my_token = my_pipeline->token_counter;
info.my_token_ready = true;
}
my_pipeline->token_counter++; 
} else {
my_pipeline->end_of_input = true;
return end_of_stream;
}
} else { 
while( !my_input_buffer->has_item() ) {
if( !is_blocking ) {
return item_not_available;
}
my_input_buffer->sema_P();
if( my_pipeline->end_of_input && !has_more_work() ) {
return end_of_stream;
}
}
if( !my_input_buffer->return_item(info, true) ) {
__TBB_ASSERT(false,"return_item failed");
}
info.my_object = (*this)(info.my_object);
}
if( next_filter_in_pipeline ) {
if ( !next_filter_in_pipeline->my_input_buffer->put_token(info,true) ) {
__TBB_ASSERT(false, "Couldn't put token after thread-bound buffer");
}
} else {
size_t ntokens_avail = ++(my_pipeline->input_tokens);
if( my_pipeline->filter_list->is_bound() ) {
if( ntokens_avail == 1 ) {
my_pipeline->filter_list->my_input_buffer->sema_V();
}
}
}

return success;
}

} 

