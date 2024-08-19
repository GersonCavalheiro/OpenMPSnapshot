

#include "rml_tbb.h"
#include "../server/thread_monitor.h"
#include "tbb/atomic.h"
#include "tbb/cache_aligned_allocator.h"
#include "scheduler_common.h"
#include "governor.h"
#include "tbb_misc.h"

using rml::internal::thread_monitor;

namespace tbb {
namespace internal {
namespace rml {

typedef thread_monitor::handle_type thread_handle;

class private_server;

class private_worker: no_copy {
private:

enum state_t {
st_init,
st_starting,
st_normal,
st_quit
};
atomic<state_t> my_state;

private_server& my_server;

tbb_client& my_client;

const size_t my_index;


thread_monitor my_thread_monitor;

thread_handle my_handle;

private_worker* my_next;

friend class private_server;

void run();

void wake_or_launch();

void start_shutdown();

static __RML_DECL_THREAD_ROUTINE thread_routine( void* arg );

static void release_handle(thread_handle my_handle, bool join);

protected:
private_worker( private_server& server, tbb_client& client, const size_t i ) :
my_server(server),
my_client(client),
my_index(i)
{
my_state = st_init;
}
};

static const size_t cache_line_size = tbb::internal::NFS_MaxLineSize;


#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable:4510 4610)
#endif
class padded_private_worker: public private_worker {
char pad[cache_line_size - sizeof(private_worker)%cache_line_size];
public:
padded_private_worker( private_server& server, tbb_client& client, const size_t i )
: private_worker(server,client,i) { suppress_unused_warning(pad); }
};
#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(pop)
#endif

class private_server: public tbb_server, no_copy {
private:
tbb_client& my_client;

const tbb_client::size_type my_n_thread;

const size_t my_stack_size;


atomic<int> my_slack;

atomic<int> my_ref_count;

padded_private_worker* my_thread_array;

tbb::atomic<private_worker*> my_asleep_list_root;

typedef scheduler_mutex_type asleep_list_mutex_type;
asleep_list_mutex_type my_asleep_list_mutex;

#if TBB_USE_ASSERT
atomic<int> my_net_slack_requests;
#endif 


void propagate_chain_reaction() {
if( my_asleep_list_root )
wake_some(0);
}

bool try_insert_in_asleep_list( private_worker& t );

void wake_some( int additional_slack );

virtual ~private_server();

void remove_server_ref() {
if( --my_ref_count==0 ) {
my_client.acknowledge_close_connection();
this->~private_server();
tbb::cache_aligned_allocator<private_server>().deallocate( this, 1 );
}
}

friend class private_worker;
public:
private_server( tbb_client& client );

version_type version() const __TBB_override {
return 0;
}

void request_close_connection( bool  ) __TBB_override {
for( size_t i=0; i<my_n_thread; ++i )
my_thread_array[i].start_shutdown();
remove_server_ref();
}

void yield() __TBB_override {__TBB_Yield();}

void independent_thread_number_changed( int ) __TBB_override {__TBB_ASSERT(false,NULL);}

unsigned default_concurrency() const __TBB_override { return governor::default_num_threads() - 1; }

void adjust_job_count_estimate( int delta ) __TBB_override;

#if _WIN32||_WIN64
void register_master ( ::rml::server::execution_resource_t& ) __TBB_override {}
void unregister_master ( ::rml::server::execution_resource_t ) __TBB_override {}
#endif 
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable:4189)
#endif
#if __MINGW32__ && __GNUC__==4 &&__GNUC_MINOR__>=2 && !__MINGW64__
__attribute__((force_align_arg_pointer))
#endif
__RML_DECL_THREAD_ROUTINE private_worker::thread_routine( void* arg ) {
private_worker* self = static_cast<private_worker*>(arg);
AVOID_64K_ALIASING( self->my_index );
self->run();
return 0;
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(pop)
#endif

void private_worker::release_handle(thread_handle handle, bool join) {
if (join)
thread_monitor::join(handle);
else
thread_monitor::detach_thread(handle);
}

void private_worker::start_shutdown() {
state_t s;

do {
s = my_state;
__TBB_ASSERT( s!=st_quit, NULL );
} while( my_state.compare_and_swap( st_quit, s )!=s );
if( s==st_normal || s==st_starting ) {
my_thread_monitor.notify();
if (s==st_normal)
release_handle(my_handle, governor::does_client_join_workers(my_client));
} else if( s==st_init ) {
my_server.remove_server_ref();
}
}

void private_worker::run() {
my_server.propagate_chain_reaction();


::rml::job& j = *my_client.create_one_job();
while( my_state!=st_quit ) {
if( my_server.my_slack>=0 ) {
my_client.process(j);
} else {
thread_monitor::cookie c;
my_thread_monitor.prepare_wait(c);
if( my_state!=st_quit && my_server.try_insert_in_asleep_list(*this) ) {
my_thread_monitor.commit_wait(c);
my_server.propagate_chain_reaction();
} else {
my_thread_monitor.cancel_wait();
}
}
}
my_client.cleanup(j);

++my_server.my_slack;
my_server.remove_server_ref();
}

inline void private_worker::wake_or_launch() {
if( my_state==st_init && my_state.compare_and_swap( st_starting, st_init )==st_init ) {
#if USE_WINTHREAD
my_handle = thread_monitor::launch( thread_routine, this, my_server.my_stack_size, &this->my_index );
#elif USE_PTHREAD
{
affinity_helper fpa;
fpa.protect_affinity_mask( true );
my_handle = thread_monitor::launch( thread_routine, this, my_server.my_stack_size );
}
#endif 
state_t s = my_state.compare_and_swap( st_normal, st_starting );
if (st_starting != s) {
__TBB_ASSERT( s==st_quit, NULL );
release_handle(my_handle, governor::does_client_join_workers(my_client));
}
}
else
my_thread_monitor.notify();
}

private_server::private_server( tbb_client& client ) :
my_client(client),
my_n_thread(client.max_job_count()),
my_stack_size(client.min_stack_size()),
my_thread_array(NULL)
{
my_ref_count = my_n_thread+1;
my_slack = 0;
#if TBB_USE_ASSERT
my_net_slack_requests = 0;
#endif 
my_asleep_list_root = NULL;
my_thread_array = tbb::cache_aligned_allocator<padded_private_worker>().allocate( my_n_thread );
memset( my_thread_array, 0, sizeof(private_worker)*my_n_thread );
for( size_t i=0; i<my_n_thread; ++i ) {
private_worker* t = new( &my_thread_array[i] ) padded_private_worker( *this, client, i );
t->my_next = my_asleep_list_root;
my_asleep_list_root = t;
}
}

private_server::~private_server() {
__TBB_ASSERT( my_net_slack_requests==0, NULL );
for( size_t i=my_n_thread; i--; )
my_thread_array[i].~padded_private_worker();
tbb::cache_aligned_allocator<padded_private_worker>().deallocate( my_thread_array, my_n_thread );
tbb::internal::poison_pointer( my_thread_array );
}

inline bool private_server::try_insert_in_asleep_list( private_worker& t ) {
asleep_list_mutex_type::scoped_lock lock;
if( !lock.try_acquire(my_asleep_list_mutex) )
return false;
int k = ++my_slack;
if( k<=0 ) {
t.my_next = my_asleep_list_root;
my_asleep_list_root = &t;
return true;
} else {
--my_slack;
return false;
}
}

void private_server::wake_some( int additional_slack ) {
__TBB_ASSERT( additional_slack>=0, NULL );
private_worker* wakee[2];
private_worker**w = wakee;
{
asleep_list_mutex_type::scoped_lock lock(my_asleep_list_mutex);
while( my_asleep_list_root && w<wakee+2 ) {
if( additional_slack>0 ) {
if (additional_slack+my_slack<=0) 
break;
--additional_slack;
} else {
int old;
do {
old = my_slack;
if( old<=0 ) goto done;
} while( my_slack.compare_and_swap(old-1,old)!=old );
}
my_asleep_list_root = (*w++ = my_asleep_list_root)->my_next;
}
if( additional_slack ) {
my_slack += additional_slack;
}
}
done:
while( w>wakee )
(*--w)->wake_or_launch();
}

void private_server::adjust_job_count_estimate( int delta ) {
#if TBB_USE_ASSERT
my_net_slack_requests+=delta;
#endif 
if( delta<0 ) {
my_slack+=delta;
} else if( delta>0 ) {
wake_some( delta );
}
}

tbb_server* make_private_server( tbb_client& client ) {
return new( tbb::cache_aligned_allocator<private_server>().allocate(1) ) private_server(client);
}

} 
} 

} 
