

#include "rml_tbb.h"
#define private public 
#include "rml_omp.h"
#undef private

#include "tbb/tbb_allocator.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/aligned_space.h"
#include "tbb/atomic.h"
#include "tbb/spin_mutex.h"
#include "tbb/tbb_misc.h"           
#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable: 4985 )
#endif
#include "tbb/concurrent_vector.h"
#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif
#if _MSC_VER && defined(_Wp64)
#pragma warning (push)
#pragma warning (disable: 4244)
#endif

#include "job_automaton.h"
#include "wait_counter.h"
#include "thread_monitor.h"

#if RML_USE_WCRM
#include <concrt.h>
#include <concrtrm.h>
using namespace Concurrency;
#include <vector>
#include <unordered_map>
#define __RML_REMOVE_VIRTUAL_PROCESSORS_DISABLED 0
#endif 

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace rml {
namespace internal {

using tbb::internal::rml::tbb_client;
using tbb::internal::rml::tbb_server;

using __kmp::rml::omp_client;
using __kmp::rml::omp_server;

typedef versioned_object::version_type version_type;

#define SERVER_VERSION 2
#define EARLIEST_COMPATIBLE_CLIENT_VERSION 2

static const size_t cache_line_size = tbb::internal::NFS_MaxLineSize;

template<typename Server, typename Client> class generic_connection;
class tbb_connection_v2;
class omp_connection_v2;

#if RML_USE_WCRM


enum thread_state_t {
ts_idle,
ts_asleep,
ts_busy,
ts_done
};

enum thread_extra_state_t {
ts_none,
ts_removed,
ts_lent
};

enum thread_grab_t {
wk_failed,
wk_from_asleep,
wk_from_idle
};

#else 


enum thread_state_t {
ts_idle,
ts_asleep,
ts_omp_busy,
ts_tbb_busy,
ts_done,
ts_created,
ts_started,
ts_visited,
ts_none,
ts_deactivated,
ts_reactivated
};
#endif 

#if TBB_USE_ASSERT
#define PRODUCE_ARG(x) ,x
#else
#define PRODUCE_ARG(x)
#endif 

class omp_dispatch_type {
typedef ::rml::job job_type;
omp_client* client;
void* cookie;
omp_client::size_type index;
tbb::atomic<job_type*> job;
#if TBB_USE_ASSERT
omp_connection_v2* server;
#endif 
public:
omp_dispatch_type() {job=NULL;}
void consume();
void produce( omp_client& c, job_type* j, void* cookie_, omp_client::size_type index_ PRODUCE_ARG( omp_connection_v2& s )) {
__TBB_ASSERT( j, NULL );
__TBB_ASSERT( !job, "job already set" );
client = &c;
#if TBB_USE_ASSERT
server = &s;
#endif 
cookie = cookie_;
index = index_;
job = j;
}
};


class ref_count: no_copy {
friend class thread_map;
tbb::atomic<int> my_ref_count;
public:
ref_count(int k ) {my_ref_count=k;}
~ref_count() {__TBB_ASSERT( !my_ref_count, "premature destruction of refcounted object" );}
int add_ref() {
int k = ++my_ref_count;
__TBB_ASSERT(k>=1,"reference count underflowed before add_ref");
return k;
}
int remove_ref() {
int k = --my_ref_count;
__TBB_ASSERT(k>=0,"reference count underflow");
return k;
}
};

#if RML_USE_WCRM

#if USE_UMS_THREAD
#define RML_THREAD_KIND UmsThreadDefault
#define RML_THREAD_KIND_STRING "UmsThread"
#else
#define RML_THREAD_KIND ThreadScheduler
#define RML_THREAD_KIND_STRING "WinThread"
#endif

class thread_map;

static const IExecutionResource* c_remove_prepare = (IExecutionResource*)0;
static const IExecutionResource* c_remove_returned = (IExecutionResource*)1;

class server_thread_rep : no_copy {
friend class thread_map;
friend class omp_connection_v2;
friend class server_thread;
friend class tbb_server_thread;
friend class omp_server_thread;
template<typename Connection> friend void make_job( Connection& c, typename Connection::server_thread_type& t );
typedef int thread_state_rep_t;
public:
server_thread_rep( bool assigned, IScheduler* s, IExecutionResource* r, thread_map& map, rml::client& cl ) :
uid( GetExecutionContextId() ), my_scheduler(s), my_proxy(NULL),
my_thread_map(map), my_client(cl), my_job(NULL)
{
my_state = assigned ? ts_busy : ts_idle;
my_extra_state = ts_none;
terminate = false;
my_execution_resource = r;
}
~server_thread_rep() {}

inline rml::job* wait_for_job() {
if( !my_job ) my_job = my_job_automaton.wait_for_job();
return my_job;
}

inline thread_state_t read_state() const { thread_state_rep_t s = my_state; return static_cast<thread_state_t>(s); }
inline void set_state( thread_state_t to ) {my_state = to;}
inline void set_removed() { __TBB_ASSERT( my_extra_state==ts_none, NULL ); my_extra_state = ts_removed; }
inline bool is_removed() const { return my_extra_state==ts_removed; }
inline bool is_lent() const {return my_extra_state==ts_lent;}
inline void set_lent() { my_extra_state=ts_lent; }
inline void set_returned() { my_extra_state=ts_none; }
inline IExecutionResource* get_execution_resource() { return my_execution_resource; }
inline IVirtualProcessorRoot* get_virtual_processor() { return (IVirtualProcessorRoot*)get_execution_resource(); }

inline bool wakeup( thread_state_t to, thread_state_t from ) {
__TBB_ASSERT( from==ts_asleep && (to==ts_idle||to==ts_busy||to==ts_done), NULL );
return my_state.compare_and_swap( to, from )==from;
}

thread_grab_t try_grab_for();

template<typename Connection> bool destroy_job( Connection* c );

void revive( IScheduler* s, IExecutionResource* r, rml::client& c ) {
__TBB_ASSERT( my_scheduler==s, "my_scheduler has been altered?\n" );
my_scheduler = s;
__TBB_ASSERT( &my_client==&c, "my_client has been altered?\n" );
if( r ) my_execution_resource = r;
my_client = c;
my_state = ts_idle;
__TBB_ASSERT( my_extra_state==ts_removed, NULL );
my_extra_state = ts_none;
}

protected:
const int uid;
IScheduler* my_scheduler;
IThreadProxy* my_proxy;
tbb::atomic<IExecutionResource*> my_execution_resource; 
thread_map& my_thread_map;
rml::client& my_client;
job* my_job;
job_automaton my_job_automaton;
tbb::atomic<bool> terminate;
tbb::atomic<thread_state_rep_t> my_state;
tbb::atomic<thread_extra_state_t> my_extra_state;
};

class server_thread : public IExecutionContext, public server_thread_rep {
friend class tbb_connection_v2;
friend class omp_connection_v2;
friend class tbb_server_thread;
friend class omp_server_thread;
friend class thread_map;
template<typename Connection> friend void make_job( Connection& c, typename Connection::server_thread_type& t );
protected:
server_thread( bool is_tbb, bool assigned, IScheduler* s, IExecutionResource* r, thread_map& map, rml::client& cl ) : server_thread_rep(assigned,s,r,map,cl), tbb_thread(is_tbb) {}
~server_thread() {}
unsigned int GetId() const __TBB_override { return uid; }
IScheduler* GetScheduler() __TBB_override { return my_scheduler; }
IThreadProxy* GetProxy()   __TBB_override { return my_proxy; }
void SetProxy( IThreadProxy* thr_proxy ) __TBB_override { my_proxy = thr_proxy; }

private:
bool tbb_thread;
};

class tbb_connection_v2;
class omp_connection_v2;

class tbb_server_thread : public server_thread {
friend class tbb_connection_v2;
public:
tbb_server_thread( bool assigned, IScheduler* s, IExecutionResource* r, tbb_connection_v2* con, thread_map& map, rml::client& cl ) : server_thread(true,assigned,s,r,map,cl), my_conn(con) {
activation_count = 0;
}
~tbb_server_thread() {}
void Dispatch( DispatchState* ) __TBB_override;
inline bool initiate_termination();
bool sleep_perhaps();
bool switch_out();
private:
tbb_connection_v2* my_conn;
public:
tbb::atomic<int> activation_count;
};

class omp_server_thread : public server_thread {
friend class omp_connection_v2;
public:
omp_server_thread( bool assigned, IScheduler* s, IExecutionResource* r, omp_connection_v2* con, thread_map& map, rml::client& cl ) :
server_thread(false,assigned,s,r,map,cl), my_conn(con), my_cookie(NULL), my_index(UINT_MAX) {}
~omp_server_thread() {}
void Dispatch( DispatchState* ) __TBB_override;
inline void* get_cookie() {return my_cookie;}
inline ::__kmp::rml::omp_client::size_type get_index() {return my_index;}

inline IExecutionResource* get_execution_resource() { return get_execution_resource(); }
inline bool initiate_termination() { return destroy_job( (omp_connection_v2*) my_conn ); }
void sleep_perhaps();
private:
omp_connection_v2* my_conn;
void* my_cookie;
::__kmp::rml::omp_client::size_type my_index;
omp_dispatch_type omp_data;
};

template<typename Connection>
class scheduler : no_copy, public IScheduler {
public:
unsigned int GetId() const __TBB_override {return uid;}
void Statistics( unsigned int* , unsigned int* , unsigned int* ) __TBB_override {}
SchedulerPolicy GetPolicy() const __TBB_override { __TBB_ASSERT(my_policy,NULL); return *my_policy; }
void AddVirtualProcessors( IVirtualProcessorRoot** vproots, unsigned int count ) __TBB_override { if( !my_conn.is_closing() ) my_conn.add_virtual_processors( vproots, count); }
void RemoveVirtualProcessors( IVirtualProcessorRoot** vproots, unsigned int count ) __TBB_override;
void NotifyResourcesExternallyIdle( IVirtualProcessorRoot** vproots, unsigned int count ) __TBB_override { __TBB_ASSERT( false, "This call is not allowed for TBB" ); }
void NotifyResourcesExternallyBusy( IVirtualProcessorRoot** vproots, unsigned int count ) __TBB_override { __TBB_ASSERT( false, "This call is not allowed for TBB" ); }
protected:
scheduler( Connection& conn );
virtual ~scheduler() { __TBB_ASSERT( my_policy, NULL ); delete my_policy; }

public:
static scheduler* create( Connection& conn ) {return new scheduler( conn );}

private:
const int uid;
Connection& my_conn;
SchedulerPolicy* my_policy;
};



class thread_scavenger_thread : public IExecutionContext, no_copy {
public:
thread_scavenger_thread( IScheduler* s, IVirtualProcessorRoot* r, thread_map& map ) :
uid( GetExecutionContextId() ), my_scheduler(s), my_virtual_processor_root(r), my_proxy(NULL), my_thread_map(map)
{
my_state = ts_busy;
#if TBB_USE_ASSERT
activation_count = 0;
#endif
}
~thread_scavenger_thread() {}
unsigned int GetId() const __TBB_override { return uid; }
IScheduler* GetScheduler() __TBB_override { return my_scheduler; }
IThreadProxy* GetProxy()   __TBB_override { return my_proxy; }
void SetProxy( IThreadProxy* thr_proxy ) __TBB_override { my_proxy = thr_proxy; }
void Dispatch( DispatchState* ) __TBB_override;
inline thread_state_t read_state() { return my_state; }
inline void set_state( thread_state_t s ) { my_state = s; }
inline IVirtualProcessorRoot* get_virtual_processor() { return my_virtual_processor_root; }
private:
const int uid;
IScheduler* my_scheduler;
IVirtualProcessorRoot* my_virtual_processor_root;
IThreadProxy* my_proxy;
thread_map& my_thread_map;
tbb::atomic<thread_state_t> my_state;
#if TBB_USE_ASSERT
public:
tbb::atomic<int> activation_count;
#endif
};

static const thread_scavenger_thread* c_claimed = reinterpret_cast<thread_scavenger_thread*>(1);

struct garbage_connection_queue {
tbb::atomic<uintptr_t> head;
tbb::atomic<uintptr_t> tail;
static const uintptr_t empty = 0; 
static const uintptr_t plugged = 1;  
static const uintptr_t plugged_acked = 2;  
};


class connection_scavenger_thread {
friend void assist_cleanup_connections();

tbb::atomic<thread_state_t> state;


thread_monitor monitor;
HANDLE thr_handle;
#if TBB_USE_ASSERT
tbb::atomic<int> n_scavenger_threads;
#endif

public:
connection_scavenger_thread() : thr_handle(NULL) {
state = ts_asleep;
#if TBB_USE_ASSERT
n_scavenger_threads = 0;
#endif
}

~connection_scavenger_thread() {}

void wakeup() {
if( state.compare_and_swap( ts_busy, ts_asleep )==ts_asleep )
monitor.notify();
}

void sleep_perhaps();

void process_requests( uintptr_t conn_ex );

static __RML_DECL_THREAD_ROUTINE thread_routine( void* arg );

void launch() {
thread_monitor::launch( connection_scavenger_thread::thread_routine, this, NULL );
}

template<typename Server, typename Client>
void add_request( generic_connection<Server,Client>* conn_to_close );

template<typename Server, typename Client>
uintptr_t grab_and_prepend( generic_connection<Server,Client>* last_conn_to_close );
};

void free_all_connections( uintptr_t );

#endif 

#if !RML_USE_WCRM
class server_thread;

struct thread_map_base {
class value_type {
public:
server_thread& thread() {
__TBB_ASSERT( my_thread, "thread_map::value_type::thread() called when !my_thread" );
return *my_thread;
}
rml::job& job() {
__TBB_ASSERT( my_job, "thread_map::value_type::job() called when !my_job" );
return *my_job;
}
value_type() : my_thread(NULL), my_job(NULL) {}
server_thread& wait_for_thread() const {
for(;;) {
server_thread* ptr=const_cast<server_thread*volatile&>(my_thread);
if( ptr )
return *ptr;
__TBB_Yield();
}
}

rml::job* wait_for_job() const {
if( !my_job ) {
my_job = my_automaton.wait_for_job();
}
return my_job;
}
private:
server_thread* my_thread;

mutable rml::job* my_job;
job_automaton my_automaton;
friend class thread_map;
};
typedef tbb::concurrent_vector<value_type,tbb::zero_allocator<value_type,tbb::cache_aligned_allocator> > array_type;
};
#endif 

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable:4510 4610)
#endif

template<typename T>
class padded: public T {
char pad[cache_line_size - sizeof(T)%cache_line_size];
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(pop)
#endif

static unsigned the_default_concurrency;
static tbb::atomic<int> the_balance;
static tbb::atomic<tbb::internal::do_once_state> rml_module_state;

#if !RML_USE_WCRM

class server_thread: public ref_count {
friend class thread_map;
template<typename Server, typename Client> friend class generic_connection;
friend class tbb_connection_v2;
friend class omp_connection_v2;
typedef int thread_state_rep_t;
tbb::atomic<thread_state_rep_t> state;
public:
thread_monitor monitor;
private:
bool    is_omp_thread;
tbb::atomic<thread_state_rep_t> my_extra_state;
server_thread* link;
thread_map_base::array_type::iterator my_map_pos;
rml::server *my_conn;
rml::job* my_job;
job_automaton* my_ja;
size_t my_index;
tbb::atomic<bool> terminate;
omp_dispatch_type omp_dispatch;

#if TBB_USE_ASSERT
bool has_active_thread;
#endif 

void sleep_perhaps( thread_state_t asleep );


template<typename Connection>
bool destroy_job( Connection& c );


bool do_termination();

void loop();
static __RML_DECL_THREAD_ROUTINE thread_routine( void* arg );

public:
server_thread();

~server_thread();

thread_state_t read_state() const {
thread_state_rep_t s = state;
__TBB_ASSERT( unsigned(s)<=unsigned(ts_done), "corrupted server thread?" );
return thread_state_t(s);
}

thread_state_t read_extra_state() const {
thread_state_rep_t s = my_extra_state;
return thread_state_t(s);
}

void launch( size_t stack_size );


bool wakeup( thread_state_t to, thread_state_t from );


bool try_grab_for( thread_state_t s );

#if _WIN32||_WIN64
void deactivate();

void reactivate();
#endif 
};

class private_thread_bag {
struct list_thread: server_thread {
list_thread* next;
};

tbb::atomic<list_thread*> my_root;
tbb::cache_aligned_allocator<padded<list_thread> > my_allocator;
public:
private_thread_bag() {my_root=NULL;}

server_thread& add_one_thread() {
list_thread* t = my_allocator.allocate(1);
new( t ) list_thread;
list_thread* old_root;
do {
old_root = my_root;
t->next = old_root;
} while( my_root.compare_and_swap( t, old_root )!=old_root );
return *t;
}

~private_thread_bag() {
while( my_root ) {
list_thread* t = my_root;
my_root = t->next;
t->~list_thread();
my_allocator.deallocate(static_cast<padded<list_thread>*>(t),1);
}
}
};

void wakeup_some_tbb_threads();


class thread_map : public thread_map_base {
public:
typedef rml::client::size_type size_type;
thread_map( wait_counter& fc, ::rml::client& client ) :
all_visited_at_least_once(false), my_min_stack_size(0), my_server_ref_count(1),
my_client_ref_count(1), my_client(client), my_factory_counter(fc)
{ my_unrealized_threads = 0; }
~thread_map() {}
typedef array_type::iterator iterator;
iterator begin() {return my_array.begin();}
iterator end() {return my_array.end();}
void bind();
void unbind();
void assist_cleanup( bool assist_null_only );


size_type wakeup_tbb_threads( size_type n );
bool wakeup_next_thread( iterator i, tbb_connection_v2& conn );
void release_tbb_threads( server_thread* t );
void adjust_balance( int delta );


value_type* add_one_thread( bool is_omp_thread_ );

void bind_one_thread( rml::server& server, value_type& x );

void remove_client_ref();
int add_server_ref() {return my_server_ref_count.add_ref();}
int remove_server_ref() {return my_server_ref_count.remove_ref();}

::rml::client& client() const {return my_client;}

size_type get_unrealized_threads() { return my_unrealized_threads; }

private:
private_thread_bag my_private_threads;
bool all_visited_at_least_once;
array_type my_array;
size_t my_min_stack_size;
tbb::atomic<size_type> my_unrealized_threads;


ref_count my_server_ref_count;


ref_count my_client_ref_count;

::rml::client& my_client;
wait_counter& my_factory_counter;
};

void thread_map::bind_one_thread( rml::server& server, value_type& x ) {
server_thread& t = x.thread();
my_server_ref_count.add_ref();
my_client_ref_count.add_ref();
#if TBB_USE_ASSERT
__TBB_ASSERT( t.add_ref()==1, NULL );
#else
t.add_ref();
#endif
t.my_conn = &server;
t.my_ja = &x.my_automaton;
t.launch( my_min_stack_size );

__TBB_ASSERT( t.state!=ts_tbb_busy, NULL );
t.wakeup( ts_idle, ts_asleep );
}

thread_map::value_type* thread_map::add_one_thread( bool is_omp_thread_ ) {
size_type u;
do {
u = my_unrealized_threads;
if( !u ) return NULL;
} while( my_unrealized_threads.compare_and_swap(u-1,u)!=u );
server_thread& t = my_private_threads.add_one_thread();
t.is_omp_thread = is_omp_thread_;
__TBB_ASSERT( u>=1, NULL );
t.my_index = u - 1;
__TBB_ASSERT( t.state!=ts_tbb_busy, NULL );
t.my_extra_state = t.is_omp_thread ? ts_none : ts_created;

iterator i = t.my_map_pos = my_array.grow_by(1);
value_type& v = *i;
v.my_thread = &t;
return &v;
}

void thread_map::bind() {
++my_factory_counter;
my_min_stack_size = my_client.min_stack_size();
__TBB_ASSERT( my_unrealized_threads==0, "already called bind?" );
my_unrealized_threads = my_client.max_job_count();
}

void thread_map::unbind() {
for( iterator i=begin(); i!=end(); ++i ) {
server_thread& t = i->thread();
t.terminate = true;
t.wakeup( ts_idle, ts_asleep );
}
remove_client_ref();
}

void thread_map::assist_cleanup( bool assist_null_only ) {
for( iterator i = begin(); i!=end(); ++i ) {
rml::job* j=0;
job_automaton& ja = i->my_automaton;
if( assist_null_only ? ja.try_plug_null() : ja.try_plug(j) ) {
if( j ) {
my_client.cleanup(*j);
} else {
}
remove_client_ref();
}
}
}

thread_map::size_type thread_map::wakeup_tbb_threads( size_type n ) {
__TBB_ASSERT(n>0,"must specify positive number of threads to wake up");
iterator e = end();
for( iterator k=begin(); k!=e; ++k ) {
server_thread& t = k->wait_for_thread();
thread_state_t thr_s = t.read_state();
if( t.read_extra_state()==ts_created || thr_s==ts_tbb_busy || thr_s==ts_done )
continue;
if( --the_balance>=0 ) { 
while( !t.try_grab_for( ts_tbb_busy ) ) {
thr_s = t.read_state();
if( thr_s==ts_tbb_busy || thr_s==ts_done ) {
++the_balance;
goto skip;
}
}
if( --n==0 )
return 0;
} else {
++the_balance;
break;
}
skip:
;
}
return n<my_unrealized_threads ? n : size_type(my_unrealized_threads);
}
#else 

class thread_map : no_copy {
friend class omp_connection_v2;
typedef ::std::unordered_map<uintptr_t,server_thread*> unordered_map_type;
size_t my_min_stack_size;
size_t my_unrealized_threads;
::rml::client& my_client;
wait_counter& my_factory_counter;
ref_count my_server_ref_count;
ref_count my_client_ref_count;
unordered_map_type my_map;
bool shutdown_in_progress;
std::vector<IExecutionResource*> original_exec_resources;
tbb::cache_aligned_allocator<padded<tbb_server_thread> > my_tbb_allocator;
tbb::cache_aligned_allocator<padded<omp_server_thread> > my_omp_allocator;
tbb::cache_aligned_allocator<padded<thread_scavenger_thread> > my_scavenger_allocator;
IResourceManager* my_concrt_resource_manager;
IScheduler* my_scheduler;
ISchedulerProxy* my_scheduler_proxy;
tbb::atomic<thread_scavenger_thread*> my_thread_scavenger_thread;
#if TBB_USE_ASSERT
tbb::atomic<int> n_add_vp_requests;
tbb::atomic<int> n_thread_scavengers_created;
#endif
public:
thread_map( wait_counter& fc, ::rml::client& client ) :
my_min_stack_size(0), my_client(client), my_factory_counter(fc),
my_server_ref_count(1), my_client_ref_count(1), shutdown_in_progress(false),
my_concrt_resource_manager(NULL), my_scheduler(NULL), my_scheduler_proxy(NULL)
{
my_thread_scavenger_thread = NULL;
#if TBB_USE_ASSERT
n_add_vp_requests = 0;
n_thread_scavengers_created;
#endif
}

~thread_map() {
__TBB_ASSERT( n_thread_scavengers_created<=1, "too many scavenger thread created" );
if( my_thread_scavenger_thread ) {
__TBB_ASSERT( my_thread_scavenger_thread!=c_claimed, NULL );
while( my_thread_scavenger_thread->read_state()==ts_busy )
__TBB_Yield();
thread_scavenger_thread* tst = my_thread_scavenger_thread;
my_scavenger_allocator.deallocate(static_cast<padded<thread_scavenger_thread>*>(tst),1);
}
for( unordered_map_type::const_iterator hi=my_map.begin(); hi!=my_map.end(); ++hi ) {
server_thread* thr = hi->second;
if( thr->tbb_thread ) {
while( ((tbb_server_thread*)thr)->activation_count>1 )
__TBB_Yield();
((tbb_server_thread*)thr)->~tbb_server_thread();
my_tbb_allocator.deallocate(static_cast<padded<tbb_server_thread>*>(thr),1);
} else {
((omp_server_thread*)thr)->~omp_server_thread();
my_omp_allocator.deallocate(static_cast<padded<omp_server_thread>*>(thr),1);
}
}
if( my_scheduler_proxy ) {
my_scheduler_proxy->Shutdown();
my_concrt_resource_manager->Release();
__TBB_ASSERT( my_scheduler, NULL );
delete my_scheduler;
} else {
__TBB_ASSERT( !my_scheduler, NULL );
}
}
typedef unordered_map_type::key_type key_type;
typedef unordered_map_type::value_type value_type;
typedef unordered_map_type::iterator iterator;
iterator begin() {return my_map.begin();}
iterator end() {return my_map.end();}
iterator find( key_type k ) {return my_map.find( k );}
iterator insert( key_type k, server_thread* v ) {
std::pair<iterator,bool> res = my_map.insert( value_type(k,v) );
return res.first;
}
void bind( IScheduler* s ) {
++my_factory_counter;
if( s ) {
my_unrealized_threads = s->GetPolicy().GetPolicyValue( MaxConcurrency );
__TBB_ASSERT( my_unrealized_threads>0, NULL );
my_scheduler = s;
my_concrt_resource_manager = CreateResourceManager(); 
my_scheduler_proxy = my_concrt_resource_manager->RegisterScheduler( s, CONCRT_RM_VERSION_1 );
my_scheduler_proxy->RequestInitialVirtualProcessors( false );
}
}
bool is_closing() { return shutdown_in_progress; }
void unbind( rml::server& server, ::tbb::spin_mutex& mtx );
void add_client_ref() { my_server_ref_count.add_ref(); }
void remove_client_ref();
void add_server_ref() {my_server_ref_count.add_ref();}
int remove_server_ref() {return my_server_ref_count.remove_ref();}
int get_server_ref_count() { int k = my_server_ref_count.my_ref_count; return k; }
void assist_cleanup( bool assist_null_only );
void adjust_balance( int delta );
int current_balance() const {int k = the_balance; return k;}
::rml::client& client() const {return my_client;}
void register_as_master( server::execution_resource_t& v ) const { (IExecutionResource*&)v = my_scheduler_proxy ? my_scheduler_proxy->SubscribeCurrentThread() : NULL; }
void unregister( server::execution_resource_t v ) const {if( v ) ((IExecutionResource*)v)->Remove( my_scheduler );}
void add_virtual_processors( IVirtualProcessorRoot** vprocs, unsigned int count, tbb_connection_v2& conn, ::tbb::spin_mutex& mtx );
void add_virtual_processors( IVirtualProcessorRoot** vprocs, unsigned int count, omp_connection_v2& conn, ::tbb::spin_mutex& mtx );
void remove_virtual_processors( IVirtualProcessorRoot** vproots, unsigned count, ::tbb::spin_mutex& mtx );
void mark_virtual_processors_as_lent( IVirtualProcessorRoot** vproots, unsigned count, ::tbb::spin_mutex& mtx );
void create_oversubscribers( unsigned n, std::vector<server_thread*>& thr_vec, omp_connection_v2& conn, ::tbb::spin_mutex& mtx );
void wakeup_tbb_threads( int c, ::tbb::spin_mutex& mtx );
void mark_virtual_processors_as_returned( IVirtualProcessorRoot** vprocs, unsigned int count, tbb::spin_mutex& mtx );
inline void addto_original_exec_resources( IExecutionResource* r, ::tbb::spin_mutex& mtx ) {
::tbb::spin_mutex::scoped_lock lck(mtx);
__TBB_ASSERT( !is_closing(), "trying to register master while connection is being shutdown?" );
original_exec_resources.push_back( r );
}
#if !__RML_REMOVE_VIRTUAL_PROCESSORS_DISABLED
void allocate_thread_scavenger( IExecutionResource* v );
#endif
inline thread_scavenger_thread* get_thread_scavenger() { return my_thread_scavenger_thread; }
};

garbage_connection_queue connections_to_reclaim;
connection_scavenger_thread connection_scavenger;

#endif 


template<typename Server, typename Client>
struct connection_traits {};

static tbb::atomic<uintptr_t> active_tbb_connections;
static tbb::atomic<int> current_tbb_conn_readers;
static size_t current_tbb_conn_reader_epoch;
static tbb::atomic<size_t> close_tbb_connection_event_count;

#if RML_USE_WCRM
template<typename Connection>
void make_job( Connection& c, server_thread& t );
#endif

template<typename Server, typename Client>
class generic_connection: public Server, no_copy {
version_type version() const __TBB_override {return SERVER_VERSION;}
void yield() __TBB_override {thread_monitor::yield();}
void independent_thread_number_changed( int delta ) __TBB_override { my_thread_map.adjust_balance( -delta ); }
unsigned default_concurrency() const __TBB_override { return the_default_concurrency; }
friend void wakeup_some_tbb_threads();
friend class connection_scavenger_thread;

protected:
thread_map my_thread_map;
generic_connection* next_conn;
size_t my_ec;
#if RML_USE_WCRM
tbb::spin_mutex map_mtx;
IScheduler* my_scheduler;
void do_open( IScheduler* s ) {
my_scheduler = s;
my_thread_map.bind( s );
}
bool is_closing() { return my_thread_map.is_closing(); }
void request_close_connection( bool existing );
#else
void do_open() {my_thread_map.bind();}
void request_close_connection( bool );
#endif 
virtual ~generic_connection() {}
#if !RML_USE_WCRM
generic_connection( wait_counter& fc, Client& c ) : my_thread_map(fc,c), next_conn(NULL), my_ec(0) {}
#else
generic_connection( wait_counter& fc, Client& c ) :
my_thread_map(fc,c), next_conn(NULL), my_ec(0), map_mtx(), my_scheduler(NULL) {}
void add_virtual_processors( IVirtualProcessorRoot** vprocs, unsigned int count );
void remove_virtual_processors( IVirtualProcessorRoot** vprocs, unsigned int count );
void notify_resources_externally_busy( IVirtualProcessorRoot** vprocs, unsigned int count ) { my_thread_map.mark_virtual_processors_as_lent( vprocs, count, map_mtx ); }
void notify_resources_externally_idle( IVirtualProcessorRoot** vprocs, unsigned int count ) {
my_thread_map.mark_virtual_processors_as_returned( vprocs, count, map_mtx );
}
#endif 

public:
typedef Server server_type;
typedef Client client_type;
Client& client() const {return static_cast<Client&>(my_thread_map.client());}
void set_scratch_ptr( job& j, void* ptr ) { ::rml::server::scratch_ptr(j) = ptr; }
#if RML_USE_WCRM
template<typename Connection>
friend void make_job( Connection& c, server_thread& t );
void add_server_ref ()   {my_thread_map.add_server_ref();}
void remove_server_ref() {if( my_thread_map.remove_server_ref()==0 ) delete this;}
void add_client_ref ()   {my_thread_map.add_client_ref();}
void remove_client_ref() {my_thread_map.remove_client_ref();}
#else 
int  add_server_ref ()   {return my_thread_map.add_server_ref();}
void remove_server_ref() {if( my_thread_map.remove_server_ref()==0 ) delete this;}
void remove_client_ref() {my_thread_map.remove_client_ref();}
void make_job( server_thread& t, job_automaton& ja );
#endif 
static generic_connection* get_addr( uintptr_t addr_ex ) {
return reinterpret_cast<generic_connection*>( addr_ex&~(uintptr_t)3 );
}
};


template<>
struct connection_traits<tbb_server,tbb_client> {
static const bool assist_null_only = true;
static const bool is_tbb = true;
};


class tbb_connection_v2: public generic_connection<tbb_server,tbb_client> {
void adjust_job_count_estimate( int delta ) __TBB_override;
#if !RML_USE_WCRM
#if _WIN32||_WIN64
void register_master ( rml::server::execution_resource_t&  ) __TBB_override {}
void unregister_master ( rml::server::execution_resource_t  ) __TBB_override {}
#endif
#else
void register_master ( rml::server::execution_resource_t& v ) __TBB_override {
my_thread_map.register_as_master(v);
if( v ) ++nesting;
}
void unregister_master ( rml::server::execution_resource_t v ) __TBB_override {
if( v ) {
__TBB_ASSERT( nesting>0, NULL );
if( --nesting==0 ) {
#if !__RML_REMOVE_VIRTUAL_PROCESSORS_DISABLED
my_thread_map.allocate_thread_scavenger( (IExecutionResource*)v );
#endif
}
}
my_thread_map.unregister(v);
}
IScheduler* create_scheduler() {return( scheduler<tbb_connection_v2>::create( *this ) );}
friend void  free_all_connections( uintptr_t );
friend class scheduler<tbb_connection_v2>;
friend class execution_context;
friend class connection_scavenger_thread;
#endif 
friend void wakeup_some_tbb_threads();
tbb::atomic<int> my_slack;
friend class dummy_class_to_shut_up_gratuitous_warning_from_gcc_3_2_3;
#if TBB_USE_ASSERT
tbb::atomic<int> my_job_count_estimate;
#endif 

tbb::atomic<int> n_adjust_job_count_requests;
#if RML_USE_WCRM
tbb::atomic<int> nesting;
#endif

~tbb_connection_v2();

public:
#if RML_USE_WCRM
typedef tbb_server_thread server_thread_type;
#endif
bool has_slack() const {return my_slack>0;}

#if RML_USE_WCRM
bool try_process( job& job )
#else
bool try_process( server_thread& t, job& job )
#endif
{
bool visited = false;
int k = --my_slack;
if( k>=0 ) {
#if !RML_USE_WCRM
t.my_extra_state = ts_visited; 
#endif
client().process(job);
visited = true;
}
++my_slack;
return visited;
}

tbb_connection_v2( wait_counter& fc, tbb_client& client ) : generic_connection<tbb_server,tbb_client>(fc,client)
{
my_slack = 0;
#if RML_USE_WCRM
nesting = 0;
#endif
#if TBB_USE_ASSERT
my_job_count_estimate = 0;
#endif 
__TBB_ASSERT( !my_slack, NULL );

#if RML_USE_WCRM
do_open( client.max_job_count()>0 ? create_scheduler() : NULL );
#else
do_open();
#endif 
n_adjust_job_count_requests = 0;

uintptr_t conn;
do {
for( ; (conn=active_tbb_connections)&1; )
__TBB_Yield();
} while( active_tbb_connections.compare_and_swap( conn|1, conn )!=conn );

this->next_conn = generic_connection<tbb_server,tbb_client>::get_addr(conn);
active_tbb_connections = (uintptr_t) this; 
}
inline void wakeup_tbb_threads( unsigned n ) {
my_thread_map.wakeup_tbb_threads( n
#if RML_USE_WCRM
, map_mtx
#endif
);
}
#if RML_USE_WCRM
inline int get_nesting_level() { return nesting; }
#else
inline bool wakeup_next_thread( thread_map::iterator i ) {return my_thread_map.wakeup_next_thread( i, *this );}
inline thread_map::size_type get_unrealized_threads () {return my_thread_map.get_unrealized_threads();}
#endif 
};


template<>
struct connection_traits<omp_server,omp_client> {
static const bool assist_null_only = false;
static const bool is_tbb = false;
};

class omp_connection_v2: public generic_connection<omp_server,omp_client> {
#if !RML_USE_WCRM
int  current_balance() const __TBB_override {return the_balance;}
#else
friend void  free_all_connections( uintptr_t );
friend class scheduler<omp_connection_v2>;
int current_balance() const __TBB_override {return my_thread_map.current_balance();}
#endif 
int  try_increase_load( size_type n, bool strict ) __TBB_override;
void decrease_load( size_type n ) __TBB_override;
void get_threads( size_type request_size, void* cookie, job* array[] ) __TBB_override;
#if !RML_USE_WCRM
#if _WIN32||_WIN64
void register_master ( rml::server::execution_resource_t&  ) __TBB_override {}
void unregister_master ( rml::server::execution_resource_t  ) __TBB_override {}
#endif
#else
void register_master ( rml::server::execution_resource_t& v ) __TBB_override {
my_thread_map.register_as_master( v );
my_thread_map.addto_original_exec_resources( (IExecutionResource*)v, map_mtx );
}
void unregister_master ( rml::server::execution_resource_t v ) __TBB_override { my_thread_map.unregister(v); }
#endif 
#if _WIN32||_WIN64
void deactivate( rml::job* j ) __TBB_override;
void reactivate( rml::job* j ) __TBB_override;
#endif 
#if RML_USE_WCRM
public:
typedef omp_server_thread server_thread_type;
private:
IScheduler* create_scheduler() {return( scheduler<omp_connection_v2>::create( *this ) );}
#endif 
public:
#if TBB_USE_ASSERT

tbb::atomic<int> net_delta;
#endif 

omp_connection_v2( wait_counter& fc, omp_client& client ) : generic_connection<omp_server,omp_client>(fc,client) {
#if TBB_USE_ASSERT
net_delta = 0;
#endif 
#if RML_USE_WCRM
do_open( create_scheduler() );
#else
do_open();
#endif 
}
~omp_connection_v2() {__TBB_ASSERT( net_delta==0, "net increase/decrease of load is nonzero" );}
};

#if !RML_USE_WCRM


bool thread_map::wakeup_next_thread( thread_map::iterator this_thr, tbb_connection_v2& conn ) {
if( all_visited_at_least_once )
return false;

iterator e = end();
retry:
bool exist = false;
iterator k=this_thr;
for( ++k; k!=e; ++k ) {
server_thread& t = k->wait_for_thread();
if( t.my_extra_state!=ts_visited )
exist = true;
if( t.read_state()!=ts_tbb_busy && t.my_extra_state==ts_started )
if( t.try_grab_for( ts_tbb_busy ) )
return true;
}
for( k=begin(); k!=this_thr; ++k ) {
server_thread& t = k->wait_for_thread();
if( t.my_extra_state!=ts_visited )
exist = true;
if( t.read_state()!=ts_tbb_busy && t.my_extra_state==ts_started )
if( t.try_grab_for( ts_tbb_busy ) )
return true;
}

if( exist )
if( conn.has_slack() )
goto retry;
else
all_visited_at_least_once = true;
return false;
}

void thread_map::release_tbb_threads( server_thread* t ) {
for( ; t; t = t->link ) {
while( t->read_state()!=ts_asleep )
__TBB_Yield();
t->my_extra_state = ts_started;
}
}
#endif 

void thread_map::adjust_balance( int delta ) {
int new_balance = the_balance += delta;
if( new_balance>0 && 0>=new_balance-delta  )
wakeup_some_tbb_threads();
}

void thread_map::remove_client_ref() {
int k = my_client_ref_count.remove_ref();
if( k==0 ) {
--my_factory_counter;
my_client.acknowledge_close_connection();
}
}

#if RML_USE_WCRM

template<typename Connection>
void make_job( Connection& c, typename Connection::server_thread_type& t ) {
if( t.my_job_automaton.try_acquire() ) {
rml::job* j = t.my_client.create_one_job();
__TBB_ASSERT( j!=NULL, "client:::create_one_job returned NULL" );
__TBB_ASSERT( (intptr_t(j)&1)==0, "client::create_one_job returned misaligned job" );
t.my_job_automaton.set_and_release( j );
c.set_scratch_ptr( *j, (void*) &t );
}
}
#endif 

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif
#if RML_USE_WCRM
template<typename Server, typename Client>
void generic_connection<Server,Client>::request_close_connection( bool exiting ) {
if( connection_traits<Server,Client>::is_tbb )
__TBB_ASSERT( !exiting, NULL);
#if TBB_USE_ASSERT
else if( exiting )
reinterpret_cast<omp_connection_v2*>(this)->net_delta = 0;
#endif
if( exiting ) {
uintptr_t tail = connections_to_reclaim.tail;
while( connections_to_reclaim.tail.compare_and_swap( garbage_connection_queue::plugged, tail )!=tail )
__TBB_Yield();
my_thread_map.unbind( *this, map_mtx );
my_thread_map.assist_cleanup( connection_traits<Server,Client>::assist_null_only );
free_all_connections( connection_scavenger.grab_and_prepend( this ) );
return;
}
#else 
template<typename Server, typename Client>
void generic_connection<Server,Client>::request_close_connection( bool ) {
#endif 
if( connection_traits<Server,Client>::is_tbb ) {
uintptr_t conn;
do {
for( ; (conn=active_tbb_connections)&1; )
__TBB_Yield();
} while( active_tbb_connections.compare_and_swap( conn|1, conn )!=conn );

generic_connection* pred_conn = NULL;
generic_connection* curr_conn = (generic_connection*) conn;
for( ; curr_conn && curr_conn!=this; curr_conn=curr_conn->next_conn )
pred_conn = curr_conn;
__TBB_ASSERT( curr_conn==this, "the current connection is not in the list?" );

if( pred_conn ) {
pred_conn->next_conn = curr_conn->next_conn;
active_tbb_connections = reinterpret_cast<uintptr_t>(generic_connection<tbb_server,tbb_client>::get_addr(active_tbb_connections)); 
} else
active_tbb_connections = (uintptr_t) curr_conn->next_conn; 
curr_conn->next_conn = NULL;
my_ec = ++close_tbb_connection_event_count;
}
#if RML_USE_WCRM
my_thread_map.unbind( *this, map_mtx );
my_thread_map.assist_cleanup( connection_traits<Server,Client>::assist_null_only );
connection_scavenger.add_request( this );
#else
my_thread_map.unbind();
my_thread_map.assist_cleanup( connection_traits<Server,Client>::assist_null_only );
remove_server_ref();
#endif
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif

#if RML_USE_WCRM

template<typename Server, typename Client>
void generic_connection<Server,Client>::add_virtual_processors( IVirtualProcessorRoot** vproots, unsigned int count )
{}

template<>
void generic_connection<tbb_server,tbb_client>::add_virtual_processors( IVirtualProcessorRoot** vproots, unsigned int count )
{
my_thread_map.add_virtual_processors( vproots, count, (tbb_connection_v2&)*this, map_mtx );
}
template<>
void generic_connection<omp_server,omp_client>::add_virtual_processors( IVirtualProcessorRoot** vproots, unsigned int count )
{
my_thread_map.add_virtual_processors( vproots, count, (omp_connection_v2&)*this, map_mtx );
}

template<typename Server, typename Client>
void generic_connection<Server,Client>::remove_virtual_processors( IVirtualProcessorRoot** vproots, unsigned int count )
{
__TBB_ASSERT( false, "should not be called" );
}


template<>
void generic_connection<tbb_server,tbb_client>::remove_virtual_processors( IVirtualProcessorRoot** vproots, unsigned int count )
{
my_thread_map.remove_virtual_processors( vproots, count, map_mtx );
}

void tbb_connection_v2::adjust_job_count_estimate( int delta ) {
#if TBB_USE_ASSERT
my_job_count_estimate += delta;
#endif 
int c = my_slack+=delta;
if( c>0 ) {
++n_adjust_job_count_requests;
my_thread_map.wakeup_tbb_threads( c, map_mtx );
--n_adjust_job_count_requests;
}
}
#endif 

tbb_connection_v2::~tbb_connection_v2() {
#if TBB_USE_ASSERT
if( my_job_count_estimate!=0 ) {
fprintf(stderr, "TBB client tried to disconnect with non-zero net job count estimate of %d\n", int(my_job_count_estimate ));
abort();
}
__TBB_ASSERT( !my_slack, "attempt to destroy tbb_server with nonzero slack" );
__TBB_ASSERT( this!=static_cast<tbb_connection_v2*>(generic_connection<tbb_server,tbb_client >::get_addr(active_tbb_connections)), "request_close_connection() must be called" );
#endif 
#if !RML_USE_WCRM
if( the_balance>0 )
wakeup_some_tbb_threads();
#endif
while( current_tbb_conn_readers>0 && (ptrdiff_t)(my_ec-current_tbb_conn_reader_epoch)>0 )
__TBB_Yield();
}

#if !RML_USE_WCRM
template<typename Server, typename Client>
void generic_connection<Server,Client>::make_job( server_thread& t, job_automaton& ja ) {
if( ja.try_acquire() ) {
rml::job* j = client().create_one_job();
__TBB_ASSERT( j!=NULL, "client:::create_one_job returned NULL" );
__TBB_ASSERT( (intptr_t(j)&1)==0, "client::create_one_job returned misaligned job" );
ja.set_and_release( j );
__TBB_ASSERT( t.my_conn && t.my_ja && t.my_job==NULL, NULL );
t.my_job  = j;
set_scratch_ptr( *j, (void*) &t );
}
}

void tbb_connection_v2::adjust_job_count_estimate( int delta ) {
#if TBB_USE_ASSERT
my_job_count_estimate += delta;
#endif 
int c = my_slack+=delta;
if( c>0 ) {
++n_adjust_job_count_requests;
thread_map::size_type n = my_thread_map.wakeup_tbb_threads(c);

server_thread* new_threads_anchor = NULL;
thread_map::size_type i;
{
tbb::internal::affinity_helper fpa;
for( i=0; i<n; ++i ) {
thread_map::value_type* k = my_thread_map.add_one_thread( false );
if( !k )
break;
fpa.protect_affinity_mask( true );
my_thread_map.bind_one_thread( *this, *k );
server_thread& t = k->thread();
__TBB_ASSERT( !t.link, NULL );
t.link = new_threads_anchor;
new_threads_anchor = &t;
}
}

thread_map::size_type j=0;
for( ; the_balance>0 && j<i; ++j ) {
if( --the_balance>=0 ) {
__TBB_ASSERT( new_threads_anchor, NULL );

server_thread* t = new_threads_anchor;
new_threads_anchor = t->link;
while( !t->try_grab_for( ts_tbb_busy ) )
__TBB_Yield();
t->my_extra_state = ts_started;
} else {
++the_balance;
break;
}
}
__TBB_ASSERT( i-j!=0||new_threads_anchor==NULL, NULL );
if( new_threads_anchor )
my_thread_map.release_tbb_threads( new_threads_anchor );

--n_adjust_job_count_requests;
}
}
#endif 

#if RML_USE_WCRM
int omp_connection_v2::try_increase_load( size_type n, bool strict ) {
__TBB_ASSERT(int(n)>=0,NULL);
if( strict ) {
the_balance -= int(n);
} else {
int avail, old;
do {
avail = the_balance;
if( avail<=0 ) {
return avail;
}
old = the_balance.compare_and_swap( int(n)<avail ? avail-n : 0, avail );
} while( old!=avail );
if( int(n)>avail )
n=avail;
}
#if TBB_USE_ASSERT
net_delta += n;
#endif 
return n;
}

void omp_connection_v2::decrease_load( size_type  ) {}

void omp_connection_v2::get_threads( size_type request_size, void* cookie, job* array[] ) {
unsigned index = 0;
std::vector<omp_server_thread*> enlisted(request_size);
std::vector<thread_grab_t> to_activate(request_size);

if( request_size==0 ) return;

{
tbb::spin_mutex::scoped_lock lock(map_mtx);

__TBB_ASSERT( !is_closing(), "try to get threads while connection is being shutdown?" );

for( int scan=0; scan<2; ++scan ) {
for( thread_map::iterator i=my_thread_map.begin(); i!=my_thread_map.end(); ++i ) {
omp_server_thread* thr = (omp_server_thread*) (*i).second;
if( scan==0 && thr->is_lent() ) continue;
thread_grab_t res = thr->try_grab_for();
if( res!=wk_failed ) {
to_activate[index] = res;
enlisted[index] = thr;
if( ++index==request_size )
goto activate_threads;
}
}
}
}

activate_threads:

for( unsigned i=0; i<index; ++i ) {
omp_server_thread* thr = enlisted[i];
if( to_activate[i]==wk_from_asleep )
thr->get_virtual_processor()->Activate( thr );
job* j = thr->wait_for_job();
array[i] = j;
thr->omp_data.produce( client(), j, cookie, i PRODUCE_ARG(*this) );
}

if( index==request_size )
return;


__TBB_ASSERT( request_size>index, NULL );
unsigned n = request_size - index;
std::vector<server_thread*> thr_vec(n);
typedef std::vector<server_thread*>::iterator iterator_thr;
my_thread_map.create_oversubscribers( n, thr_vec, *this, map_mtx );
for( iterator_thr ti=thr_vec.begin(); ti!=thr_vec.end(); ++ti ) {
omp_server_thread* thr = (omp_server_thread*) *ti;
__TBB_ASSERT( thr, "thread not created?" );
thr->get_virtual_processor()->Activate( thr );
job* j = thr->wait_for_job();
array[index] = j;
thr->omp_data.produce( client(), j, cookie, index PRODUCE_ARG(*this) );
++index;
}
}

#if _WIN32||_WIN64
void omp_connection_v2::deactivate( rml::job* j )
{
my_thread_map.adjust_balance(1);
#if TBB_USE_ASSERT
net_delta -= 1;
#endif
omp_server_thread* thr = (omp_server_thread*) scratch_ptr( *j );
(thr->get_virtual_processor())->Deactivate( thr );
}

void omp_connection_v2::reactivate( rml::job* j )
{
omp_server_thread* thr = (omp_server_thread*) scratch_ptr( *j );
(thr->get_virtual_processor())->Activate( thr );
}
#endif 

#endif  

void wakeup_some_tbb_threads()
{

if( generic_connection<tbb_server,tbb_client >::get_addr(active_tbb_connections)==0 )
return;

start_it_over:
int n_curr_readers = ++current_tbb_conn_readers;
if( n_curr_readers>1 ) 
return;

current_tbb_conn_reader_epoch = close_tbb_connection_event_count;

generic_connection<tbb_server,tbb_client>* next_conn_wake_up = generic_connection<tbb_server,tbb_client>::get_addr( active_tbb_connections );

for( ; next_conn_wake_up; ) {


while( ((tbb_connection_v2*)next_conn_wake_up)->n_adjust_job_count_requests>0 )
__TBB_Yield();

int bal = the_balance;
n_curr_readers = current_tbb_conn_readers; 
if( bal<=0 ) break;

tbb_connection_v2* tbb_conn = (tbb_connection_v2*)next_conn_wake_up;
int my_slack = tbb_conn->my_slack;
if( my_slack>0 ) tbb_conn->wakeup_tbb_threads( my_slack );
next_conn_wake_up = next_conn_wake_up->next_conn;
}

int delta = current_tbb_conn_readers -= n_curr_readers;
if( delta>0 ) {
current_tbb_conn_readers = 0;
if( the_balance>0 && generic_connection<tbb_server,tbb_client >::get_addr(active_tbb_connections)!=0 )
goto start_it_over;
}

current_tbb_conn_reader_epoch = close_tbb_connection_event_count;
}

#if !RML_USE_WCRM
int omp_connection_v2::try_increase_load( size_type n, bool strict ) {
__TBB_ASSERT(int(n)>=0,NULL);
if( strict ) {
the_balance -= int(n);
} else {
int avail, old;
do {
avail = the_balance;
if( avail<=0 ) {
return avail;
}
old = the_balance.compare_and_swap( int(n)<avail ? avail-n : 0, avail );
} while( old!=avail );
if( int(n)>avail )
n=avail;
}
#if TBB_USE_ASSERT
net_delta += n;
#endif 
return n;
}

void omp_connection_v2::decrease_load( size_type n ) {
__TBB_ASSERT(int(n)>=0,NULL);
my_thread_map.adjust_balance(int(n));
#if TBB_USE_ASSERT
net_delta -= n;
#endif 
}

void omp_connection_v2::get_threads( size_type request_size, void* cookie, job* array[] ) {

if( !request_size )
return;

unsigned index = 0;
for(;;) { 
thread_map::iterator k_end=my_thread_map.end();
for( thread_map::iterator k=my_thread_map.begin(); k!=k_end; ++k ) {
server_thread& t = k->wait_for_thread();
if( t.try_grab_for( ts_omp_busy ) ) {
job* j = k->wait_for_job();
array[index] = j;
t.omp_dispatch.produce( client(), j, cookie, index PRODUCE_ARG(*this) );
if( ++index==request_size )
return;
}
}
for( unsigned i=index; i<request_size; ++i ) {
__TBB_ASSERT( index<request_size, NULL );
thread_map::value_type* k = my_thread_map.add_one_thread( true );
#if TBB_USE_ASSERT
if( !k ) {
__TBB_ASSERT(false, "server::get_threads: exceeded job_count\n");
}
#endif
my_thread_map.bind_one_thread( *this, *k );
server_thread& t = k->thread();
if( t.try_grab_for( ts_omp_busy ) ) {
job* j = k->wait_for_job();
array[index] = j;
t.omp_dispatch.produce( client(), j, cookie, index PRODUCE_ARG(*this) );
if( ++index==request_size )
return;
} 
}
}
}
#endif 

void omp_dispatch_type::consume() {
job_type* j;
tbb::internal::atomic_backoff backoff;
while( (j = job)==NULL ) backoff.pause();
job = static_cast<job_type*>(NULL);
client->process(*j,cookie,index);
#if TBB_USE_ASSERT
--server->net_delta;
#endif 
}

#if !RML_USE_WCRM
#if _WIN32||_WIN64
void omp_connection_v2::deactivate( rml::job* j )
{
#if TBB_USE_ASSERT
net_delta -= 1;
#endif
__TBB_ASSERT( j, NULL );
server_thread* thr = (server_thread*) scratch_ptr( *j );
thr->deactivate();
}

void omp_connection_v2::reactivate( rml::job* j )
{
__TBB_ASSERT( j, NULL );
server_thread* thr = (server_thread*) scratch_ptr( *j );
thr->reactivate();
}
#endif 


server_thread::server_thread() :
ref_count(0),
link(NULL),
my_map_pos(),
my_conn(NULL), my_job(NULL), my_ja(NULL)
{
state = ts_idle;
terminate = false;
#if TBB_USE_ASSERT
has_active_thread = false;
#endif 
}

server_thread::~server_thread() {
__TBB_ASSERT( !has_active_thread, NULL );
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable:4189)
#endif
__RML_DECL_THREAD_ROUTINE server_thread::thread_routine( void* arg ) {
server_thread* self = static_cast<server_thread*>(arg);
AVOID_64K_ALIASING( self->my_index );
#if TBB_USE_ASSERT
__TBB_ASSERT( !self->has_active_thread, NULL );
self->has_active_thread = true;
#endif 
self->loop();
return 0;
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(pop)
#endif

void server_thread::launch( size_t stack_size ) {
#if USE_WINTHREAD
thread_monitor::launch( thread_routine, this, stack_size, &this->my_index );
#else
thread_monitor::launch( thread_routine, this, stack_size );
#endif 
}

void server_thread::sleep_perhaps( thread_state_t asleep ) {
if( terminate ) return;
__TBB_ASSERT( asleep==ts_asleep, NULL );
thread_monitor::cookie c;
monitor.prepare_wait(c);
if( state.compare_and_swap( asleep, ts_idle )==ts_idle ) {
if( !terminate ) {
monitor.commit_wait(c);
} else {
monitor.cancel_wait();
}
thread_state_t s = read_state();
if( s==ts_asleep ) {
state.compare_and_swap( ts_idle, ts_asleep );
} else {
if( !is_omp_thread ) {
__TBB_ASSERT( s==ts_tbb_busy||s==ts_idle, NULL );
}
}
} else {
__TBB_ASSERT( state==ts_omp_busy||state==ts_tbb_busy, NULL );
monitor.cancel_wait();
}
__TBB_ASSERT( read_state()!=asleep, "a thread can only put itself to sleep" );
}

bool server_thread::wakeup( thread_state_t to, thread_state_t from ) {
bool success = false;
__TBB_ASSERT( from==ts_asleep && (to==ts_idle||to==ts_omp_busy||to==ts_tbb_busy), NULL );
if( state.compare_and_swap( to, from )==from ) {
if( !is_omp_thread ) __TBB_ASSERT( to==ts_idle||to==ts_tbb_busy, NULL );
monitor.notify();
success = true;
}
return success;
}

bool server_thread::try_grab_for( thread_state_t target_state ) {
bool success = false;
switch( read_state() ) {
case ts_asleep:
success = wakeup( target_state, ts_asleep );
break;
case ts_idle:
success = state.compare_and_swap( target_state, ts_idle )==ts_idle;
break;
default:
break;
}
return success;
}

#if _WIN32||_WIN64
void server_thread::deactivate() {
thread_state_t es = (thread_state_t) my_extra_state.fetch_and_store( ts_deactivated );
__TBB_ASSERT( my_extra_state==ts_deactivated, "someone else tampered with my_extra_state?" );
if( es==ts_none )
state = ts_idle;
else
__TBB_ASSERT( es==ts_reactivated, "Cannot call deactivate() while in ts_deactivated" );
__TBB_ASSERT( my_extra_state==ts_deactivated, "someone else tampered with my_extra_state?" );
my_extra_state = ts_none; 
int bal = ++the_balance;
if( bal>0 )
wakeup_some_tbb_threads();
if( es==ts_none )
sleep_perhaps( ts_asleep );
}

void server_thread::reactivate() {
thread_state_t es;
do {
while( (es=read_extra_state())==ts_deactivated )
__TBB_Yield();
if( es==ts_reactivated ) {
__TBB_ASSERT( false, "two Reactivate() calls in a row.  Should not happen" );
return;
}
__TBB_ASSERT( es==ts_none, NULL );
} while( (thread_state_t)my_extra_state.compare_and_swap( ts_reactivated, ts_none )!=ts_none );
if( state!=ts_omp_busy ) {
my_extra_state = ts_none;
while( !try_grab_for( ts_omp_busy ) )
__TBB_Yield();
}
}
#endif 


template<typename Connection>
bool server_thread::destroy_job( Connection& c ) {
__TBB_ASSERT( !is_omp_thread||(state==ts_idle||state==ts_omp_busy), NULL );
__TBB_ASSERT(  is_omp_thread||(state==ts_idle||state==ts_tbb_busy), NULL );
if( !is_omp_thread ) {
__TBB_ASSERT( state==ts_idle||state==ts_tbb_busy, NULL );
if( state==ts_idle )
state.compare_and_swap( ts_done, ts_idle );

if( state==ts_tbb_busy ) { 
++the_balance;
state = ts_done; 
}
}
if( job_automaton* ja = my_ja ) {
rml::job* j;
if( ja->try_plug(j) ) {
__TBB_ASSERT( j, NULL );
c.client().cleanup(*j);
c.remove_client_ref();
} else {
}
}
int k = remove_ref();
__TBB_ASSERT_EX( k==0, "more than one references?" );
#if TBB_USE_ASSERT
has_active_thread = false;
#endif 
c.remove_server_ref();
return true;
}

bool server_thread::do_termination() {
if( is_omp_thread )
return destroy_job( *static_cast<omp_connection_v2*>(my_conn) );
else
return destroy_job( *static_cast<tbb_connection_v2*>(my_conn) );
}

void server_thread::loop() {
if( is_omp_thread )
static_cast<omp_connection_v2*>(my_conn)->make_job( *this, *my_ja );
else
static_cast<tbb_connection_v2*>(my_conn)->make_job( *this, *my_ja );
for(;;) {
__TBB_Yield();
if( state==ts_idle )
sleep_perhaps( ts_asleep );

if( terminate )
if( do_termination() )
return;

thread_state_t s = read_state();
__TBB_ASSERT( s==ts_idle||s==ts_omp_busy||s==ts_tbb_busy, NULL );

if( s==ts_omp_busy ) {
omp_dispatch.consume();

if( ++the_balance>0 )
wakeup_some_tbb_threads();
state = ts_idle;
} else if( s==ts_tbb_busy ) {
__TBB_ASSERT( my_conn && my_job, NULL );
tbb_connection_v2& conn = *static_cast<tbb_connection_v2*>(my_conn);
bool has_coin = true;
if( conn.has_slack() ) {
do {
if( conn.try_process( *this, *my_job ) )
if( conn.has_slack() && the_balance>=0 )
has_coin = !conn.wakeup_next_thread( my_map_pos );
} while( has_coin && conn.has_slack() && the_balance>=0 );
}
state = ts_idle;
if( has_coin ) {
++the_balance; 
if( conn.has_slack() ) { 
if( --the_balance>=0 ) { 
if( state.compare_and_swap( ts_tbb_busy, ts_idle )!=ts_idle )
++the_balance; 
} else {
++the_balance;
}
} 
}

if( the_balance>0 )
wakeup_some_tbb_threads();
}
}
}
#endif 

#if RML_USE_WCRM

class tbb_connection_v2;
class omp_connection_v2;

#define CREATE_SCHEDULER_POLICY(policy,min_thrs,max_thrs,stack_size) \
try {                                                                 \
policy = new SchedulerPolicy (7,                                  \
SchedulerKind, RML_THREAD_KIND,  \
MinConcurrency, min_thrs,                       \
MaxConcurrency, max_thrs,                       \
TargetOversubscriptionFactor, 1,                \
ContextStackSize, stack_size/1000,  \
ContextPriority, THREAD_PRIORITY_NORMAL,        \
DynamicProgressFeedback, ProgressFeedbackDisabled ); \
} catch ( invalid_scheduler_policy_key & ) {                               \
__TBB_ASSERT( false, "invalid scheduler policy key exception caught" );\
} catch ( invalid_scheduler_policy_value & ) {                        \
__TBB_ASSERT( false, "invalid scheduler policy value exception caught" );\
}

static unsigned int core_count;
static tbb::atomic<int> core_count_inited;


static unsigned int get_processor_count()
{
if( core_count_inited!=2 ) {
if( core_count_inited.compare_and_swap( 1, 0 )==0 ) {
core_count = GetProcessorCount();
core_count_inited = 2;
} else {
tbb::internal::spin_wait_until_eq( core_count_inited, 2 );
}
}
return core_count;
}

template<typename Connection>
scheduler<Connection>::scheduler( Connection& conn ) : uid(GetSchedulerId()), my_conn(conn) {}

template<>
scheduler<tbb_connection_v2>::scheduler( tbb_connection_v2& conn ) : uid(GetSchedulerId()), my_conn(conn)
{
rml::client& cl = my_conn.client();
unsigned max_job_count = cl.max_job_count();
unsigned count = get_processor_count();
__TBB_ASSERT( max_job_count>0, "max job count must be positive" );
__TBB_ASSERT( count>1, "The processor count must be greater than 1" );
if( max_job_count>count-1) max_job_count = count-1;
CREATE_SCHEDULER_POLICY( my_policy, 0, max_job_count, cl.min_stack_size() );
}

#if __RML_REMOVE_VIRTUAL_PROCESSORS_DISABLED
template<>
void scheduler<tbb_connection_v2>::RemoveVirtualProcessors( IVirtualProcessorRoot**, unsigned int)
{
}
#else
template<>
void scheduler<tbb_connection_v2>::RemoveVirtualProcessors( IVirtualProcessorRoot** vproots, unsigned int count )
{
if( !my_conn.is_closing() )
my_conn.remove_virtual_processors( vproots, count );
}
#endif

template<>
void scheduler<tbb_connection_v2>::NotifyResourcesExternallyIdle( IVirtualProcessorRoot** , unsigned int )
{
__TBB_ASSERT( false, "NotifyResourcesExternallyIdle() is not allowed for TBB" );
}

template<>
void scheduler<tbb_connection_v2>::NotifyResourcesExternallyBusy( IVirtualProcessorRoot** , unsigned int  )
{
__TBB_ASSERT( false, "NotifyResourcesExternallyBusy() is not allowed for TBB" );
}

template<>
scheduler<omp_connection_v2>::scheduler( omp_connection_v2& conn ) : uid(GetSchedulerId()), my_conn(conn)
{
unsigned count = get_processor_count();
rml::client& cl = my_conn.client();
__TBB_ASSERT( count>1, "The processor count must be greater than 1" );
CREATE_SCHEDULER_POLICY( my_policy, count-1, count-1, cl.min_stack_size() );
}

template<>
void scheduler<omp_connection_v2>::RemoveVirtualProcessors( IVirtualProcessorRoot** , unsigned int  ) {
__TBB_ASSERT( false, "RemoveVirtualProcessors() is not allowed for OMP" );
}

template<>
void scheduler<omp_connection_v2>::NotifyResourcesExternallyIdle( IVirtualProcessorRoot** vproots, unsigned int count ){
if( !my_conn.is_closing() )
my_conn.notify_resources_externally_idle( vproots, count );
}

template<>
void scheduler<omp_connection_v2>::NotifyResourcesExternallyBusy( IVirtualProcessorRoot** vproots, unsigned int count ){
if( !my_conn.is_closing() )
my_conn.notify_resources_externally_busy( vproots, count );
}


void tbb_server_thread::Dispatch( DispatchState* ) {
tbb_connection_v2* tbb_conn = static_cast<tbb_connection_v2*>(my_conn);
make_job( *tbb_conn, *this );

for( ;; ) {
if( the_balance>0 )
wakeup_some_tbb_threads();
if( read_state()!=ts_busy )
if( sleep_perhaps() )
return;
if( terminate )
if( initiate_termination() )
return;
if( read_state()==ts_busy ) {
if ( tbb_conn->has_slack() ) {
do {
tbb_conn->try_process( *wait_for_job() );
} while( tbb_conn->has_slack() && the_balance>=0 && !is_removed() );
}
__TBB_ASSERT( read_state()==ts_busy, "thread is not in busy state after returning from process()" );
if( my_state.compare_and_swap( ts_idle, ts_busy )==ts_busy ) {
int bal = ++the_balance;
if( tbb_conn->has_slack() ) {
bal = --the_balance;  
if( bal>=0 ) { 
if( my_state.compare_and_swap( ts_busy, ts_idle )!=ts_idle )
++the_balance; 
} else {
++the_balance;
}
} 
} else {
__TBB_ASSERT( false, "someone tampered with my state" );
}
} 
}
}

void omp_server_thread::Dispatch( DispatchState* ) {
make_job( *static_cast<omp_connection_v2*>(my_conn), *this );

for( ;; ) {
if( read_state()!=ts_busy )
sleep_perhaps();
if( terminate ) {
if( initiate_termination() )
return;
}
if( read_state()==ts_busy ) {
omp_data.consume();
__TBB_ASSERT( read_state()==ts_busy, "thread is not in busy state after returning from process()" );
my_thread_map.adjust_balance( 1 );
set_state( ts_idle );
}
}
}

thread_grab_t server_thread_rep::try_grab_for() {
thread_grab_t res = wk_failed;
thread_state_t s = read_state();
switch( s ) {
case ts_asleep:
if( wakeup( ts_busy, ts_asleep ) )
res = wk_from_asleep;
__TBB_ASSERT( res==wk_failed||read_state()==ts_busy, NULL );
break;
case ts_idle:
if( my_state.compare_and_swap( ts_busy, ts_idle )==ts_idle )
res = wk_from_idle;
break;
default:
break;
}
return res;
}

bool tbb_server_thread::switch_out() {
thread_state_t s = read_state();
__TBB_ASSERT( s==ts_asleep||s==ts_busy, NULL );
IExecutionResource* old_vp = my_execution_resource;
if( s==ts_busy ) {
++the_balance;
my_state = ts_asleep;
}
IThreadProxy* proxy = my_proxy;
__TBB_ASSERT( proxy, NULL );
my_execution_resource = (IExecutionResource*) c_remove_prepare;
old_vp->Remove( my_scheduler );
my_execution_resource = (IExecutionResource*) c_remove_returned;
int cnt = --activation_count;
__TBB_ASSERT_EX( cnt==0||cnt==1, "too many activations?" );
proxy->SwitchOut();
if( terminate ) {
bool activated = activation_count==1;
#if TBB_USE_ASSERT

if( activated ) {
while( is_removed() )
__TBB_Yield();
thread_state_t state = read_state();
__TBB_ASSERT( state==ts_idle || state==ts_done, NULL );
}
#endif
__TBB_ASSERT( my_state==ts_asleep||my_state==ts_idle, NULL );
rml::job* j;
if( my_job_automaton.try_plug(j) ) {
__TBB_ASSERT( j, NULL );
my_client.cleanup(*j);
my_conn->remove_client_ref();
}
if( !activated )
proxy->SwitchTo( my_thread_map.get_thread_scavenger(), Idle );
my_conn->remove_server_ref();
return true;
}
while( get_virtual_processor()==c_remove_returned )
__TBB_Yield();
return false;
}

bool tbb_server_thread::sleep_perhaps () {
if( terminate ) return false;
thread_state_t s = read_state();
if( s==ts_idle ) {
if( my_state.compare_and_swap( ts_asleep, ts_idle )==ts_idle ) {
if( !terminate ) {
if( !is_removed() ) {
--activation_count;
get_virtual_processor()->Deactivate( this );
}
if( is_removed() ) {
if( switch_out() )
return true;
__TBB_ASSERT( my_execution_resource>c_remove_returned, NULL );
}
while( read_state()==ts_asleep )
__TBB_Yield();
} else {
if( my_state.compare_and_swap( ts_done, ts_asleep )!=ts_asleep ) {
--activation_count;
get_virtual_processor()->Deactivate( this );
}
}
}
} else {
__TBB_ASSERT( s==ts_busy, NULL );
}
return false;
}

void omp_server_thread::sleep_perhaps () {
if( terminate ) return;
thread_state_t s = read_state();
if( s==ts_idle ) {
if( my_state.compare_and_swap( ts_asleep, ts_idle )==ts_idle ) {
if( !terminate ) {
get_virtual_processor()->Deactivate( this );
__TBB_ASSERT( !is_removed(), "OMP threads should not be deprived of a virtual processor" );
__TBB_ASSERT( read_state()!=ts_asleep, NULL );
} else {
if( my_state.compare_and_swap( ts_done, ts_asleep )!=ts_asleep )
get_virtual_processor()->Deactivate( this );
}
}
} else {
__TBB_ASSERT( s==ts_busy, NULL );
}
}

bool tbb_server_thread::initiate_termination() {
if( read_state()==ts_busy ) {
int bal = ++the_balance;
if( bal>0 ) wakeup_some_tbb_threads();
}
return destroy_job( (tbb_connection_v2*) my_conn );
}

template<typename Connection>
bool server_thread_rep::destroy_job( Connection* c ) {
__TBB_ASSERT( my_state!=ts_asleep, NULL );
rml::job* j;
if( my_job_automaton.try_plug(j) ) {
__TBB_ASSERT( j, NULL );
my_client.cleanup(*j);
c->remove_client_ref();
}
c->remove_server_ref();
return true;
}

void thread_map::assist_cleanup( bool assist_null_only ) {
for( iterator i = begin(); i!=end(); ++i ) {
rml::job* j=0;
server_thread* thr = (*i).second;
job_automaton& ja = thr->my_job_automaton;
if( assist_null_only ? ja.try_plug_null() : ja.try_plug(j) ) {
if( j ) {
my_client.cleanup(*j);
} else {
}
remove_client_ref();
}
}
}

void thread_map::add_virtual_processors( IVirtualProcessorRoot** vproots, unsigned int count, tbb_connection_v2& conn, ::tbb::spin_mutex& mtx )
{
#if TBB_USE_ASSERT
int req_cnt = ++n_add_vp_requests;
__TBB_ASSERT( req_cnt==1, NULL );
#endif
std::vector<thread_map::iterator> vec(count);
std::vector<tbb_server_thread*> tvec(count);
iterator end;

{
tbb::spin_mutex::scoped_lock lck( mtx );
__TBB_ASSERT( my_map.size()==0||count==1, NULL );
end = my_map.end(); 
for( size_t i=0; i<count; ++i ) {
vec[i] = my_map.find( (key_type) vproots[i] );
#if TBB_USE_DEBUG
if( vec[i]!=end ) {
tbb_server_thread* t = (tbb_server_thread*) (*vec[i]).second;
IVirtualProcessorRoot* v = t->get_virtual_processor();
__TBB_ASSERT( v==c_remove_prepare||v==c_remove_returned, NULL );
}
#endif
}

iterator nxt = my_map.begin();
for( size_t i=0; i<count; ++i ) {
if( vec[i]!=end ) {
#if TBB_USE_ASSERT
tbb_server_thread* t = (tbb_server_thread*) (*vec[i]).second;
__TBB_ASSERT( t->read_state()==ts_asleep, NULL );
IVirtualProcessorRoot* r = t->get_virtual_processor();
__TBB_ASSERT( r==c_remove_prepare||r==c_remove_returned, NULL );
#endif
continue;
}

if( my_unrealized_threads>0 ) {
--my_unrealized_threads;
} else {
__TBB_ASSERT( nxt!=end, "nxt should not be thread_map::iterator::end" );
for( ; nxt!=end; ++nxt ) {
tbb_server_thread* t = (tbb_server_thread*) (*nxt).second;
if( t->is_removed() && t->read_state()==ts_asleep && t->get_virtual_processor()==c_remove_returned ) {
vec[i] = nxt++;
break;
}
}
if( vec[i]==end ) 
vproots[i] = NULL;
}
}
}

for( size_t i=0; i<count; ++i ) {
__TBB_ASSERT( !tvec[i], NULL );
if( vec[i]==end ) {
if( vproots[i] ) {
tvec[i] = my_tbb_allocator.allocate(1);
new ( tvec[i] ) tbb_server_thread( false, my_scheduler, (IExecutionResource*)vproots[i], &conn, *this, my_client );
}
#if TBB_USE_ASSERT
} else {
tbb_server_thread* t = (tbb_server_thread*) (*vec[i]).second;
__TBB_ASSERT( t->GetProxy(), "Proxy is cleared?" );
#endif
}
}

{
tbb::spin_mutex::scoped_lock lck( mtx );

bool closing = is_closing();

for( size_t i=0; i<count; ++i ) {
if( vec[i]==end ) {
if( vproots[i] ) {
thread_map::key_type key = (thread_map::key_type) vproots[i];
vec[i] = insert( key, (server_thread*) tvec[i] );
my_client_ref_count.add_ref();
my_server_ref_count.add_ref();
}
} else if( !closing ) {
tbb_server_thread* t = (tbb_server_thread*) (*vec[i]).second;

if( (*vec[i]).first!=(thread_map::key_type)vproots[i] ) {
my_map.erase( vec[i] );
thread_map::key_type key = (thread_map::key_type) vproots[i];
__TBB_ASSERT( key, NULL );
vec[i] = insert( key, t );
}
__TBB_ASSERT( t->read_state()==ts_asleep, NULL );
}
}

__TBB_ASSERT( closing==is_closing(), NULL );
if( closing ) return;
}

for( size_t i=0; i<count; ++i ) {
if( vproots[i] ) {
tbb_server_thread* t = (tbb_server_thread*) (*vec[i]).second;
__TBB_ASSERT( tvec[i]!=NULL||t->GetProxy(), "Proxy is cleared?" );
if( t->is_removed() )
__TBB_ASSERT( t->get_virtual_processor()==c_remove_returned, NULL );
int cnt = ++t->activation_count;
__TBB_ASSERT_EX( cnt==0||cnt==1, NULL );
vproots[i]->Activate( t );
if( t->is_removed() )
t->revive( my_scheduler, vproots[i], my_client );
}
}
#if TBB_USE_ASSERT
req_cnt = --n_add_vp_requests;
__TBB_ASSERT( req_cnt==0, NULL );
#endif
}

void thread_map::remove_virtual_processors( IVirtualProcessorRoot** vproots, unsigned count, ::tbb::spin_mutex& mtx ) {
if( my_map.size()==0 )
return;
tbb::spin_mutex::scoped_lock lck( mtx );

if( is_closing() ) return;

for( unsigned int c=0; c<count; ++c ) {
iterator i = my_map.find( (key_type) vproots[c] );
if( i==my_map.end() ) {
thread_scavenger_thread* tst = my_thread_scavenger_thread;
if( !tst ) {
vproots[c]->Remove( my_scheduler );
} else {
while( (tst=my_thread_scavenger_thread)==c_claimed )
__TBB_Yield();
if( vproots[c]!=tst->get_virtual_processor() )
vproots[c]->Remove( my_scheduler );
}
continue;
}
tbb_server_thread* thr = (tbb_server_thread*) (*i).second;
__TBB_ASSERT( thr->tbb_thread, "incorrect type of server_thread" );
thr->set_removed();
if( thr->read_state()==ts_asleep ) {
while( thr->activation_count>0 ) {
if( thr->get_virtual_processor()<=c_remove_returned )
break;
__TBB_Yield();
}
if( thr->get_virtual_processor()>c_remove_returned ) {
++thr->activation_count;
thr->get_virtual_processor()->Activate( thr );
} 
} 
}
}

void thread_map::add_virtual_processors( IVirtualProcessorRoot** vproots, unsigned int count, omp_connection_v2& conn, ::tbb::spin_mutex& mtx )
{
std::vector<thread_map::iterator> vec(count);
std::vector<server_thread*> tvec(count);
iterator end;

{
tbb::spin_mutex::scoped_lock lck( mtx );
end = my_map.end(); 
for( size_t i=0; i<count; ++i )
vec[i] = my_map.find( (key_type) vproots[i] );
}

for( size_t i=0; i<count; ++i ) {
__TBB_ASSERT( !tvec[i], NULL );
if( vec[i]==end ) {
tvec[i] = my_omp_allocator.allocate(1);
new ( tvec[i] ) omp_server_thread( false, my_scheduler, (IExecutionResource*)vproots[i], &conn, *this, my_client );
}
}

{
tbb::spin_mutex::scoped_lock lck( mtx );

for( size_t i=0; i<count; ++i ) {
if( vec[i]==my_map.end() ) {
thread_map::key_type key = (thread_map::key_type) vproots[i];
vec[i] = insert( key, tvec[i] );
my_client_ref_count.add_ref();
my_server_ref_count.add_ref();
}
}

if( is_closing() ) return;
}

for( size_t i=0; i<count; ++i )
vproots[i]->Activate( (*vec[i]).second );

{
tbb::spin_mutex::scoped_lock lck( mtx );
for( size_t i=0; i<count; ++i )
original_exec_resources.push_back( vproots[i] );
}
}

void thread_map::mark_virtual_processors_as_lent( IVirtualProcessorRoot** vproots, unsigned count, ::tbb::spin_mutex& mtx ) {
tbb::spin_mutex::scoped_lock lck( mtx );

if( is_closing() ) return;

iterator end = my_map.end();
for( unsigned int c=0; c<count; ++c ) {
iterator i = my_map.find( (key_type) vproots[c] );
if( i==end ) {
my_map.insert( unordered_map_type::value_type( (key_type) vproots[c], (server_thread*)1 ) );
} else {
server_thread* thr = (*i).second;
if( ((uintptr_t)thr)&~(uintptr_t)1 ) {
__TBB_ASSERT( !thr->is_removed(), "incorrectly removed" );
((omp_server_thread*)thr)->set_lent();
}
}
}
}

void thread_map::create_oversubscribers( unsigned n, std::vector<server_thread*>& thr_vec, omp_connection_v2& conn, ::tbb::spin_mutex& mtx ) {
std::vector<IExecutionResource*> curr_exec_rsc;
{
tbb::spin_mutex::scoped_lock lck( mtx );
curr_exec_rsc = original_exec_resources; 
}
typedef std::vector<IExecutionResource*>::iterator iterator_er;
typedef ::std::vector<std::pair<unordered_map_type::key_type, unordered_map_type::mapped_type> > map_val_vector_t;
map_val_vector_t v_vec(n);
iterator_er begin = curr_exec_rsc.begin();
iterator_er end   = curr_exec_rsc.end();
iterator_er i = begin;
for( unsigned c=0; c<n; ++c ) {
IVirtualProcessorRoot* vpr = my_scheduler_proxy->CreateOversubscriber( *i );
omp_server_thread* t = new ( my_omp_allocator.allocate(1) ) omp_server_thread( true, my_scheduler, (IExecutionResource*)vpr, &conn, *this, my_client );
thr_vec[c] = t;
v_vec[c] = unordered_map_type::value_type( (key_type) vpr, t );
if( ++i==end ) i = begin;
}

{
tbb::spin_mutex::scoped_lock lck( mtx );

if( is_closing() ) return;

iterator map_end = my_map.end();
unsigned c = 0;
for( map_val_vector_t::iterator vi=v_vec.begin(); vi!=v_vec.end(); ++vi, ++c ) {
iterator j = my_map.find( (key_type) (*vi).first );
if( j==map_end ) {
my_map.insert( *vi );
} else {
uintptr_t lent = (uintptr_t) (*j).second;
__TBB_ASSERT( lent<=1, "vproc map entry added incorrectly?");
(*j).second = thr_vec[c];
if( lent )
((omp_server_thread*)thr_vec[c])->set_lent();
else
((omp_server_thread*)thr_vec[c])->set_returned();
}
my_client_ref_count.add_ref();
my_server_ref_count.add_ref();
}
}
}

void thread_map::wakeup_tbb_threads( int c, ::tbb::spin_mutex& mtx ) {
std::vector<tbb_server_thread*> vec(c);

size_t idx = 0;
{
tbb::spin_mutex::scoped_lock lck( mtx );

if( is_closing() ) return;

int bal = the_balance;
int cnt = c<bal ? c : bal;

if( cnt<=0 ) { return; }

for( iterator i=begin(); i!=end(); ++i ) {
tbb_server_thread* thr = (tbb_server_thread*) (*i).second;
if( thr->is_removed() )
continue;

if( --the_balance>=0 ) {
thread_grab_t res;
while( (res=thr->try_grab_for())!=wk_from_idle ) {
if( res==wk_from_asleep ) {
vec[idx++] = thr;
break;
} else {
thread_state_t s = thr->read_state();
if( s==ts_busy ) {
++the_balance;
goto skip;
}
}
}
thread_state_t s = thr->read_state();
__TBB_ASSERT_EX( s==ts_busy, "should have set the state to ts_busy" );
if( --cnt==0 )
break;
} else {
++the_balance;
break;
}
skip:
;
}
}

for( size_t i=0; i<idx; ++i ) {
tbb_server_thread* thr = vec[i];
__TBB_ASSERT( thr, NULL );
thread_state_t s = thr->read_state();
__TBB_ASSERT_EX( s==ts_busy, "should have set the state to ts_busy" );
++thr->activation_count;
thr->get_virtual_processor()->Activate( thr );
}

}

void thread_map::mark_virtual_processors_as_returned( IVirtualProcessorRoot** vprocs, unsigned int count, tbb::spin_mutex& mtx ) {
{
tbb::spin_mutex::scoped_lock lck( mtx );

if( is_closing() ) return;

iterator end = my_map.end();
for(unsigned c=0; c<count; ++c ) {
iterator i = my_map.find( (key_type) vprocs[c] );
if( i==end ) {
my_map.insert( unordered_map_type::value_type( (key_type) vprocs[c], static_cast<server_thread*>(0) ) );
} else {
omp_server_thread* thr = (omp_server_thread*) (*i).second;
if( ((uintptr_t)thr)&~(uintptr_t)1 ) {
__TBB_ASSERT( !thr->is_removed(), "incorrectly removed" );
thr->set_returned();
}
}
}
}
}


void thread_map::unbind( rml::server& , tbb::spin_mutex& mtx ) {
{
tbb::spin_mutex::scoped_lock lck( mtx );
shutdown_in_progress = true;  

for( iterator i = begin(); i!=end(); ++i ) {
server_thread* t = (*i).second;
t->terminate = true;
if( t->is_removed() ) {
if( t->read_state()==ts_asleep ) {
__TBB_ASSERT( my_thread_scavenger_thread, "this is TBB connection; thread_scavenger_thread must be allocated" );
#if TBB_USE_ASSERT
while( t->get_virtual_processor()>c_remove_returned )
__TBB_Yield();
#endif
}
} else {
if( t->wakeup( ts_done, ts_asleep ) ) {
if( t->tbb_thread )
++((tbb_server_thread*)t)->activation_count;
t->get_virtual_processor()->Activate( t );
}
}
}
}
remove_client_ref();

if( my_thread_scavenger_thread ) {
thread_scavenger_thread* tst;
while( (tst=my_thread_scavenger_thread)==c_claimed )
__TBB_Yield();
#if TBB_USE_ASSERT
++my_thread_scavenger_thread->activation_count;
#endif
tst->get_virtual_processor()->Activate( tst );
}
}

#if !__RML_REMOVE_VIRTUAL_PROCESSORS_DISABLED
void thread_map::allocate_thread_scavenger( IExecutionResource* v )
{
if( my_thread_scavenger_thread>c_claimed ) return;
thread_scavenger_thread* c = my_thread_scavenger_thread.fetch_and_store((thread_scavenger_thread*)c_claimed);
if( c==NULL ) { 
add_server_ref();
#if TBB_USE_ASSERT
++n_thread_scavengers_created;
#endif
__TBB_ASSERT( v, NULL );
IVirtualProcessorRoot* vpr = my_scheduler_proxy->CreateOversubscriber( v );
my_thread_scavenger_thread = c = new ( my_scavenger_allocator.allocate(1) ) thread_scavenger_thread( my_scheduler, vpr, *this );
#if TBB_USE_ASSERT
++c->activation_count;
#endif
vpr->Activate( c );
} else if( c>c_claimed ) {
my_thread_scavenger_thread = c;
}
}
#endif

void thread_scavenger_thread::Dispatch( DispatchState* )
{
__TBB_ASSERT( my_proxy, NULL );
#if TBB_USE_ASSERT
--activation_count;
#endif
get_virtual_processor()->Deactivate( this );
for( thread_map::iterator i=my_thread_map.begin(); i!=my_thread_map.end(); ++i ) {
tbb_server_thread* t = (tbb_server_thread*) (*i).second;
if( t->read_state()==ts_asleep && t->is_removed() ) {
while( t->get_execution_resource()!=c_remove_returned )
__TBB_Yield();
my_proxy->SwitchTo( t, Blocking );
}
}
get_virtual_processor()->Remove( my_scheduler );
my_thread_map.remove_server_ref();
__TBB_ASSERT( activation_count==1, NULL );
set_state( ts_done );
}

extern "C" bool WINAPI DllMain( HINSTANCE , DWORD fwdReason, LPVOID lpvReserved ) {
void assist_cleanup_connections();
if( fwdReason==DLL_PROCESS_DETACH ) {
if( !lpvReserved ) 
assist_cleanup_connections();
}
return true;
}

void free_all_connections( uintptr_t conn_ex ) {
while( conn_ex ) {
bool is_tbb = (conn_ex&2)>0;
uintptr_t curr_conn = conn_ex & ~(uintptr_t)3;
__TBB_ASSERT( curr_conn, NULL );

if( is_tbb ) {
tbb_connection_v2* tbb_conn = reinterpret_cast<tbb_connection_v2*>(curr_conn);
conn_ex = reinterpret_cast<uintptr_t>(tbb_conn->next_conn);
while( tbb_conn->my_thread_map.remove_server_ref()>0 )
__TBB_Yield();
delete tbb_conn;
} else {
omp_connection_v2* omp_conn = reinterpret_cast<omp_connection_v2*>(curr_conn);
conn_ex = reinterpret_cast<uintptr_t>(omp_conn->next_conn);
while( omp_conn->my_thread_map.remove_server_ref()>0 )
__TBB_Yield();
delete omp_conn;
}
}
}

void assist_cleanup_connections()
{
uintptr_t tail = connections_to_reclaim.tail;
while( connections_to_reclaim.tail.compare_and_swap( garbage_connection_queue::plugged, tail )!=tail ) {
__TBB_Yield();
tail = connections_to_reclaim.tail;
}

__TBB_ASSERT( connection_scavenger.state==ts_busy || connection_scavenger.state==ts_asleep, NULL );
DWORD thr_exit_code = STILL_ACTIVE;
while( connection_scavenger.state==ts_busy ) {
if( GetExitCodeThread( connection_scavenger.thr_handle, &thr_exit_code )>0 )
if( thr_exit_code!=STILL_ACTIVE )
break;
__TBB_Yield();
thr_exit_code = STILL_ACTIVE;
}
if( connection_scavenger.state==ts_asleep && thr_exit_code==STILL_ACTIVE )
connection_scavenger.wakeup(); 

if( tail && connections_to_reclaim.tail!=garbage_connection_queue::plugged_acked ) {
uintptr_t head = connections_to_reclaim.head.fetch_and_store( garbage_connection_queue::empty );
if( head==garbage_connection_queue::empty )
head = tail;
connection_scavenger.process_requests( head );
}
__TBB_ASSERT( connections_to_reclaim.tail==garbage_connection_queue::plugged||connections_to_reclaim.tail==garbage_connection_queue::plugged_acked, "someone else added a request after termination has initiated" );
__TBB_ASSERT( (unsigned)the_balance==the_default_concurrency, NULL );
}

void connection_scavenger_thread::sleep_perhaps() {
uintptr_t tail = connections_to_reclaim.tail;
if( tail>=garbage_connection_queue::plugged ) return;
__TBB_ASSERT( !tail, NULL );
thread_monitor::cookie c;
monitor.prepare_wait(c);
if( state.compare_and_swap( ts_asleep, ts_busy )==ts_busy ) {
if( connections_to_reclaim.tail!=garbage_connection_queue::plugged ) {
monitor.commit_wait(c);
} else {
monitor.cancel_wait();
}
thread_state_t s = state;
if( s==ts_asleep ) 
state.compare_and_swap( ts_busy, ts_asleep );
} else {
__TBB_ASSERT( false, "someone else tampered with my state" );
}
__TBB_ASSERT( state==ts_busy, "a thread can only put itself to sleep" );
}

void connection_scavenger_thread::process_requests( uintptr_t conn_ex )
{
__TBB_ASSERT( conn_ex>1, NULL );
__TBB_ASSERT( n_scavenger_threads==1||connections_to_reclaim.tail==garbage_connection_queue::plugged, "more than one connection_scavenger_thread being active?" );

bool done = false;
while( !done ) {
bool is_tbb = (conn_ex&2)>0;
uintptr_t curr_conn = conn_ex & ~(uintptr_t)3;

uintptr_t next_conn;
tbb_connection_v2* tbb_conn = NULL;
omp_connection_v2* omp_conn = NULL;
if( is_tbb ) {
tbb_conn = reinterpret_cast<tbb_connection_v2*>(curr_conn);
next_conn = reinterpret_cast<uintptr_t>(tbb_conn->next_conn);
while( tbb_conn->my_thread_map.get_server_ref_count()>1 )
__TBB_Yield();
} else {
omp_conn = reinterpret_cast<omp_connection_v2*>(curr_conn);
next_conn = reinterpret_cast<uintptr_t>(omp_conn->next_conn);
while( omp_conn->my_thread_map.get_server_ref_count()>1 )
__TBB_Yield();
}


if( next_conn==0 ) {
uintptr_t tail = connections_to_reclaim.tail;
if( tail==garbage_connection_queue::plugged ) {
tail = garbage_connection_queue::plugged_acked; 
done = true;
} else if( tail==conn_ex ) {
if( connections_to_reclaim.tail.compare_and_swap( garbage_connection_queue::empty, tail )==tail ) {
__TBB_ASSERT( !connections_to_reclaim.head, NULL );
done = true;
}
}

if( !done ) {
if( is_tbb ) {
while( !tbb_conn->next_conn )
__TBB_Yield();
conn_ex = reinterpret_cast<uintptr_t>(tbb_conn->next_conn);
} else {
while( !omp_conn->next_conn )
__TBB_Yield();
conn_ex = reinterpret_cast<uintptr_t>(omp_conn->next_conn);
}
}
} else {
conn_ex = next_conn;
}
__TBB_ASSERT( conn_ex, NULL );
if( is_tbb )
tbb_conn->remove_server_ref();
else
omp_conn->remove_server_ref();
}
}

__RML_DECL_THREAD_ROUTINE connection_scavenger_thread::thread_routine( void* arg ) {
connection_scavenger_thread* thr = (connection_scavenger_thread*) arg;
thr->state = ts_busy;
thr->thr_handle = GetCurrentThread();
#if TBB_USE_ASSERT
++thr->n_scavenger_threads;
#endif
for(;;) {
__TBB_Yield();
thr->sleep_perhaps();
if( connections_to_reclaim.tail==garbage_connection_queue::plugged || connections_to_reclaim.tail==garbage_connection_queue::plugged_acked ) {
thr->state = ts_asleep;
return 0;
}

__TBB_ASSERT( connections_to_reclaim.tail!=garbage_connection_queue::plugged_acked, NULL );
__TBB_ASSERT( connections_to_reclaim.tail>garbage_connection_queue::plugged && (connections_to_reclaim.tail&garbage_connection_queue::plugged)==0 , NULL );
while( connections_to_reclaim.head==garbage_connection_queue::empty )
__TBB_Yield();
uintptr_t head = connections_to_reclaim.head.fetch_and_store( garbage_connection_queue::empty );
thr->process_requests( head );
wakeup_some_tbb_threads();
}
}

template<typename Server, typename Client>
void connection_scavenger_thread::add_request( generic_connection<Server,Client>* conn_to_close )
{
uintptr_t conn_ex = (uintptr_t)conn_to_close | (connection_traits<Server,Client>::is_tbb<<1);
__TBB_ASSERT( !conn_to_close->next_conn, NULL );
const uintptr_t old_tail_ex = connections_to_reclaim.tail.fetch_and_store(conn_ex);
__TBB_ASSERT( old_tail_ex==0||old_tail_ex>garbage_connection_queue::plugged_acked, "Unloading DLL called while this connection is being closed?" );

if( old_tail_ex==garbage_connection_queue::empty )
connections_to_reclaim.head = conn_ex;
else {
bool is_tbb = (old_tail_ex&2)>0;
uintptr_t old_tail = old_tail_ex & ~(uintptr_t)3;
if( is_tbb )
reinterpret_cast<tbb_connection_v2*>(old_tail)->next_conn = reinterpret_cast<tbb_connection_v2*>(conn_ex);
else
reinterpret_cast<omp_connection_v2*>(old_tail)->next_conn = reinterpret_cast<omp_connection_v2*>(conn_ex);
}

if( state==ts_asleep )
wakeup();
}

template<>
uintptr_t connection_scavenger_thread::grab_and_prepend( generic_connection<tbb_server,tbb_client>*  ) { return 0;}

template<>
uintptr_t connection_scavenger_thread::grab_and_prepend( generic_connection<omp_server,omp_client>* last_conn_to_close )
{
uintptr_t conn_ex = (uintptr_t)last_conn_to_close;
uintptr_t head = connections_to_reclaim.head.fetch_and_store( garbage_connection_queue::empty );
reinterpret_cast<omp_connection_v2*>(last_conn_to_close)->next_conn = reinterpret_cast<omp_connection_v2*>(head);
return conn_ex;
}

extern "C" ULONGLONG NTAPI VerSetConditionMask( ULONGLONG, DWORD, BYTE);

bool is_windows7_or_later ()
{
try {
return GetOSVersion()>=IResourceManager::Win7OrLater;
} catch( ... ) {
return false;
}
}

#endif 

template<typename Connection, typename Server, typename Client>
static factory::status_type connect( factory& f, Server*& server, Client& client ) {
server = new Connection(*static_cast<wait_counter*>(f.scratch_ptr),client);
return factory::st_success;
}

void init_rml_module () {
the_balance = the_default_concurrency = tbb::internal::AvailableHwConcurrency() - 1;
#if RML_USE_WCRM
connection_scavenger.launch();
#endif
}

extern "C" factory::status_type __RML_open_factory( factory& f, version_type& server_version, version_type client_version ) {
static tbb::atomic<bool> one_time_flag;
if( one_time_flag.compare_and_swap(true,false)==false) {
__TBB_ASSERT( (size_t)f.library_handle!=factory::c_dont_unload, NULL );
#if _WIN32||_WIN64
f.library_handle = reinterpret_cast<HMODULE>(factory::c_dont_unload);
#else
f.library_handle = reinterpret_cast<void*>(factory::c_dont_unload);
#endif
}

tbb::internal::atomic_do_once ( &init_rml_module, rml_module_state );

server_version = SERVER_VERSION;
f.scratch_ptr = 0;
if( client_version==0 ) {
return factory::st_incompatible;
#if RML_USE_WCRM
} else if ( !is_windows7_or_later() ) {
#if TBB_USE_DEBUG
fprintf(stderr, "This version of the RML library requires Windows 7 to run on.\nConnection request denied.\n");
#endif
return factory::st_incompatible;
#endif
} else {
#if TBB_USE_DEBUG
if( client_version<EARLIEST_COMPATIBLE_CLIENT_VERSION )
fprintf(stderr, "This client library is too old for the current RML server.\nThe connection request is granted but oversubscription/undersubscription may occur.\n");
#endif
f.scratch_ptr = new wait_counter;
return factory::st_success;
}
}

extern "C" void __RML_close_factory( factory& f ) {
if( wait_counter* fc = static_cast<wait_counter*>(f.scratch_ptr) ) {
f.scratch_ptr = 0;
fc->wait();
size_t bal = the_balance;
f.scratch_ptr = (void*)bal;
delete fc;
}
}

void call_with_build_date_str( ::rml::server_info_callback_t cb, void* arg );

}} 

namespace tbb {
namespace internal {
namespace rml {

extern "C" tbb_factory::status_type __TBB_make_rml_server( tbb_factory& f, tbb_server*& server, tbb_client& client ) {
return ::rml::internal::connect< ::rml::internal::tbb_connection_v2>(f,server,client);
}

extern "C" void __TBB_call_with_my_server_info( ::rml::server_info_callback_t cb, void* arg ) {
return ::rml::internal::call_with_build_date_str( cb, arg );
}

}}}

namespace __kmp {
namespace rml {

extern "C" omp_factory::status_type __KMP_make_rml_server( omp_factory& f, omp_server*& server, omp_client& client ) {
return ::rml::internal::connect< ::rml::internal::omp_connection_v2>(f,server,client);
}

extern "C" void __KMP_call_with_my_server_info( ::rml::server_info_callback_t cb, void* arg ) {
return ::rml::internal::call_with_build_date_str( cb, arg );
}

}}


#include "version_string.ver"

#ifndef __TBB_VERSION_STRINGS
#pragma message("Warning: version_string.ver isn't generated properly by version_info.sh script!")
#endif

#ifndef __TBB_DATETIME
#define __TBB_DATETIME __DATE__ " " __TIME__
#endif

#if !RML_USE_WCRM
#define RML_SERVER_BUILD_TIME "Intel(R) RML library built: " __TBB_DATETIME
#define RML_SERVER_VERSION_ST "Intel(R) RML library version: v" TOSTRING(SERVER_VERSION)
#else
#define RML_SERVER_BUILD_TIME "Intel(R) RML library built: " __TBB_DATETIME
#define RML_SERVER_VERSION_ST "Intel(R) RML library version: v" TOSTRING(SERVER_VERSION) " on ConcRT RM with " RML_THREAD_KIND_STRING
#endif

namespace rml {
namespace internal {

void call_with_build_date_str( ::rml::server_info_callback_t cb, void* arg )
{
(*cb)( arg, RML_SERVER_BUILD_TIME );
(*cb)( arg, RML_SERVER_VERSION_ST );
}
}} 
