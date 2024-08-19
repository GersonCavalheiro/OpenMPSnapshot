

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_task_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_task_H
#pragma message("TBB Warning: tbb/task.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_task_H
#define __TBB_task_H

#define __TBB_task_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"
#include "tbb_machine.h"
#include "tbb_profiling.h"
#include <climits>

typedef struct ___itt_caller *__itt_caller;

namespace tbb {

class task;
class task_list;
class task_group_context;

#if _MSC_VER || (__GNUC__==3 && __GNUC_MINOR__<3)
#define __TBB_TASK_BASE_ACCESS public
#else
#define __TBB_TASK_BASE_ACCESS private
#endif

namespace internal { 

class allocate_additional_child_of_proxy: no_assign {
task* self;
task& parent;
public:
explicit allocate_additional_child_of_proxy( task& parent_ ) : self(NULL), parent(parent_) {
suppress_unused_warning( self );
}
task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
void __TBB_EXPORTED_METHOD free( task& ) const;
};

struct cpu_ctl_env_space { int space[sizeof(internal::uint64_t)/sizeof(int)]; };
} 

namespace interface5 {
namespace internal {

class task_base: tbb::internal::no_copy {
__TBB_TASK_BASE_ACCESS:
friend class tbb::task;

static void spawn( task& t );

static void spawn( task_list& list );


static tbb::internal::allocate_additional_child_of_proxy allocate_additional_child_of( task& t ) {
return tbb::internal::allocate_additional_child_of_proxy(t);
}


static void __TBB_EXPORTED_FUNC destroy( task& victim );
};
} 
} 

namespace internal {

class scheduler: no_copy {
public:
virtual void spawn( task& first, task*& next ) = 0;

virtual void wait_for_all( task& parent, task* child ) = 0;

virtual void spawn_root_and_wait( task& first, task*& next ) = 0;

virtual ~scheduler() = 0;

virtual void enqueue( task& t, void* reserved ) = 0;
};


typedef intptr_t reference_count;

#if __TBB_PREVIEW_RESUMABLE_TASKS
static const reference_count abandon_flag = reference_count(1) << (sizeof(reference_count)*CHAR_BIT - 2);
#endif

typedef unsigned short affinity_id;

#if __TBB_TASK_ISOLATION
typedef intptr_t isolation_tag;
const isolation_tag no_isolation = 0;
#endif 

#if __TBB_TASK_GROUP_CONTEXT
class generic_scheduler;

struct context_list_node_t {
context_list_node_t *my_prev,
*my_next;
};

class allocate_root_with_context_proxy: no_assign {
task_group_context& my_context;
public:
allocate_root_with_context_proxy ( task_group_context& ctx ) : my_context(ctx) {}
task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
void __TBB_EXPORTED_METHOD free( task& ) const;
};
#endif 

class allocate_root_proxy: no_assign {
public:
static task& __TBB_EXPORTED_FUNC allocate( size_t size );
static void __TBB_EXPORTED_FUNC free( task& );
};

class allocate_continuation_proxy: no_assign {
public:
task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
void __TBB_EXPORTED_METHOD free( task& ) const;
};

class allocate_child_proxy: no_assign {
public:
task& __TBB_EXPORTED_METHOD allocate( size_t size ) const;
void __TBB_EXPORTED_METHOD free( task& ) const;
};

#if __TBB_PREVIEW_CRITICAL_TASKS
void make_critical( task& t );
bool is_critical( task& t );
#endif


class task_prefix {
private:
friend class tbb::task;
friend class tbb::interface5::internal::task_base;
friend class tbb::task_list;
friend class internal::scheduler;
friend class internal::allocate_root_proxy;
friend class internal::allocate_child_proxy;
friend class internal::allocate_continuation_proxy;
friend class internal::allocate_additional_child_of_proxy;
#if __TBB_PREVIEW_CRITICAL_TASKS
friend void make_critical( task& );
friend bool is_critical( task& );
#endif

#if __TBB_TASK_ISOLATION
isolation_tag isolation;
#else
intptr_t reserved_space_for_task_isolation_tag;
#endif 

#if __TBB_TASK_GROUP_CONTEXT

task_group_context  *context;
#endif 


scheduler* origin;

#if __TBB_TASK_PRIORITY || __TBB_PREVIEW_RESUMABLE_TASKS
union {
#endif 

scheduler* owner;

#if __TBB_TASK_PRIORITY

task* next_offloaded;
#endif

#if __TBB_PREVIEW_RESUMABLE_TASKS
scheduler* abandoned_scheduler;
#endif
#if __TBB_TASK_PRIORITY || __TBB_PREVIEW_RESUMABLE_TASKS
};
#endif 


tbb::task* parent;


__TBB_atomic reference_count ref_count;


int depth;


unsigned char state;


unsigned char extra_state;

affinity_id affinity;

tbb::task* next;

tbb::task& task() {return *reinterpret_cast<tbb::task*>(this+1);}
};

} 

#if __TBB_TASK_GROUP_CONTEXT

#if __TBB_TASK_PRIORITY
namespace internal {
static const int priority_stride_v4 = INT_MAX / 4;
#if __TBB_PREVIEW_CRITICAL_TASKS
static const int priority_critical = priority_stride_v4 * 3 + priority_stride_v4 / 3 * 2;
#endif
}

enum priority_t {
priority_normal = internal::priority_stride_v4 * 2,
priority_low = priority_normal - internal::priority_stride_v4,
priority_high = priority_normal + internal::priority_stride_v4
};

#endif 

#if TBB_USE_CAPTURED_EXCEPTION
class tbb_exception;
#else
namespace internal {
class tbb_exception_ptr;
}
#endif 

class task_scheduler_init;
namespace interface7 { class task_arena; }
using interface7::task_arena;


class task_group_context : internal::no_copy {
private:
friend class internal::generic_scheduler;
friend class task_scheduler_init;
friend class task_arena;

#if TBB_USE_CAPTURED_EXCEPTION
typedef tbb_exception exception_container_type;
#else
typedef internal::tbb_exception_ptr exception_container_type;
#endif

enum version_traits_word_layout {
traits_offset = 16,
version_mask = 0xFFFF,
traits_mask = 0xFFFFul << traits_offset
};

public:
enum kind_type {
isolated,
bound
};

enum traits_type {
exact_exception = 0x0001ul << traits_offset,
#if __TBB_FP_CONTEXT
fp_settings     = 0x0002ul << traits_offset,
#endif
concurrent_wait = 0x0004ul << traits_offset,
#if TBB_USE_CAPTURED_EXCEPTION
default_traits = 0
#else
default_traits = exact_exception
#endif 
};

private:
enum state {
may_have_children = 1,
next_state_value, low_unused_state_bit = (next_state_value-1)*2
};

union {
__TBB_atomic kind_type my_kind;
uintptr_t _my_kind_aligner;
};

task_group_context *my_parent;


internal::context_list_node_t my_node;

__itt_caller itt_caller;


char _leading_padding[internal::NFS_MaxLineSize
- 2 * sizeof(uintptr_t)- sizeof(void*) - sizeof(internal::context_list_node_t)
- sizeof(__itt_caller)
#if __TBB_FP_CONTEXT
- sizeof(internal::cpu_ctl_env_space)
#endif
];

#if __TBB_FP_CONTEXT

internal::cpu_ctl_env_space my_cpu_ctl_env;
#endif

uintptr_t my_cancellation_requested;


uintptr_t my_version_and_traits;

exception_container_type *my_exception;

internal::generic_scheduler *my_owner;

uintptr_t my_state;

#if __TBB_TASK_PRIORITY
intptr_t my_priority;
#endif 

internal::string_index my_name;


char _trailing_padding[internal::NFS_MaxLineSize - 2 * sizeof(uintptr_t) - 2 * sizeof(void*)
#if __TBB_TASK_PRIORITY
- sizeof(intptr_t)
#endif 
- sizeof(internal::string_index)
];

public:

task_group_context ( kind_type relation_with_parent = bound,
uintptr_t t = default_traits )
: my_kind(relation_with_parent)
, my_version_and_traits(3 | t)
, my_name(internal::CUSTOM_CTX)
{
init();
}

task_group_context ( internal::string_index name )
: my_kind(bound)
, my_version_and_traits(3 | default_traits)
, my_name(name)
{
init();
}

__TBB_EXPORTED_METHOD ~task_group_context ();


void __TBB_EXPORTED_METHOD reset ();


bool __TBB_EXPORTED_METHOD cancel_group_execution ();

bool __TBB_EXPORTED_METHOD is_group_execution_cancelled () const;


void __TBB_EXPORTED_METHOD register_pending_exception ();

#if __TBB_FP_CONTEXT

void __TBB_EXPORTED_METHOD capture_fp_settings ();
#endif

#if __TBB_TASK_PRIORITY
__TBB_DEPRECATED_IN_VERBOSE_MODE void set_priority ( priority_t );

__TBB_DEPRECATED_IN_VERBOSE_MODE priority_t priority () const;
#endif 

uintptr_t traits() const { return my_version_and_traits & traits_mask; }

protected:

void __TBB_EXPORTED_METHOD init ();

private:
friend class task;
friend class internal::allocate_root_with_context_proxy;

static const kind_type binding_required = bound;
static const kind_type binding_completed = kind_type(bound+1);
static const kind_type detached = kind_type(binding_completed+1);
static const kind_type dying = kind_type(detached+1);

template <typename T>
void propagate_task_group_state ( T task_group_context::*mptr_state, task_group_context& src, T new_state );

void bind_to ( internal::generic_scheduler *local_sched );

void register_with ( internal::generic_scheduler *local_sched );

#if __TBB_FP_CONTEXT
void copy_fp_settings( const task_group_context &src );
#endif 
}; 

#endif 


class __TBB_DEPRECATED_IN_VERBOSE_MODE task: __TBB_TASK_BASE_ACCESS interface5::internal::task_base {

void __TBB_EXPORTED_METHOD internal_set_ref_count( int count );

internal::reference_count __TBB_EXPORTED_METHOD internal_decrement_ref_count();

protected:
task() {prefix().extra_state=1;}

public:
virtual ~task() {}

virtual task* execute() = 0;

enum state_type {
executing,
reexecute,
ready,
allocated,
freed,
recycle
#if __TBB_RECYCLE_TO_ENQUEUE
,to_enqueue
#endif
#if __TBB_PREVIEW_RESUMABLE_TASKS
,to_resume
#endif
};


static internal::allocate_root_proxy allocate_root() {
return internal::allocate_root_proxy();
}

#if __TBB_TASK_GROUP_CONTEXT
static internal::allocate_root_with_context_proxy allocate_root( task_group_context& ctx ) {
return internal::allocate_root_with_context_proxy(ctx);
}
#endif 


internal::allocate_continuation_proxy& allocate_continuation() {
return *reinterpret_cast<internal::allocate_continuation_proxy*>(this);
}

internal::allocate_child_proxy& allocate_child() {
return *reinterpret_cast<internal::allocate_child_proxy*>(this);
}

using task_base::allocate_additional_child_of;

#if __TBB_DEPRECATED_TASK_INTERFACE

void __TBB_EXPORTED_METHOD destroy( task& t );
#else 
using task_base::destroy;
#endif 



void recycle_as_continuation() {
__TBB_ASSERT( prefix().state==executing, "execute not running?" );
prefix().state = allocated;
}


void recycle_as_safe_continuation() {
__TBB_ASSERT( prefix().state==executing, "execute not running?" );
prefix().state = recycle;
}

void recycle_as_child_of( task& new_parent ) {
internal::task_prefix& p = prefix();
__TBB_ASSERT( prefix().state==executing||prefix().state==allocated, "execute not running, or already recycled" );
__TBB_ASSERT( prefix().ref_count==0, "no child tasks allowed when recycled as a child" );
__TBB_ASSERT( p.parent==NULL, "parent must be null" );
__TBB_ASSERT( new_parent.prefix().state<=recycle, "corrupt parent's state" );
__TBB_ASSERT( new_parent.prefix().state!=freed, "parent already freed" );
p.state = allocated;
p.parent = &new_parent;
#if __TBB_TASK_GROUP_CONTEXT
p.context = new_parent.prefix().context;
#endif 
}


void recycle_to_reexecute() {
__TBB_ASSERT( prefix().state==executing, "execute not running, or already recycled" );
__TBB_ASSERT( prefix().ref_count==0, "no child tasks allowed when recycled for reexecution" );
prefix().state = reexecute;
}

#if __TBB_RECYCLE_TO_ENQUEUE

void recycle_to_enqueue() {
__TBB_ASSERT( prefix().state==executing, "execute not running, or already recycled" );
prefix().state = to_enqueue;
}
#endif 


void set_ref_count( int count ) {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
internal_set_ref_count(count);
#else
prefix().ref_count = count;
#endif 
}


void increment_ref_count() {
__TBB_FetchAndIncrementWacquire( &prefix().ref_count );
}


int add_ref_count( int count ) {
internal::call_itt_notify( internal::releasing, &prefix().ref_count );
internal::reference_count k = count+__TBB_FetchAndAddW( &prefix().ref_count, count );
__TBB_ASSERT( k>=0, "task's reference count underflowed" );
if( k==0 )
internal::call_itt_notify( internal::acquired, &prefix().ref_count );
return int(k);
}


int decrement_ref_count() {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
return int(internal_decrement_ref_count());
#else
return int(__TBB_FetchAndDecrementWrelease( &prefix().ref_count ))-1;
#endif 
}

using task_base::spawn;

void spawn_and_wait_for_all( task& child ) {
prefix().owner->wait_for_all( *this, &child );
}

void __TBB_EXPORTED_METHOD spawn_and_wait_for_all( task_list& list );

static void spawn_root_and_wait( task& root ) {
root.prefix().owner->spawn_root_and_wait( root, root.prefix().next );
}


static void spawn_root_and_wait( task_list& root_list );


void wait_for_all() {
prefix().owner->wait_for_all( *this, NULL );
}

#if __TBB_TASK_PRIORITY

#endif 
static void enqueue( task& t ) {
t.prefix().owner->enqueue( t, NULL );
}

#if __TBB_TASK_PRIORITY
static void enqueue( task& t, priority_t p ) {
#if __TBB_PREVIEW_CRITICAL_TASKS
__TBB_ASSERT(p == priority_low || p == priority_normal || p == priority_high
|| p == internal::priority_critical, "Invalid priority level value");
#else
__TBB_ASSERT(p == priority_low || p == priority_normal || p == priority_high, "Invalid priority level value");
#endif
t.prefix().owner->enqueue( t, (void*)p );
}
#endif 

#if __TBB_TASK_PRIORITY
inline static void enqueue( task& t, task_arena& arena, priority_t p = priority_t(0) );
#else
inline static void enqueue( task& t, task_arena& arena);
#endif

static task& __TBB_EXPORTED_FUNC self();

task* parent() const {return prefix().parent;}

void set_parent(task* p) {
#if __TBB_TASK_GROUP_CONTEXT
__TBB_ASSERT(!p || prefix().context == p->prefix().context, "The tasks must be in the same context");
#endif
prefix().parent = p;
}

#if __TBB_TASK_GROUP_CONTEXT

task_group_context* context() {return prefix().context;}

task_group_context* group () { return prefix().context; }
#endif 

bool is_stolen_task() const {
return (prefix().extra_state & 0x80)!=0;
}

bool is_enqueued_task() const {
return (prefix().extra_state & 0x10)!=0;
}

#if __TBB_PREVIEW_RESUMABLE_TASKS
typedef void* suspend_point;

template <typename F>
static void suspend(F f);

static void resume(suspend_point tag);
#endif


state_type state() const {return state_type(prefix().state);}

int ref_count() const {
#if TBB_USE_ASSERT
#if __TBB_PREVIEW_RESUMABLE_TASKS
internal::reference_count ref_count_ = prefix().ref_count & ~internal::abandon_flag;
#else
internal::reference_count ref_count_ = prefix().ref_count;
#endif
__TBB_ASSERT( ref_count_==int(ref_count_), "integer overflow error");
#endif
#if __TBB_PREVIEW_RESUMABLE_TASKS
return int(prefix().ref_count & ~internal::abandon_flag);
#else
return int(prefix().ref_count);
#endif
}

bool __TBB_EXPORTED_METHOD is_owned_by_current_thread() const;



typedef internal::affinity_id affinity_id;

void set_affinity( affinity_id id ) {prefix().affinity = id;}

affinity_id affinity() const {return prefix().affinity;}


virtual void __TBB_EXPORTED_METHOD note_affinity( affinity_id id );

#if __TBB_TASK_GROUP_CONTEXT

void __TBB_EXPORTED_METHOD change_group ( task_group_context& ctx );


bool cancel_group_execution () { return prefix().context->cancel_group_execution(); }

bool is_cancelled () const { return prefix().context->is_group_execution_cancelled(); }
#else
bool is_cancelled () const { return false; }
#endif 

#if __TBB_TASK_PRIORITY
__TBB_DEPRECATED void set_group_priority ( priority_t p ) {  prefix().context->set_priority(p); }

__TBB_DEPRECATED priority_t group_priority () const { return prefix().context->priority(); }

#endif 

private:
friend class interface5::internal::task_base;
friend class task_list;
friend class internal::scheduler;
friend class internal::allocate_root_proxy;
#if __TBB_TASK_GROUP_CONTEXT
friend class internal::allocate_root_with_context_proxy;
#endif 
friend class internal::allocate_continuation_proxy;
friend class internal::allocate_child_proxy;
friend class internal::allocate_additional_child_of_proxy;


internal::task_prefix& prefix( internal::version_tag* = NULL ) const {
return reinterpret_cast<internal::task_prefix*>(const_cast<task*>(this))[-1];
}
#if __TBB_PREVIEW_CRITICAL_TASKS
friend void internal::make_critical( task& );
friend bool internal::is_critical( task& );
#endif
}; 

#if __TBB_PREVIEW_CRITICAL_TASKS
namespace internal {
inline void make_critical( task& t ) { t.prefix().extra_state |= 0x8; }
inline bool is_critical( task& t ) { return bool((t.prefix().extra_state & 0x8) != 0); }
} 
#endif 

#if __TBB_PREVIEW_RESUMABLE_TASKS
namespace internal {
template <typename F>
static void suspend_callback(void* user_callback, task::suspend_point tag) {
F user_callback_copy = *static_cast<F*>(user_callback);
user_callback_copy(tag);
}
void __TBB_EXPORTED_FUNC internal_suspend(void* suspend_callback, void* user_callback);
void __TBB_EXPORTED_FUNC internal_resume(task::suspend_point);
task::suspend_point __TBB_EXPORTED_FUNC internal_current_suspend_point();
}

template <typename F>
inline void task::suspend(F f) {
internal::internal_suspend((void*)internal::suspend_callback<F>, &f);
}
inline void task::resume(suspend_point tag) {
internal::internal_resume(tag);
}
#endif


class __TBB_DEPRECATED_IN_VERBOSE_MODE empty_task: public task {
task* execute() __TBB_override {
return NULL;
}
};

namespace internal {
template<typename F>
class function_task : public task {
#if __TBB_ALLOW_MUTABLE_FUNCTORS
F my_func;
#else
const F my_func;
#endif
task* execute() __TBB_override {
my_func();
return NULL;
}
public:
function_task( const F& f ) : my_func(f) {}
#if __TBB_CPP11_RVALUE_REF_PRESENT
function_task( F&& f ) : my_func( std::move(f) ) {}
#endif
};
} 


class __TBB_DEPRECATED_IN_VERBOSE_MODE task_list: internal::no_copy {
private:
task* first;
task** next_ptr;
friend class task;
friend class interface5::internal::task_base;
public:
task_list() : first(NULL), next_ptr(&first) {}

~task_list() {}

bool empty() const {return !first;}

void push_back( task& task ) {
task.prefix().next = NULL;
*next_ptr = &task;
next_ptr = &task.prefix().next;
}
#if __TBB_TODO
void push_front( task& task ) {
if( empty() ) {
push_back(task);
} else {
task.prefix().next = first;
first = &task;
}
}
#endif
task& pop_front() {
__TBB_ASSERT( !empty(), "attempt to pop item from empty task_list" );
task* result = first;
first = result->prefix().next;
if( !first ) next_ptr = &first;
return *result;
}

void clear() {
first=NULL;
next_ptr=&first;
}
};

inline void interface5::internal::task_base::spawn( task& t ) {
t.prefix().owner->spawn( t, t.prefix().next );
}

inline void interface5::internal::task_base::spawn( task_list& list ) {
if( task* t = list.first ) {
t->prefix().owner->spawn( *t, *list.next_ptr );
list.clear();
}
}

inline void task::spawn_root_and_wait( task_list& root_list ) {
if( task* t = root_list.first ) {
t->prefix().owner->spawn_root_and_wait( *t, *root_list.next_ptr );
root_list.clear();
}
}

} 

inline void *operator new( size_t bytes, const tbb::internal::allocate_root_proxy& ) {
return &tbb::internal::allocate_root_proxy::allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_root_proxy& ) {
tbb::internal::allocate_root_proxy::free( *static_cast<tbb::task*>(task) );
}

#if __TBB_TASK_GROUP_CONTEXT
inline void *operator new( size_t bytes, const tbb::internal::allocate_root_with_context_proxy& p ) {
return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_root_with_context_proxy& p ) {
p.free( *static_cast<tbb::task*>(task) );
}
#endif 

inline void *operator new( size_t bytes, const tbb::internal::allocate_continuation_proxy& p ) {
return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_continuation_proxy& p ) {
p.free( *static_cast<tbb::task*>(task) );
}

inline void *operator new( size_t bytes, const tbb::internal::allocate_child_proxy& p ) {
return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_child_proxy& p ) {
p.free( *static_cast<tbb::task*>(task) );
}

inline void *operator new( size_t bytes, const tbb::internal::allocate_additional_child_of_proxy& p ) {
return &p.allocate(bytes);
}

inline void operator delete( void* task, const tbb::internal::allocate_additional_child_of_proxy& p ) {
p.free( *static_cast<tbb::task*>(task) );
}

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_task_H_include_area

#endif 
