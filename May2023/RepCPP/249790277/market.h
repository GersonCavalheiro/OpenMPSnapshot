

#ifndef _TBB_market_H
#define _TBB_market_H

#include "tbb/tbb_stddef.h"

#include "scheduler_common.h"
#include "tbb/atomic.h"
#include "tbb/spin_rw_mutex.h"
#include "../rml/include/rml_tbb.h"

#include "intrusive_list.h"

#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (push)
#pragma warning (disable: 4244)
#endif

namespace tbb {

class task_group_context;

namespace internal {


class market : no_copy, rml::tbb_client {
friend class generic_scheduler;
friend class arena;
friend class tbb::interface7::internal::task_arena_base;
template<typename SchedulerTraits> friend class custom_scheduler;
friend class tbb::task_group_context;
private:
friend void ITT_DoUnsafeOneTimeInitialization ();

typedef intrusive_list<arena> arena_list_type;
typedef intrusive_list<generic_scheduler> scheduler_list_type;

static market* theMarket;

typedef scheduler_mutex_type global_market_mutex_type;

static global_market_mutex_type  theMarketMutex;

typedef spin_rw_mutex arenas_list_mutex_type;
arenas_list_mutex_type my_arenas_list_mutex;

rml::tbb_server* my_server;


unsigned my_num_workers_hard_limit;


unsigned my_num_workers_soft_limit;

int my_num_workers_requested;


atomic<unsigned> my_first_unused_worker_idx;

int my_total_demand;

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
int my_mandatory_num_requested;
#endif

#if __TBB_TASK_PRIORITY

intptr_t my_global_top_priority;


intptr_t my_global_bottom_priority;


uintptr_t my_global_reload_epoch;

struct priority_level_info {
arena_list_type arenas;


arena *next_arena;

int workers_requested;

int workers_available;
}; 

priority_level_info my_priority_levels[num_priority_levels];

#else 

arena_list_type my_arenas;


arena *my_next_arena;
#endif 

uintptr_t my_arenas_aba_epoch;

unsigned my_ref_count;

unsigned my_public_ref_count;

size_t my_stack_size;

bool my_join_workers;

static const unsigned skip_soft_limit_warning = ~0U;

unsigned my_workers_soft_limit_to_report;
#if __TBB_COUNT_TASK_NODES

atomic<intptr_t> my_task_node_count;
#endif 

market ( unsigned workers_soft_limit, unsigned workers_hard_limit, size_t stack_size );

static market& global_market ( bool is_public, unsigned max_num_workers = 0, size_t stack_size = 0 );

void destroy ();

#if __TBB_TASK_PRIORITY
arena* arena_in_need ( arena* prev_arena );


void update_allotment ( intptr_t highest_affected_priority );

void update_arena_top_priority ( arena& a, intptr_t newPriority );

inline void update_global_top_priority ( intptr_t newPriority );

inline void reset_global_priority ();

inline void advance_global_reload_epoch () {
__TBB_store_with_release( my_global_reload_epoch, my_global_reload_epoch + 1 );
}

void assert_market_valid () const {
__TBB_ASSERT( (my_priority_levels[my_global_top_priority].workers_requested > 0
&& !my_priority_levels[my_global_top_priority].arenas.empty())
|| (my_global_top_priority == my_global_bottom_priority &&
my_global_top_priority == normalized_normal_priority), NULL );
}

#else 


void update_allotment () {
if ( my_total_demand )
update_allotment( my_arenas, my_total_demand, (int)my_num_workers_soft_limit );
}

arena* arena_in_need (arena*) {
if(__TBB_load_with_acquire(my_total_demand) <= 0)
return NULL;
arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex, false);
return arena_in_need(my_arenas, my_next_arena);
}
void assert_market_valid () const {}
#endif 


void insert_arena_into_list ( arena& a );

void remove_arena_from_list ( arena& a );

arena* arena_in_need ( arena_list_type &arenas, arena *&next );

static int update_allotment ( arena_list_type& arenas, int total_demand, int max_workers );



version_type version () const __TBB_override { return 0; }

unsigned max_job_count () const __TBB_override { return my_num_workers_hard_limit; }

size_t min_stack_size () const __TBB_override { return worker_stack_size(); }

policy_type policy () const __TBB_override { return throughput; }

job* create_one_job () __TBB_override;

void cleanup( job& j ) __TBB_override;

void acknowledge_close_connection () __TBB_override;

void process( job& j ) __TBB_override;

public:

static arena* create_arena ( int num_slots, int num_reserved_slots, size_t stack_size );

void try_destroy_arena ( arena*, uintptr_t aba_epoch );

void detach_arena ( arena& );

bool release ( bool is_public, bool blocking_terminate );

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
bool mandatory_concurrency_enable_impl ( arena *a, bool *enabled = NULL );

bool mandatory_concurrency_enable ( arena *a );

void mandatory_concurrency_disable ( arena *a );
#endif 


void adjust_demand ( arena&, int delta );

bool must_join_workers () const { return my_join_workers; }

size_t worker_stack_size () const { return my_stack_size; }

static void set_active_num_workers( unsigned w );

static unsigned app_parallelism_limit();

#if _WIN32||_WIN64
void register_master( ::rml::server::execution_resource_t& rsc_handle ) {
__TBB_ASSERT( my_server, "RML server not defined?" );
my_server->register_master( rsc_handle );
}

void unregister_master( ::rml::server::execution_resource_t& rsc_handle ) const {
my_server->unregister_master( rsc_handle );
}
#endif 

#if __TBB_TASK_GROUP_CONTEXT

template <typename T>
bool propagate_task_group_state ( T task_group_context::*mptr_state, task_group_context& src, T new_state );
#endif 

#if __TBB_TASK_PRIORITY

bool lower_arena_priority ( arena& a, intptr_t new_priority, uintptr_t old_reload_epoch );


bool update_arena_priority ( arena& a, intptr_t new_priority );
#endif 

#if __TBB_COUNT_TASK_NODES

void update_task_node_count( intptr_t delta ) { my_task_node_count += delta; }
#endif 

#if __TBB_TASK_GROUP_CONTEXT
scheduler_list_type my_masters;


generic_scheduler* my_workers[1];
#endif 

static unsigned max_num_workers() {
global_market_mutex_type::scoped_lock lock( theMarketMutex );
return theMarket? theMarket->my_num_workers_hard_limit : 0;
}
}; 

} 
} 

#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (pop)
#endif 

#endif 
