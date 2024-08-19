


#if __TBB_CPF_BUILD
#define TBB_PREVIEW_AGGREGATOR 1
#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#define TBB_PREVIEW_VARIADIC_PARALLEL_INVOKE 1
#define TBB_PREVIEW_FLOW_GRAPH_NODES 1
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#endif

#if __TBB_TEST_SECONDARY
#if TBB_USE_DEBUG
#ifdef _DEBUG
#undef _DEBUG
#endif 
#define _DEBUG
static bool isDebugExpected = true;
#else
#define _DEBUG 0x0
static bool isDebugExpected = false;
#endif 
#define DO_TEST_DEBUG_MACRO 1
#else
#if _DEBUG
static bool isDebugExpected = true;
#define DO_TEST_DEBUG_MACRO 1
#elif _MSC_VER
static bool isDebugExpected = false;
#define DO_TEST_DEBUG_MACRO 1
#endif 
#endif 

#if DO_TEST_DEBUG_MACRO
#undef TBB_USE_DEBUG
#endif 
#include "tbb/tbb_config.h"

#if !TBB_USE_DEBUG && defined(_DEBUG)
#undef _DEBUG
#endif 

#include "harness_defs.h"
#if !(__TBB_TEST_SECONDARY && __TBB_CPP11_STD_PLACEHOLDERS_LINKAGE_BROKEN)

#if _MSC_VER
#pragma warning (disable : 4503)      
#if !TBB_USE_EXCEPTIONS
#pragma warning (push)
#pragma warning (disable: 4530)
#endif
#endif

#include "tbb/tbb.h"

static volatile size_t g_sink;

#define TestTypeDefinitionPresence( Type ) g_sink = sizeof(tbb::Type);
#define TestTypeDefinitionPresence2(TypeStart, TypeEnd) g_sink = sizeof(tbb::TypeStart,TypeEnd);
#define TestFuncDefinitionPresence(Fn, Args, ReturnType) { ReturnType (*pfn)Args = &tbb::Fn; (void)pfn; }

struct Body {
void operator() () const {}
};
struct Body1 {
void operator() ( int ) const {}
};
struct Body1a {
int operator() ( const tbb::blocked_range<int>&, const int ) const { return 0; }
};
struct Body1b {
int operator() ( const int, const int ) const { return 0; }
};
struct Body2 {
Body2 () {}
Body2 ( const Body2&, tbb::split ) {}
void operator() ( const tbb::blocked_range<int>& ) const {}
void join( const Body2& ) {}
};
struct Body3 {
Body3 () {}
Body3 ( const Body3&, tbb::split ) {}
void operator() ( const tbb::blocked_range2d<int>&, tbb::pre_scan_tag ) const {}
void operator() ( const tbb::blocked_range2d<int>&, tbb::final_scan_tag ) const {}
void reverse_join( Body3& ) {}
void assign( const Body3& ) {}
};

#if !__TBB_TEST_SECONDARY

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"

#include <stdexcept>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif


template <typename E>
void TestExceptionClassExports ( const E& exc, tbb::internal::exception_id eid ) {
ASSERT( eid<tbb::internal::eid_max, NULL );
#if TBB_USE_EXCEPTIONS
for ( int i = 0; i < 2; ++i ) {
try {
if ( i == 0 )
throw exc;
#if !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
else
tbb::internal::throw_exception( eid );
#endif
}
catch ( E& e ) {
ASSERT ( e.what(), "Missing what() string" );
}
catch ( ... ) {
ASSERT ( __TBB_EXCEPTION_TYPE_INFO_BROKEN, "Unrecognized exception. Likely RTTI related exports are missing" );
}
}
#else 
(void)exc;
#endif 
}

void TestExceptionClassesExports () {
TestExceptionClassExports( std::bad_alloc(), tbb::internal::eid_bad_alloc );
TestExceptionClassExports( tbb::bad_last_alloc(), tbb::internal::eid_bad_last_alloc );
TestExceptionClassExports( std::invalid_argument("test"), tbb::internal::eid_nonpositive_step );
TestExceptionClassExports( std::out_of_range("test"), tbb::internal::eid_out_of_range );
TestExceptionClassExports( std::range_error("test"), tbb::internal::eid_segment_range_error );
TestExceptionClassExports( std::range_error("test"), tbb::internal::eid_index_range_error );
TestExceptionClassExports( tbb::missing_wait(), tbb::internal::eid_missing_wait );
TestExceptionClassExports( tbb::invalid_multiple_scheduling(), tbb::internal::eid_invalid_multiple_scheduling );
TestExceptionClassExports( tbb::improper_lock(), tbb::internal::eid_improper_lock );
TestExceptionClassExports( std::runtime_error("test"), tbb::internal::eid_possible_deadlock );
TestExceptionClassExports( std::runtime_error("test"), tbb::internal::eid_operation_not_permitted );
TestExceptionClassExports( std::runtime_error("test"), tbb::internal::eid_condvar_wait_failed );
TestExceptionClassExports( std::out_of_range("test"), tbb::internal::eid_invalid_load_factor );
TestExceptionClassExports( std::invalid_argument("test"), tbb::internal::eid_invalid_swap );
TestExceptionClassExports( std::length_error("test"), tbb::internal::eid_reservation_length_error );
TestExceptionClassExports( std::out_of_range("test"), tbb::internal::eid_invalid_key );
TestExceptionClassExports( tbb::user_abort(), tbb::internal::eid_user_abort );
TestExceptionClassExports( std::runtime_error("test"), tbb::internal::eid_bad_tagged_msg_cast );
}
#endif 

#if __TBB_CPF_BUILD
struct Handler {
void operator()( tbb::aggregator_operation* ) {}
};
static void TestPreviewNames() {
TestTypeDefinitionPresence( aggregator );
TestTypeDefinitionPresence( aggregator_ext<Handler> );
TestTypeDefinitionPresence2(concurrent_lru_cache<int, int> );
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
TestTypeDefinitionPresence2( flow::composite_node<tbb::flow::tuple<int>, tbb::flow::tuple<int> > );
#endif
TestTypeDefinitionPresence( static_partitioner );
}
#endif

#if __TBB_TEST_SECONDARY

#include "harness_assert.h"
bool Secondary()
#else
bool Secondary();
int TestMain ()
#endif
{
#if __TBB_CPP11_STD_PLACEHOLDERS_LINKAGE_BROKEN
REPORT("Known issue: \"multiple definition\" linker error detection test skipped.\n");
#endif
TestTypeDefinitionPresence( aligned_space<int> );
TestTypeDefinitionPresence( atomic<int> );
TestTypeDefinitionPresence( cache_aligned_allocator<int> );
TestTypeDefinitionPresence( tbb_hash_compare<int> );
TestTypeDefinitionPresence2(concurrent_hash_map<int, int> );
TestTypeDefinitionPresence2(concurrent_unordered_map<int, int> );
TestTypeDefinitionPresence2(concurrent_unordered_multimap<int, int> );
TestTypeDefinitionPresence( concurrent_unordered_set<int> );
TestTypeDefinitionPresence( concurrent_unordered_multiset<int> );
TestTypeDefinitionPresence( concurrent_bounded_queue<int> );
TestTypeDefinitionPresence( concurrent_queue<int> );
TestTypeDefinitionPresence( strict_ppl::concurrent_queue<int> );
TestTypeDefinitionPresence( concurrent_priority_queue<int> );
TestTypeDefinitionPresence( combinable<int> );
TestTypeDefinitionPresence( concurrent_vector<int> );
TestTypeDefinitionPresence( enumerable_thread_specific<int> );

TestTypeDefinitionPresence( flow::graph );
TestTypeDefinitionPresence( flow::source_node<int> );
TestTypeDefinitionPresence2(flow::function_node<int, int> );
typedef tbb::flow::tuple<int, int> intpair;
TestTypeDefinitionPresence2(flow::multifunction_node<int, intpair> );
TestTypeDefinitionPresence( flow::split_node<intpair> );
TestTypeDefinitionPresence( flow::continue_node<int> );
TestTypeDefinitionPresence( flow::overwrite_node<int> );
TestTypeDefinitionPresence( flow::write_once_node<int> );
TestTypeDefinitionPresence( flow::broadcast_node<int> );
TestTypeDefinitionPresence( flow::buffer_node<int> );
TestTypeDefinitionPresence( flow::queue_node<int> );
TestTypeDefinitionPresence( flow::sequencer_node<int> );
TestTypeDefinitionPresence( flow::priority_queue_node<int> );
TestTypeDefinitionPresence( flow::limiter_node<int> );
TestTypeDefinitionPresence2(flow::indexer_node<int, int> );
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
TestTypeDefinitionPresence2( flow::composite_node<tbb::flow::tuple<int>, tbb::flow::tuple<int> > );
#endif
using tbb::flow::queueing;
TestTypeDefinitionPresence2( flow::join_node< intpair, queueing > );

TestTypeDefinitionPresence( mutex );
TestTypeDefinitionPresence( null_mutex );
TestTypeDefinitionPresence( null_rw_mutex );
TestTypeDefinitionPresence( queuing_mutex );
TestTypeDefinitionPresence( queuing_rw_mutex );
TestTypeDefinitionPresence( recursive_mutex );
TestTypeDefinitionPresence( spin_mutex );
TestTypeDefinitionPresence( spin_rw_mutex );
TestTypeDefinitionPresence( speculative_spin_mutex );
TestTypeDefinitionPresence( speculative_spin_rw_mutex );
TestTypeDefinitionPresence( critical_section );
TestTypeDefinitionPresence( reader_writer_lock );
#if __TBB_TASK_GROUP_CONTEXT
TestTypeDefinitionPresence( tbb_exception );
TestTypeDefinitionPresence( captured_exception );
TestTypeDefinitionPresence( movable_exception<int> );
#if !TBB_USE_CAPTURED_EXCEPTION
TestTypeDefinitionPresence( internal::tbb_exception_ptr );
#endif 
TestTypeDefinitionPresence( task_group_context );
TestTypeDefinitionPresence( task_group );
TestTypeDefinitionPresence( structured_task_group );
TestTypeDefinitionPresence( task_handle<Body> );
#endif 
TestTypeDefinitionPresence( blocked_range3d<int> );
TestFuncDefinitionPresence( parallel_invoke, (const Body&, const Body&), void );
TestFuncDefinitionPresence( parallel_do, (int*, int*, const Body1&), void );
TestFuncDefinitionPresence( parallel_for_each, (int*, int*, const Body1&), void );
TestFuncDefinitionPresence( parallel_for, (int, int, int, const Body1&), void );
TestFuncDefinitionPresence( parallel_for, (const tbb::blocked_range<int>&, const Body2&, const tbb::simple_partitioner&), void );
TestFuncDefinitionPresence( parallel_reduce, (const tbb::blocked_range<int>&, const int&, const Body1a&, const Body1b&, const tbb::auto_partitioner&), int );
TestFuncDefinitionPresence( parallel_reduce, (const tbb::blocked_range<int>&, Body2&, tbb::affinity_partitioner&), void );
TestFuncDefinitionPresence( parallel_deterministic_reduce, (const tbb::blocked_range<int>&, const int&, const Body1a&, const Body1b&), int );
TestFuncDefinitionPresence( parallel_deterministic_reduce, (const tbb::blocked_range<int>&, Body2&), void );
TestFuncDefinitionPresence( parallel_scan, (const tbb::blocked_range2d<int>&, Body3&, const tbb::auto_partitioner&), void );
TestFuncDefinitionPresence( parallel_sort, (int*, int*), void );
TestTypeDefinitionPresence( pipeline );
TestFuncDefinitionPresence( parallel_pipeline, (size_t, const tbb::filter_t<void,void>&), void );
TestTypeDefinitionPresence( task );
TestTypeDefinitionPresence( empty_task );
TestTypeDefinitionPresence( task_list );
TestTypeDefinitionPresence( task_arena );
TestTypeDefinitionPresence( task_scheduler_init );
TestTypeDefinitionPresence( task_scheduler_observer );
TestTypeDefinitionPresence( tbb_thread );
TestTypeDefinitionPresence( tbb_allocator<int> );
TestTypeDefinitionPresence( zero_allocator<int> );
TestTypeDefinitionPresence( tick_count );
#if TBB_PREVIEW_GLOBAL_CONTROL
TestTypeDefinitionPresence( global_control );
#endif
TestFuncDefinitionPresence( parallel_for, (int, int, int, const Body1&, const tbb::static_partitioner&), void );
TestFuncDefinitionPresence( parallel_reduce, (const tbb::blocked_range<int>&, Body2&, const tbb::static_partitioner&), void );

#if __TBB_CPF_BUILD
TestPreviewNames();
#endif
#ifdef DO_TEST_DEBUG_MACRO
#if TBB_USE_DEBUG
ASSERT( isDebugExpected, "Debug mode is observed while release mode is expected." );
#else
ASSERT( !isDebugExpected, "Release mode is observed while debug mode is expected." );
#endif 
#endif 
#if __TBB_TEST_SECONDARY
return true;
#else
TestExceptionClassesExports();
Secondary();
return Harness::Done;
#endif 
}
#endif 
