

#ifndef __TBB_parallel_for_H
#define __TBB_parallel_for_H

#include <new>
#include "task.h"
#include "partitioner.h"
#include "blocked_range.h"
#include "tbb_exception.h"

namespace tbb {

namespace interface6 {
namespace internal {


template<typename Range, typename Body, typename Partitioner>
class start_for: public task {
Range my_range;
const Body my_body;
typename Partitioner::task_partition_type my_partition;
task* execute();

public:
start_for( const Range& range, const Body& body, Partitioner& partitioner ) :
my_range(range),    
my_body(body),
my_partition(partitioner)
{
}

start_for( start_for& parent_, split ) :
my_range(parent_.my_range,split()),
my_body(parent_.my_body),
my_partition(parent_.my_partition, split())
{
my_partition.set_affinity(*this);
}

start_for( start_for& parent_, const Range& r, depth_t d ) :
my_range(r),
my_body(parent_.my_body),
my_partition(parent_.my_partition,split())
{
my_partition.set_affinity(*this);
my_partition.align_depth( d );
}
void note_affinity( affinity_id id ) {
my_partition.note_affinity( id );
}
static void run(  const Range& range, const Body& body, Partitioner& partitioner ) {
if( !range.empty() ) {
#if !__TBB_TASK_GROUP_CONTEXT || TBB_JOIN_OUTER_TASK_GROUP
start_for& a = *new(task::allocate_root()) start_for(range,body,partitioner);
#else
task_group_context context;
start_for& a = *new(task::allocate_root(context)) start_for(range,body,partitioner);
#endif 
task::spawn_root_and_wait(a);
}
}
#if __TBB_TASK_GROUP_CONTEXT
static void run(  const Range& range, const Body& body, Partitioner& partitioner, task_group_context& context ) {
if( !range.empty() ) {
start_for& a = *new(task::allocate_root(context)) start_for(range,body,partitioner);
task::spawn_root_and_wait(a);
}
}
#endif 
flag_task *create_continuation() {
return new( allocate_continuation() ) flag_task();
}
void run_body( Range &r ) { my_body( r ); }
};

template<typename Range, typename Body, typename Partitioner>
task* start_for<Range,Body,Partitioner>::execute() {
my_partition.check_being_stolen( *this );
my_partition.execute(*this, my_range);
return NULL;
} 
} 
} 

namespace internal {
using interface6::internal::start_for;

template<typename Function, typename Index>
class parallel_for_body : internal::no_assign {
const Function &my_func;
const Index my_begin;
const Index my_step; 
public:
parallel_for_body( const Function& _func, Index& _begin, Index& _step) 
: my_func(_func), my_begin(_begin), my_step(_step) {}

void operator()( tbb::blocked_range<Index>& r ) const {
#if __INTEL_COMPILER
#pragma ivdep
#endif
for( Index i = r.begin(),  k = my_begin + i * my_step; i < r.end(); i++, k = k + my_step)
my_func( k );
}
};
} 







template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body ) {
internal::start_for<Range,Body,const __TBB_DEFAULT_PARTITIONER>::run(range,body,__TBB_DEFAULT_PARTITIONER());
}


template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const simple_partitioner& partitioner ) {
internal::start_for<Range,Body,const simple_partitioner>::run(range,body,partitioner);
}


template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const auto_partitioner& partitioner ) {
internal::start_for<Range,Body,const auto_partitioner>::run(range,body,partitioner);
}


template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, affinity_partitioner& partitioner ) {
internal::start_for<Range,Body,affinity_partitioner>::run(range,body,partitioner);
}

#if __TBB_TASK_GROUP_CONTEXT

template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, task_group_context& context ) {
internal::start_for<Range,Body,const __TBB_DEFAULT_PARTITIONER>::run(range, body, __TBB_DEFAULT_PARTITIONER(), context);
}


template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const simple_partitioner& partitioner, task_group_context& context ) {
internal::start_for<Range,Body,const simple_partitioner>::run(range, body, partitioner, context);
}


template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, const auto_partitioner& partitioner, task_group_context& context ) {
internal::start_for<Range,Body,const auto_partitioner>::run(range, body, partitioner, context);
}


template<typename Range, typename Body>
void parallel_for( const Range& range, const Body& body, affinity_partitioner& partitioner, task_group_context& context ) {
internal::start_for<Range,Body,affinity_partitioner>::run(range,body,partitioner, context);
}
#endif 

namespace strict_ppl {

template <typename Index, typename Function, typename Partitioner>
void parallel_for_impl(Index first, Index last, Index step, const Function& f, Partitioner& partitioner) {
if (step <= 0 )
internal::throw_exception(internal::eid_nonpositive_step); 
else if (last > first) {
Index end = (last - first - Index(1)) / step + Index(1);
tbb::blocked_range<Index> range(static_cast<Index>(0), end);
internal::parallel_for_body<Function, Index> body(f, first, step);
tbb::parallel_for(range, body, partitioner);
}
}    

template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, auto_partitioner());
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const simple_partitioner& partitioner) {
parallel_for_impl<Index,Function,const simple_partitioner>(first, last, step, f, partitioner);
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const auto_partitioner& partitioner) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, partitioner);
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, affinity_partitioner& partitioner) {
parallel_for_impl(first, last, step, f, partitioner);
}

template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, auto_partitioner());
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const simple_partitioner& partitioner) {
parallel_for_impl<Index,Function,const simple_partitioner>(first, last, static_cast<Index>(1), f, partitioner);
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, const auto_partitioner& partitioner) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, partitioner);
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, affinity_partitioner& partitioner) {
parallel_for_impl(first, last, static_cast<Index>(1), f, partitioner);
}

#if __TBB_TASK_GROUP_CONTEXT
template <typename Index, typename Function, typename Partitioner>
void parallel_for_impl(Index first, Index last, Index step, const Function& f, Partitioner& partitioner, tbb::task_group_context &context) {
if (step <= 0 )
internal::throw_exception(internal::eid_nonpositive_step); 
else if (last > first) {
Index end = (last - first - Index(1)) / step + Index(1);
tbb::blocked_range<Index> range(static_cast<Index>(0), end);
internal::parallel_for_body<Function, Index> body(f, first, step);
tbb::parallel_for(range, body, partitioner, context);
}
}

template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, tbb::task_group_context &context) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, auto_partitioner(), context);
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const simple_partitioner& partitioner, tbb::task_group_context &context) {
parallel_for_impl<Index,Function,const simple_partitioner>(first, last, step, f, partitioner, context);
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, const auto_partitioner& partitioner, tbb::task_group_context &context) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, partitioner, context);
}
template <typename Index, typename Function>
void parallel_for(Index first, Index last, Index step, const Function& f, affinity_partitioner& partitioner, tbb::task_group_context &context) {
parallel_for_impl(first, last, step, f, partitioner, context);
}


template <typename Index, typename Function>
void parallel_for(Index first, Index last, const Function& f, tbb::task_group_context &context) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, auto_partitioner(), context);
}
template <typename Index, typename Function, typename Partitioner>
void parallel_for(Index first, Index last, const Function& f, const simple_partitioner& partitioner, tbb::task_group_context &context) {
parallel_for_impl<Index,Function,const simple_partitioner>(first, last, static_cast<Index>(1), f, partitioner, context);
}
template <typename Index, typename Function, typename Partitioner>
void parallel_for(Index first, Index last, const Function& f, const auto_partitioner& partitioner, tbb::task_group_context &context) {
parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, partitioner, context);
}
template <typename Index, typename Function, typename Partitioner>
void parallel_for(Index first, Index last, const Function& f, affinity_partitioner& partitioner, tbb::task_group_context &context) {
parallel_for_impl(first, last, static_cast<Index>(1), f, partitioner, context);
}

#endif 

} 

using strict_ppl::parallel_for;

} 

#if TBB_PREVIEW_SERIAL_SUBSET
#define __TBB_NORMAL_EXECUTION
#include "../serial/tbb/parallel_for.h"
#undef __TBB_NORMAL_EXECUTION
#endif

#endif 

