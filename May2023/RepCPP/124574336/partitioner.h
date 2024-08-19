

#ifndef __TBB_partitioner_H
#define __TBB_partitioner_H

#define __TBB_partitioner_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#ifndef __TBB_INITIAL_CHUNKS
#define __TBB_INITIAL_CHUNKS 2
#endif
#ifndef __TBB_RANGE_POOL_CAPACITY
#define __TBB_RANGE_POOL_CAPACITY 8
#endif
#ifndef __TBB_INIT_DEPTH
#define __TBB_INIT_DEPTH 5
#endif
#ifndef __TBB_DEMAND_DEPTH_ADD
#define __TBB_DEMAND_DEPTH_ADD 1
#endif
#ifndef __TBB_STATIC_THRESHOLD
#define __TBB_STATIC_THRESHOLD 40000
#endif
#if __TBB_DEFINE_MIC
#define __TBB_NONUNIFORM_TASK_CREATION 1
#ifdef __TBB_time_stamp
#define __TBB_USE_MACHINE_TIME_STAMPS 1
#define __TBB_task_duration() __TBB_STATIC_THRESHOLD
#endif 
#endif 

#include "task.h"
#include "task_arena.h"
#include "aligned_space.h"
#include "atomic.h"
#include "internal/_template_helpers.h"

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4244)
#endif

namespace tbb {

class auto_partitioner;
class simple_partitioner;
class static_partitioner;
class affinity_partitioner;

namespace interface9 {
namespace internal {
class affinity_partition_type;
}
}

namespace internal { 
size_t __TBB_EXPORTED_FUNC get_initial_auto_partitioner_divisor();

class affinity_partitioner_base_v3: no_copy {
friend class tbb::affinity_partitioner;
friend class tbb::interface9::internal::affinity_partition_type;

affinity_id* my_array;
size_t my_size;
affinity_partitioner_base_v3() : my_array(NULL), my_size(0) {}
~affinity_partitioner_base_v3() {resize(0);}

void __TBB_EXPORTED_METHOD resize( unsigned factor );
};

class partition_type_base {
public:
void set_affinity( task & ) {}
void note_affinity( task::affinity_id ) {}
task* continue_after_execute_range() {return NULL;}
bool decide_whether_to_delay() {return false;}
void spawn_or_delay( bool, task& b ) {
task::spawn(b);
}
};

template<typename Range, typename Body, typename Partitioner> class start_scan;

} 

namespace serial {
namespace interface9 {
template<typename Range, typename Body, typename Partitioner> class start_for;
}
}

namespace interface9 {
namespace internal {
using namespace tbb::internal;
template<typename Range, typename Body, typename Partitioner> class start_for;
template<typename Range, typename Body, typename Partitioner> class start_reduce;
template<typename Range, typename Body, typename Partitioner> class start_deterministic_reduce;

class flag_task: public task {
public:
tbb::atomic<bool> my_child_stolen;
flag_task() { my_child_stolen = false; }
task* execute() __TBB_override { return NULL; }
static void mark_task_stolen(task &t) {
tbb::atomic<bool> &flag = static_cast<flag_task*>(t.parent())->my_child_stolen;
#if TBB_USE_THREADING_TOOLS
flag.fetch_and_store<release>(true);
#else
flag = true;
#endif 
}
static bool is_peer_stolen(task &t) {
return static_cast<flag_task*>(t.parent())->my_child_stolen;
}
};

typedef unsigned char depth_t;

template <typename T, depth_t MaxCapacity>
class range_vector {
depth_t my_head;
depth_t my_tail;
depth_t my_size;
depth_t my_depth[MaxCapacity]; 
tbb::aligned_space<T, MaxCapacity> my_pool;

public:
range_vector(const T& elem) : my_head(0), my_tail(0), my_size(1) {
my_depth[0] = 0;
new( static_cast<void *>(my_pool.begin()) ) T(elem);
}
~range_vector() {
while( !empty() ) pop_back();
}
bool empty() const { return my_size == 0; }
depth_t size() const { return my_size; }
void split_to_fill(depth_t max_depth) {
while( my_size < MaxCapacity && is_divisible(max_depth) ) {
depth_t prev = my_head;
my_head = (my_head + 1) % MaxCapacity;
new(my_pool.begin()+my_head) T(my_pool.begin()[prev]); 
my_pool.begin()[prev].~T(); 
new(my_pool.begin()+prev) T(my_pool.begin()[my_head], split()); 
my_depth[my_head] = ++my_depth[prev];
my_size++;
}
}
void pop_back() {
__TBB_ASSERT(my_size > 0, "range_vector::pop_back() with empty size");
my_pool.begin()[my_head].~T();
my_size--;
my_head = (my_head + MaxCapacity - 1) % MaxCapacity;
}
void pop_front() {
__TBB_ASSERT(my_size > 0, "range_vector::pop_front() with empty size");
my_pool.begin()[my_tail].~T();
my_size--;
my_tail = (my_tail + 1) % MaxCapacity;
}
T& back() {
__TBB_ASSERT(my_size > 0, "range_vector::back() with empty size");
return my_pool.begin()[my_head];
}
T& front() {
__TBB_ASSERT(my_size > 0, "range_vector::front() with empty size");
return my_pool.begin()[my_tail];
}
depth_t front_depth() {
__TBB_ASSERT(my_size > 0, "range_vector::front_depth() with empty size");
return my_depth[my_tail];
}
depth_t back_depth() {
__TBB_ASSERT(my_size > 0, "range_vector::back_depth() with empty size");
return my_depth[my_head];
}
bool is_divisible(depth_t max_depth) {
return back_depth() < max_depth && back().is_divisible();
}
};

template <typename Partition>
struct partition_type_base {
typedef split split_type;
void set_affinity( task & ) {}
void note_affinity( task::affinity_id ) {}
bool check_being_stolen(task &) { return false; } 
bool check_for_demand(task &) { return false; }
bool is_divisible() { return true; } 
depth_t max_depth() { return 0; }
void align_depth(depth_t) { }
template <typename Range> split_type get_split() { return split(); }
Partition& self() { return *static_cast<Partition*>(this); } 

template<typename StartType, typename Range>
void work_balance(StartType &start, Range &range) {
start.run_body( range ); 
}

template<typename StartType, typename Range>
void execute(StartType &start, Range &range) {
if ( range.is_divisible() ) {
if ( self().is_divisible() ) {
do { 
typename Partition::split_type split_obj = self().template get_split<Range>();
start.offer_work( split_obj );
} while ( range.is_divisible() && self().is_divisible() );
}
}
self().work_balance(start, range);
}
};

template <typename Partition>
struct adaptive_mode : partition_type_base<Partition> {
typedef Partition my_partition;
size_t my_divisor;
static const unsigned factor = 1;
adaptive_mode() : my_divisor(tbb::internal::get_initial_auto_partitioner_divisor() / 4 * my_partition::factor) {}
adaptive_mode(adaptive_mode &src, split) : my_divisor(do_split(src, split())) {}

size_t do_split(adaptive_mode &src, split) {
return src.my_divisor /= 2u;
}
};


template <typename Range, typename = void>
struct proportion_helper {
static proportional_split get_split(size_t) { return proportional_split(1,1); }
};
template <typename Range>
struct proportion_helper<Range, typename enable_if<Range::is_splittable_in_proportion, void>::type> {
static proportional_split get_split(size_t n) {
#if __TBB_NONUNIFORM_TASK_CREATION
size_t right = (n + 2) / 3;
#else
size_t right = n / 2;
#endif
size_t left = n - right;
return proportional_split(left, right);
}
};

template <typename Partition>
struct proportional_mode : adaptive_mode<Partition> {
typedef Partition my_partition;
using partition_type_base<Partition>::self; 

proportional_mode() : adaptive_mode<Partition>() {}
proportional_mode(proportional_mode &src, split) : adaptive_mode<Partition>(src, split()) {}
proportional_mode(proportional_mode &src, const proportional_split& split_obj) { self().my_divisor = do_split(src, split_obj); }
size_t do_split(proportional_mode &src, const proportional_split& split_obj) {
#if __TBB_ENABLE_RANGE_FEEDBACK
size_t portion = size_t(float(src.my_divisor) * float(split_obj.right())
/ float(split_obj.left() + split_obj.right()) + 0.5f);
#else
size_t portion = split_obj.right() * my_partition::factor;
#endif
portion = (portion + my_partition::factor/2) & (0ul - my_partition::factor);
#if __TBB_ENABLE_RANGE_FEEDBACK

if (!portion)
portion = my_partition::factor;
else if (portion == src.my_divisor)
portion = src.my_divisor - my_partition::factor;
#endif
src.my_divisor -= portion;
return portion;
}
bool is_divisible() { 
return self().my_divisor > my_partition::factor;
}
template <typename Range>
proportional_split get_split() {
return proportion_helper<Range>::get_split( self().my_divisor / my_partition::factor );
}
};

static size_t get_initial_partition_head() {
int current_index = tbb::this_task_arena::current_thread_index();
if (current_index == tbb::task_arena::not_initialized)
current_index = 0;
return size_t(current_index);
}

template <typename Partition>
struct linear_affinity_mode : proportional_mode<Partition> {
size_t my_head;
size_t my_max_affinity;
using proportional_mode<Partition>::self;
linear_affinity_mode() : proportional_mode<Partition>(), my_head(get_initial_partition_head()),
my_max_affinity(self().my_divisor) {}
linear_affinity_mode(linear_affinity_mode &src, split) : proportional_mode<Partition>(src, split())
, my_head((src.my_head + src.my_divisor) % src.my_max_affinity), my_max_affinity(src.my_max_affinity) {}
linear_affinity_mode(linear_affinity_mode &src, const proportional_split& split_obj) : proportional_mode<Partition>(src, split_obj)
, my_head((src.my_head + src.my_divisor) % src.my_max_affinity), my_max_affinity(src.my_max_affinity) {}
void set_affinity( task &t ) {
if( self().my_divisor )
t.set_affinity( affinity_id(my_head) + 1 );
}
};


template<class Mode>
struct dynamic_grainsize_mode : Mode {
using Mode::self;
#ifdef __TBB_USE_MACHINE_TIME_STAMPS
tbb::internal::machine_tsc_t my_dst_tsc;
#endif
enum {
begin = 0,
run,
pass
} my_delay;
depth_t my_max_depth;
static const unsigned range_pool_size = __TBB_RANGE_POOL_CAPACITY;
dynamic_grainsize_mode(): Mode()
#ifdef __TBB_USE_MACHINE_TIME_STAMPS
, my_dst_tsc(0)
#endif
, my_delay(begin)
, my_max_depth(__TBB_INIT_DEPTH) {}
dynamic_grainsize_mode(dynamic_grainsize_mode& p, split)
: Mode(p, split())
#ifdef __TBB_USE_MACHINE_TIME_STAMPS
, my_dst_tsc(0)
#endif
, my_delay(pass)
, my_max_depth(p.my_max_depth) {}
dynamic_grainsize_mode(dynamic_grainsize_mode& p, const proportional_split& split_obj)
: Mode(p, split_obj)
#ifdef __TBB_USE_MACHINE_TIME_STAMPS
, my_dst_tsc(0)
#endif
, my_delay(begin)
, my_max_depth(p.my_max_depth) {}
bool check_being_stolen(task &t) { 
if( !(self().my_divisor / Mode::my_partition::factor) ) { 
self().my_divisor = 1; 
if( t.is_stolen_task() && t.parent()->ref_count() >= 2 ) { 
#if __TBB_USE_OPTIONAL_RTTI
__TBB_ASSERT(dynamic_cast<flag_task*>(t.parent()), 0);
#endif
flag_task::mark_task_stolen(t);
if( !my_max_depth ) my_max_depth++;
my_max_depth += __TBB_DEMAND_DEPTH_ADD;
return true;
}
}
return false;
}
depth_t max_depth() { return my_max_depth; }
void align_depth(depth_t base) {
__TBB_ASSERT(base <= my_max_depth, 0);
my_max_depth -= base;
}
template<typename StartType, typename Range>
void work_balance(StartType &start, Range &range) {
if( !range.is_divisible() || !self().max_depth() ) {
start.run_body( range ); 
}
else { 
internal::range_vector<Range, range_pool_size> range_pool(range);
do {
range_pool.split_to_fill(self().max_depth()); 
if( self().check_for_demand( start ) ) {
if( range_pool.size() > 1 ) {
start.offer_work( range_pool.front(), range_pool.front_depth() );
range_pool.pop_front();
continue;
}
if( range_pool.is_divisible(self().max_depth()) ) 
continue; 
}
start.run_body( range_pool.back() );
range_pool.pop_back();
} while( !range_pool.empty() && !start.is_cancelled() );
}
}
bool check_for_demand( task &t ) {
if( pass == my_delay ) {
if( self().my_divisor > 1 ) 
return true; 
else if( self().my_divisor && my_max_depth ) { 
self().my_divisor = 0; 
return true;
}
else if( flag_task::is_peer_stolen(t) ) {
my_max_depth += __TBB_DEMAND_DEPTH_ADD;
return true;
}
} else if( begin == my_delay ) {
#ifndef __TBB_USE_MACHINE_TIME_STAMPS
my_delay = pass;
#else
my_dst_tsc = __TBB_time_stamp() + __TBB_task_duration();
my_delay = run;
} else if( run == my_delay ) {
if( __TBB_time_stamp() < my_dst_tsc ) {
__TBB_ASSERT(my_max_depth > 0, NULL);
my_max_depth--; 
return false;
}
my_delay = pass;
return true;
#endif 
}
return false;
}
};

class auto_partition_type: public dynamic_grainsize_mode<adaptive_mode<auto_partition_type> > {
public:
auto_partition_type( const auto_partitioner& )
: dynamic_grainsize_mode<adaptive_mode<auto_partition_type> >() {
my_divisor *= __TBB_INITIAL_CHUNKS;
}
auto_partition_type( auto_partition_type& src, split)
: dynamic_grainsize_mode<adaptive_mode<auto_partition_type> >(src, split()) {}
bool is_divisible() { 
if( my_divisor > 1 ) return true;
if( my_divisor && my_max_depth ) { 
my_max_depth--;
my_divisor = 0; 
return true;
} else return false;
}
bool check_for_demand(task &t) {
if( flag_task::is_peer_stolen(t) ) {
my_max_depth += __TBB_DEMAND_DEPTH_ADD;
return true;
} else return false;
}
};

class simple_partition_type: public partition_type_base<simple_partition_type> {
public:
simple_partition_type( const simple_partitioner& ) {}
simple_partition_type( const simple_partition_type&, split ) {}
template<typename StartType, typename Range>
void execute(StartType &start, Range &range) {
split_type split_obj = split(); 
while( range.is_divisible() )
start.offer_work( split_obj );
start.run_body( range );
}
};

class static_partition_type : public linear_affinity_mode<static_partition_type> {
public:
typedef proportional_split split_type;
static_partition_type( const static_partitioner& )
: linear_affinity_mode<static_partition_type>() {}
static_partition_type( static_partition_type& p, split )
: linear_affinity_mode<static_partition_type>(p, split()) {}
static_partition_type( static_partition_type& p, const proportional_split& split_obj )
: linear_affinity_mode<static_partition_type>(p, split_obj) {}
};

class affinity_partition_type : public dynamic_grainsize_mode<linear_affinity_mode<affinity_partition_type> > {
static const unsigned factor_power = 4; 
tbb::internal::affinity_id* my_array;
public:
static const unsigned factor = 1 << factor_power; 
typedef proportional_split split_type;
affinity_partition_type( tbb::internal::affinity_partitioner_base_v3& ap )
: dynamic_grainsize_mode<linear_affinity_mode<affinity_partition_type> >() {
__TBB_ASSERT( (factor&(factor-1))==0, "factor must be power of two" );
ap.resize(factor);
my_array = ap.my_array;
my_max_depth = factor_power + 1;
__TBB_ASSERT( my_max_depth < __TBB_RANGE_POOL_CAPACITY, 0 );
}
affinity_partition_type(affinity_partition_type& p, split)
: dynamic_grainsize_mode<linear_affinity_mode<affinity_partition_type> >(p, split())
, my_array(p.my_array) {}
affinity_partition_type(affinity_partition_type& p, const proportional_split& split_obj)
: dynamic_grainsize_mode<linear_affinity_mode<affinity_partition_type> >(p, split_obj)
, my_array(p.my_array) {}
void set_affinity( task &t ) {
if( my_divisor ) {
if( !my_array[my_head] )
t.set_affinity( affinity_id(my_head / factor + 1) );
else
t.set_affinity( my_array[my_head] );
}
}
void note_affinity( task::affinity_id id ) {
if( my_divisor )
my_array[my_head] = id;
}
};

class old_auto_partition_type: public tbb::internal::partition_type_base {
size_t num_chunks;
static const size_t VICTIM_CHUNKS = 4;
public:
bool should_execute_range(const task &t) {
if( num_chunks<VICTIM_CHUNKS && t.is_stolen_task() )
num_chunks = VICTIM_CHUNKS;
return num_chunks==1;
}
old_auto_partition_type( const auto_partitioner& )
: num_chunks(internal::get_initial_auto_partitioner_divisor()*__TBB_INITIAL_CHUNKS/4) {}
old_auto_partition_type( const affinity_partitioner& )
: num_chunks(internal::get_initial_auto_partitioner_divisor()*__TBB_INITIAL_CHUNKS/4) {}
old_auto_partition_type( old_auto_partition_type& pt, split ) {
num_chunks = pt.num_chunks = (pt.num_chunks+1u) / 2u;
}
};

} 
} 


class simple_partitioner {
public:
simple_partitioner() {}
private:
template<typename Range, typename Body, typename Partitioner> friend class serial::interface9::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_reduce;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_deterministic_reduce;
template<typename Range, typename Body, typename Partitioner> friend class internal::start_scan;
class partition_type: public internal::partition_type_base {
public:
bool should_execute_range(const task& ) {return false;}
partition_type( const simple_partitioner& ) {}
partition_type( const partition_type&, split ) {}
};
typedef interface9::internal::simple_partition_type task_partition_type;

typedef interface9::internal::simple_partition_type::split_type split_type;
};


class auto_partitioner {
public:
auto_partitioner() {}

private:
template<typename Range, typename Body, typename Partitioner> friend class serial::interface9::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_reduce;
template<typename Range, typename Body, typename Partitioner> friend class internal::start_scan;
typedef interface9::internal::old_auto_partition_type partition_type;
typedef interface9::internal::auto_partition_type task_partition_type;

typedef interface9::internal::auto_partition_type::split_type split_type;
};

class static_partitioner {
public:
static_partitioner() {}
private:
template<typename Range, typename Body, typename Partitioner> friend class serial::interface9::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_reduce;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_deterministic_reduce;
template<typename Range, typename Body, typename Partitioner> friend class internal::start_scan;
typedef interface9::internal::old_auto_partition_type partition_type;
typedef interface9::internal::static_partition_type task_partition_type;

typedef interface9::internal::static_partition_type::split_type split_type;
};

class affinity_partitioner: internal::affinity_partitioner_base_v3 {
public:
affinity_partitioner() {}

private:
template<typename Range, typename Body, typename Partitioner> friend class serial::interface9::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_for;
template<typename Range, typename Body, typename Partitioner> friend class interface9::internal::start_reduce;
template<typename Range, typename Body, typename Partitioner> friend class internal::start_scan;
typedef interface9::internal::old_auto_partition_type partition_type;
typedef interface9::internal::affinity_partition_type task_partition_type;

typedef interface9::internal::affinity_partition_type::split_type split_type;
};

} 

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif 
#undef __TBB_INITIAL_CHUNKS
#undef __TBB_RANGE_POOL_CAPACITY
#undef __TBB_INIT_DEPTH

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_partitioner_H_include_area

#endif 
