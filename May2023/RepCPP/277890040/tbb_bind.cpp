

#include "../tbb/tbb_assert_impl.h" 
#include "tbb/tbb_stddef.h"

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning( push )
#pragma warning( disable : 4100 )
#endif
#include <hwloc.h>
#if _MSC_VER && !__INTEL_COMPILER
#pragma warning( pop )
#endif

#include <vector>

#define assertion_hwloc_wrapper(command, ...) \
__TBB_ASSERT_EX( (command(__VA_ARGS__)) >= 0, "Error occurred during call to hwloc API.");

namespace tbb {
namespace internal {

class platform_topology {
friend class numa_affinity_handler;

static hwloc_topology_t topology;
static hwloc_cpuset_t   process_cpu_affinity_mask;
static hwloc_nodeset_t  process_node_affinity_mask;
static std::vector<hwloc_cpuset_t>  affinity_masks_list;

static std::vector<int> default_concurrency_list;
static std::vector<int> numa_indexes_list;
static int  numa_nodes_count;

enum init_stages { uninitialized, started, topology_allocated, topology_loaded, topology_parsed };
static init_stages initialization_state;

static bool intergroup_binding_allowed(size_t groups_num) { return groups_num > 1; }

public:
typedef hwloc_cpuset_t             affinity_mask;
typedef hwloc_const_cpuset_t const_affinity_mask;

static bool is_topology_parsed() { return initialization_state == topology_parsed; }

static void initialize( size_t groups_num ) {
if ( initialization_state != uninitialized )
return;
initialization_state = started;

if ( hwloc_topology_init( &topology ) == 0 ) {
initialization_state = topology_allocated;
if ( hwloc_topology_load( topology ) == 0 ) {
initialization_state = topology_loaded;
}
}

if ( initialization_state != topology_loaded ) {
if ( initialization_state == topology_allocated ) {
hwloc_topology_destroy(topology);
}
numa_nodes_count = 1;
numa_indexes_list.push_back(-1);
default_concurrency_list.push_back(-1);
return;
}

if ( intergroup_binding_allowed(groups_num) ) {
process_cpu_affinity_mask  = hwloc_bitmap_dup(hwloc_topology_get_complete_cpuset (topology));
process_node_affinity_mask = hwloc_bitmap_dup(hwloc_topology_get_complete_nodeset(topology));
} else {
process_cpu_affinity_mask  = hwloc_bitmap_alloc();
process_node_affinity_mask = hwloc_bitmap_alloc();

assertion_hwloc_wrapper(hwloc_get_cpubind, topology, process_cpu_affinity_mask, 0);
hwloc_cpuset_to_nodeset(topology, process_cpu_affinity_mask, process_node_affinity_mask);
}

if (hwloc_bitmap_weight(process_node_affinity_mask) < 0) {
numa_nodes_count = 1;
numa_indexes_list.push_back(0);
default_concurrency_list.push_back(hwloc_bitmap_weight(process_cpu_affinity_mask));

affinity_masks_list.push_back(hwloc_bitmap_dup(process_cpu_affinity_mask));
initialization_state = topology_parsed;
return;
}

numa_nodes_count = hwloc_bitmap_weight(process_node_affinity_mask);
__TBB_ASSERT(numa_nodes_count > 0, "Any system must contain one or more NUMA nodes");

unsigned counter = 0;
int i = 0;
int max_numa_index = -1;
numa_indexes_list.resize(numa_nodes_count);
hwloc_obj_t node_buffer;
hwloc_bitmap_foreach_begin(i, process_node_affinity_mask) {
node_buffer = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
numa_indexes_list[counter] = static_cast<int>(node_buffer->logical_index);

if ( numa_indexes_list[counter] > max_numa_index ) {
max_numa_index = numa_indexes_list[counter];
}

counter++;
} hwloc_bitmap_foreach_end();
__TBB_ASSERT(max_numa_index >= 0, "Maximal NUMA index must not be negative");

default_concurrency_list.resize(max_numa_index + 1);
affinity_masks_list.resize(max_numa_index + 1);

int index = 0;
hwloc_bitmap_foreach_begin(i, process_node_affinity_mask) {
node_buffer = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
index = static_cast<int>(node_buffer->logical_index);

hwloc_cpuset_t& current_mask = affinity_masks_list[index];
current_mask = hwloc_bitmap_dup(node_buffer->cpuset);

hwloc_bitmap_and(current_mask, current_mask, process_cpu_affinity_mask);
__TBB_ASSERT(!hwloc_bitmap_iszero(current_mask), "hwloc detected unavailable NUMA node");
default_concurrency_list[index] = hwloc_bitmap_weight(current_mask);
} hwloc_bitmap_foreach_end();
initialization_state = topology_parsed;
}

~platform_topology() {
if ( is_topology_parsed() ) {
for (int i = 0; i < numa_nodes_count; i++) {
hwloc_bitmap_free(affinity_masks_list[numa_indexes_list[i]]);
}
hwloc_bitmap_free(process_node_affinity_mask);
hwloc_bitmap_free(process_cpu_affinity_mask);
}

if ( initialization_state >= topology_allocated ) {
hwloc_topology_destroy(topology);
}

initialization_state = uninitialized;
}

static void fill(int& nodes_count, int*& indexes_list, int*& concurrency_list ) {
__TBB_ASSERT(is_topology_parsed(), "Trying to get access to uninitialized platform_topology");
nodes_count = numa_nodes_count;
indexes_list = &numa_indexes_list.front();
concurrency_list = &default_concurrency_list.front();
}

static affinity_mask allocate_process_affinity_mask() {
__TBB_ASSERT(is_topology_parsed(), "Trying to get access to uninitialized platform_topology");
return hwloc_bitmap_dup(process_cpu_affinity_mask);
}

static void free_affinity_mask( affinity_mask mask_to_free ) {
hwloc_bitmap_free(mask_to_free); 
}

static void store_current_affinity_mask( affinity_mask current_mask ) {
assertion_hwloc_wrapper(hwloc_get_cpubind, topology, current_mask, HWLOC_CPUBIND_THREAD);

hwloc_bitmap_and(current_mask, current_mask, process_cpu_affinity_mask);
__TBB_ASSERT(!hwloc_bitmap_iszero(current_mask),
"Current affinity mask must intersects with process affinity mask");
}

static void set_new_affinity_mask( const_affinity_mask new_mask ) {
assertion_hwloc_wrapper(hwloc_set_cpubind, topology, new_mask, HWLOC_CPUBIND_THREAD);
}

static const_affinity_mask get_node_affinity_mask( int node_index ) {
__TBB_ASSERT((int)affinity_masks_list.size() > node_index,
"Trying to get affinity mask for uninitialized NUMA node");
return affinity_masks_list[node_index];
}
};

hwloc_topology_t platform_topology::topology = NULL;
hwloc_cpuset_t   platform_topology::process_cpu_affinity_mask = NULL;
hwloc_nodeset_t  platform_topology::process_node_affinity_mask = NULL;
std::vector<hwloc_cpuset_t>  platform_topology::affinity_masks_list;

std::vector<int> platform_topology::default_concurrency_list;
std::vector<int> platform_topology::numa_indexes_list;
int  platform_topology::numa_nodes_count = 0;

platform_topology::init_stages platform_topology::initialization_state = uninitialized;

class binding_handler {
typedef std::vector<platform_topology::affinity_mask> affinity_masks_container;
affinity_masks_container affinity_backup;

public:
binding_handler( size_t size ) : affinity_backup(size) {
for (affinity_masks_container::iterator it = affinity_backup.begin();
it != affinity_backup.end(); it++) {
*it = platform_topology::allocate_process_affinity_mask();
}
}

~binding_handler() {
for (affinity_masks_container::iterator it = affinity_backup.begin();
it != affinity_backup.end(); it++) {
platform_topology::free_affinity_mask(*it);
}
}

void bind_thread_to_node( unsigned slot_num, unsigned numa_node_id ) {
__TBB_ASSERT(slot_num < affinity_backup.size(),
"The slot number is greater than the number of slots in the arena");
__TBB_ASSERT(platform_topology::is_topology_parsed(),
"Trying to get access to uninitialized platform_topology");
platform_topology::store_current_affinity_mask(affinity_backup[slot_num]);

platform_topology::set_new_affinity_mask(
platform_topology::get_node_affinity_mask(numa_node_id));
}

void restore_previous_affinity_mask( unsigned slot_num ) {
__TBB_ASSERT(platform_topology::is_topology_parsed(),
"Trying to get access to uninitialized platform_topology");
platform_topology::set_new_affinity_mask(affinity_backup[slot_num]);
};

};

extern "C" { 

void initialize_numa_topology( size_t groups_num,
int& nodes_count, int*& indexes_list, int*& concurrency_list ) {
platform_topology::initialize(groups_num);
platform_topology::fill(nodes_count, indexes_list, concurrency_list);
}

binding_handler* allocate_binding_handler(int slot_num) {
__TBB_ASSERT(slot_num > 0, "Trying to create numa handler for 0 threads.");
return new binding_handler(slot_num);
}

void deallocate_binding_handler(binding_handler* handler_ptr) {
__TBB_ASSERT(handler_ptr != NULL, "Trying to deallocate NULL pointer.");
delete handler_ptr;
}

void bind_to_node(binding_handler* handler_ptr, int slot_num, int numa_id) {
__TBB_ASSERT(handler_ptr != NULL, "Trying to get access to uninitialized metadata.");
__TBB_ASSERT(platform_topology::is_topology_parsed(), "Trying to get access "
"to uninitialized platform_topology.");
handler_ptr->bind_thread_to_node(slot_num, numa_id);
}

void restore_affinity(binding_handler* handler_ptr, int slot_num) {
__TBB_ASSERT(handler_ptr != NULL, "Trying to get access to uninitialized metadata.");
__TBB_ASSERT(platform_topology::is_topology_parsed(), "Trying to get access "
"to uninitialized platform_topology.");
handler_ptr->restore_previous_affinity_mask(slot_num);
}

} 

} 
} 

#undef assertion_hwloc_wrapper
