

#include <GASPI.h>
#include <GASPI_Lowlevel.h>

#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"

#include <cassert>

using namespace tagaspi;

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

gaspi_return_t
tagaspi_write_list(const gaspi_number_t num,
gaspi_segment_id_t * const segment_id_local,
gaspi_offset_t * const offset_local,
const gaspi_rank_t rank,
gaspi_segment_id_t * const segment_id_remote,
gaspi_offset_t * const offset_remote,
gaspi_size_t * const size,
const gaspi_queue_id_t queue)
{
assert(_env.enabled);
gaspi_return_t eret;

void *counter = TaskingModel::getCurrentEventCounter();
assert(counter != NULL);

gaspi_tag_t tag = (gaspi_tag_t) counter;

gaspi_number_t numRequests = 0;
eret = gaspi_operation_get_num_requests(GASPI_OP_WRITE_LIST, num, &numRequests);
assert(eret == GASPI_SUCCESS);
assert(numRequests > 0);

TaskingModel::increaseCurrentTaskEventCounter(counter, numRequests);

eret = gaspi_operation_list_submit(GASPI_OP_WRITE_LIST, tag,
num, segment_id_local, offset_local, rank,
segment_id_remote, offset_remote, size,
0, 0, 0, queue, GASPI_BLOCK);
assert(eret != GASPI_TIMEOUT);

if (eret != GASPI_SUCCESS) {
TaskingModel::decreaseTaskEventCounter(counter, numRequests);
}

return eret;
}

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop
