

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
tagaspi_read(const gaspi_segment_id_t segment_id_local,
const gaspi_offset_t offset_local,
const gaspi_rank_t rank,
const gaspi_segment_id_t segment_id_remote,
const gaspi_offset_t offset_remote,
const gaspi_size_t size,
const gaspi_queue_id_t queue)
{
assert(_env.enabled);
gaspi_return_t eret;

void *counter = TaskingModel::getCurrentEventCounter();
assert(counter != NULL);

gaspi_tag_t tag = (gaspi_tag_t) counter;

gaspi_number_t numRequests = _env.numRequests[Operation::READ];
assert(numRequestes > 0);

TaskingModel::increaseCurrentTaskEventCounter(counter, numRequests);

eret = gaspi_operation_submit(GASPI_OP_READ, tag,
segment_id_local, offset_local, rank,
segment_id_remote, offset_remote, size,
0, 0, queue, GASPI_BLOCK);
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
