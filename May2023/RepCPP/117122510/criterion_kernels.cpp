

#include "core/stop/criterion_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace omp {

namespace set_all_statuses {


void set_all_statuses(std::shared_ptr<const OmpExecutor> exec, uint8 stoppingId,
bool setFinalized, array<stopping_status>* stop_status)
{
#pragma omp parallel for
for (int i = 0; i < stop_status->get_num_elems(); i++) {
stop_status->get_data()[i].stop(stoppingId, setFinalized);
}
}


}  
}  
}  
}  
