

#include "core/stop/residual_norm_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace omp {

namespace residual_norm {


template <typename ValueType>
void residual_norm(std::shared_ptr<const OmpExecutor> exec,
const matrix::Dense<ValueType>* tau,
const matrix::Dense<ValueType>* orig_tau,
ValueType rel_residual_goal, uint8 stoppingId,
bool setFinalized, array<stopping_status>* stop_status,
array<bool>* device_storage, bool* all_converged,
bool* one_changed)
{
static_assert(is_complex_s<ValueType>::value == false,
"ValueType must not be complex in this function!");
bool local_one_changed = false;
#pragma omp parallel for reduction(|| : local_one_changed)
for (size_type i = 0; i < tau->get_size()[1]; ++i) {
if (tau->at(i) < rel_residual_goal * orig_tau->at(i)) {
stop_status->get_data()[i].converge(stoppingId, setFinalized);
local_one_changed = true;
}
}
*one_changed = local_one_changed;
bool local_all_converged = true;
#pragma omp parallel for reduction(&& : local_all_converged)
for (size_type i = 0; i < stop_status->get_num_elems(); ++i) {
if (!stop_status->get_const_data()[i].has_stopped()) {
local_all_converged = false;
}
}
*all_converged = local_all_converged;
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
GKO_DECLARE_RESIDUAL_NORM_KERNEL);


}  



namespace implicit_residual_norm {


template <typename ValueType>
void implicit_residual_norm(
std::shared_ptr<const OmpExecutor> exec,
const matrix::Dense<ValueType>* tau,
const matrix::Dense<remove_complex<ValueType>>* orig_tau,
remove_complex<ValueType> rel_residual_goal, uint8 stoppingId,
bool setFinalized, array<stopping_status>* stop_status,
array<bool>* device_storage, bool* all_converged, bool* one_changed)
{
bool local_one_changed = false;
#pragma omp parallel for reduction(|| : local_one_changed)
for (size_type i = 0; i < tau->get_size()[1]; ++i) {
if (sqrt(abs(tau->at(i))) < rel_residual_goal * orig_tau->at(i)) {
stop_status->get_data()[i].converge(stoppingId, setFinalized);
local_one_changed = true;
}
}
*one_changed = local_one_changed;
bool local_all_converged = true;
#pragma omp parallel for reduction(&& : local_all_converged)
for (size_type i = 0; i < stop_status->get_num_elems(); ++i) {
if (!stop_status->get_const_data()[i].has_stopped()) {
local_all_converged = false;
}
}
*all_converged = local_all_converged;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL);


}  
}  
}  
}  
