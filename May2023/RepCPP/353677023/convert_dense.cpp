#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void Dense<T>::convert(const COO<T> &coo) {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

set_row(coo.get_row());
set_col(coo.get_col());
val_create_flag = true;
resize(get_row() * get_col());

#pragma omp parallel for
for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
data()[i] = 0.0;
}

for (auto i = decltype(coo.get_nnz()){0}; i < coo.get_nnz(); i++) {
insert(coo.row_index[i], coo.col_index[i], coo.data()[i]);
}
logger.util_out();
}
template void Dense<double>::convert(const COO<double> &coo);
template void Dense<float>::convert(const COO<float> &coo);

template <typename T> void Dense<T>::convert(const Dense<T> &mat) {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

val_create_flag = true;
resize(mat.get_nnz());

rowN = mat.get_row();
colN = mat.get_col();

#if MONOLISH_USE_NVIDIA_GPU
if (mat.get_device_mem_stat()) {
throw std::runtime_error(
"error can not convert Dense->Dense when gpu_status == true");
}
#endif
internal::vcopy(get_nnz(), mat.data(), data(), false);

logger.util_out();
}
template void Dense<double>::convert(const Dense<double> &mat);
template void Dense<float>::convert(const Dense<float> &mat);

} 
} 
