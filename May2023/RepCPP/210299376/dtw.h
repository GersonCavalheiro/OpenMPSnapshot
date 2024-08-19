#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

std::vector<std::pair<dim_t, dim_t>> negative_dtw(const StorageView& x);

}
