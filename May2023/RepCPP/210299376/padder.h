#pragma once

#include "ops/gather.h"
#include "storage_view.h"

namespace ctranslate2 {

class Padder {
public:
static inline bool allow_padding_removal(const Device device,
const ComputeType compute_type) {
return device == Device::CPU || compute_type != ComputeType::FLOAT16;
}

Padder(const StorageView& lengths,
const dim_t max_time = -1,
const dim_t pad_batch_to_multiple = 1);

void remove_padding(StorageView& x) const;

void add_padding(StorageView& x) const;

private:
dim_t _batch_size;
dim_t _max_time;
StorageView _padded_to_flat;
StorageView _flat_to_padded;
const ops::Gather _gather_op;
};

}
