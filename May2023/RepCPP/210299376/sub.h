#pragma once

#include "op.h"

namespace ctranslate2 {
namespace ops {

class Sub : public BinaryOp {
public:
void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;

private:
template <Device D, typename T>
void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
c.resize_as(a);
if (b.is_scalar()) {
primitives<D>::sub(b.data<T>()[0], a.data<T>(), c.data<T>(), c.size());
} else {
primitives<D>::sub(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
}
}
};

}
}
