#pragma once

#include <cassert>
#include <vector>

#include "ctranslate2/storage_view.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/profiler.h"

namespace ctranslate2 {
namespace ops {

class Op {
public:
virtual ~Op() = default;
};


class UnaryOp : public Op {
public:
virtual void operator()(const StorageView&, StorageView&) const = 0;
};

class BinaryOp : public Op {
public:
virtual void operator()(const StorageView&, const StorageView&, StorageView&) const = 0;
};

class TernaryOp : public Op {
public:
virtual void operator()(const StorageView&,
const StorageView&,
const StorageView&,
StorageView&) const = 0;
};

}
}
