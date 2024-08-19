#pragma once

#include "op.h"

namespace ctranslate2 {
namespace ops {

enum class ActivationType {
ReLU,
GELUTanh,
Swish,
GELU,
GELUSigmoid,
};

const UnaryOp& get_activation_op(ActivationType type);

}
}
