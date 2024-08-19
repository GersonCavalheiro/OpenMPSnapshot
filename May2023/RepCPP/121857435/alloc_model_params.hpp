#pragma once

#include "defs.hpp"
#include <vector>

namespace bnmf_algs {
namespace alloc_model {


template <typename Scalar> struct Params { 
Scalar a;

Scalar b;

std::vector<Scalar> alpha;

std::vector<Scalar> beta;


Params(Scalar a, Scalar b, const std::vector<Scalar>& alpha,
const std::vector<Scalar>& beta)
: a(a), b(b), alpha(alpha), beta(beta) {}


explicit Params(const shape<3>& tensor_shape)
: a(static_cast<Scalar>(1)), b(static_cast<Scalar>(10)),
alpha(tensor_shape[0], static_cast<Scalar>(1)),
beta(tensor_shape[2], static_cast<Scalar>(1)) {}

Params() = default;
};


template <typename Scalar>
Params<Scalar> make_bld_params(const shape<3>& tensor_shape) {
return Params<Scalar>(tensor_shape);
}


template <typename Scalar>
std::vector<Params<Scalar>> make_EM_params(const shape<3>& tensor_shape) {
Params<Scalar> bld_params(tensor_shape);
bld_params.beta =
std::vector<Scalar>(tensor_shape[1], static_cast<Scalar>(1));

return std::vector<Params<Scalar>>(tensor_shape[2], bld_params);
}

} 
} 
