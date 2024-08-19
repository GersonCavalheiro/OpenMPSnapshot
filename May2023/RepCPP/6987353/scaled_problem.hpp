#ifndef AMGCL_ADAPTER_SCALED_PROBLEM_HPP
#define AMGCL_ADAPTER_SCALED_PROBLEM_HPP





#include <vector>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/interface.hpp>

namespace amgcl {
namespace adapter {

template <class Matrix, class Scale>
struct scaled_matrix {
typedef typename backend::value_type<Matrix>::type value_type;
typedef typename backend::value_type<Scale>::type  scale_type;

const Matrix &A;
const Scale  &s;

scaled_matrix(const Matrix &A, const Scale &s) : A(A), s(s) {}

size_t rows()     const { return backend::rows(A);     }
size_t cols()     const { return backend::cols(A);     }
size_t nonzeros() const { return backend::nonzeros(A); }

struct row_iterator : public backend::row_iterator<Matrix>::type {
typedef typename backend::row_iterator<Matrix>::type Base;

scale_type  si;
const Scale &s;

row_iterator(const Matrix &A, const Scale &s, size_t i)
: Base(A, i), si(s[i]), s(s) {}

value_type value() const {
return si * static_cast<const Base*>(this)->value() * s[this->col()];
}
};

row_iterator row_begin(size_t i) const {
return row_iterator(A, s, i);
}
};

template <class Backend, class Scale>
struct scaled_problem {
typedef typename Backend::params backend_params;

const std::shared_ptr<Scale> s;
const backend_params &bprm;

scaled_problem(std::shared_ptr<Scale> s, const backend_params &bprm = backend_params())
: s(s), bprm(bprm) {}

template <class Matrix>
scaled_matrix<Matrix, Scale> matrix(const Matrix &A) const {
return scaled_matrix<Matrix, Scale>(A, *s);
}

template <class Vector>
std::shared_ptr<typename Backend::vector> rhs(const Vector &v) const {
auto t = Backend::copy_vector(v, bprm);
(*this)(*t);
return t;
}

template <class Vector>
void operator()(Vector &x) const {
typedef typename backend::value_type<Vector>::type value_type;
typedef typename math::scalar_of<value_type>::type scalar_type;

const auto one  = math::identity<scalar_type>();
const auto zero = math::zero<scalar_type>();

if (backend::is_builtin_vector<Vector>::value) {
backend::vmul(one, *s, x, zero, x);
} else {
backend::vmul(one, *Backend::copy_vector(*s, bprm), x, zero, x);
}
}
};

template <class Backend, class Matrix>
scaled_problem<
Backend,
std::vector<
typename math::scalar_of<
typename backend::value_type<Matrix>::type
>::type>
>
scale_diagonal(
const Matrix &A,
const typename Backend::params &bprm = typename Backend::params()
)
{
typedef typename backend::value_type<Matrix>::type value_type;
typedef typename math::scalar_of<value_type>::type scalar_type;
ptrdiff_t n = backend::rows(A);
auto      s = std::make_shared<std::vector<scalar_type>>(n);

#pragma omp parallel for
for(ptrdiff_t i = 0; i < n; ++i) {
for(auto a = backend::row_begin(A, i); a; ++a) {
if (a.col() == i) {
(*s)[i] = math::inverse(sqrt(math::norm(a.value())));
break;
}
}
}

return scaled_problem<Backend, std::vector<scalar_type>>(s, bprm);
}

} 

} 

#endif
