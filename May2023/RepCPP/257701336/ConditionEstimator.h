
#ifndef EIGEN_CONDITIONESTIMATOR_H
#define EIGEN_CONDITIONESTIMATOR_H

namespace Eigen {

namespace internal {

template <typename Vector, typename RealVector, bool IsComplex>
struct rcond_compute_sign {
static inline Vector run(const Vector& v) {
const RealVector v_abs = v.cwiseAbs();
return (v_abs.array() == static_cast<typename Vector::RealScalar>(0))
.select(Vector::Ones(v.size()), v.cwiseQuotient(v_abs));
}
};

template <typename Vector>
struct rcond_compute_sign<Vector, Vector, false> {
static inline Vector run(const Vector& v) {
return (v.array() < static_cast<typename Vector::RealScalar>(0))
.select(-Vector::Ones(v.size()), Vector::Ones(v.size()));
}
};


template <typename Decomposition>
typename Decomposition::RealScalar rcond_invmatrix_L1_norm_estimate(const Decomposition& dec)
{
typedef typename Decomposition::MatrixType MatrixType;
typedef typename Decomposition::Scalar Scalar;
typedef typename Decomposition::RealScalar RealScalar;
typedef typename internal::plain_col_type<MatrixType>::type Vector;
typedef typename internal::plain_col_type<MatrixType, RealScalar>::type RealVector;
const bool is_complex = (NumTraits<Scalar>::IsComplex != 0);

eigen_assert(dec.rows() == dec.cols());
const Index n = dec.rows();
if (n == 0)
return 0;

#ifdef __INTEL_COMPILER
#pragma warning push
#pragma warning ( disable : 2259 )
#endif
Vector v = dec.solve(Vector::Ones(n) / Scalar(n));
#ifdef __INTEL_COMPILER
#pragma warning pop
#endif

RealScalar lower_bound = v.template lpNorm<1>();
if (n == 1)
return lower_bound;

RealScalar old_lower_bound = lower_bound;
Vector sign_vector(n);
Vector old_sign_vector;
Index v_max_abs_index = -1;
Index old_v_max_abs_index = v_max_abs_index;
for (int k = 0; k < 4; ++k)
{
sign_vector = internal::rcond_compute_sign<Vector, RealVector, is_complex>::run(v);
if (k > 0 && !is_complex && sign_vector == old_sign_vector) {
break;
}
v = dec.adjoint().solve(sign_vector);
v.real().cwiseAbs().maxCoeff(&v_max_abs_index);
if (v_max_abs_index == old_v_max_abs_index) {
break;
}
v = dec.solve(Vector::Unit(n, v_max_abs_index));  
lower_bound = v.template lpNorm<1>();
if (lower_bound <= old_lower_bound) {
break;
}
if (!is_complex) {
old_sign_vector = sign_vector;
}
old_v_max_abs_index = v_max_abs_index;
old_lower_bound = lower_bound;
}
Scalar alternating_sign(RealScalar(1));
for (Index i = 0; i < n; ++i) {
v[i] = alternating_sign * static_cast<RealScalar>(RealScalar(1) + (RealScalar(i) / (RealScalar(n - 1))));
alternating_sign = -alternating_sign;
}
v = dec.solve(v);
const RealScalar alternate_lower_bound = (2 * v.template lpNorm<1>()) / (3 * RealScalar(n));
return numext::maxi(lower_bound, alternate_lower_bound);
}


template <typename Decomposition>
typename Decomposition::RealScalar
rcond_estimate_helper(typename Decomposition::RealScalar matrix_norm, const Decomposition& dec)
{
typedef typename Decomposition::RealScalar RealScalar;
eigen_assert(dec.rows() == dec.cols());
if (dec.rows() == 0)              return NumTraits<RealScalar>::infinity();
if (matrix_norm == RealScalar(0)) return RealScalar(0);
if (dec.rows() == 1)              return RealScalar(1);
const RealScalar inverse_matrix_norm = rcond_invmatrix_L1_norm_estimate(dec);
return (inverse_matrix_norm == RealScalar(0) ? RealScalar(0)
: (RealScalar(1) / inverse_matrix_norm) / matrix_norm);
}

}  

}  

#endif
