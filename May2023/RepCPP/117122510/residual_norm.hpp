

#ifndef GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_
#define GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace stop {



enum class mode { absolute, initial_resnorm, rhs_norm };



template <typename ValueType>
class ResidualNormBase
: public EnablePolymorphicObject<ResidualNormBase<ValueType>, Criterion> {
friend class EnablePolymorphicObject<ResidualNormBase<ValueType>,
Criterion>;

protected:
using absolute_type = remove_complex<ValueType>;
using ComplexVector = matrix::Dense<to_complex<ValueType>>;
using NormVector = matrix::Dense<absolute_type>;
using Vector = matrix::Dense<ValueType>;
bool check_impl(uint8 stoppingId, bool setFinalized,
array<stopping_status>* stop_status, bool* one_changed,
const Criterion::Updater& updater) override;

explicit ResidualNormBase(std::shared_ptr<const gko::Executor> exec)
: EnablePolymorphicObject<ResidualNormBase, Criterion>(exec),
device_storage_{exec, 2}
{}

explicit ResidualNormBase(std::shared_ptr<const gko::Executor> exec,
const CriterionArgs& args,
absolute_type reduction_factor, mode baseline);

remove_complex<ValueType> reduction_factor_{};
std::unique_ptr<NormVector> starting_tau_{};
std::unique_ptr<NormVector> u_dense_tau_{};

array<bool> device_storage_;

private:
mode baseline_{mode::rhs_norm};
std::shared_ptr<const LinOp> system_matrix_{};
std::shared_ptr<const LinOp> b_{};

std::shared_ptr<const Vector> one_{};
std::shared_ptr<const Vector> neg_one_{};
};



template <typename ValueType = default_precision>
class ResidualNorm : public ResidualNormBase<ValueType> {
public:
using ComplexVector = matrix::Dense<to_complex<ValueType>>;
using NormVector = matrix::Dense<remove_complex<ValueType>>;
using Vector = matrix::Dense<ValueType>;

GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
{

remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
reduction_factor, static_cast<remove_complex<ValueType>>(1e-15));


mode GKO_FACTORY_PARAMETER_SCALAR(baseline, mode::rhs_norm);
};
GKO_ENABLE_CRITERION_FACTORY(ResidualNorm<ValueType>, parameters, Factory);
GKO_ENABLE_BUILD_METHOD(Factory);

protected:
explicit ResidualNorm(std::shared_ptr<const gko::Executor> exec)
: ResidualNormBase<ValueType>(exec)
{}

explicit ResidualNorm(const Factory* factory, const CriterionArgs& args)
: ResidualNormBase<ValueType>(
factory->get_executor(), args,
factory->get_parameters().reduction_factor,
factory->get_parameters().baseline),
parameters_{factory->get_parameters()}
{}
};



template <typename ValueType = default_precision>
class ImplicitResidualNorm : public ResidualNormBase<ValueType> {
public:
using ComplexVector = matrix::Dense<to_complex<ValueType>>;
using NormVector = matrix::Dense<remove_complex<ValueType>>;
using Vector = matrix::Dense<ValueType>;

GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
{

remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
reduction_factor, static_cast<remove_complex<ValueType>>(1e-15));


mode GKO_FACTORY_PARAMETER_SCALAR(baseline, mode::rhs_norm);
};
GKO_ENABLE_CRITERION_FACTORY(ImplicitResidualNorm<ValueType>, parameters,
Factory);
GKO_ENABLE_BUILD_METHOD(Factory);

protected:
bool check_impl(uint8 stoppingId, bool setFinalized,
array<stopping_status>* stop_status, bool* one_changed,
const Criterion::Updater& updater) override;

explicit ImplicitResidualNorm(std::shared_ptr<const gko::Executor> exec)
: ResidualNormBase<ValueType>(exec)
{}

explicit ImplicitResidualNorm(const Factory* factory,
const CriterionArgs& args)
: ResidualNormBase<ValueType>(
factory->get_executor(), args,
factory->get_parameters().reduction_factor,
factory->get_parameters().baseline),
parameters_{factory->get_parameters()}
{}
};


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_BUILD) || defined(__INTEL_LLVM_COMPILER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif



template <typename ValueType = default_precision>
class [[deprecated(
"Please use the class ResidualNorm with the factory parameter baseline = "
"mode::initial_resnorm")]] ResidualNormReduction
: public ResidualNormBase<ValueType>
{
public:
using ComplexVector = matrix::Dense<to_complex<ValueType>>;
using NormVector = matrix::Dense<remove_complex<ValueType>>;
using Vector = matrix::Dense<ValueType>;

GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
{

remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
reduction_factor, static_cast<remove_complex<ValueType>>(1e-15));
};
GKO_ENABLE_CRITERION_FACTORY(ResidualNormReduction<ValueType>, parameters,
Factory);
GKO_ENABLE_BUILD_METHOD(Factory);

protected:
explicit ResidualNormReduction(std::shared_ptr<const gko::Executor> exec)
: ResidualNormBase<ValueType>(exec)
{}

explicit ResidualNormReduction(const Factory* factory,
const CriterionArgs& args)
: ResidualNormBase<ValueType>(
factory->get_executor(), args,
factory->get_parameters().reduction_factor,
mode::initial_resnorm),
parameters_{factory->get_parameters()}
{}
};



template <typename ValueType = default_precision>
class [[deprecated(
"Please use the class ResidualNorm with the factory parameter baseline = "
"mode::rhs_norm")]] RelativeResidualNorm
: public ResidualNormBase<ValueType>
{
public:
using ComplexVector = matrix::Dense<to_complex<ValueType>>;
using NormVector = matrix::Dense<remove_complex<ValueType>>;
using Vector = matrix::Dense<ValueType>;

GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
{

remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
tolerance, static_cast<remove_complex<ValueType>>(1e-15));
};
GKO_ENABLE_CRITERION_FACTORY(RelativeResidualNorm<ValueType>, parameters,
Factory);
GKO_ENABLE_BUILD_METHOD(Factory);

protected:
explicit RelativeResidualNorm(std::shared_ptr<const gko::Executor> exec)
: ResidualNormBase<ValueType>(exec)
{}

explicit RelativeResidualNorm(const Factory* factory,
const CriterionArgs& args)
: ResidualNormBase<ValueType>(factory->get_executor(), args,
factory->get_parameters().tolerance,
mode::rhs_norm),
parameters_{factory->get_parameters()}
{}
};



template <typename ValueType = default_precision>
class [[deprecated(
"Please use the class ResidualNorm with the factory parameter baseline = "
"mode::absolute")]] AbsoluteResidualNorm
: public ResidualNormBase<ValueType>
{
public:
using NormVector = matrix::Dense<remove_complex<ValueType>>;
using Vector = matrix::Dense<ValueType>;

GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
{

remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
tolerance, static_cast<remove_complex<ValueType>>(1e-15));
};
GKO_ENABLE_CRITERION_FACTORY(AbsoluteResidualNorm<ValueType>, parameters,
Factory);
GKO_ENABLE_BUILD_METHOD(Factory);

protected:
explicit AbsoluteResidualNorm(std::shared_ptr<const gko::Executor> exec)
: ResidualNormBase<ValueType>(exec)
{}

explicit AbsoluteResidualNorm(const Factory* factory,
const CriterionArgs& args)
: ResidualNormBase<ValueType>(factory->get_executor(), args,
factory->get_parameters().tolerance,
mode::absolute),
parameters_{factory->get_parameters()}
{}
};


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_BUILD) || defined(__INTEL_LLVM_COMPILER)
#pragma warning(pop)
#endif


}  
}  


#endif  
