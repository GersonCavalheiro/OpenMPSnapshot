

#ifndef GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_


#include <memory>
#include <utility>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/workspace.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5211, 4973, 4974)
#endif


namespace gko {
namespace solver {



enum class initial_guess_mode {

zero,

rhs,

provided
};


namespace multigrid {
namespace detail {


class MultigridState;


}  
}  



class ApplyWithInitialGuess {
protected:
friend class multigrid::detail::MultigridState;


virtual void apply_with_initial_guess(const LinOp* b, LinOp* x,
initial_guess_mode guess) const = 0;

void apply_with_initial_guess(ptr_param<const LinOp> b, ptr_param<LinOp> x,
initial_guess_mode guess) const
{
apply_with_initial_guess(b.get(), x.get(), guess);
}


virtual void apply_with_initial_guess(const LinOp* alpha, const LinOp* b,
const LinOp* beta, LinOp* x,
initial_guess_mode guess) const = 0;


void apply_with_initial_guess(ptr_param<const LinOp> alpha,
ptr_param<const LinOp> b,
ptr_param<const LinOp> beta,
ptr_param<LinOp> x,
initial_guess_mode guess) const
{
apply_with_initial_guess(alpha.get(), b.get(), beta.get(), x.get(),
guess);
}


initial_guess_mode get_default_initial_guess() const { return guess_; }


explicit ApplyWithInitialGuess(
initial_guess_mode guess = initial_guess_mode::provided)
: guess_(guess)
{}


void set_default_initial_guess(initial_guess_mode guess) { guess_ = guess; }

private:
initial_guess_mode guess_;
};



template <typename DerivedType>
class EnableApplyWithInitialGuess : public ApplyWithInitialGuess {
protected:
friend class multigrid::detail::MultigridState;

explicit EnableApplyWithInitialGuess(
initial_guess_mode guess = initial_guess_mode::provided)
: ApplyWithInitialGuess(guess)
{}


void apply_with_initial_guess(const LinOp* b, LinOp* x,
initial_guess_mode guess) const override
{
self()->template log<log::Logger::linop_apply_started>(self(), b, x);
auto exec = self()->get_executor();
GKO_ASSERT_CONFORMANT(self(), b);
GKO_ASSERT_EQUAL_ROWS(self(), x);
GKO_ASSERT_EQUAL_COLS(b, x);
this->apply_with_initial_guess_impl(make_temporary_clone(exec, b).get(),
make_temporary_clone(exec, x).get(),
guess);
self()->template log<log::Logger::linop_apply_completed>(self(), b, x);
}


void apply_with_initial_guess(const LinOp* alpha, const LinOp* b,
const LinOp* beta, LinOp* x,
initial_guess_mode guess) const override
{
self()->template log<log::Logger::linop_advanced_apply_started>(
self(), alpha, b, beta, x);
auto exec = self()->get_executor();
GKO_ASSERT_CONFORMANT(self(), b);
GKO_ASSERT_EQUAL_ROWS(self(), x);
GKO_ASSERT_EQUAL_COLS(b, x);
GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
this->apply_with_initial_guess_impl(
make_temporary_clone(exec, alpha).get(),
make_temporary_clone(exec, b).get(),
make_temporary_clone(exec, beta).get(),
make_temporary_clone(exec, x).get(), guess);
self()->template log<log::Logger::linop_advanced_apply_completed>(
self(), alpha, b, beta, x);
}


virtual void apply_with_initial_guess_impl(
const LinOp* b, LinOp* x, initial_guess_mode guess) const = 0;


virtual void apply_with_initial_guess_impl(
const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x,
initial_guess_mode guess) const = 0;

GKO_ENABLE_SELF(DerivedType);
};



template <typename Solver>
struct workspace_traits {
static int num_vectors(const Solver&) { return 0; }
static int num_arrays(const Solver&) { return 0; }
static std::vector<std::string> op_names(const Solver&) { return {}; }
static std::vector<std::string> array_names(const Solver&) { return {}; }
static std::vector<int> scalars(const Solver&) { return {}; }
static std::vector<int> vectors(const Solver&) { return {}; }
};



template <typename DerivedType>
class EnablePreconditionable : public Preconditionable {
public:

void set_preconditioner(std::shared_ptr<const LinOp> new_precond) override
{
auto exec = self()->get_executor();
if (new_precond) {
GKO_ASSERT_EQUAL_DIMENSIONS(self(), new_precond);
GKO_ASSERT_IS_SQUARE_MATRIX(new_precond);
if (new_precond->get_executor() != exec) {
new_precond = gko::clone(exec, new_precond);
}
}
Preconditionable::set_preconditioner(new_precond);
}


EnablePreconditionable& operator=(const EnablePreconditionable& other)
{
if (&other != this) {
set_preconditioner(other.get_preconditioner());
}
return *this;
}


EnablePreconditionable& operator=(EnablePreconditionable&& other)
{
if (&other != this) {
set_preconditioner(other.get_preconditioner());
other.set_preconditioner(nullptr);
}
return *this;
}

EnablePreconditionable() = default;

EnablePreconditionable(std::shared_ptr<const LinOp> preconditioner)
{
set_preconditioner(std::move(preconditioner));
}


EnablePreconditionable(const EnablePreconditionable& other)
{
*this = other;
}


EnablePreconditionable(EnablePreconditionable&& other)
{
*this = std::move(other);
}

private:
DerivedType* self() { return static_cast<DerivedType*>(this); }

const DerivedType* self() const
{
return static_cast<const DerivedType*>(this);
}
};


namespace detail {



class SolverBaseLinOp {
public:
SolverBaseLinOp(std::shared_ptr<const Executor> exec)
: workspace_{std::move(exec)}
{}

virtual ~SolverBaseLinOp() = default;


std::shared_ptr<const LinOp> get_system_matrix() const
{
return system_matrix_;
}

const LinOp* get_workspace_op(int vector_id) const
{
return workspace_.get_op(vector_id);
}

virtual int get_num_workspace_ops() const { return 0; }

virtual std::vector<std::string> get_workspace_op_names() const
{
return {};
}


virtual std::vector<int> get_workspace_scalars() const { return {}; }


virtual std::vector<int> get_workspace_vectors() const { return {}; }

protected:
void set_system_matrix_base(std::shared_ptr<const LinOp> system_matrix)
{
system_matrix_ = std::move(system_matrix);
}

void set_workspace_size(int num_operators, int num_arrays) const
{
workspace_.set_size(num_operators, num_arrays);
}

template <typename LinOpType>
LinOpType* create_workspace_op(int vector_id, gko::dim<2> size) const
{
return workspace_.template create_or_get_op<LinOpType>(
vector_id,
[&] {
return LinOpType::create(this->workspace_.get_executor(), size);
},
typeid(LinOpType), size, size[1]);
}

template <typename LinOpType>
LinOpType* create_workspace_op_with_config_of(int vector_id,
const LinOpType* vec) const
{
return workspace_.template create_or_get_op<LinOpType>(
vector_id, [&] { return LinOpType::create_with_config_of(vec); },
typeid(*vec), vec->get_size(), vec->get_stride());
}

template <typename LinOpType>
LinOpType* create_workspace_op_with_type_of(int vector_id,
const LinOpType* vec,
dim<2> size) const
{
return workspace_.template create_or_get_op<LinOpType>(
vector_id,
[&] {
return LinOpType::create_with_type_of(
vec, workspace_.get_executor(), size, size[1]);
},
typeid(*vec), size, size[1]);
}

template <typename LinOpType>
LinOpType* create_workspace_op_with_type_of(int vector_id,
const LinOpType* vec,
dim<2> global_size,
dim<2> local_size) const
{
return workspace_.template create_or_get_op<LinOpType>(
vector_id,
[&] {
return LinOpType::create_with_type_of(
vec, workspace_.get_executor(), global_size, local_size,
local_size[1]);
},
typeid(*vec), global_size, local_size[1]);
}

template <typename ValueType>
matrix::Dense<ValueType>* create_workspace_scalar(int vector_id,
size_type size) const
{
return workspace_.template create_or_get_op<matrix::Dense<ValueType>>(
vector_id,
[&] {
return matrix::Dense<ValueType>::create(
workspace_.get_executor(), dim<2>{1, size});
},
typeid(matrix::Dense<ValueType>), gko::dim<2>{1, size}, size);
}

template <typename ValueType>
array<ValueType>& create_workspace_array(int array_id, size_type size) const
{
return workspace_.template create_or_get_array<ValueType>(array_id,
size);
}

template <typename ValueType>
array<ValueType>& create_workspace_array(int array_id) const
{
return workspace_.template init_or_get_array<ValueType>(array_id);
}

private:
mutable detail::workspace workspace_;

std::shared_ptr<const LinOp> system_matrix_;
};


}  


template <typename MatrixType>
class
[[deprecated("This class will be replaced by the template-less detail::SolverBaseLinOp in a future release")]] SolverBase
: public detail::SolverBaseLinOp
{
public:
using detail::SolverBaseLinOp::SolverBaseLinOp;


std::shared_ptr<const MatrixType> get_system_matrix() const
{
return std::dynamic_pointer_cast<const MatrixType>(
SolverBaseLinOp::get_system_matrix());
}

protected:
void set_system_matrix_base(std::shared_ptr<const MatrixType> system_matrix)
{
SolverBaseLinOp::set_system_matrix_base(std::move(system_matrix));
}
};



template <typename DerivedType, typename MatrixType = LinOp>
class EnableSolverBase : public SolverBase<MatrixType> {
public:

EnableSolverBase& operator=(const EnableSolverBase& other)
{
if (&other != this) {
set_system_matrix(other.get_system_matrix());
}
return *this;
}


EnableSolverBase& operator=(EnableSolverBase&& other)
{
if (&other != this) {
set_system_matrix(other.get_system_matrix());
other.set_system_matrix(nullptr);
}
return *this;
}

EnableSolverBase() : SolverBase<MatrixType>{self()->get_executor()} {}

EnableSolverBase(std::shared_ptr<const MatrixType> system_matrix)
: SolverBase<MatrixType>{self()->get_executor()}
{
set_system_matrix(std::move(system_matrix));
}


EnableSolverBase(const EnableSolverBase& other)
: SolverBase<MatrixType>{other.self()->get_executor()}
{
*this = other;
}


EnableSolverBase(EnableSolverBase&& other)
: SolverBase<MatrixType>{other.self()->get_executor()}
{
*this = std::move(other);
}

int get_num_workspace_ops() const override
{
using traits = workspace_traits<DerivedType>;
return traits::num_vectors(*self());
}

std::vector<std::string> get_workspace_op_names() const override
{
using traits = workspace_traits<DerivedType>;
return traits::op_names(*self());
}


std::vector<int> get_workspace_scalars() const override
{
using traits = workspace_traits<DerivedType>;
return traits::scalars(*self());
}


std::vector<int> get_workspace_vectors() const override
{
using traits = workspace_traits<DerivedType>;
return traits::vectors(*self());
}

protected:
void set_system_matrix(std::shared_ptr<const MatrixType> new_system_matrix)
{
auto exec = self()->get_executor();
if (new_system_matrix) {
GKO_ASSERT_EQUAL_DIMENSIONS(self(), new_system_matrix);
GKO_ASSERT_IS_SQUARE_MATRIX(new_system_matrix);
if (new_system_matrix->get_executor() != exec) {
new_system_matrix = gko::clone(exec, new_system_matrix);
}
}
this->set_system_matrix_base(new_system_matrix);
}

void setup_workspace() const
{
using traits = workspace_traits<DerivedType>;
this->set_workspace_size(traits::num_vectors(*self()),
traits::num_arrays(*self()));
}

private:
DerivedType* self() { return static_cast<DerivedType*>(this); }

const DerivedType* self() const
{
return static_cast<const DerivedType*>(this);
}
};



class IterativeBase {
public:

std::shared_ptr<const stop::CriterionFactory> get_stop_criterion_factory()
const
{
return stop_factory_;
}


virtual void set_stop_criterion_factory(
std::shared_ptr<const stop::CriterionFactory> new_stop_factory)
{
stop_factory_ = new_stop_factory;
}

private:
std::shared_ptr<const stop::CriterionFactory> stop_factory_;
};



template <typename DerivedType>
class EnableIterativeBase : public IterativeBase {
public:

EnableIterativeBase& operator=(const EnableIterativeBase& other)
{
if (&other != this) {
set_stop_criterion_factory(other.get_stop_criterion_factory());
}
return *this;
}


EnableIterativeBase& operator=(EnableIterativeBase&& other)
{
if (&other != this) {
set_stop_criterion_factory(other.get_stop_criterion_factory());
other.set_stop_criterion_factory(nullptr);
}
return *this;
}

EnableIterativeBase() = default;

EnableIterativeBase(
std::shared_ptr<const stop::CriterionFactory> stop_factory)
{
set_stop_criterion_factory(std::move(stop_factory));
}


EnableIterativeBase(const EnableIterativeBase& other) { *this = other; }


EnableIterativeBase(EnableIterativeBase&& other)
{
*this = std::move(other);
}

void set_stop_criterion_factory(
std::shared_ptr<const stop::CriterionFactory> new_stop_factory) override
{
auto exec = self()->get_executor();
if (new_stop_factory && new_stop_factory->get_executor() != exec) {
new_stop_factory = gko::clone(exec, new_stop_factory);
}
IterativeBase::set_stop_criterion_factory(new_stop_factory);
}

private:
DerivedType* self() { return static_cast<DerivedType*>(this); }

const DerivedType* self() const
{
return static_cast<const DerivedType*>(this);
}
};



template <typename ValueType, typename DerivedType>
class EnablePreconditionedIterativeSolver
: public EnableSolverBase<DerivedType>,
public EnableIterativeBase<DerivedType>,
public EnablePreconditionable<DerivedType> {
public:
EnablePreconditionedIterativeSolver() = default;

EnablePreconditionedIterativeSolver(
std::shared_ptr<const LinOp> system_matrix,
std::shared_ptr<const stop::CriterionFactory> stop_factory,
std::shared_ptr<const LinOp> preconditioner)
: EnableSolverBase<DerivedType>(std::move(system_matrix)),
EnableIterativeBase<DerivedType>{std::move(stop_factory)},
EnablePreconditionable<DerivedType>{std::move(preconditioner)}
{}

template <typename FactoryParameters>
EnablePreconditionedIterativeSolver(
std::shared_ptr<const LinOp> system_matrix,
const FactoryParameters& params)
: EnablePreconditionedIterativeSolver{
system_matrix, stop::combine(params.criteria),
generate_preconditioner(system_matrix, params)}
{}

private:
template <typename FactoryParameters>
static std::shared_ptr<const LinOp> generate_preconditioner(
std::shared_ptr<const LinOp> system_matrix,
const FactoryParameters& params)
{
if (params.generated_preconditioner) {
return params.generated_preconditioner;
} else if (params.preconditioner) {
return params.preconditioner->generate(system_matrix);
} else {
return matrix::Identity<ValueType>::create(
system_matrix->get_executor(), system_matrix->get_size());
}
}
};


}  
}  


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif  
