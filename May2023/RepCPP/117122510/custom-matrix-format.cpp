

#include <iostream>
#include <map>
#include <string>


#include <omp.h>
#include <ginkgo/ginkgo.hpp>


template <typename ValueType>
void stencil_kernel(std::size_t size, const ValueType* coefs,
const ValueType* b, ValueType* x);


template <typename ValueType>
class StencilMatrix : public gko::EnableLinOp<StencilMatrix<ValueType>>,
public gko::EnableCreateMethod<StencilMatrix<ValueType>> {
public:
StencilMatrix(std::shared_ptr<const gko::Executor> exec,
gko::size_type size = 0, ValueType left = -1.0,
ValueType center = 2.0, ValueType right = -1.0)
: gko::EnableLinOp<StencilMatrix>(exec, gko::dim<2>{size}),
coefficients(exec, {left, center, right})
{}

protected:
using vec = gko::matrix::Dense<ValueType>;
using coef_type = gko::array<ValueType>;

void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
{
auto dense_b = gko::as<vec>(b);
auto dense_x = gko::as<vec>(x);

struct stencil_operation : gko::Operation {
stencil_operation(const coef_type& coefficients, const vec* b,
vec* x)
: coefficients{coefficients}, b{b}, x{x}
{}

void run(std::shared_ptr<const gko::OmpExecutor>) const override
{
auto b_values = b->get_const_values();
auto x_values = x->get_values();
#pragma omp parallel for
for (std::size_t i = 0; i < x->get_size()[0]; ++i) {
auto coefs = coefficients.get_const_data();
auto result = coefs[1] * b_values[i];
if (i > 0) {
result += coefs[0] * b_values[i - 1];
}
if (i < x->get_size()[0] - 1) {
result += coefs[2] * b_values[i + 1];
}
x_values[i] = result;
}
}

void run(std::shared_ptr<const gko::CudaExecutor>) const override
{
stencil_kernel(x->get_size()[0], coefficients.get_const_data(),
b->get_const_values(), x->get_values());
}


const coef_type& coefficients;
const vec* b;
vec* x;
};
this->get_executor()->run(
stencil_operation(coefficients, dense_b, dense_x));
}

void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
const gko::LinOp* beta, gko::LinOp* x) const override
{
auto dense_b = gko::as<vec>(b);
auto dense_x = gko::as<vec>(x);
auto tmp_x = dense_x->clone();
this->apply_impl(b, tmp_x.get());
dense_x->scale(beta);
dense_x->add_scaled(alpha, tmp_x);
}

private:
coef_type coefficients;
};


template <typename ValueType, typename IndexType>
void generate_stencil_matrix(gko::matrix::Csr<ValueType, IndexType>* matrix)
{
const auto discretization_points = matrix->get_size()[0];
auto row_ptrs = matrix->get_row_ptrs();
auto col_idxs = matrix->get_col_idxs();
auto values = matrix->get_values();
IndexType pos = 0;
const ValueType coefs[] = {-1, 2, -1};
row_ptrs[0] = pos;
for (int i = 0; i < discretization_points; ++i) {
for (auto ofs : {-1, 0, 1}) {
if (0 <= i + ofs && i + ofs < discretization_points) {
values[pos] = coefs[ofs + 1];
col_idxs[pos] = i + ofs;
++pos;
}
}
row_ptrs[i + 1] = pos;
}
}


template <typename Closure, typename ValueType>
void generate_rhs(Closure f, ValueType u0, ValueType u1,
gko::matrix::Dense<ValueType>* rhs)
{
const auto discretization_points = rhs->get_size()[0];
auto values = rhs->get_values();
const ValueType h = 1.0 / (discretization_points + 1);
for (int i = 0; i < discretization_points; ++i) {
const ValueType xi = ValueType(i + 1) * h;
values[i] = -f(xi) * h * h;
}
values[0] += u0;
values[discretization_points - 1] += u1;
}


template <typename ValueType>
void print_solution(ValueType u0, ValueType u1,
const gko::matrix::Dense<ValueType>* u)
{
std::cout << u0 << '\n';
for (int i = 0; i < u->get_size()[0]; ++i) {
std::cout << u->get_const_values()[i] << '\n';
}
std::cout << u1 << std::endl;
}


template <typename Closure, typename ValueType>
double calculate_error(int discretization_points,
const gko::matrix::Dense<ValueType>* u,
Closure correct_u)
{
const auto h = 1.0 / (discretization_points + 1);
auto error = 0.0;
for (int i = 0; i < discretization_points; ++i) {
using std::abs;
const auto xi = (i + 1) * h;
error +=
abs(u->get_const_values()[i] - correct_u(xi)) / abs(correct_u(xi));
}
return error;
}


int main(int argc, char* argv[])
{
using ValueType = double;
using RealValueType = gko::remove_complex<ValueType>;
using IndexType = int;

using vec = gko::matrix::Dense<ValueType>;
using mtx = gko::matrix::Csr<ValueType, IndexType>;
using cg = gko::solver::Cg<ValueType>;

if (argc == 2 && (std::string(argv[1]) == "--help")) {
std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
std::exit(-1);
}

const auto executor_string = argc >= 2 ? argv[1] : "reference";
const unsigned int discretization_points =
argc >= 3 ? std::atoi(argv[2]) : 100u;
std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
exec_map{
{"omp", [] { return gko::OmpExecutor::create(); }},
{"cuda",
[] {
return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
true);
}},
{"hip",
[] {
return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
true);
}},
{"dpcpp",
[] {
return gko::DpcppExecutor::create(0,
gko::OmpExecutor::create());
}},
{"reference", [] { return gko::ReferenceExecutor::create(); }}};

const auto exec = exec_map.at(executor_string)();  
const auto app_exec = exec->get_master();

auto correct_u = [](ValueType x) { return x * x * x; };
auto f = [](ValueType x) { return ValueType{6} * x; };
auto u0 = correct_u(0);
auto u1 = correct_u(1);

auto rhs = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
generate_rhs(f, u0, u1, rhs.get());
auto u = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
for (int i = 0; i < u->get_size()[0]; ++i) {
u->get_values()[i] = 0.0;
}

const RealValueType reduction_factor{1e-7};
cg::build()
.with_criteria(gko::stop::Iteration::build()
.with_max_iters(discretization_points)
.on(exec),
gko::stop::ResidualNorm<ValueType>::build()
.with_reduction_factor(reduction_factor)
.on(exec))
.on(exec)
->generate(StencilMatrix<ValueType>::create(exec, discretization_points,
-1, 2, -1))
->apply(rhs, u);

std::cout << "\nSolve complete."
<< "\nThe average relative error is "
<< calculate_error(discretization_points, u.get(), correct_u) /
discretization_points
<< std::endl;
}
