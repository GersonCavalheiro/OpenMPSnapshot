#ifndef MJOLNIR_OMP_GJ_F_LANGEVIN_INTEGRATOR_HPP
#define MJOLNIR_OMP_GJ_F_LANGEVIN_INTEGRATOR_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/omp/System.hpp>
#include <mjolnir/omp/RandomNumberGenerator.hpp>
#include <mjolnir/omp/SystemMotionRemover.hpp>
#include <mjolnir/core/GJFNVTLangevinIntegrator.hpp>

namespace mjolnir
{


template<typename realT, template<typename, typename> class boundaryT>
class GJFNVTLangevinIntegrator<OpenMPSimulatorTraits<realT, boundaryT>>
{
public:
using traits_type     = OpenMPSimulatorTraits<realT, boundaryT>;
using boundary_type   = typename traits_type::boundary_type;
using real_type       = typename traits_type::real_type;
using coordinate_type = typename traits_type::coordinate_type;
using system_type     = System<traits_type>;
using forcefield_type = std::unique_ptr<ForceFieldBase<traits_type>>;
using rng_type        = RandomNumberGenerator<traits_type>;
using remover_type    = SystemMotionRemover<traits_type>;
using variable_key_type = typename system_type::variable_key_type;

public:

GJFNVTLangevinIntegrator(const real_type dt, std::vector<real_type>&& alpha,
remover_type&& remover)
: dt_(dt), halfdt_(dt / 2), alphas_(std::move(alpha)),
betas_(alphas_.size()),
bs_(alphas_.size()),
vel_coefs_(alphas_.size()),
remover_(std::move(remover))
{}
~GJFNVTLangevinIntegrator() = default;

void initialize(system_type& sys, forcefield_type& ff, rng_type&)
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();

if( ! ff->constraint().empty())
{
MJOLNIR_LOG_WARN("G-JF langevin integrator does not support"
"  constraints. [[forcefields.constraint]] will be ignored.");
}

this->update(sys);

if( ! sys.force_initialized())
{
#pragma omp parallel for
for(std::size_t i=0; i<sys.size(); ++i)
{
sys.force(i) = math::make_coordinate<coordinate_type>(0, 0, 0);
}
for(auto& kv : sys.variables())
{
auto& var = kv.second;
var.update(var.x(), var.v(), real_type(0));
}
ff->calc_force(sys);
}
return;
}

real_type step(const real_type time, system_type& sys, forcefield_type& ff, rng_type& rng)
{
real_type largest_disp2(0);
#pragma omp parallel for reduction(max:largest_disp2)
for(std::size_t i=0; i<sys.size(); ++i)
{
const auto rm = sys.rmass(i);  
const auto r2m = real_type(0.5) * rm; 

auto&      p = sys.position(i);
auto&      v = sys.velocity(i);
auto&      f = sys.force(i);
const auto b = this->bs_[i];
const auto a = this->vel_coefs_[i];


const auto beta = this->gen_R(rng) * betas_[i];

const auto dp = (b * dt_) * (v + r2m * (dt_ * f + beta));

p = sys.adjust_position(p + dp);
v = a * (v + (dt_ * r2m) * f) + b * rm * beta;

f = math::make_coordinate<coordinate_type>(0, 0, 0); 

largest_disp2 = std::max(largest_disp2, math::length_sq(dp));
}
for(auto& kv : sys.variables()) 
{
const auto& key = kv.first;
auto& param = params_for_dynvar_.at(key);
auto& var = kv.second;

const auto beta = rng.gaussian() * param.beta;

const auto next_x = var.x() + (param.b * dt_) * (var.v() +
(real_type(0.5) / var.m()) * (dt_ * var.f() + beta));

const auto next_v = param.vel_coef * (var.v() +
(real_type(0.5) * dt_ / var.m()) * var.f()) +
(param.b / var.m()) * beta;

var.update(next_x, next_v, real_type(0));
}

ff->reduce_margin(2 * std::sqrt(largest_disp2), sys);
ff->calc_force(sys);

#pragma omp parallel for
for(std::size_t i=0; i<sys.size(); ++i)
{
const auto  rm   = sys.rmass(i);  
const auto& f    = sys.force(i);
auto&       v    = sys.velocity(i);

v += (dt_ * real_type(0.5) * rm) * f;
}
for(auto& kv : sys.variables()) 
{
auto& var = kv.second;

const auto next_v = var.v() +
(dt_ * real_type(0.5) / var.m()) * var.f();

var.update(var.x(), next_v, var.f());
}

remover_.remove(sys);

return time + dt_;
}

void update(const system_type& sys)
{
if(!sys.has_attribute("temperature"))
{
throw std::out_of_range("mjolnir::GJFNVTLangevinIntegrator: "
"Langevin Integrator requires reference temperature, but "
"`temperature` is not found in `system.attribute`.");
}
this->temperature_ = sys.attribute("temperature");
this->reset_parameters(sys);
return;
}
void update(const system_type& sys, const real_type dt)
{
this->dt_ = dt;
this->halfdt_ = dt / 2;
this->update(sys);
return;
}

real_type delta_t() const noexcept {return dt_;}
std::vector<real_type> const& parameters() const noexcept {return alphas_;}

private:

void reset_parameters(const system_type& sys) noexcept
{
alphas_   .resize(sys.size());
betas_    .resize(sys.size());
bs_       .resize(sys.size());
vel_coefs_.resize(sys.size());
const auto kBT = physics::constants<real_type>::kB() * this->temperature_;
#pragma omp parallel for
for(std::size_t i=0; i<sys.size(); ++i)
{
const auto alpha   = this->alphas_.at(i);
const auto m       = sys.mass(i);
this->bs_.at(i)    = 1.0 / (1.0 + alpha * dt_ * 0.5 / m);
this->betas_.at(i) = std::sqrt(2 * alpha * kBT * dt_);
this->vel_coefs_.at(i) = 1.0 - alpha * bs_.at(i) * dt_ / m;
}

for(const auto& kv : sys.variables())
{
const auto& key = kv.first;
const auto& var = kv.second;

dynvar_params_type param;
param.alpha    = var.m() * var.gamma();
param.beta     = std::sqrt(2 * param.alpha * kBT * dt_);
param.b        = 1.0 / (1.0 + param.alpha * dt_ * 0.5 / var.m());
param.vel_coef = 1.0 - param.alpha * param.b * dt_ / var.m();
params_for_dynvar_[key] = param;
}
return;
}

coordinate_type gen_R(rng_type& rng) noexcept
{
const auto x = rng.gaussian();
const auto y = rng.gaussian();
const auto z = rng.gaussian();
return math::make_coordinate<coordinate_type>(x, y, z);
}

private:
real_type dt_;
real_type halfdt_;
real_type temperature_;

std::vector<real_type> alphas_;
std::vector<real_type> betas_;
std::vector<real_type> bs_;
std::vector<real_type> vel_coefs_;

remover_type remover_;

struct dynvar_params_type
{
real_type alpha; 
real_type beta;
real_type b;
real_type vel_coef;
};
std::map<variable_key_type, dynvar_params_type> params_for_dynvar_;
};

#ifdef MJOLNIR_SEPARATE_BUILD
extern template class GJFNVTLangevinIntegrator<OpenMPSimulatorTraits<double, UnlimitedBoundary>>;
extern template class GJFNVTLangevinIntegrator<OpenMPSimulatorTraits<float,  UnlimitedBoundary>>;
extern template class GJFNVTLangevinIntegrator<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>>;
extern template class GJFNVTLangevinIntegrator<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>>;
#endif

} 
#endif
