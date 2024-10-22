#ifndef MJOLNIR_OMP_BASE_PAIR_INTERACTION_HPP
#define MJOLNIR_OMP_BASE_PAIR_INTERACTION_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/omp/System.hpp>
#include <mjolnir/omp/UnlimitedGridCellList.hpp>
#include <mjolnir/omp/PeriodicGridCellList.hpp>
#include <mjolnir/forcefield/3SPN2/ThreeSPN2BasePairInteraction.hpp>

namespace mjolnir
{

template<typename realT, template<typename, typename> class boundaryT>
class ThreeSPN2BasePairInteraction<
OpenMPSimulatorTraits<realT, boundaryT>
> final : public GlobalInteractionBase<OpenMPSimulatorTraits<realT, boundaryT>>
{
public:

using traits_type         = OpenMPSimulatorTraits<realT, boundaryT>;
using base_type           = GlobalInteractionBase<traits_type>;
using real_type           = typename base_type::real_type;
using coordinate_type     = typename base_type::coordinate_type;
using system_type         = typename base_type::system_type;
using topology_type       = typename base_type::topology_type;
using boundary_type       = typename base_type::boundary_type;
using potential_type      = ThreeSPN2BasePairPotential<real_type>;
using parameter_list_type = ThreeSPN2BasePairParameterList<traits_type>;
using partition_type      = SpatialPartition<traits_type, potential_type>;

using base_kind        = parameter_3SPN2::bead_kind;
using base_pair_kind   = parameter_3SPN2::base_pair_kind;
using cross_stack_kind = parameter_3SPN2::cross_stack_kind;

public:

ThreeSPN2BasePairInteraction(potential_type&& pot, parameter_list_type&& para, partition_type&& part)
: potential_(std::move(pot)), parameters_(std::move(para)), partition_(std::move(part))
{}
~ThreeSPN2BasePairInteraction() {}

void initialize(const system_type& sys, const topology_type& topol) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
MJOLNIR_LOG_INFO("potential is ", this->name());

this->potential_ .initialize(sys);
this->parameters_.initialize(sys, topol, potential_);
this->partition_ .initialize(sys, this->parameters_);
}

void update(const system_type& sys, const topology_type& topol) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
MJOLNIR_LOG_INFO("potential is ", this->name());

this->potential_ .update(sys);
this->parameters_.update(sys, topol, potential_);
this->partition_ .initialize(sys, this->parameters_);
}

void reduce_margin(const real_type dmargin, const system_type& sys) override
{
this->partition_.reduce_margin(dmargin, sys, this->parameters_);
return;
}
void scale_margin(const real_type scale, const system_type& sys) override
{
this->partition_.scale_margin(scale, sys, this->parameters_);
return;
}

void calc_force(system_type& sys)           const noexcept override
{
this->template calc_force_energy_virial_impl<false, false>(sys);
return ;
}
void calc_force_and_virial(system_type& sys) const noexcept override
{
this->template calc_force_energy_virial_impl<false, true>(sys);
return;
}
real_type calc_force_and_energy(system_type& sys) const noexcept override
{
return this->template calc_force_energy_virial_impl<true, false>(sys);
}
real_type calc_force_virial_energy(system_type& sys) const noexcept override
{
return this->template calc_force_energy_virial_impl<true, true>(sys);
}
real_type calc_energy(const system_type& sys)     const noexcept override
{
constexpr auto pi     = math::constants<real_type>::pi();
constexpr auto two_pi = math::constants<real_type>::two_pi();

real_type E_BP = 0.0;

const auto leading_participants = this->parameters_.leading_participants();
#pragma omp parallel for reduction(+:E_BP)
for(std::size_t i=0; i<leading_participants.size(); ++i)
{
const std::size_t Bi = leading_participants[i];
const auto& rBi = sys.position(Bi);

for(const auto& ptnr : this->partition_.partners(Bi))
{
const auto  Bj   = ptnr.index;
const auto& para = ptnr.parameter();
const auto& rBj  = sys.position(Bj);

const auto  bp_kind = para.bp_kind;

const auto  Bij = sys.adjust_direction(rBi, rBj); 

const auto lBij_sq = math::length_sq(Bij);
if(lBij_sq > parameters_.cutoff_sq())
{
continue;
}


const auto rlBij = math::rsqrt(lBij_sq); 
const auto lBij  = lBij_sq * rlBij;      


const real_type epsilon_bp = parameters_.epsilon(bp_kind);
const real_type alpha_bp   = parameters_.alpha(bp_kind);
const real_type r0_bp      = parameters_.r0(bp_kind);

E_BP += potential_.U_rep(epsilon_bp, alpha_bp, lBij, r0_bp);


const auto   Si = para.Si;
const auto   Sj = para.Sj;
const auto& rSi = sys.position(Si);
const auto& rSj = sys.position(Sj);

const auto SBi = sys.adjust_direction(rSi, rBi); 
const auto SBj = sys.adjust_direction(rSj, rBj); 

const auto lSBi_sq = math::length_sq(SBi); 
const auto lSBj_sq = math::length_sq(SBj); 

const auto rlSBi = math::rsqrt(lSBi_sq); 
const auto rlSBj = math::rsqrt(lSBj_sq); 

const auto dot_SBiBj = -math::dot_product(SBi, Bij);
const auto dot_SBjBi =  math::dot_product(SBj, Bij);

const auto cos_theta1 = dot_SBiBj * rlSBi * rlBij;
const auto cos_theta2 = dot_SBjBi * rlSBj * rlBij;

const auto theta1 = std::acos(math::clamp<real_type>(cos_theta1, -1, 1));
const auto theta2 = std::acos(math::clamp<real_type>(cos_theta2, -1, 1));


const auto K_BP         = parameters_.K_BP();
const auto pi_over_K_BP = parameters_.pi_over_K_BP();
const auto theta1_0     = parameters_.theta1_0(bp_kind);
const auto theta2_0     = parameters_.theta2_0(bp_kind);

const auto f1 = potential_.f(K_BP, pi_over_K_BP, theta1, theta1_0);
const auto f2 = potential_.f(K_BP, pi_over_K_BP, theta2, theta2_0);

if(f1 == real_type(0.0) || f2 == real_type(0.0))
{
continue;
}


const auto Bij_reg = Bij * rlBij;
const auto R = math::dot_product(SBi, Bij_reg) * Bij_reg - SBi;
const auto S = math::dot_product(SBj, Bij_reg) * Bij_reg - SBj;

const auto dot_RS  = math::dot_product(R, S) *
math::rsqrt(math::length_sq(R) * math::length_sq(S));
const auto cos_phi = math::clamp<real_type>(dot_RS, -1, 1);

const auto n    =  math::cross_product(Bij, SBj);
const auto sign = -math::dot_product(SBi, n);
const auto phi  = std::copysign(std::acos(cos_phi), sign);

auto dphi = phi - parameters_.phi_0(bp_kind);
if     (pi   < dphi) {dphi -= two_pi;}
else if(dphi <  -pi) {dphi += two_pi;}

const auto cos_dphi = std::cos(dphi);

const auto U_attr = potential_.U_attr(epsilon_bp, alpha_bp, lBij, r0_bp);

E_BP += 0.5 * (1 + cos_dphi) * f1 * f2 * U_attr;
}
}
return E_BP;
}

std::string name() const override {return "3SPN2BasePair";}

parameter_list_type const& parameters() const noexcept {return parameters_;}
parameter_list_type&       parameters()       noexcept {return parameters_;}

base_type* clone() const override
{
return new ThreeSPN2BasePairInteraction(*this);
}

private:

template<bool NeedEnergy, bool NeedVirial>
real_type calc_force_energy_virial_impl(system_type& sys) const noexcept
{
constexpr auto pi        = math::constants<real_type>::pi();
constexpr auto two_pi    = math::constants<real_type>::two_pi();
constexpr auto tolerance = math::abs_tolerance<real_type>();

const auto leading_participants = this->parameters_.leading_participants();
real_type energy = 0;
#pragma omp parallel for reduction(+:energy)
for(std::size_t i=0; i<leading_participants.size(); ++i)
{
const std::size_t Bi = leading_participants[i];
const auto& rBi = sys.position(Bi);
for(const auto& ptnr : this->partition_.partners(Bi))
{
const auto  Bj   = ptnr.index;
const auto& para = ptnr.parameter();
const auto& rBj  = sys.position(Bj);

const auto Bij = sys.adjust_direction(rBi, rBj); 
const auto lBij_sq = math::length_sq(Bij);
if(lBij_sq > parameters_.cutoff_sq())
{
continue;
}


const auto rlBij = math::rsqrt(lBij_sq); 
const auto lBij  = lBij_sq * rlBij;      

const auto Bij_reg =  rlBij * Bij;
const auto Bji_reg = -rlBij * Bij;

const auto bp_kind    = para.bp_kind;
const auto epsilon_bp = parameters_.epsilon(bp_kind);
const auto alpha_bp   = parameters_.alpha(bp_kind);
const auto r0_bp      = parameters_.r0(bp_kind);

if  (NeedEnergy)
{
energy += potential_.U_rep(epsilon_bp, alpha_bp, lBij, r0_bp);
}
const auto dU_rep = potential_.dU_rep(epsilon_bp, alpha_bp, lBij, r0_bp);
if(dU_rep != real_type(0))
{
const auto F = -dU_rep * Bji_reg;
sys.force(Bi) += F;
sys.force(Bj) -= F;

if(NeedVirial)
{
sys.virial() += math::tensor_product(Bij, -F); 
}
}


const auto   Si = para.Si;
const auto   Sj = para.Sj;
const auto& rSi = sys.position(Si);
const auto& rSj = sys.position(Sj);

const auto SBi = sys.adjust_direction(rSi, rBi); 
const auto SBj = sys.adjust_direction(rSj, rBj); 

const auto lSBi_sq = math::length_sq(SBi); 
const auto lSBj_sq = math::length_sq(SBj); 
const auto rlSBi   = math::rsqrt(lSBi_sq); 
const auto rlSBj   = math::rsqrt(lSBj_sq); 
const auto BSi_reg = -rlSBi * SBi;
const auto BSj_reg = -rlSBj * SBj;

const auto dot_SBiBj  = -math::dot_product(SBi, Bij);
const auto dot_SBjBi  =  math::dot_product(SBj, Bij);
const auto cos_theta1 = dot_SBiBj * rlSBi * rlBij;
const auto cos_theta2 = dot_SBjBi * rlSBj * rlBij;
const auto theta1 = std::acos(math::clamp<real_type>(cos_theta1, -1, 1));
const auto theta2 = std::acos(math::clamp<real_type>(cos_theta2, -1, 1));


const auto K_BP         = parameters_.K_BP();
const auto pi_over_K_BP = parameters_.pi_over_K_BP();
const auto theta1_0     = parameters_.theta1_0(bp_kind);
const auto theta2_0     = parameters_.theta2_0(bp_kind);

const auto f1 = potential_.f(K_BP, pi_over_K_BP, theta1, theta1_0);
const auto f2 = potential_.f(K_BP, pi_over_K_BP, theta2, theta2_0);

if(f1 == real_type(0.0) || f2 == real_type(0.0))
{
continue;
}


const auto df1 = potential_.df(K_BP, pi_over_K_BP, theta1, theta1_0);
const auto df2 = potential_.df(K_BP, pi_over_K_BP, theta2, theta2_0);

const auto m = math::cross_product(-SBi, Bij);
const auto n = math::cross_product( Bij, SBj);
const auto m_lsq = math::length_sq(m);
const auto n_lsq = math::length_sq(n);

const auto dot_phi = math::dot_product(m, n) *
math::rsqrt(m_lsq * n_lsq);
const auto cos_phi = math::clamp<real_type>(dot_phi, -1, 1);

const auto phi = std::copysign(std::acos(cos_phi),
-math::dot_product(SBi, n));

auto dphi = phi - this->parameters_.phi_0(bp_kind);
if(dphi < -pi) {dphi += two_pi;}
if(pi <= dphi) {dphi -= two_pi;}
const auto cos_dphi = std::cos(dphi);
const auto sin_dphi = std::sin(dphi);

const auto U_dU_attr = potential_.U_dU_attr(epsilon_bp, alpha_bp, lBij, r0_bp);

if  (NeedEnergy)
{
energy += 0.5 * (1 + cos_dphi) * f1 * f2 * U_dU_attr.first;
}


if(cos_dphi == real_type(-1.0)) 
{
continue;
}

auto f_Si = math::make_coordinate<coordinate_type>(0,0,0);
auto f_Sj = math::make_coordinate<coordinate_type>(0,0,0);
auto f_Bi = math::make_coordinate<coordinate_type>(0,0,0);
auto f_Bj = math::make_coordinate<coordinate_type>(0,0,0);

if(sin_dphi != real_type(0.0))
{
const auto coef = real_type(0.5) * sin_dphi *
f1 * f2 * U_dU_attr.first;

const auto rlBij_sq = rlBij * rlBij; 
const auto fSi = ( coef * lBij / m_lsq) * m;
const auto fSj = (-coef * lBij / n_lsq) * n;

const auto coef_Bi = dot_SBiBj * rlBij_sq;
const auto coef_Bj = dot_SBjBi * rlBij_sq;

f_Si += fSi;
f_Bi += (coef_Bi - real_type(1.0)) * fSi - coef_Bj * fSj;
f_Bj += (coef_Bj - real_type(1.0)) * fSj - coef_Bi * fSi;
f_Sj += fSj;
}

const auto dihd_term = real_type(0.5) * (real_type(1.0) + cos_dphi);

if(df1 != real_type(0.0))
{
const auto coef = -dihd_term * df1 * f2 * U_dU_attr.first;

const auto sin_theta1 = std::sin(theta1);
const auto coef_rsin  = (sin_theta1 > tolerance) ?
(coef / sin_theta1) : (coef / tolerance);

const auto fSi = (coef_rsin * rlSBi) *
(cos_theta1 * BSi_reg - Bij_reg);
const auto fBj = (coef_rsin * rlBij) *
(cos_theta1 * Bij_reg - BSi_reg);
f_Si += fSi;
f_Bi -= (fSi + fBj);
f_Bj += fBj;
}
if(df2 != real_type(0.0))
{
const auto coef = -dihd_term * f1 * df2 * U_dU_attr.first;

const auto sin_theta2 = std::sin(theta2);
const auto coef_rsin  = (sin_theta2 > tolerance) ?
(coef / sin_theta2) : (coef / tolerance);

const auto fBi = (coef_rsin * rlBij) *
(cos_theta2 * Bji_reg - BSj_reg);
const auto fSj = (coef_rsin * rlSBj) *
(cos_theta2 * BSj_reg - Bji_reg);
f_Bi += fBi;
f_Bj -= (fBi + fSj);
f_Sj += fSj;
}
if(U_dU_attr.second != real_type(0.0))
{
const auto coef = -dihd_term * f1 * f2 * U_dU_attr.second;
f_Bi += coef * Bji_reg;
f_Bj += coef * Bij_reg;
}

sys.force(Si) += f_Si;
sys.force(Bi) += f_Bi;
sys.force(Bj) += f_Bj;
sys.force(Sj) += f_Sj;

if  (NeedVirial)
{
sys.virial()  += math::tensor_product(sys.transpose(rSi, rBi), f_Si) +
math::tensor_product(                   rBi , f_Bi) +
math::tensor_product(              rBi + Bij, f_Bj) +
math::tensor_product(sys.transpose(rSj, rBi), f_Sj) ;
}
}
}
return energy;
}

private:

potential_type      potential_;
parameter_list_type parameters_;
partition_type      partition_;
};

} 

#ifdef MJOLNIR_SEPARATE_BUILD
#include <mjolnir/core/BoundaryCondition.hpp>
#include <mjolnir/core/SimulatorTraits.hpp>

namespace mjolnir
{

extern template class ThreeSPN2BasePairInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>       >;
extern template class ThreeSPN2BasePairInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>       >;
extern template class ThreeSPN2BasePairInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>>;
extern template class ThreeSPN2BasePairInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>>;

} 
#endif 

#endif 
