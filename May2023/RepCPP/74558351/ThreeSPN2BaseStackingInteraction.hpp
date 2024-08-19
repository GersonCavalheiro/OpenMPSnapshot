#ifndef MJOLNIR_OMP_3SPN2_BASE_STACKING_INTERACTION_HPP
#define MJOLNIR_OMP_3SPN2_BASE_STACKING_INTERACTION_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/omp/System.hpp>
#include <mjolnir/forcefield/3SPN2/ThreeSPN2BaseStackingInteraction.hpp>

namespace mjolnir
{

template<typename realT, template<typename, typename> class boundaryT>
class ThreeSPN2BaseStackingInteraction<
OpenMPSimulatorTraits<realT, boundaryT>
> final : public LocalInteractionBase<OpenMPSimulatorTraits<realT, boundaryT>>
{
public:

using traits_type          = OpenMPSimulatorTraits<realT, boundaryT>;
using base_type            = LocalInteractionBase<traits_type>;
using real_type            = typename base_type::real_type;
using coordinate_type      = typename base_type::coordinate_type;
using system_type          = typename base_type::system_type;
using topology_type        = typename base_type::topology_type;
using connection_kind_type = typename base_type::connection_kind_type;

using potential_type       = ThreeSPN2BaseStackingPotential<real_type>;
using base_stack_kind      = parameter_3SPN2::base_stack_kind;

using indices_type         = std::array<std::size_t, 3>;
using parameter_type       = base_stack_kind;
using parameter_index_pair = std::pair<indices_type, parameter_type>;
using container_type       = std::vector<parameter_index_pair>;

using nucleotide_info_type = parameter_3SPN2::NucleotideInfo;

public:

ThreeSPN2BaseStackingInteraction(const connection_kind_type kind,
container_type&& para, potential_type&& pot,
std::vector<nucleotide_info_type>&& nuc_idx)
: kind_(kind), parameters_(std::move(para)), potential_(std::move(pot)),
nucleotide_info_(std::move(nuc_idx))
{}
~ThreeSPN2BaseStackingInteraction() override {};
ThreeSPN2BaseStackingInteraction(const ThreeSPN2BaseStackingInteraction&) = default;
ThreeSPN2BaseStackingInteraction(ThreeSPN2BaseStackingInteraction&&)      = default;
ThreeSPN2BaseStackingInteraction& operator=(const ThreeSPN2BaseStackingInteraction&) = default;
ThreeSPN2BaseStackingInteraction& operator=(ThreeSPN2BaseStackingInteraction&&)      = default;


void initialize(const system_type& sys) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
this->potential_.initialize(sys);
}

void update(const system_type& sys) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
this->potential_.update(sys);
}

void reduce_margin(const real_type, const system_type&) override {return;}
void  scale_margin(const real_type, const system_type&) override {return;}

void calc_force(system_type& sys) const noexcept override
{
this->template calc_force_energy_virial_impl<false, false>(sys);
return;
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

real_type calc_energy(const system_type& sys) const noexcept override
{
real_type E = 0.0;
#pragma omp parallel for reduction(+:E)
for(std::size_t idx=0; idx<this->parameters_.size(); ++idx)
{
const auto& idxp = this->parameters_[idx];

const std::size_t Si = idxp.first[0];
const std::size_t Bi = idxp.first[1];
const std::size_t Bj = idxp.first[2];
const auto   bs_kind = idxp.second;

const auto& rBi = sys.position(Bi);
const auto& rBj = sys.position(Bj);
const auto& rSi = sys.position(Si);

const auto Bji = sys.adjust_direction(rBj, rBi); 
const auto SBi = sys.adjust_direction(rSi, rBi); 

const auto lBji_sq = math::length_sq(Bji);
const auto rlBji   = math::rsqrt(lBji_sq);
const auto lBji    = lBji_sq * rlBji;


E += potential_.U_rep(bs_kind, lBji);


const auto lSBi_sq = math::length_sq(SBi);
const auto rlSBi   = math::rsqrt(lSBi_sq);

const auto dot_SBij = math::dot_product(SBi, Bji);
const auto cos_SBij = dot_SBij * rlBji * rlSBi;
const auto theta = std::acos(math::clamp<real_type>(cos_SBij, -1, 1));


const auto theta_0 = potential_.theta_0(bs_kind);
const auto f_theta = potential_.f(theta, theta_0);
if(f_theta == real_type(0.0))
{
continue; 
}
const auto U_attr = potential_.U_attr(bs_kind, lBji);
E += U_attr * f_theta;
}
return E;
}

std::string name() const override {return "3SPN2BaseStacking"_s;}

void write_topology(topology_type& topol) const override
{
MJOLNIR_GET_DEFAULT_LOGGER();
if(this->kind_.empty() || this->kind_ == "none")
{
MJOLNIR_LOG_WARN("3SPN2 Base-Base Interaction (base pairing + "
"cross stacking) requires the number of nucleotides"
" that separates bases but topology is not set.");
MJOLNIR_LOG_WARN("I trust that you know what you are doing.");
return;
}

for(std::size_t i=1; i<nucleotide_info_.size(); ++i)
{
constexpr auto nil = nucleotide_info_type::nil();
const auto& Base5 = nucleotide_info_.at(i-1);
const auto& Base3 = nucleotide_info_.at(i);

if(Base5.strand != Base3.strand) {continue;}

topol.add_connection(Base5.S, Base3.S, this->kind_);
topol.add_connection(Base5.S, Base3.B, this->kind_);
topol.add_connection(Base5.B, Base3.S, this->kind_);
topol.add_connection(Base5.B, Base3.B, this->kind_);

if(Base5.P != nil)
{
topol.add_connection(Base5.P, Base3.S, this->kind_);
topol.add_connection(Base5.P, Base3.B, this->kind_);
}
if(Base3.P != nil)
{
topol.add_connection(Base5.S, Base3.P, this->kind_);
topol.add_connection(Base5.B, Base3.P, this->kind_);
}
if(Base5.P != nil && Base3.P != nil)
{
topol.add_connection(Base5.P, Base3.P, this->kind_);
}
}
return;
}

connection_kind_type const& connection_kind() const noexcept {return kind_;}

container_type const& parameters() const noexcept {return parameters_;}
container_type&       parameters()       noexcept {return parameters_;}

potential_type const& potential() const noexcept {return potential_;}
potential_type&       potential()       noexcept {return potential_;}

base_type* clone() const override
{
return new ThreeSPN2BaseStackingInteraction(kind_,
container_type(parameters_), potential_type(potential_),
std::vector<nucleotide_info_type>(nucleotide_info_));
}

private:

template<bool NeedEnergy, bool NeedVirial>
real_type calc_force_energy_virial_impl(system_type& sys) const noexcept
{
constexpr auto tolerance = math::abs_tolerance<real_type>();

real_type E = 0.0;
#pragma omp parallel for reduction(+:E)
for(std::size_t idx=0; idx<this->parameters_.size(); ++idx)
{
const std::size_t thread_id = omp_get_thread_num();
const auto& idxp = this->parameters_[idx];

const std::size_t Si = idxp.first[0];
const std::size_t Bi = idxp.first[1];
const std::size_t Bj = idxp.first[2];
const auto   bs_kind = idxp.second;

const auto& rBi = sys.position(Bi);
const auto& rBj = sys.position(Bj);
const auto& rSi = sys.position(Si);

const auto Bji = sys.adjust_direction(rBj, rBi); 
const auto SBi = sys.adjust_direction(rSi, rBi); 

const auto lBji_sq = math::length_sq(Bji); 
const auto rlBji   = math::rsqrt(lBji_sq); 
const auto lBji    = lBji_sq * rlBji;      
const auto Bji_reg = Bji * rlBji;          


const auto dU_rep = potential_.dU_rep(bs_kind, lBji);
if(dU_rep != real_type(0.0))
{
const auto f = dU_rep * Bji_reg;
sys.force_thread(thread_id, Bi) -= f;
sys.force_thread(thread_id, Bj) += f;

if(NeedVirial)
{
sys.virial_thread(thread_id) += math::tensor_product(Bji, -f);
}
}

if(NeedEnergy)
{
E += potential_.U_rep(bs_kind, lBji);
}


const auto lSBi_sq = math::length_sq(SBi);
const auto rlSBi   = math::rsqrt(lSBi_sq);
const auto SBi_reg = SBi * rlSBi;

const auto cos_theta = math::dot_product(SBi_reg, Bji_reg);
const auto theta = std::acos(math::clamp<real_type>(cos_theta, -1, 1));


const auto theta_0 = potential_.theta_0(bs_kind);
const auto f_theta = potential_.f(theta, theta_0);
if(f_theta == real_type(0.0))
{
continue;
}
const auto df_theta  = potential_.df(theta, theta_0);
const auto U_dU_attr = potential_.U_dU_attr(bs_kind, lBji);

if(NeedEnergy)
{
E += U_dU_attr.first * f_theta;
}

auto f_Si = math::make_coordinate<coordinate_type>(0,0,0);
auto f_Bi = math::make_coordinate<coordinate_type>(0,0,0);
auto f_Bj = math::make_coordinate<coordinate_type>(0,0,0);

if(df_theta != real_type(0.0))
{
const auto coef = -df_theta * U_dU_attr.first;

const auto sin_theta = std::sin(theta);
const auto coef_rsin = (sin_theta > tolerance) ?
coef / sin_theta : coef / tolerance;

const auto fSi = (coef_rsin * rlSBi) * (Bji_reg - cos_theta * SBi_reg);
const auto fBj = (coef_rsin * rlBji) * (SBi_reg - cos_theta * Bji_reg);

f_Si +=  fSi;
f_Bi -= (fSi + fBj);
f_Bj +=        fBj;
}


if(U_dU_attr.second != real_type(0.0))
{
const auto coef = f_theta * U_dU_attr.second;
f_Bi -= coef * Bji_reg;
f_Bj += coef * Bji_reg;
}
sys.force_thread(thread_id, Si) += f_Si;
sys.force_thread(thread_id, Bi) += f_Bi;
sys.force_thread(thread_id, Bj) += f_Bj;

if(NeedVirial)
{
sys.virial_thread(thread_id) +=
math::tensor_product(rBi, f_Bi) +
math::tensor_product(rBi - Bji, f_Bj) +
math::tensor_product(rBi - SBi, f_Si) ;
}
}
return E;
}

private:

connection_kind_type kind_;
container_type parameters_;
potential_type potential_;
std::vector<nucleotide_info_type> nucleotide_info_;
};

} 

#ifdef MJOLNIR_SEPARATE_BUILD
#include <mjolnir/core/BoundaryCondition.hpp>

namespace mjolnir
{


extern template class ThreeSPN2BaseStackingInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>       >;
extern template class ThreeSPN2BaseStackingInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>       >;
extern template class ThreeSPN2BaseStackingInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>>;
extern template class ThreeSPN2BaseStackingInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>>;

} 
#endif 

#endif 
