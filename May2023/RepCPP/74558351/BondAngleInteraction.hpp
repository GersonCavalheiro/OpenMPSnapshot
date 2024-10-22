#ifndef MJOLNIR_OMP_BOND_ANGLE_INTERACTION_HPP
#define MJOLNIR_OMP_BOND_ANGLE_INTERACTION_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/omp/System.hpp>
#include <mjolnir/forcefield/local/BondAngleInteraction.hpp>

namespace mjolnir
{

template<typename realT, template<typename, typename> class boundaryT,
typename potentialT>
class BondAngleInteraction<OpenMPSimulatorTraits<realT, boundaryT>, potentialT>
final: public LocalInteractionBase<OpenMPSimulatorTraits<realT, boundaryT>>
{
public:
using traits_type          = OpenMPSimulatorTraits<realT, boundaryT>;
using potential_type       = potentialT;
using base_type            = LocalInteractionBase<traits_type>;
using real_type            = typename base_type::real_type;
using coordinate_type      = typename base_type::coordinate_type;
using system_type          = typename base_type::system_type;
using topology_type        = typename base_type::topology_type;
using connection_kind_type = typename base_type::connection_kind_type;

using indices_type         = std::array<std::size_t, 3>;
using potential_index_pair = std::pair<indices_type, potentialT>;
using container_type       = std::vector<potential_index_pair>;
using iterator             = typename container_type::iterator;
using const_iterator       = typename container_type::const_iterator;

public:

BondAngleInteraction(const connection_kind_type kind,
const container_type& pot)
: kind_(kind), potentials(pot)
{}
BondAngleInteraction(const connection_kind_type kind,
container_type&& pot)
: kind_(kind), potentials(std::move(pot))
{}
~BondAngleInteraction() override {}

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
for(std::size_t i=0; i<this->potentials.size(); ++i)
{
const auto& idxp = this->potentials[i];
const std::size_t idx0 = idxp.first[0];
const std::size_t idx1 = idxp.first[1];
const std::size_t idx2 = idxp.first[2];

const coordinate_type v_2to1 =
sys.adjust_direction(sys.position(idx1), sys.position(idx0));
const coordinate_type v_2to3 =
sys.adjust_direction(sys.position(idx1), sys.position(idx2));

const real_type lensq_v21   = math::length_sq(v_2to1);
const real_type lensq_v23   = math::length_sq(v_2to3);
const real_type dot_v21_v23 = math::dot_product(v_2to1, v_2to3);

const real_type dot_ijk   = dot_v21_v23 * math::rsqrt(lensq_v21 * lensq_v23);
const real_type cos_theta = math::clamp(dot_ijk, real_type(-1.0), real_type(1.0));
const real_type theta     = std::acos(cos_theta);

E += idxp.second.potential(theta);
}
return E;
}

void initialize(const system_type& sys) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
MJOLNIR_LOG_INFO("With OpenMP: potential = ", potential_type::name(),
", number of angles = ", potentials.size());
for(auto& potential : potentials)
{
potential.second.initialize(sys);
}
return;
}

void update(const system_type& sys) override
{
for(auto& item : potentials)
{
item.second.update(sys);
}
}

void reduce_margin(const real_type, const system_type&) override {return;}
void  scale_margin(const real_type, const system_type&) override {return;}

std::string name() const override
{return "BondAngle:"_s + potential_type::name();}

void write_topology(topology_type& topol) const override
{
if(this->kind_.empty() || this->kind_ == "none") {return;}

for(const auto& idxp : this->potentials)
{
const auto i = idxp.first[0];
const auto j = idxp.first[1];
const auto k = idxp.first[2];
topol.add_connection(i, j, this->kind_);
topol.add_connection(j, k, this->kind_);
}
return;
}

base_type* clone() const override
{
return new BondAngleInteraction(kind_, container_type(potentials));
}

private:

template<bool NeedEnergy, bool NeedVirial>
real_type calc_force_energy_virial_impl(system_type& sys) const noexcept
{
real_type E = 0.0;
#pragma omp parallel for reduction(+:E)
for(std::size_t i=0; i<this->potentials.size(); ++i)
{
const auto& idxp = this->potentials[i];
const std::size_t idx0 = idxp.first[0];
const std::size_t idx1 = idxp.first[1];
const std::size_t idx2 = idxp.first[2];
const auto& p0 = sys.position(idx0);
const auto& p1 = sys.position(idx1);
const auto& p2 = sys.position(idx2);

const coordinate_type r_ij         = sys.adjust_direction(p1, p0);
const real_type       inv_len_r_ij = math::rlength(r_ij);
const coordinate_type r_ij_reg     = r_ij * inv_len_r_ij;

const coordinate_type r_kj         = sys.adjust_direction(p1, p2);
const real_type       inv_len_r_kj = math::rlength(r_kj);
const coordinate_type r_kj_reg     = r_kj * inv_len_r_kj;

const real_type dot_ijk   = math::dot_product(r_ij_reg, r_kj_reg);
const real_type cos_theta = math::clamp(dot_ijk, real_type(-1.0), real_type(1.0));

const real_type theta = std::acos(cos_theta);
const real_type coef  = -(idxp.second.derivative(theta));

if (NeedEnergy)
{
E += idxp.second.potential(theta);
}

const real_type sin_theta    = std::sin(theta);
const real_type coef_inv_sin = (sin_theta > math::abs_tolerance<real_type>()) ?
coef / sin_theta : coef / math::abs_tolerance<real_type>();

const auto Fi = (coef_inv_sin * inv_len_r_ij) * (cos_theta * r_ij_reg - r_kj_reg);
const auto Fk = (coef_inv_sin * inv_len_r_kj) * (cos_theta * r_kj_reg - r_ij_reg);
const auto Fj = -Fi - Fk;

const std::size_t thread_id = omp_get_thread_num();

sys.force_thread(thread_id, idx0) += Fi;
sys.force_thread(thread_id, idx1) += Fj;
sys.force_thread(thread_id, idx2) += Fk;

if(NeedVirial)
{
sys.virial_thread(thread_id) +=
math::tensor_product(p1 + r_ij, Fi) +
math::tensor_product(p1,        Fj) +
math::tensor_product(p1 + r_kj, Fk);
}
}
return E;
}

private:
connection_kind_type kind_;
container_type potentials;
};

}

#ifdef MJOLNIR_SEPARATE_BUILD
#include <mjolnir/core/BoundaryCondition.hpp>
#include <mjolnir/forcefield/local/HarmonicPotential.hpp>
#include <mjolnir/forcefield/local/GaussianPotential.hpp>
#include <mjolnir/forcefield/FLP/FlexibleLocalAnglePotential.hpp>

namespace mjolnir
{

extern template class BondAngleInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>, HarmonicPotential<double>>;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>, HarmonicPotential<float> >;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>, HarmonicPotential<double>>;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>, HarmonicPotential<float> >;

extern template class BondAngleInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>, GaussianPotential<double>>;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>, GaussianPotential<float> >;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>, GaussianPotential<double>>;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>, GaussianPotential<float> >;

extern template class BondAngleInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>, FlexibleLocalAnglePotential<double>>;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>, FlexibleLocalAnglePotential<float> >;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>, FlexibleLocalAnglePotential<double>>;
extern template class BondAngleInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>, FlexibleLocalAnglePotential<float> >;

} 
#endif 

#endif 
