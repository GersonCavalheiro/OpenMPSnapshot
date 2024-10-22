#ifndef MJOLNIR_OMP_DIHEDRAL_ANGLE_INTERACTION_HPP
#define MJOLNIR_OMP_DIHEDRAL_ANGLE_INTERACTION_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/omp/System.hpp>
#include <mjolnir/forcefield/local/DihedralAngleInteraction.hpp>

namespace mjolnir
{

template<typename realT, template<typename, typename> class boundaryT,
typename potentialT>
class DihedralAngleInteraction<OpenMPSimulatorTraits<realT, boundaryT>, potentialT>
final : public LocalInteractionBase<OpenMPSimulatorTraits<realT, boundaryT>>
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

using indices_type         = std::array<std::size_t, 4>;
using potential_index_pair = std::pair<indices_type, potentialT>;
using container_type       = std::vector<potential_index_pair>;
using iterator             = typename container_type::iterator;
using const_iterator       = typename container_type::const_iterator;

public:

DihedralAngleInteraction(const connection_kind_type kind,
const container_type& pot)
: kind_(kind), potentials(pot)
{}
DihedralAngleInteraction(const connection_kind_type kind,
container_type&& pot)
: kind_(kind), potentials(std::move(pot))
{}
~DihedralAngleInteraction() override {}

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
const coordinate_type r_ij = sys.adjust_direction(
sys.position(idxp.first[1]), sys.position(idxp.first[0]));
const coordinate_type r_kj = sys.adjust_direction(
sys.position(idxp.first[1]), sys.position(idxp.first[2]));
const coordinate_type r_lk = sys.adjust_direction(
sys.position(idxp.first[2]), sys.position(idxp.first[3]));
const real_type r_kj_lensq_inv = real_type(1.0) / math::length_sq(r_kj);

const coordinate_type n = math::cross_product(r_kj, real_type(-1.0) * r_lk);

const coordinate_type R = r_ij -
(math::dot_product(r_ij, r_kj) * r_kj_lensq_inv) * r_kj;
const coordinate_type S = r_lk -
(math::dot_product(r_lk, r_kj) * r_kj_lensq_inv) * r_kj;
const real_type R_lensq = math::length_sq(R);
const real_type S_lensq = math::length_sq(S);

const real_type dot_RS  = math::dot_product(R, S) * math::rsqrt(R_lensq * S_lensq);
const real_type cos_phi = math::clamp(dot_RS, real_type(-1.0), real_type(1.0));
const real_type phi     =
std::copysign(std::acos(cos_phi), math::dot_product(r_ij, n));

E += idxp.second.potential(phi);
}
return E;
}

void initialize(const system_type& sys) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
MJOLNIR_LOG_INFO("With OpenMP: potential = ", potential_type::name(),
", number of dihedrals = ", potentials.size());
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
{return "DihedralAngle:"_s + potential_type::name();}

void write_topology(topology_type& topol) const override
{
if(this->kind_.empty() || this->kind_ == "none") {return;}

for(const auto& idxp : this->potentials)
{
const auto i = idxp.first[0];
const auto j = idxp.first[1];
const auto k = idxp.first[2];
const auto l = idxp.first[3];
topol.add_connection(i, j, this->kind_);
topol.add_connection(j, k, this->kind_);
topol.add_connection(k, l, this->kind_);
}
return;
}

base_type* clone() const override
{
return new DihedralAngleInteraction(kind_, container_type(potentials));
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
const std::size_t idx3 = idxp.first[3];

const auto& r_i = sys.position(idx0);
const auto& r_j = sys.position(idx1);
const auto& r_k = sys.position(idx2);
const auto& r_l = sys.position(idx3);

const coordinate_type r_ij = sys.adjust_direction(r_j, r_i);
const coordinate_type r_kj = sys.adjust_direction(r_j, r_k);
const coordinate_type r_lk = sys.adjust_direction(r_k, r_l);
const coordinate_type r_kl = real_type(-1.0) * r_lk;

const real_type r_kj_lensq  = math::length_sq(r_kj);
const real_type r_kj_rlen   = math::rsqrt(r_kj_lensq);
const real_type r_kj_rlensq = r_kj_rlen * r_kj_rlen;
const real_type r_kj_len    = r_kj_rlen * r_kj_lensq;

const coordinate_type m = math::cross_product(r_ij, r_kj);
const coordinate_type n = math::cross_product(r_kj, r_kl);
const real_type m_lensq = math::length_sq(m);
const real_type n_lensq = math::length_sq(n);

const real_type dot_mn  = math::dot_product(m, n) *
math::rsqrt(m_lensq * n_lensq);
const real_type cos_phi = math::clamp<real_type>(dot_mn, -1, 1);
const real_type phi     =
std::copysign(std::acos(cos_phi), math::dot_product(r_ij, n));

const real_type coef = -(idxp.second.derivative(phi));

if(NeedEnergy)
{
E += idxp.second.potential(phi);
}

const coordinate_type Fi = ( coef * r_kj_len / m_lensq) * m;
const coordinate_type Fl = (-coef * r_kj_len / n_lensq) * n;

const real_type coef_ijk = math::dot_product(r_ij, r_kj) * r_kj_rlensq;
const real_type coef_jkl = math::dot_product(r_kl, r_kj) * r_kj_rlensq;

const auto Fj = (coef_ijk - real_type(1.0)) * Fi - coef_jkl * Fl;
const auto Fk = (coef_jkl - real_type(1.0)) * Fl - coef_ijk * Fi;

const auto thread_id = omp_get_thread_num();

sys.force_thread(thread_id, idx0) += Fi;
sys.force_thread(thread_id, idx1) += Fj;
sys.force_thread(thread_id, idx2) += Fk;
sys.force_thread(thread_id, idx3) += Fl;

if(NeedVirial)
{
sys.virial_thread(thread_id) +=
math::tensor_product(r_j + r_ij,        Fi) +
math::tensor_product(r_j,               Fj) +
math::tensor_product(r_j + r_kj,        Fk) +
math::tensor_product(r_j + r_kj + r_lk, Fl);
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
#include <mjolnir/forcefield/local/ClementiDihedralPotential.hpp>
#include <mjolnir/forcefield/local/PeriodicGaussianPotential.hpp>
#include <mjolnir/forcefield/local/CosinePotential.hpp>
#include <mjolnir/forcefield/FLP/FlexibleLocalDihedralPotential.hpp>

namespace mjolnir
{

extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>, ClementiDihedralPotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>, ClementiDihedralPotential<float> >;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>, ClementiDihedralPotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>, ClementiDihedralPotential<float> >;

extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>, PeriodicGaussianPotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>, PeriodicGaussianPotential<float> >;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>, PeriodicGaussianPotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>, PeriodicGaussianPotential<float> >;

extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>, FlexibleLocalDihedralPotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>, FlexibleLocalDihedralPotential<float> >;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>, FlexibleLocalDihedralPotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>, FlexibleLocalDihedralPotential<float> >;

extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>, CosinePotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>, CosinePotential<float> >;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>, CosinePotential<double>>;
extern template class DihedralAngleInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>, CosinePotential<float> >;

} 
#endif 


#endif 
