#ifndef MJOLNIR_OMP_EXTERNAL_POSITION_RESTRAINT_INTERACTION_HPP
#define MJOLNIR_OMP_EXTERNAL_POSITION_RESTRAINT_INTERACTION_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/forcefield/external/PositionRestraintInteraction.hpp>

namespace mjolnir
{

template<typename realT, template<typename, typename> class boundaryT,
typename potentialT>
class PositionRestraintInteraction<
OpenMPSimulatorTraits<realT, boundaryT>, potentialT
> final: public ExternalForceInteractionBase<OpenMPSimulatorTraits<realT, boundaryT>>
{
public:
using traits_type     = OpenMPSimulatorTraits<realT, boundaryT>;
using potential_type  = potentialT;
using base_type       = ExternalForceInteractionBase<traits_type>;
using real_type       = typename base_type::real_type;
using coordinate_type = typename base_type::coordinate_type;
using system_type     = typename base_type::system_type;
using boundary_type   = typename base_type::boundary_type;

using potential_container_type = std::vector<
std::tuple<std::size_t, coordinate_type, potential_type>
>;

public:

explicit PositionRestraintInteraction(potential_container_type&& pots)
: potentials_(std::move(pots))
{}
~PositionRestraintInteraction() override = default;

void calc_force (system_type& sys) const noexcept override
{
#pragma omp parallel for
for(std::size_t i=0; i<this->potentials_.size(); ++i)
{
const auto& pots = this->potentials_[i];

const auto  pid = std::get<0>(pots); 
const auto& pos = std::get<1>(pots); 
const auto& pot = std::get<2>(pots); 

const auto& r_i = sys.position(pid);
const auto  dr  = sys.adjust_direction(r_i, pos);

const auto rlen = math::rlength(dr); 
const auto dist = math::length_sq(dr) * rlen;
const auto dV   = pot.derivative(dist);
if(dV == 0.0){continue;}

sys.force_thread(omp_get_thread_num(), pid) += (dV * rlen) * dr;
}
return ;
}

real_type calc_energy(system_type const& sys) const noexcept override
{
real_type E = 0.0;
#pragma omp parallel for reduction(+:E)
for(std::size_t i=0; i<this->potentials_.size(); ++i)
{
const auto& pots = this->potentials_[i];

const auto  pid = std::get<0>(pots); 
const auto& pos = std::get<1>(pots); 
const auto& pot = std::get<2>(pots); 

const auto& r_i = sys.position(pid);
const auto  dr  = sys.adjust_direction(r_i, pos);

const auto dist = math::length(dr);
E += pot.potential(dist);
}
return E;
}

real_type calc_force_and_energy(system_type& sys) const noexcept override
{
real_type E = 0.0;
#pragma omp parallel for reduction(+:E)
for(std::size_t i=0; i<this->potentials_.size(); ++i)
{
const auto& pots = this->potentials_[i];

const auto  pid = std::get<0>(pots); 
const auto& pos = std::get<1>(pots); 
const auto& pot = std::get<2>(pots); 

const auto& r_i = sys.position(pid);
const auto  dr  = sys.adjust_direction(r_i, pos);

const auto rlen = math::rlength(dr); 
const auto dist = math::length_sq(dr) * rlen;
const auto dV   = pot.derivative(dist);
if(dV == 0.0){continue;}

E += pot.potential(dist);
sys.force_thread(omp_get_thread_num(), pid) += (dV * rlen) * dr;
}
return E;
}


void initialize(const system_type& sys) override
{
for(auto& pot : potentials_)
{
std::get<2>(pot).update(sys);
}
return;
}

void update(const system_type& sys) override
{
for(auto& pot : potentials_)
{
std::get<2>(pot).update(sys);
}
return;
}

void reduce_margin(const real_type, const system_type&) override {return;}
void  scale_margin(const real_type, const system_type&) override {return;}

std::string name() const override
{return "PositionRestraint:"_s + potential_type::name();}


base_type* clone() const override
{
return new PositionRestraintInteraction(
potential_container_type(potentials_));
}


private:

potential_container_type potentials_;
};

} 
#endif
