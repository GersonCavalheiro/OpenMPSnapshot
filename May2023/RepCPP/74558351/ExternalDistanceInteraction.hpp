#ifndef MJOLNIR_OMP_EXTERNAL_DISTANCE_INTERACTION_HPP
#define MJOLNIR_OMP_EXTERNAL_DISTANCE_INTERACTION_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/forcefield/external/ExternalDistanceInteraction.hpp>

namespace mjolnir
{


template<typename realT, template<typename, typename> class boundaryT,
typename potentialT, typename shapeT>
class ExternalDistanceInteraction<
OpenMPSimulatorTraits<realT, boundaryT>, potentialT, shapeT>
final : public ExternalForceInteractionBase<OpenMPSimulatorTraits<realT, boundaryT>>
{
public:
using traits_type     = OpenMPSimulatorTraits<realT, boundaryT>;
using potential_type  = potentialT;
using shape_type      = shapeT;
using base_type       = ExternalForceInteractionBase<traits_type>;
using real_type       = typename base_type::real_type;
using coordinate_type = typename base_type::coordinate_type;
using system_type     = typename base_type::system_type;
using boundary_type   = typename base_type::boundary_type;

public:

ExternalDistanceInteraction(shape_type&& shape, potential_type&& pot)
: shape_(std::move(shape)), potential_(std::move(pot))
{}
~ExternalDistanceInteraction() override {}

void calc_force (system_type& sys) const noexcept override
{
const auto& neighbors = this->shape_.neighbors();
#pragma omp parallel for
for(std::size_t idx=0; idx < neighbors.size(); ++idx)
{
const std::size_t i = neighbors[idx];
const auto& ri = sys.position(i);

const real_type dist = this->shape_.calc_distance(ri, sys.boundary());
const real_type dV   = this->potential_.derivative(i, dist);
if(dV == 0.0){continue;}

const auto f = shape_.calc_force_direction(ri, sys.boundary());
sys.force_thread(omp_get_thread_num(), i) += -dV * f;
}
return ;
}

real_type calc_energy(system_type const& sys) const noexcept override
{
real_type E = 0.0;
const auto& neighbors = this->shape_.neighbors();
#pragma omp parallel for reduction(+:E)
for(std::size_t idx=0; idx < neighbors.size(); ++idx)
{
const std::size_t i = neighbors[idx];
const auto&    ri = sys.position(i);
const real_type d = this->shape_.calc_distance(ri, sys.boundary());
E += this->potential_.potential(i, d);
}
return E;
}

real_type calc_force_and_energy(system_type& sys) const noexcept override
{
real_type E = 0.0;
const auto& neighbors = this->shape_.neighbors();
#pragma omp parallel for reduction(+:E)
for(std::size_t idx=0; idx < neighbors.size(); ++idx)
{
const std::size_t i = neighbors[idx];
const auto&     ri   = sys.position(i);
const real_type dist = this->shape_.calc_distance(ri, sys.boundary());
const real_type dV   = this->potential_.derivative(i, dist);

if(dV == 0.0){continue;}

E += this->potential_.potential(i, dist);

const auto f = shape_.calc_force_direction(ri, sys.boundary());
sys.force_thread(omp_get_thread_num(), i) += -dV * f;
}
return E;
}



void initialize(const system_type& sys) override
{
this->potential_.update(sys); 
this->shape_.initialize(sys, this->potential_);
}


void update(const system_type& sys) override
{
this->potential_.update(sys); 
this->shape_.initialize(sys, this->potential_);
}

void reduce_margin(const real_type dmargin, const system_type& sys) override
{
this->shape_.reduce_margin(dmargin, sys);
}
void scale_margin(const real_type scale, const system_type& sys) override
{
this->shape_.scale_margin(scale, sys);
}


std::string name() const override
{return "ExternalDistance:"_s + potential_.name();}

base_type* clone() const override
{
return new ExternalDistanceInteraction(
shape_type(shape_), potential_type(potential_));
}

private:

shape_type     shape_;
potential_type potential_;
};

} 
#endif
