#ifndef MJOLNIR_OMP_PROTEIN_DNA_NON_SPECIFIC_INTERACTION_HPP
#define MJOLNIR_OMP_PROTEIN_DNA_NON_SPECIFIC_INTERACTION_HPP
#include <mjolnir/omp/OpenMPSimulatorTraits.hpp>
#include <mjolnir/omp/System.hpp>
#include <mjolnir/forcefield/PDNS/ProteinDNANonSpecificInteraction.hpp>

namespace mjolnir
{

template<typename realT, template<typename, typename> class boundaryT>
class ProteinDNANonSpecificInteraction<OpenMPSimulatorTraits<realT, boundaryT>>
final : public GlobalInteractionBase<OpenMPSimulatorTraits<realT, boundaryT>>
{
public:

using traits_type     = OpenMPSimulatorTraits<realT, boundaryT>;
using base_type       = GlobalInteractionBase<traits_type>;
using real_type       = typename base_type::real_type;
using coordinate_type = typename base_type::coordinate_type;
using system_type     = typename base_type::system_type;
using topology_type   = typename base_type::topology_type;
using boundary_type   = typename base_type::boundary_type;
using potential_type  = ProteinDNANonSpecificPotential<real_type>;
using partition_type  = SpatialPartition<traits_type, potential_type>;
using parameter_list_type = ProteinDNANonSpecificParameterList<traits_type>;

public:

ProteinDNANonSpecificInteraction(potential_type&& pot, parameter_list_type&& para, partition_type&& part)
: potential_(std::move(pot)), parameters_(std::move(para)), partition_(std::move(part))
{}
~ProteinDNANonSpecificInteraction() {}

void initialize(const system_type& sys, const topology_type& topol) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
MJOLNIR_LOG_INFO("potential is PDNS");

this->potential_ .initialize(sys);
this->parameters_.initialize(sys, topol, this->potential_);
this->partition_ .initialize(sys, this->parameters_);
return;
}

void update(const system_type& sys, const topology_type& topol) override
{
MJOLNIR_GET_DEFAULT_LOGGER();
MJOLNIR_LOG_FUNCTION();
MJOLNIR_LOG_INFO("potential is PDNS");

this->potential_ .update(sys);
this->parameters_.update(sys, topol, this->potential_);
this->partition_ .initialize(sys, this->parameters_);
return;
}

void reduce_margin(const real_type dmargin, const system_type& sys) override
{
this->partition_.reduce_margin(dmargin, sys, this->parameters_);
return ;
}
void scale_margin(const real_type scale, const system_type& sys) override
{
this->partition_.scale_margin(scale, sys, this->parameters_);
return ;
}

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
MJOLNIR_GET_DEFAULT_LOGGER_DEBUG();
MJOLNIR_LOG_FUNCTION_DEBUG();

const auto rsigma         = parameters_.rsigma();
const auto delta          = parameters_.delta();
const auto delta2         = parameters_.delta2();
const auto pi_over_2delta = parameters_.pi_over_2delta();

real_type E = 0.0;
#pragma omp parallel for reduction(+:E)
for(std::size_t i=0; i < this->parameters_.contacts().size(); ++i)
{
const auto& contact = parameters_.contacts()[i];

const auto  P  = contact.P;
const auto& rP = sys.position(P);
for(const auto& ptnr : this->partition_.partners(P))
{
const auto  D  = ptnr.index;  
const auto& para = ptnr.parameter();
const auto  S3 = para.S3; 
const auto& rD = sys.position(D);



const auto rPD    = sys.adjust_direction(rP, rD); 
const auto lPD_sq = math::length_sq(rPD);
if(contact.r_cut_sq < lPD_sq)
{
continue;
}
const auto rlPD   = math::rsqrt(lPD_sq);
const auto  lPD   = lPD_sq * rlPD;
const auto f      = potential_.f(contact.r0, lPD, rsigma);


const auto& rPC    = sys.position(contact.PC);
const auto& rPN    = sys.position(contact.PN);
const auto rPNC    = sys.adjust_direction(rPN, rPC); 
const auto rlPNC   = math::rlength(rPNC);
const auto dotPNC  = math::dot_product(rPNC, rPD);
const auto cosPNC  = dotPNC * rlPD * rlPNC;
const auto theta   = std::acos(math::clamp<real_type>(cosPNC,-1,1));
const auto g_theta = potential_.g(contact.theta0, theta, delta, delta2, pi_over_2delta);

if(g_theta == real_type(0)) {continue;}


const auto& rS3   = sys.position(S3);
const auto rS3D   = sys.adjust_direction(rS3, rD); 
const auto rlS3D  = math::rlength(rS3D);
const auto dotS3D = math::dot_product(rPD, rS3D);
const auto cosS3D = dotS3D * rlS3D * rlPD;
const auto phi    = std::acos(math::clamp<real_type>(cosS3D,-1,1));
const auto g_phi  = potential_.g(contact.phi0, phi, delta, delta2, pi_over_2delta);

if(g_phi == real_type(0)) {continue;}


const auto k = contact.k;

MJOLNIR_LOG_DEBUG("protein = ", P, ", DNA = ", D, ", r0 = ", contact.r0);

E += k * f * g_theta * g_phi;
}
}
return E;
}

std::string name() const override {return "PDNSInteraction";}

parameter_list_type const& parameters() const noexcept {return parameters_;}
parameter_list_type&       parameters()       noexcept {return parameters_;}

base_type* clone() const override
{
return new ProteinDNANonSpecificInteraction(*this);
}

private: 

template<bool NeedEnergy, bool NeedVirial>
real_type calc_force_energy_virial_impl(system_type& sys) const noexcept
{
MJOLNIR_GET_DEFAULT_LOGGER_DEBUG();
MJOLNIR_LOG_FUNCTION_DEBUG();

constexpr auto tolerance = math::abs_tolerance<real_type>();

const auto rsigma         = parameters_.rsigma();
const auto delta          = parameters_.delta();
const auto delta2         = parameters_.delta2();
const auto pi_over_2delta = parameters_.pi_over_2delta();

real_type energy = 0;
#pragma omp parallel for reduction(+:energy)
for(std::size_t i=0; i < this->parameters_.contacts().size(); ++i)
{
const auto& contact = parameters_.contacts()[i];

const auto  P  = contact.P;
const auto& rP = sys.position(P);
for(const auto& ptnr : this->partition_.partners(P))
{
const auto  D  = ptnr.index;  
const auto& para = ptnr.parameter();
const auto  S3 = para.S3; 
const auto& rD = sys.position(D);

MJOLNIR_LOG_DEBUG("protein = ", P, ", DNA = ", D, ", r0 = ", contact.r0);



const auto rPD    = sys.adjust_direction(rP, rD); 
const auto lPD_sq = math::length_sq(rPD);
if(contact.r_cut_sq < lPD_sq)
{
continue;
}
const auto rlPD   = math::rsqrt(lPD_sq);
const auto  lPD   = lPD_sq * rlPD;
const auto f_df   = potential_.f_df(contact.r0, lPD, rsigma);

MJOLNIR_LOG_DEBUG("f = ", f_df.first, ", df = ", f_df.second);


const auto& rPC   = sys.position(contact.PC);
const auto& rPN   = sys.position(contact.PN);
const auto rPNC   = sys.adjust_direction(rPN, rPC); 
const auto rlPNC  = math::rlength(rPNC);
const auto dotPNC = math::dot_product(rPNC, rPD);
const auto cosPNC = dotPNC * rlPD * rlPNC;
const auto theta  = std::acos(math::clamp<real_type>(cosPNC,-1,1));

const auto g_dg_theta = potential_.g_dg(contact.theta0, theta, delta, delta2, pi_over_2delta);

MJOLNIR_LOG_DEBUG("g(theta) = ", g_dg_theta.first,
", dg(theta) = ", g_dg_theta.second);


const auto& rS3   = sys.position(S3);
const auto rS3D   = sys.adjust_direction(rS3, rD); 
const auto rlS3D  = math::rlength(rS3D);
const auto dotS3D = math::dot_product(rPD, rS3D);
const auto cosS3D = dotS3D * rlS3D * rlPD;
const auto phi    = std::acos(math::clamp<real_type>(cosS3D,-1,1));

const auto g_dg_phi = potential_.g_dg(contact.phi0, phi, delta, delta2, pi_over_2delta);

MJOLNIR_LOG_DEBUG("g(phi) = ", g_dg_phi.first,
", dg(phi) = ", g_dg_phi.second);

const auto k = contact.k;
const auto thread_id = omp_get_thread_num();

auto f_P  = math::make_coordinate<coordinate_type>(0,0,0);
auto f_D  = math::make_coordinate<coordinate_type>(0,0,0);
auto f_S3 = math::make_coordinate<coordinate_type>(0,0,0);
auto f_PN = math::make_coordinate<coordinate_type>(0,0,0);
auto f_PC = math::make_coordinate<coordinate_type>(0,0,0);

if(NeedEnergy)
{
energy += k * f_df.first * g_dg_theta.first * g_dg_phi.first;
}

if(g_dg_theta.first  != 0 && g_dg_phi.first  != 0)
{
MJOLNIR_LOG_DEBUG("calculating distance force");

const auto coef = rlPD * k *
f_df.second * g_dg_theta.first * g_dg_phi.first;
const auto F = -coef * rPD;

f_P -= F;
f_D += F;
}

if(g_dg_theta.second != 0 && g_dg_phi.first  != 0)
{
MJOLNIR_LOG_DEBUG("calculating theta force");

const auto deriv =
k * f_df.first * g_dg_theta.second * g_dg_phi.first;

const auto sin_theta = std::sin(theta);
const auto coef_sin  = deriv / std::max(sin_theta, tolerance);

const auto rPD_reg  = rlPD  * rPD;
const auto rPNC_reg = rlPNC * rPNC;

const auto F_P  = -coef_sin * rlPD  * (rPNC_reg - cosPNC * rPD_reg );
const auto F_PN = -coef_sin * rlPNC * (rPD_reg  - cosPNC * rPNC_reg);

f_D  -= F_P;
f_P  += F_P;
f_PN += F_PN;
f_PC -= F_PN;
}

if(g_dg_theta.first  != 0 && g_dg_phi.second != 0)
{
MJOLNIR_LOG_DEBUG("calculating phi force");

const auto deriv =
k * f_df.first * g_dg_theta.first * g_dg_phi.second;

const auto sin_phi  = std::sin(phi);
const auto coef_sin = deriv / std::max(sin_phi, tolerance);

const auto rPD_reg  = rlPD  * rPD;
const auto rS3D_reg = rlS3D * rS3D;

const auto F_P = -coef_sin * rlPD  * (rS3D_reg - cosS3D * rPD_reg);
const auto F_S = -coef_sin * rlS3D * (rPD_reg  - cosS3D * rS3D_reg);

f_P  += F_P;
f_D  -= F_P + F_S;
f_S3 += F_S;
}
sys.force_thread(thread_id, P)       += f_P;
sys.force_thread(thread_id, D)       += f_D;
sys.force_thread(thread_id, S3)      += f_S3;
sys.force_thread(thread_id, contact.PN) += f_PN;
sys.force_thread(thread_id, contact.PC) += f_PC;

if(NeedVirial)
{
sys.virial_thread(thread_id) +=
math::tensor_product(rP,       f_P) +
math::tensor_product(rP + rPD, f_D) +
math::tensor_product(sys.transpose(rS3, rP), f_S3) +
math::tensor_product(sys.transpose(rPN, rP), f_PN) +
math::tensor_product(sys.transpose(rPC, rP), f_PC) ;
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

namespace mjolnir
{
extern template class ProteinDNANonSpecificInteraction<OpenMPSimulatorTraits<double, UnlimitedBoundary>       >;
extern template class ProteinDNANonSpecificInteraction<OpenMPSimulatorTraits<float,  UnlimitedBoundary>       >;
extern template class ProteinDNANonSpecificInteraction<OpenMPSimulatorTraits<double, CuboidalPeriodicBoundary>>;
extern template class ProteinDNANonSpecificInteraction<OpenMPSimulatorTraits<float,  CuboidalPeriodicBoundary>>;
} 
#endif 
#endif
