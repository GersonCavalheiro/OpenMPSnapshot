

#ifndef BOOST_UUID_RANDOM_GENERATOR_HPP
#define BOOST_UUID_RANDOM_GENERATOR_HPP

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/move/core.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/tti/has_member_function.hpp>
#include <boost/uuid/detail/random_provider.hpp>
#include <boost/uuid/uuid.hpp>
#include <limits>

namespace boost {
namespace uuids {

namespace detail {
template<class U>
U& set_uuid_random_vv(U& u)
{
*(u.begin() + 8) &= 0xBF;
*(u.begin() + 8) |= 0x80;

*(u.begin() + 6) &= 0x4F; 
*(u.begin() + 6) |= 0x40; 

return u;
}

BOOST_TTI_HAS_MEMBER_FUNCTION(seed)
}

template <typename UniformRandomNumberGenerator>
class basic_random_generator
{
BOOST_MOVABLE_BUT_NOT_COPYABLE(basic_random_generator)

private:
typedef uniform_int<unsigned long> distribution_type;
typedef variate_generator<UniformRandomNumberGenerator*, distribution_type> generator_type;

struct impl
{
generator_type generator;

explicit impl(UniformRandomNumberGenerator* purng_arg) :
generator(purng_arg, distribution_type((std::numeric_limits<unsigned long>::min)(), (std::numeric_limits<unsigned long>::max)()))
{
}

virtual ~impl() {}

BOOST_DELETED_FUNCTION(impl(impl const&))
BOOST_DELETED_FUNCTION(impl& operator= (impl const&))
};

struct urng_holder
{
UniformRandomNumberGenerator urng;
};

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4355)
#endif

struct self_contained_impl :
public urng_holder,
public impl
{
self_contained_impl() : impl(&this->urng_holder::urng)
{
}
};

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

public:
typedef uuid result_type;

basic_random_generator() : m_impl(new self_contained_impl())
{
seed(static_cast< self_contained_impl* >(m_impl)->urng);
}

explicit basic_random_generator(UniformRandomNumberGenerator& gen) : m_impl(new impl(&gen))
{
}

explicit basic_random_generator(UniformRandomNumberGenerator* gen) : m_impl(new impl(gen))
{
BOOST_ASSERT(!!gen);
}

basic_random_generator(BOOST_RV_REF(basic_random_generator) that) BOOST_NOEXCEPT : m_impl(that.m_impl)
{
that.m_impl = 0;
}

basic_random_generator& operator= (BOOST_RV_REF(basic_random_generator) that) BOOST_NOEXCEPT
{
delete m_impl;
m_impl = that.m_impl;
that.m_impl = 0;
return *this;
}

~basic_random_generator()
{
delete m_impl;
}

result_type operator()()
{
result_type u;

int i=0;
unsigned long random_value = m_impl->generator();
for (uuid::iterator it = u.begin(), end = u.end(); it != end; ++it, ++i) {
if (i==sizeof(unsigned long)) {
random_value = m_impl->generator();
i = 0;
}

*it = static_cast<uuid::value_type>((random_value >> (i*8)) & 0xFF);
}

return detail::set_uuid_random_vv(u);
}

private:

template<class MaybePseudoRandomNumberGenerator>
typename boost::enable_if<detail::has_member_function_seed<MaybePseudoRandomNumberGenerator, void> >::type
seed(MaybePseudoRandomNumberGenerator& rng)
{
detail::random_provider seeder;
rng.seed(seeder);
}

template<class MaybePseudoRandomNumberGenerator>
typename boost::disable_if<detail::has_member_function_seed<MaybePseudoRandomNumberGenerator, void> >::type
seed(MaybePseudoRandomNumberGenerator&)
{
}

impl* m_impl;
};

class random_generator_pure
{
BOOST_MOVABLE_BUT_NOT_COPYABLE(random_generator_pure)

public:
typedef uuid result_type;

BOOST_DEFAULTED_FUNCTION(random_generator_pure(), {})

random_generator_pure(BOOST_RV_REF(random_generator_pure) that) BOOST_NOEXCEPT :
prov_(boost::move(that.prov_))
{
}

random_generator_pure& operator= (BOOST_RV_REF(random_generator_pure) that) BOOST_NOEXCEPT
{
prov_ = boost::move(that.prov_);
return *this;
}

result_type operator()()
{
result_type result;
prov_.get_random_bytes(&result, sizeof(result_type));
return detail::set_uuid_random_vv(result);
}

private:
detail::random_provider prov_;
};

#if defined(BOOST_UUID_RANDOM_GENERATOR_COMPAT)
typedef basic_random_generator<mt19937> random_generator;
#else
typedef random_generator_pure random_generator;
typedef basic_random_generator<mt19937> random_generator_mt19937;
#endif

}} 

#endif 
