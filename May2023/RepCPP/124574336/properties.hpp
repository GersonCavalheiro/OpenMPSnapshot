

#ifndef BOOST_FIBERS_PROPERTIES_HPP
#define BOOST_FIBERS_PROPERTIES_HPP

#include <boost/fiber/detail/config.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif

# if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable:4275)
# endif

namespace boost {
namespace fibers {

class context;

namespace algo {

class algorithm;

}

class BOOST_FIBERS_DECL fiber_properties {
protected:
context         *   ctx_;
algo::algorithm *   algo_{ nullptr };

void notify() noexcept;

public:

explicit fiber_properties( context * ctx) noexcept :
ctx_{ ctx } {
}

virtual ~fiber_properties() = default;

void set_algorithm( algo::algorithm * algo) noexcept {
algo_ = algo;
}
};

}} 

# if defined(BOOST_MSVC)
# pragma warning(pop)
# endif

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif

#endif 
