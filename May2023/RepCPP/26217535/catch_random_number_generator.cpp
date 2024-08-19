

#include <catch2/internal/catch_random_number_generator.hpp>

namespace Catch {

namespace {

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4146) 
#endif
uint32_t rotate_right(uint32_t val, uint32_t count) {
const uint32_t mask = 31;
count &= mask;
return (val >> count) | (val << (-count & mask));
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}


SimplePcg32::SimplePcg32(result_type seed_) {
seed(seed_);
}


void SimplePcg32::seed(result_type seed_) {
m_state = 0;
(*this)();
m_state += seed_;
(*this)();
}

void SimplePcg32::discard(uint64_t skip) {
for (uint64_t s = 0; s < skip; ++s) {
static_cast<void>((*this)());
}
}

SimplePcg32::result_type SimplePcg32::operator()() {
const uint32_t xorshifted = static_cast<uint32_t>(((m_state >> 18u) ^ m_state) >> 27u);
const auto output = rotate_right(xorshifted, m_state >> 59u);

m_state = m_state * 6364136223846793005ULL + s_inc;

return output;
}

bool operator==(SimplePcg32 const& lhs, SimplePcg32 const& rhs) {
return lhs.m_state == rhs.m_state;
}

bool operator!=(SimplePcg32 const& lhs, SimplePcg32 const& rhs) {
return lhs.m_state != rhs.m_state;
}
}
