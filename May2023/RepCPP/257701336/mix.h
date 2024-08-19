
#ifndef UTIL_MIX_H_
#define UTIL_MIX_H_

#include <stddef.h>
#include <limits>

namespace re2 {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif

class HashMix {
public:
HashMix() : hash_(1) {}
explicit HashMix(size_t val) : hash_(val + 83) {}
void Mix(size_t val) {
static const size_t kMul = static_cast<size_t>(0xdc3eb94af8ab4c93ULL);
hash_ *= kMul;
hash_ = ((hash_ << 19) |
(hash_ >> (std::numeric_limits<size_t>::digits - 19))) + val;
}
size_t get() const { return hash_; }
private:
size_t hash_;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  

#endif  
