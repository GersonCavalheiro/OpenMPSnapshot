
#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <cstdio>

class out_buff : public std::stringbuf {
std::FILE* m_stream;
public:
out_buff(std::FILE* stream):m_stream(stream) {}
~out_buff();
int sync() override {
int ret = 0;
for (unsigned char c : str()) {
if (putc(c, m_stream) == EOF) {
ret = -1;
break;
}
}
str("");
return ret;
}
};

out_buff::~out_buff() { pubsync(); }

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wexit-time-destructors" 
#endif

namespace Catch {
std::ostream& cout() {
static std::ostream ret(new out_buff(stdout));
return ret;
}
std::ostream& clog() {
static std::ostream ret(new out_buff(stderr));
return ret;
}
std::ostream& cerr() {
return clog();
}
}


TEST_CASE("This binary uses putc to write out output", "[compilation-only]") {
SUCCEED("Nothing to test.");
}
