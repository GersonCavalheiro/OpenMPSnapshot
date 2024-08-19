#pragma once

#include <sstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

#include "algo/interfaces/Instrumental.h"

#define ASSERT_EQUAL(x, y) {                            \
std::ostringstream __assert_equal_private_os;       \
__assert_equal_private_os                           \
<< #x << " != " << #y << ", "                       \
<< __FILE__ << ":" << __LINE__;                     \
AssertEqual(x, y, __assert_equal_private_os.str()); \
}


#define ASSERT(x) {                        \
std::ostringstream os;                 \
os << #x << " is false, "              \
<< __FILE__ << ":" << __LINE__;     \
Assert(x, os.str());                   \
}

#define RUN_TEST(tr, func) tr.RunTest(func, #func)

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& s) {
os << "{";
bool first = true;
for (const auto& x : s) {
if (!first) {
os << ", ";
}
first = false;
os << x;
}

return os << "}";
}

template<class T, class U>
void AssertEqual(const T& t, const U& u, const std::string& hint = {}) {
if (t != u) {
std::ostringstream os;
os << "Assertion failed.";
if (!hint.empty()) {
os << " hint: " << hint;
}

throw std::runtime_error(os.str());
}
}

inline void Assert(bool b, const std::string& hint) {
AssertEqual(b, true, hint);
}

class TestRunner {
private:
int fail_count = 0;

public:
void RunTest(const std::function<void()>& func, const std::string& testName) {
try {
func();
std::cout << testName << " OK" << std::endl;
}
catch (std::exception& e) {
++fail_count;
std::cerr << testName << " fail: " << e.what() << std::endl;

}
catch (...) {
++fail_count;
std::cerr << "Unknown exception caught" << std::endl;
}
}

~TestRunner() {
if (fail_count > 0) {
std::cerr << fail_count << " unit tests failed. Terminate" << std::endl;
exit(1);
}
}
};