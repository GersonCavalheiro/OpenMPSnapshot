

#ifndef TPP_TEST_TESTSUITE_PARALLEL_HPP
#define TPP_TEST_TESTSUITE_PARALLEL_HPP

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "test/testsuite.hpp"

namespace tpp
{
namespace intern
{
namespace test
{
class testsuite_parallel : public testsuite
{
public:
static auto
create(char const* name_) -> testsuite_ptr {
return std::make_shared<testsuite_parallel>(enable{}, name_);
}

void
run() override {
if (m_state != IS_DONE) {
duration d;
if (m_testcases.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
throw std::overflow_error("Too many testcases! Size would overflow loop variant.");
}
auto const tc_size{static_cast<std::int64_t>(m_testcases.size())};
m_stats.m_num_tests = m_testcases.size();
streambuf_proxies<streambuf_proxy_omp> bufs;
m_setup_fn();
#pragma omp parallel default(shared)
{  
std::size_t fails{0};
std::size_t errs{0};
#pragma omp for schedule(dynamic)
for (std::int64_t i = 0; i < tc_size; ++i) {
auto& tc{m_testcases[static_cast<std::size_t>(i)]};
if (tc.result() == testcase::IS_UNDONE) {
m_pretest_fn();
tc();
switch (tc.result()) {
case testcase::HAS_FAILED: ++fails; break;
case testcase::HAD_ERROR: ++errs; break;
default: break;
}
m_posttest_fn();
tc.cout(bufs.cout.str());
tc.cerr(bufs.cerr.str());
}
}
#pragma omp critical
{  
m_stats.m_num_fails += fails;
m_stats.m_num_errs += errs;
}  
}      
m_teardown_fn();
m_state = IS_DONE;
m_stats.m_elapsed_t += d.get();
}
}

testsuite_parallel(enable e_, char const* name_) : testsuite(e_, name_) {}
};
}  
}  
}  

#endif  
