

#include <GASPI.h>

#include "common/Environment.hpp"

using namespace tagaspi;

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

gaspi_return_t
tagaspi_proc_init(const gaspi_timeout_t timeout_ms)
{
assert(!_env.enabled);

gaspi_return_t eret;
eret = gaspi_proc_init(timeout_ms);
if (eret == GASPI_SUCCESS) {
Environment::initialize();
}
return eret;
}

gaspi_return_t
tagaspi_proc_term(const gaspi_timeout_t timeout_ms)
{
if (_env.enabled) {
Environment::finalize();
}
return gaspi_proc_term(timeout_ms);
}

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop
