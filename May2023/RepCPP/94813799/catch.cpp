
#include <pybind11/embed.h>

#ifdef _MSC_VER
#  pragma warning(disable: 4996)
#endif

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

namespace py = pybind11;

int main(int argc, char *argv[]) {
py::scoped_interpreter guard{};
auto result = Catch::Session().run(argc, argv);

return result < 0xff ? result : 0xff;
}
