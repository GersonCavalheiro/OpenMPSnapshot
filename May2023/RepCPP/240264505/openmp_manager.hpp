

#ifndef LBT_OPENMP_MANAGER
#define LBT_OPENMP_MANAGER
#pragma once

#include <ostream>


namespace lbt {


class OpenMpManager {
public:

static OpenMpManager& getInstance() noexcept;


static bool setNestedParallelism(bool const is_nested) noexcept;


static bool setThreadsNum(int const number_of_threads) noexcept;


static int getThreadsMax() noexcept;


static int getThreadsNum() noexcept;


static int getThreadsCurrent() noexcept;


friend std::ostream& operator<< (std::ostream& os, OpenMpManager const& openmp_manager) noexcept;

OpenMpManager& operator=(OpenMpManager const&) = default;

protected:

OpenMpManager() noexcept;
OpenMpManager(OpenMpManager const&) = delete;
OpenMpManager(OpenMpManager&&) = delete;
OpenMpManager& operator=(OpenMpManager&&) = delete;

static int threads_max; 
static int threads_num; 
};

}

#endif 
