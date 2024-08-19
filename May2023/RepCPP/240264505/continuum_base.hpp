

#ifndef LBT_CONTINUUM_BASE
#define LBT_CONTINUUM_BASE
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>


namespace lbt {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
class ContinuumBase {
public:

virtual void setP(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept = 0;


virtual void setU(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept = 0;


virtual void setV(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept = 0;


virtual void setW(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept = 0;


virtual T getP(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;


virtual T getU(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;


virtual T getV(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;


virtual T getW(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept = 0;


virtual void save(double const timestamp) const noexcept = 0;

protected:

ContinuumBase(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ, std::filesystem::path const& output_path) noexcept
: NX{NX}, NY{NY}, NZ{NZ}, output_path{output_path} {
return;
}

ContinuumBase() = delete;
ContinuumBase(ContinuumBase const&) = default;
ContinuumBase& operator = (ContinuumBase const&) = default;
ContinuumBase(ContinuumBase&&) = default;
ContinuumBase& operator = (ContinuumBase&&) = default;

std::int32_t NX;
std::int32_t NY;
std::int32_t NZ;
std::filesystem::path output_path;
};

}

#endif 
