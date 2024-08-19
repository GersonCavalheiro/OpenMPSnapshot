

#ifndef LBT_SIMULATION
#define LBT_SIMULATION
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>

#include <nlohmann/json.hpp>

#include "continuum/continuum.hpp"

#include "general/vtk_utilities.hpp"
#include "geometry/vtk_import.hpp"
#include "base_simulation.hpp"


namespace lbt {


template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
std::array<T, 3> parseArray(nlohmann::json const& j) {
std::array<T, 3> const arr {j["x"].get<T>(), j["y"].get<T>(), j["z"].get<T>()};
return arr;
}


template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
nlohmann::json toJson(std::array<T, 3> const& arr) noexcept {
nlohmann::json j{};
j["x"] = arr.at(0);
j["y"] = arr.at(1);
j["z"] = arr.at(2);
return j;
}

namespace settings {


class Discretisation {
using json = nlohmann::json;

public:

constexpr Discretisation(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ) noexcept 
: NX{NX}, NY{NY}, NZ{NZ} {
return;
};


Discretisation(json const& j) {
NX = j["NX"].get<std::int32_t>();
NY = j["NY"].get<std::int32_t>();
NZ = j["NZ"].get<std::int32_t>();
return;
}


json toJson() const noexcept {
json j {};
j["NX"] = NX;
j["NY"] = NY;
j["NZ"] = NZ;
return j;
}


constexpr std::tuple<std::int32_t, std::int32_t, std::int32_t> getDiscretisation() const noexcept {
return std::make_tuple(NX, NY, NZ);
}


constexpr auto getNx() const noexcept {
return NX;
}


constexpr auto getNy() const noexcept {
return NY;
}


constexpr auto getNz() const noexcept {
return NZ;
}
protected:
std::int32_t NX; 
std::int32_t NY; 
std::int32_t NZ; 
};


class Physics {
using json = nlohmann::json;

public:

constexpr Physics(double const density, double const kinematic_viscosity) noexcept 
: density{density}, kinematic_viscosity{kinematic_viscosity} {
return;
}


Physics(json const& j) {
density = j["density"].get<double>();
kinematic_viscosity = j["kinematicViscosity"].get<double>();
return;
}


json toJson() const noexcept {
json j {};
j["density"] = density;
j["kinematicViscosity"] = kinematic_viscosity;
return j;
}


constexpr auto getDensity() const noexcept {
return density;
}


constexpr auto getKinematicViscosity() const noexcept {
return kinematic_viscosity;
}

protected:
double density; 
double kinematic_viscosity; 
};


template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
class InitialConditions {
using json = nlohmann::json;

public:

constexpr InitialConditions(std::array<T, 3> const& initial_velocity) noexcept 
: initial_velocity{initial_velocity} {
return;
}


InitialConditions(json const& j) {
initial_velocity = parseArray<T>(j["velocity"]);
return;
}


json toJson() const noexcept {
return lbt::toJson(initial_velocity);
}


constexpr auto getInitialVelocity() const noexcept {
return initial_velocity;
}

protected:
std::array<T, 3> initial_velocity; 
};


class Geometry {
using json = nlohmann::json;

public:

Geometry(std::vector<std::string> const& files, std::array<double, 6> const& bounding_box,
double const reduction_rate) noexcept
: files{files}, bounding_box{bounding_box}, reduction_rate{reduction_rate} {
return;
}


Geometry(json const& j) {
json const& j_files {j["models"]};
for (auto const& f: j_files) {
files.emplace_back(f.get<std::string>());
}
json const bounding_box_min {j["boundingBox"]["min"]};
json const bounding_box_max {j["boundingBox"]["max"]};
std::array<double, 6> const bounding_box {bounding_box_min["x"], bounding_box_max["x"],
bounding_box_min["y"], bounding_box_max["y"],
bounding_box_min["z"], bounding_box_max["z"]};
if (j.contains("reductionRate") == true) {
reduction_rate = j["reductionRate"];
}
return;
}


json toJson() const {
json ja {};
for (auto const& f:files) {
ja.push_back(f);
}
json j {};
j["models"] = ja;

j["boundingBox"]["min"] = lbt::toJson(std::array<double,3>{bounding_box.at(0), bounding_box.at(2), bounding_box.at(4)});
j["boundingBox"]["max"] = lbt::toJson(std::array<double,3>{bounding_box.at(1), bounding_box.at(3), bounding_box.at(5)});

if (reduction_rate != std::nullopt) {
j["reductionRate"] = reduction_rate.value();
}
return j;
}


std::vector<std::filesystem::path> getFilesWithPath(std::filesystem::path const& parent_directory) const noexcept {
std::vector<std::filesystem::path> files_with_path {};
for (auto const& f: files) {
files_with_path.emplace_back(parent_directory / f);
}
return files_with_path;
}


constexpr auto getBoundingBox() const noexcept {
return bounding_box;
}


constexpr auto getReductionRate() const noexcept {
return reduction_rate;
}

protected:
std::vector<std::string> files; 
std::array<double, 6> bounding_box; 
std::optional<double> reduction_rate; 
};


class Parallelism {
using json = nlohmann::json;

public:

constexpr Parallelism(int const number_of_threads) noexcept 
: number_of_threads{number_of_threads} {
return;
}


Parallelism(json const& j) {
number_of_threads = j["numberOfThreads"].get<int>();
return;
}


json toJson() const noexcept {
json j {};
j["numberOfThreads"] = number_of_threads;
return j;
}


constexpr auto getNumberOfThreads() const noexcept {
return number_of_threads;
}

protected:
int number_of_threads; 
};


class Output {
using json = nlohmann::json;

public:

Output(DataType const data_type, std::string const& folder, double const first_output, 
double const write_interval) noexcept 
: data_type{data_type}, folder{folder}, first_output{first_output}, write_interval{write_interval} {
return;
}


Output(json const& j) {
data_type = DataType::MHD;
if (j["dataFormat"] == "vtk") {
data_type = DataType::VTK;
}
folder = j["outputFolder"].get<std::string>();
first_output = j["firstOutput"].get<double>();
write_interval = j["writeInterval"].get<double>();
return;
}


json toJson() const noexcept {
json j {};
if (data_type == DataType::VTK) {
j["dataFormat"] = "vtk";
} else if (data_type == DataType::MHD) {
j["dataFormat"] == "mhd";
}
j["outputFolder"] = folder;
j["firstOutput"] = first_output;
j["writeInterval"] = write_interval;
return j;
}


std::filesystem::path getFullOutputPath(std::filesystem::path const& parent_directory) const noexcept {
std::filesystem::path const full_path {parent_directory / folder};
return full_path;
}


constexpr auto getFormat() const noexcept {
return data_type;
}


constexpr auto getFirstOutput() const noexcept {
return first_output;
}


constexpr auto getWriteInterval() const noexcept {
return write_interval;
}

protected:
DataType data_type; 
std::string folder; 
double first_output; 
double write_interval; 
};


class Times {
using json = nlohmann::json;

public:

constexpr Times(double const warmup, double const end) noexcept 
: warmup{warmup}, end{end} {
return;
}


Times(json const& j) {
warmup = j["warmUp"].get<double>();
if (j.contains("startTime") == true) {
start = j["startTime"];
}
end = j["endTime"].get<double>();
return;
}


json toJson() const noexcept {
json j {};
j["warmUp"] = warmup;
if (start != std::nullopt) {
j["startTime"] = start.value();
}
j["endTime"] = end;
return j;
}


constexpr auto getWarmupTime() const noexcept {
return warmup;
}


constexpr auto getStartTime() const noexcept {
return start;
}


constexpr auto getEndTime() const noexcept {
return end;
}

protected:
double warmup; 
std::optional<double> start; 
double end; 
};
}


template <template <typename T> class LT, typename T, unsigned int NPOP = 1>
class Simulation final: public BaseSimulation {
using json = nlohmann::json;

public:
Simulation() = delete;


Simulation(json const& settings, std::filesystem::path const& parent_directory)
: discretisation{settings["discretisation"]}, physics{settings["physics"]},
initial_conditions{settings["initial_conditions"]}, geometry{settings["geometry"]},
parallelism{settings["target"]}, output{settings["output"]}, times{settings["times"]} {








return;
}


json toJson() const noexcept override final {
json settings {};

settings["discretisation"] = discretisation.toJson();
settings["physics"] = physics.toJson();
settings["initial_conditions"] = initial_conditions.toJson();
settings["geometry"] = geometry.toJson();
settings["target"] = parallelism.toJson();
settings["output"] = output.toJson();
settings["times"] = times.toJson();

return settings;
}


void run() noexcept override final {
return;
}

private:
settings::Discretisation discretisation;
settings::Physics physics;
settings::InitialConditions<T> initial_conditions;
settings::Geometry geometry;
settings::Parallelism parallelism;
settings::Output output;
settings::Times times;

std::shared_ptr<Continuum<T>> continuum;

};

}

#endif 
