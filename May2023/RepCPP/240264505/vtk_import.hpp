

#ifndef LBT_VTK_IMPORT
#define LBT_VTK_IMPORT
#pragma once

#include "../general/use_vtk.hpp"

#ifdef LBT_USE_VTK
#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include "../general/vtk_utilities.hpp"


namespace lbt {

enum class Axis {x, y, z};


class Importer {
public:
Importer(Importer const&) = delete;
Importer(Importer&&) = delete;
Importer& operator= (Importer const&) = delete;
Importer& operator= (Importer&&) = delete;


static vtkSmartPointer<vtkImageData> newImageData(int const resolution_x, std::array<double, 6> const& bounding_box,
char const fill_colour = Importer::background_colour) noexcept;


static vtkSmartPointer<vtkImageData> mergeImageData(vtkSmartPointer<vtkImageData> const& image_data_1,
vtkSmartPointer<vtkImageData> const& image_data_2) noexcept;


static vtkSmartPointer<vtkImageData> importFromFile(std::filesystem::path const& filename, int const resolution_x, double const reduction_rate, 
std::optional<std::array<double, 6>> const& opt_bounding_box) noexcept;


static vtkSmartPointer<vtkPolyData> load(std::filesystem::path const& filename) noexcept;


static vtkSmartPointer<vtkImageData> caveOutImageData(vtkSmartPointer<vtkImageData> image_data) noexcept;

static constexpr unsigned char background_colour {0};
static constexpr unsigned char foreground_colour {1};
static constexpr auto scalar_data_type {VTK_UNSIGNED_CHAR};

protected:
Importer() = default;


static vtkSmartPointer<vtkPolyData> loadObj(std::filesystem::path const& filename) noexcept;


static vtkSmartPointer<vtkPolyData> loadPly(std::filesystem::path const& filename) noexcept;


static vtkSmartPointer<vtkPolyData> loadStl(std::filesystem::path const& filename) noexcept;


static vtkSmartPointer<vtkPolyData> loadVtk(std::filesystem::path const& filename) noexcept;


static vtkSmartPointer<vtkPolyData> loadVtp(std::filesystem::path const& filename) noexcept;


static vtkSmartPointer<vtkPolyData> rotatePolyData(vtkSmartPointer<vtkPolyData> input_poly_data, double const rotation_angle, 
Axis const rotation_axis = Axis::z) noexcept;


static vtkSmartPointer<vtkPolyData> translatePolyData(vtkSmartPointer<vtkPolyData> input_poly_data, 
std::array<double,3> const& translation) noexcept;


static vtkSmartPointer<vtkPolyData> cleanPolyData(vtkSmartPointer<vtkPolyData> poly_data) noexcept;


static vtkSmartPointer<vtkPolyData> reducePolyData(vtkSmartPointer<vtkPolyData> poly_data, 
double const reduction_rate) noexcept;


static vtkSmartPointer<vtkImageData> voxelisePolyData(vtkSmartPointer<vtkPolyData> const& poly_data, int const resolution_x,
std::optional<std::array<double, 6>> const& opt_bounding_box) noexcept;


static vtkSmartPointer<vtkImageData> cleanImageData(vtkSmartPointer<vtkImageData> image_data) noexcept;
};



class Geometry {
public:

Geometry(vtkSmartPointer<vtkImageData> image_data = vtkSmartPointer<vtkImageData>::New()) noexcept
: image_data{image_data} {
return;
}


Geometry(Geometry const& geometry) noexcept {
this->image_data = vtkSmartPointer<vtkImageData>::New();
this->image_data->DeepCopy(geometry.image_data);
return;
}


Geometry(Geometry&& geometry) noexcept
: image_data{geometry.image_data} {
geometry.image_data = vtkSmartPointer<vtkImageData>::New();
return;
}


Geometry& operator= (Geometry const& geometry) noexcept {
this->image_data = vtkSmartPointer<vtkImageData>::New();
this->image_data->DeepCopy(geometry.image_data);
return *this;
}


Geometry& operator= (Geometry&& geometry) noexcept {
this->image_data = geometry.image_data;
geometry.image_data = vtkSmartPointer<vtkImageData>::New();
return *this;
}


static Geometry importFromFiles(std::vector<std::filesystem::path> const& filenames, int const resolution_x, double const reduction_rate = 0.0,
std::optional<std::array<double, 6>> const& opt_bounding_box = std::nullopt) noexcept;


static Geometry importFromMhd(std::filesystem::path const& filename) noexcept;


void saveToMhd(std::filesystem::path const& output_path, std::string const& filename, bool const is_compress) const noexcept;


void saveToVtk(std::filesystem::path const& output_path, std::string const& filename) const noexcept;


void saveToFile(std::filesystem::path const& output_path, std::string const& filename, 
DataType const data_type = DataType::MHD, bool const is_compress = true) const noexcept;

protected:
vtkSmartPointer<vtkImageData> image_data;
};

}
#endif 

#endif 
