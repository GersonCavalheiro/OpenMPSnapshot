

#ifndef LBT_VTK_UTILITITES
#define LBT_VTK_UTILITITES
#pragma once

#include "use_vtk.hpp"

#ifdef LBT_USE_VTK
#include <filesystem>
#include <string>

#include <vtkImageData.h>
#include <vtkSmartPointer.h>


namespace lbt {

enum class DataType {VTK, MHD};


void saveImageDataToVtk(vtkSmartPointer<vtkImageData> const& image_data, 
std::filesystem::path const& output_path, std::string const& filename) noexcept;


void saveImageDataToMhd(vtkSmartPointer<vtkImageData> const& image_data, 
std::filesystem::path const& output_path, std::string const& filename, bool const is_compress) noexcept;
}
#endif 

#endif 
