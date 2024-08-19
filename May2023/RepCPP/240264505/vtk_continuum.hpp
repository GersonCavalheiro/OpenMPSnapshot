

#ifndef LBT_VTK_CONTINUUM
#define LBT_VTK_CONTINUUM
#pragma once

#include "../general/use_vtk.hpp"

#ifdef LBT_USE_VTK
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <string>
#include <type_traits>

#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include "../general/output_utilities.hpp"
#include "../general/vtk_utilities.hpp"
#include "continuum_base.hpp"


namespace lbt {


template <typename T>
class VtkContinuum : public ContinuumBase<T> {
static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>, "'T' is neither of type 'double' or 'float'.");

public:

VtkContinuum(std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ, std::filesystem::path const& output_path,
DataType const export_data_type = DataType::MHD) noexcept
: ContinuumBase<T>{NX, NY, NZ, output_path}, export_data_type{export_data_type}, p{vtkSmartPointer<vtkImageData>::New()}, 
u{vtkSmartPointer<vtkImageData>::New()}, v{vtkSmartPointer<vtkImageData>::New()}, w{vtkSmartPointer<vtkImageData>::New()} {
allocateScalar_(p);
allocateScalar_(u);
allocateScalar_(v);
allocateScalar_(w);
return;
}
VtkContinuum() = delete;
VtkContinuum(VtkContinuum const& c) 
: ContinuumBase<T>{c} {
this->export_data_type = c.export_data_type;
this->p = vtkSmartPointer<vtkImageData>::New();
this->p->DeepCopy(c.p);
this->u = vtkSmartPointer<vtkImageData>::New();
this->u->DeepCopy(c.u);
this->v = vtkSmartPointer<vtkImageData>::New();
this->v->DeepCopy(c.v);
this->w = vtkSmartPointer<vtkImageData>::New();
this->w->DeepCopy(c.w);
return;
}
VtkContinuum& operator= (VtkContinuum const& c) {
ContinuumBase<T>::operator=(c);
this->export_data_type = c.export_data_type;
this->p = vtkSmartPointer<vtkImageData>::New();
this->p->DeepCopy(c.p);
this->u = vtkSmartPointer<vtkImageData>::New();
this->u->DeepCopy(c.u);
this->v = vtkSmartPointer<vtkImageData>::New();
this->v->DeepCopy(c.v);
this->w = vtkSmartPointer<vtkImageData>::New();
this->w->DeepCopy(c.w);
return *this;
}
VtkContinuum(VtkContinuum&& c)
: ContinuumBase<T>{std::move(c)} {
this->export_data_type = c.export_data_type;
this->p = c.p;
this->u = c.u;
this->v = c.v;
this->w = c.w;
c.p = vtkSmartPointer<vtkImageData>::New();
c.u = vtkSmartPointer<vtkImageData>::New();
c.v = vtkSmartPointer<vtkImageData>::New();
c.w = vtkSmartPointer<vtkImageData>::New();
return;
}
VtkContinuum& operator= (VtkContinuum&& c) {
ContinuumBase<T>::operator=(std::move(c));
this->export_data_type = c.export_data_type;
this->p = c.p;
this->u = c.u;
this->v = c.v;
this->w = c.w;
c.p = vtkSmartPointer<vtkImageData>::New();
c.u = vtkSmartPointer<vtkImageData>::New();
c.v = vtkSmartPointer<vtkImageData>::New();
c.w = vtkSmartPointer<vtkImageData>::New();
return *this;
}

void setP(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept override;
void setU(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept override;
void setV(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept override;
void setW(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept override;
T getP(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept override;
T getU(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept override;
T getV(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept override;
T getW(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept override;
void save(double const timestamp) const noexcept override;


void saveToVtk(double const timestamp) const noexcept;


void saveToMhd(double const timestamp, bool const is_compress = true) const noexcept;

protected:

void allocateScalar_(vtkSmartPointer<vtkImageData>& image_data) noexcept;


inline void setImageDataComponent_(vtkSmartPointer<vtkImageData>& image_data, 
std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept;


inline T getImageDataComponent_(vtkSmartPointer<vtkImageData> const& image_data, 
std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept;

DataType export_data_type;
vtkSmartPointer<vtkImageData> p;
vtkSmartPointer<vtkImageData> u;
vtkSmartPointer<vtkImageData> v;
vtkSmartPointer<vtkImageData> w;
};

template <typename T>
void VtkContinuum<T>::setP(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept {
setImageDataComponent_(p, x, y, z, value);
return;
}

template <typename T>
void VtkContinuum<T>::setU(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept {
setImageDataComponent_(u, x, y, z, value);
return;
}

template <typename T>
void VtkContinuum<T>::setV(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept {
setImageDataComponent_(v, x, y, z, value);
return;
}

template <typename T>
void VtkContinuum<T>::setW(std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept {
setImageDataComponent_(w, x, y, z, value);
return;
}

template <typename T>
T VtkContinuum<T>::getP(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept {
return getImageDataComponent_(p, x, y, z);
}

template <typename T>
T VtkContinuum<T>::getU(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept {
return getImageDataComponent_(u, x, y, z);
}

template <typename T>
T VtkContinuum<T>::getV(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept {
return getImageDataComponent_(v, x, y, z);
}

template <typename T>
T VtkContinuum<T>::getW(std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept {
return getImageDataComponent_(w, x, y, z);
}

template <typename T>
void VtkContinuum<T>::save(double const timestamp) const noexcept {
if (export_data_type == DataType::MHD) {
saveToMhd(timestamp, true);
} else if (export_data_type == DataType::VTK) {
saveToVtk(timestamp);
}
return;
}

template <typename T>
void VtkContinuum<T>::saveToVtk(double const timestamp) const noexcept {
std::string const filename_p {"p_" + toString(timestamp)}; 
saveImageDataToVtk(p, ContinuumBase<T>::output_path, filename_p);
std::string const filename_u {"u_" + toString(timestamp)}; 
saveImageDataToVtk(u, ContinuumBase<T>::output_path, filename_u);
std::string const filename_v {"v_" + toString(timestamp)}; 
saveImageDataToVtk(v, ContinuumBase<T>::output_path, filename_v);
std::string const filename_w {"w_" + toString(timestamp)}; 
saveImageDataToVtk(w, ContinuumBase<T>::output_path, filename_w);
return;
}

template <typename T>
void VtkContinuum<T>::saveToMhd(double const timestamp, bool const is_compress) const noexcept {
std::string const filename_p {"p_" + toString(timestamp)}; 
saveImageDataToMhd(p, ContinuumBase<T>::output_path, filename_p, is_compress);
std::string const filename_u {"u_" + toString(timestamp)}; 
saveImageDataToMhd(u, ContinuumBase<T>::output_path, filename_u, is_compress);
std::string const filename_v {"v_" + toString(timestamp)}; 
saveImageDataToMhd(v, ContinuumBase<T>::output_path, filename_v, is_compress);
std::string const filename_w {"w_" + toString(timestamp)}; 
saveImageDataToMhd(w, ContinuumBase<T>::output_path, filename_w, is_compress);
return;
}

template <typename T>
void VtkContinuum<T>::allocateScalar_(vtkSmartPointer<vtkImageData>& image_data) noexcept {
double const domain_size[3] = {static_cast<double>(ContinuumBase<T>::NX), static_cast<double>(ContinuumBase<T>::NY), 
static_cast<double>(ContinuumBase<T>::NZ)};

int const resolution[3] = {ContinuumBase<T>::NX, ContinuumBase<T>::NY, ContinuumBase<T>::NZ};
image_data->SetDimensions(resolution);
double const spacing[3] = {domain_size[0]/resolution[0], domain_size[1]/resolution[1], domain_size[2]/resolution[2]};
image_data->SetSpacing(spacing);


if constexpr (std::is_same_v<T,float>) {
image_data->AllocateScalars(VTK_FLOAT, 1);
} else if constexpr (std::is_same_v<T,double>) {
image_data->AllocateScalars(VTK_DOUBLE, 1);
}
image_data->GetPointData()->GetScalars()->Fill(0);

return;
}

template <typename T>
void VtkContinuum<T>::setImageDataComponent_(vtkSmartPointer<vtkImageData>& image_data, 
std::int32_t const x, std::int32_t const y, std::int32_t const z, T const value) noexcept {
if constexpr (std::is_same_v<T,float>) {
image_data->SetScalarComponentFromFloat(x, y, z, 0, value);
} else if constexpr (std::is_same_v<T,double>) {
image_data->SetScalarComponentFromDouble(x, y, z, 0, value);
} else {
static_assert(std::is_same_v<T,T>, "Invalid template parameter 'T'.");
}
return;
}

template <typename T>
T VtkContinuum<T>::getImageDataComponent_(vtkSmartPointer<vtkImageData> const& image_data, 
std::int32_t const x, std::int32_t const y, std::int32_t const z) const noexcept {
if constexpr (std::is_same_v<T,float>) {
return image_data->GetScalarComponentAsFloat(x, y, z, 0);
} else if constexpr (std::is_same_v<T,double>) {
return image_data->GetScalarComponentAsDouble(x, y, z, 0);
} else {
static_assert(std::is_same_v<T,T>, "Invalid template parameter 'T'.");
}
}

}
#endif 

#endif 
