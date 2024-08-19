

#pragma once

#include "rawspeedconfig.h"  
#include "metadata/Camera.h" 
#include <cstdint>           
#include <map>               
#include <memory>            
#include <string>            
#include <string_view>       
#include <tuple>             

namespace rawspeed {
class Camera;

struct CameraId {
std::string make;
std::string model;
std::string mode;

bool operator<(const CameraId& rhs) const {
return std::tie(make, model, mode) <
std::tie(rhs.make, rhs.model, rhs.mode);
}
};

class CameraMetaData {
public:
CameraMetaData() = default;

#ifdef HAVE_PUGIXML
explicit CameraMetaData(const char* docname);
#endif

std::map<CameraId, std::unique_ptr<Camera>> cameras;
std::map<uint32_t, Camera*> chdkCameras;

[[nodiscard]] const Camera* getCamera(const std::string& make,
const std::string& model,
const std::string& mode) const;

[[nodiscard]] const Camera* getCamera(const std::string& make,
const std::string& model) const;

[[nodiscard]] bool hasCamera(const std::string& make,
const std::string& model,
const std::string& mode) const;
[[nodiscard]] const Camera* RAWSPEED_READONLY
getChdkCamera(uint32_t filesize) const;
[[nodiscard]] bool RAWSPEED_READONLY hasChdkCamera(uint32_t filesize) const;
void disableMake(std::string_view make) const;
void disableCamera(std::string_view make, std::string_view model) const;

private:
const Camera* addCamera(std::unique_ptr<Camera> cam);
};

} 
