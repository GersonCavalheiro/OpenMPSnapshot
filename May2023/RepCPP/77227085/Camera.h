

#pragma once

#include "rawspeedconfig.h"            
#include "adt/NotARational.h"          
#include "adt/Point.h"                 
#include "metadata/BlackArea.h"        
#include "metadata/CameraSensorInfo.h" 
#include "metadata/ColorFilterArray.h" 
#include <algorithm>                   
#include <cstdint>                     
#include <functional>                  
#include <map>                         
#include <sstream>                     
#include <string>                      
#include <utility>                     
#include <vector>                      

#ifdef HAVE_PUGIXML

namespace pugi {
class xml_node;
} 

#endif

namespace rawspeed {

class Hints {
std::map<std::string, std::string, std::less<>> data;

public:
void add(const std::string& key, const std::string& value) {
data.try_emplace(key, value);
}

[[nodiscard]] bool has(const std::string& key) const {
return data.find(key) != data.end();
}

template <typename T>
[[nodiscard]] T get(const std::string& key, T defaultValue) const {
if (auto hint = data.find(key);
hint != data.end() && !hint->second.empty()) {
std::istringstream iss(hint->second);
iss >> defaultValue;
}
return defaultValue;
}

[[nodiscard]] bool get(const std::string& key, bool defaultValue) const {
auto hint = data.find(key);
if (hint == data.end())
return defaultValue;
return "true" == hint->second;
}
};

class Camera {
public:
enum class SupportStatus {
Unsupported,
Supported,
NoSamples,
};

#ifdef HAVE_PUGIXML
explicit Camera(const pugi::xml_node& camera);
#endif

Camera(const Camera* camera, uint32_t alias_num);
[[nodiscard]] const CameraSensorInfo* getSensorInfo(int iso) const;
std::string make;
std::string model;
std::string mode;
std::string canonical_make;
std::string canonical_model;
std::string canonical_alias;
std::string canonical_id;
std::vector<std::string> aliases;
std::vector<std::string> canonical_aliases;
ColorFilterArray cfa;
SupportStatus supportStatus;
iPoint2D cropSize;
iPoint2D cropPos;
std::vector<BlackArea> blackAreas;
std::vector<CameraSensorInfo> sensorInfo;
int decoderVersion;
Hints hints;
std::vector<NotARational<int>> color_matrix;

protected:
#ifdef HAVE_PUGIXML
void parseCFA(const pugi::xml_node& node);
void parseCrop(const pugi::xml_node& node);
void parseBlackAreas(const pugi::xml_node& node);
void parseAliases(const pugi::xml_node& node);
void parseHints(const pugi::xml_node& node);
void parseID(const pugi::xml_node& node);
void parseSensor(const pugi::xml_node& node);
void parseColorMatrix(const pugi::xml_node& node);
void parseColorMatrices(const pugi::xml_node& node);

void parseCameraChild(const pugi::xml_node& node);
#endif
};

} 
