
#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Error.hpp"
#include "StringTools.hpp"

namespace CLI {

class App;

struct ConfigItem {
std::vector<std::string> parents{};

std::string name{};

std::vector<std::string> inputs{};

std::string fullname() const {
std::vector<std::string> tmp = parents;
tmp.emplace_back(name);
return detail::join(tmp, ".");
}
};

class Config {
protected:
std::vector<ConfigItem> items{};

public:
virtual std::string to_config(const App *, bool, bool, std::string) const = 0;

virtual std::vector<ConfigItem> from_config(std::istream &) const = 0;

virtual std::string to_flag(const ConfigItem &item) const {
if(item.inputs.size() == 1) {
return item.inputs.at(0);
}
throw ConversionError::TooManyInputsFlag(item.fullname());
}

std::vector<ConfigItem> from_file(const std::string &name) {
std::ifstream input{name};
if(!input.good())
throw FileError::Missing(name);

return from_config(input);
}

virtual ~Config() = default;
};

class ConfigBase : public Config {
protected:
char commentChar = '#';
char arrayStart = '[';
char arrayEnd = ']';
char arraySeparator = ',';
char valueDelimiter = '=';
char stringQuote = '"';
char characterQuote = '\'';
uint8_t maximumLayers{255};
char parentSeparatorChar{'.'};
int16_t configIndex{-1};
std::string configSection{};

public:
std::string
to_config(const App * , bool default_also, bool write_description, std::string prefix) const override;

std::vector<ConfigItem> from_config(std::istream &input) const override;
ConfigBase *comment(char cchar) {
commentChar = cchar;
return this;
}
ConfigBase *arrayBounds(char aStart, char aEnd) {
arrayStart = aStart;
arrayEnd = aEnd;
return this;
}
ConfigBase *arrayDelimiter(char aSep) {
arraySeparator = aSep;
return this;
}
ConfigBase *valueSeparator(char vSep) {
valueDelimiter = vSep;
return this;
}
ConfigBase *quoteCharacter(char qString, char qChar) {
stringQuote = qString;
characterQuote = qChar;
return this;
}
ConfigBase *maxLayers(uint8_t layers) {
maximumLayers = layers;
return this;
}
ConfigBase *parentSeparator(char sep) {
parentSeparatorChar = sep;
return this;
}
std::string &sectionRef() { return configSection; }
const std::string &section() const { return configSection; }
ConfigBase *section(const std::string &sectionName) {
configSection = sectionName;
return this;
}

int16_t &indexRef() { return configIndex; }
int16_t index() const { return configIndex; }
ConfigBase *index(int16_t sectionIndex) {
configIndex = sectionIndex;
return this;
}
};

using ConfigTOML = ConfigBase;

class ConfigINI : public ConfigTOML {

public:
ConfigINI() {
commentChar = ';';
arrayStart = '\0';
arrayEnd = '\0';
arraySeparator = ' ';
valueDelimiter = '=';
}
};
}  
