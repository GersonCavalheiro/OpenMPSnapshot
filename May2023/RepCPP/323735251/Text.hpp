#pragma once
#include <string>
#include <ostream>
#include <sstream>
#include <iterator>
#include <assert.h>


namespace Text {

inline std::string repeat(char text, size_t numTimes) {
return std::string(numTimes, text);
}

inline std::string repeat(const std::string& text, size_t numTimes) {
std::ostringstream os;
std::fill_n(std::ostream_iterator<std::string>(os), numTimes, text);
return os.str();
}


inline std::string padLeft(const std::string& text,char padChar, size_t width) {
return text.size() < width ?
text + std::string(width - text.size(), padChar) :
text;
}

inline std::string padRight(const std::string& text, char padChar, size_t width) {
return text.size() < width ?
std::string(width - text.size(), padChar) + text :
text;
}

inline std::string padSides(const std::string& text, char padChar, size_t width) {
const size_t paddingAmount = width - text.length();
if (paddingAmount <= 0) {
return text;
}
std::string extraPadding = std::string(paddingAmount % 2, padChar);
std::string sidePadding  = std::string(paddingAmount / 2, padChar);
return extraPadding + sidePadding + text + sidePadding;
}
}
