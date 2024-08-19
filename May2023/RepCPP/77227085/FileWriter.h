

#pragma once

#include <cstdint> 

namespace rawspeed {

class Buffer;

class FileWriter {
public:
explicit FileWriter(const char* filename);

void writeFile(Buffer fileMap, uint32_t size = 0) const;
[[nodiscard]] const char* Filename() const { return mFilename; }

private:
const char* mFilename;
};

} 
