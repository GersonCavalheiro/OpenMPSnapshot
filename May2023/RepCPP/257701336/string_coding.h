
#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_STRING_CODING_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_STRING_CODING_H_


#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace port {

class StringListEncoder {
public:
explicit StringListEncoder(string* out) : out_(out) {}

void Append(const protobuf::MessageLite& m) {
core::PutVarint32(out_, m.ByteSize());
m.AppendToString(&rest_);
}

void Append(const string& s) {
core::PutVarint32(out_, s.length());
strings::StrAppend(&rest_, s);
}

void Finalize() { strings::StrAppend(out_, rest_); }

private:
string* out_;
string rest_;
};

class StringListDecoder {
public:
explicit StringListDecoder(const string& in) : reader_(in) {}

bool ReadSizes(std::vector<uint32>* sizes) {
int64 total = 0;
for (auto& size : *sizes) {
if (!core::GetVarint32(&reader_, &size)) return false;
total += size;
}
if (total != static_cast<int64>(reader_.size())) {
return false;
}
return true;
}

const char* Data(uint32 size) {
const char* data = reader_.data();
reader_.remove_prefix(size);
return data;
}

private:
StringPiece reader_;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in);

}  
}  

#endif  
