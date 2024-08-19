

#ifndef GOOGLE_PROTOBUF_IO_CODED_STREAM_H__
#define GOOGLE_PROTOBUF_IO_CODED_STREAM_H__

#include <assert.h>
#include <atomic>
#include <climits>
#include <string>
#include <utility>
#ifdef _MSC_VER
#if !defined(PROTOBUF_DISABLE_LITTLE_ENDIAN_OPT_FOR_TEST)
#define PROTOBUF_LITTLE_ENDIAN 1
#endif
#if _MSC_VER >= 1300 && !defined(__INTEL_COMPILER)
#pragma runtime_checks("c", off)
#endif
#else
#include <sys/param.h>   
#if ((defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)) || \
(defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN)) && \
!defined(PROTOBUF_DISABLE_LITTLE_ENDIAN_OPT_FOR_TEST)
#define PROTOBUF_LITTLE_ENDIAN 1
#endif
#endif
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/port.h>

namespace google {

namespace protobuf {

class DescriptorPool;
class MessageFactory;

namespace internal { void MapTestForceDeterministic(); }

namespace io {

class CodedInputStream;
class CodedOutputStream;

class ZeroCopyInputStream;           
class ZeroCopyOutputStream;          

class LIBPROTOBUF_EXPORT CodedInputStream {
public:
explicit CodedInputStream(ZeroCopyInputStream* input);

explicit CodedInputStream(const uint8* buffer, int size);

~CodedInputStream();

inline bool IsFlat() const;

inline bool Skip(int count);

bool GetDirectBufferPointer(const void** data, int* size);

GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
void GetDirectBufferPointerInline(const void** data, int* size);

bool ReadRaw(void* buffer, int size);

GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
bool InternalReadRawInline(void* buffer, int size);

bool ReadString(string* buffer, int size);
GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
bool InternalReadStringInline(string* buffer, int size);


bool ReadLittleEndian32(uint32* value);
bool ReadLittleEndian64(uint64* value);

static const uint8* ReadLittleEndian32FromArray(const uint8* buffer,
uint32* value);
static const uint8* ReadLittleEndian64FromArray(const uint8* buffer,
uint64* value);

bool ReadVarint32(uint32* value);
bool ReadVarint64(uint64* value);

bool ReadVarintSizeAsInt(int* value);

GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE uint32 ReadTag() {
return last_tag_ = ReadTagNoLastTag();
}

GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE uint32 ReadTagNoLastTag();


GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
std::pair<uint32, bool> ReadTagWithCutoff(uint32 cutoff) {
std::pair<uint32, bool> result = ReadTagWithCutoffNoLastTag(cutoff);
last_tag_ = result.first;
return result;
}

GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
std::pair<uint32, bool> ReadTagWithCutoffNoLastTag(uint32 cutoff);

GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE bool ExpectTag(uint32 expected);

GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
static const uint8* ExpectTagFromArray(const uint8* buffer, uint32 expected);

bool ExpectAtEnd();

bool LastTagWas(uint32 expected);
void SetLastTag(uint32 tag) { last_tag_ = tag; }

bool ConsumedEntireMessage();


typedef int Limit;

Limit PushLimit(int byte_limit);

void PopLimit(Limit limit);

int BytesUntilLimit() const;

int CurrentPosition() const;


void SetTotalBytesLimit(int total_bytes_limit);

PROTOBUF_RUNTIME_DEPRECATED(
"Please use the single parameter version of SetTotalBytesLimit(). The "
"second parameter is ignored.")
void SetTotalBytesLimit(int total_bytes_limit, int) {
SetTotalBytesLimit(total_bytes_limit);
}

int BytesUntilTotalBytesLimit() const;


void SetRecursionLimit(int limit);


bool IncrementRecursionDepth();

void DecrementRecursionDepth();

void UnsafeDecrementRecursionDepth();

std::pair<CodedInputStream::Limit, int> IncrementRecursionDepthAndPushLimit(
int byte_limit);

Limit ReadLengthAndPushLimit();

bool DecrementRecursionDepthAndPopLimit(Limit limit);

bool CheckEntireMessageConsumedAndPopLimit(Limit limit);


void SetExtensionRegistry(const DescriptorPool* pool,
MessageFactory* factory);

const DescriptorPool* GetExtensionPool();

MessageFactory* GetExtensionFactory();

private:
GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(CodedInputStream);

const uint8* buffer_;
const uint8* buffer_end_;     
ZeroCopyInputStream* input_;
int total_bytes_read_;  

int overflow_bytes_;

uint32 last_tag_;         

bool legitimate_message_end_;

bool aliasing_enabled_;

Limit current_limit_;   

int buffer_size_after_limit_;

int total_bytes_limit_;

int recursion_budget_;
int recursion_limit_;

const DescriptorPool* extension_pool_;
MessageFactory* extension_factory_;


bool SkipFallback(int count, int original_buffer_size);

void Advance(int amount);

void BackUpInputToCurrentPosition();

void RecomputeBufferLimits();

void PrintTotalBytesLimitError();

bool Refresh();

int64 ReadVarint32Fallback(uint32 first_byte_or_zero);
int ReadVarintSizeAsIntFallback();
std::pair<uint64, bool> ReadVarint64Fallback();
bool ReadVarint32Slow(uint32* value);
bool ReadVarint64Slow(uint64* value);
int ReadVarintSizeAsIntSlow();
bool ReadLittleEndian32Fallback(uint32* value);
bool ReadLittleEndian64Fallback(uint64* value);

uint32 ReadTagFallback(uint32 first_byte_or_zero);
uint32 ReadTagSlow();
bool ReadStringFallback(string* buffer, int size);

int BufferSize() const;

static const int kDefaultTotalBytesLimit = INT_MAX;

static int default_recursion_limit_;  
};

class LIBPROTOBUF_EXPORT CodedOutputStream {
public:
explicit CodedOutputStream(ZeroCopyOutputStream* output);
CodedOutputStream(ZeroCopyOutputStream* output, bool do_eager_refresh);

~CodedOutputStream();

void Trim();

bool Skip(int count);

bool GetDirectBufferPointer(void** data, int* size);

inline uint8* GetDirectBufferForNBytesAndAdvance(int size);

void WriteRaw(const void* buffer, int size);
void WriteRawMaybeAliased(const void* data, int size);
static uint8* WriteRawToArray(const void* buffer, int size, uint8* target);

void WriteString(const string& str);
static uint8* WriteStringToArray(const string& str, uint8* target);
static uint8* WriteStringWithSizeToArray(const string& str, uint8* target);


void EnableAliasing(bool enabled);

void WriteLittleEndian32(uint32 value);
static uint8* WriteLittleEndian32ToArray(uint32 value, uint8* target);
void WriteLittleEndian64(uint64 value);
static uint8* WriteLittleEndian64ToArray(uint64 value, uint8* target);

void WriteVarint32(uint32 value);
static uint8* WriteVarint32ToArray(uint32 value, uint8* target);
void WriteVarint64(uint64 value);
static uint8* WriteVarint64ToArray(uint64 value, uint8* target);

void WriteVarint32SignExtended(int32 value);
static uint8* WriteVarint32SignExtendedToArray(int32 value, uint8* target);

void WriteTag(uint32 value);
GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
static uint8* WriteTagToArray(uint32 value, uint8* target);

static size_t VarintSize32(uint32 value);
static size_t VarintSize64(uint64 value);

static size_t VarintSize32SignExtended(int32 value);

template <uint32 Value>
struct StaticVarintSize32 {
static const size_t value =
(Value < (1 << 7))
? 1
: (Value < (1 << 14))
? 2
: (Value < (1 << 21))
? 3
: (Value < (1 << 28))
? 4
: 5;
};

inline int ByteCount() const;

bool HadError() const { return had_error_; }

void SetSerializationDeterministic(bool value) {
is_serialization_deterministic_ = value;
}
bool IsSerializationDeterministic() const {
return is_serialization_deterministic_;
}

static bool IsDefaultSerializationDeterministic() {
return default_serialization_deterministic_.load(std::memory_order_relaxed) != 0;
}

private:
GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(CodedOutputStream);

ZeroCopyOutputStream* output_;
uint8* buffer_;
int buffer_size_;
int total_bytes_;  
bool had_error_;   
bool aliasing_enabled_;  
bool is_serialization_deterministic_;
static std::atomic<bool> default_serialization_deterministic_;

void Advance(int amount);

bool Refresh();

void WriteAliasedRaw(const void* buffer, int size);

void WriteVarint32SlowPath(uint32 value);
void WriteVarint64SlowPath(uint64 value);

friend void ::google::protobuf::internal::MapTestForceDeterministic();
static void SetDefaultSerializationDeterministic() {
default_serialization_deterministic_.store(true, std::memory_order_relaxed);
}
};


inline bool CodedInputStream::ReadVarint32(uint32* value) {
uint32 v = 0;
if (GOOGLE_PREDICT_TRUE(buffer_ < buffer_end_)) {
v = *buffer_;
if (v < 0x80) {
*value = v;
Advance(1);
return true;
}
}
int64 result = ReadVarint32Fallback(v);
*value = static_cast<uint32>(result);
return result >= 0;
}

inline bool CodedInputStream::ReadVarint64(uint64* value) {
if (GOOGLE_PREDICT_TRUE(buffer_ < buffer_end_) && *buffer_ < 0x80) {
*value = *buffer_;
Advance(1);
return true;
}
std::pair<uint64, bool> p = ReadVarint64Fallback();
*value = p.first;
return p.second;
}

inline bool CodedInputStream::ReadVarintSizeAsInt(int* value) {
if (GOOGLE_PREDICT_TRUE(buffer_ < buffer_end_)) {
int v = *buffer_;
if (v < 0x80) {
*value = v;
Advance(1);
return true;
}
}
*value = ReadVarintSizeAsIntFallback();
return *value >= 0;
}

inline const uint8* CodedInputStream::ReadLittleEndian32FromArray(
const uint8* buffer,
uint32* value) {
#if defined(PROTOBUF_LITTLE_ENDIAN)
memcpy(value, buffer, sizeof(*value));
return buffer + sizeof(*value);
#else
*value = (static_cast<uint32>(buffer[0])      ) |
(static_cast<uint32>(buffer[1]) <<  8) |
(static_cast<uint32>(buffer[2]) << 16) |
(static_cast<uint32>(buffer[3]) << 24);
return buffer + sizeof(*value);
#endif
}
inline const uint8* CodedInputStream::ReadLittleEndian64FromArray(
const uint8* buffer,
uint64* value) {
#if defined(PROTOBUF_LITTLE_ENDIAN)
memcpy(value, buffer, sizeof(*value));
return buffer + sizeof(*value);
#else
uint32 part0 = (static_cast<uint32>(buffer[0])      ) |
(static_cast<uint32>(buffer[1]) <<  8) |
(static_cast<uint32>(buffer[2]) << 16) |
(static_cast<uint32>(buffer[3]) << 24);
uint32 part1 = (static_cast<uint32>(buffer[4])      ) |
(static_cast<uint32>(buffer[5]) <<  8) |
(static_cast<uint32>(buffer[6]) << 16) |
(static_cast<uint32>(buffer[7]) << 24);
*value = static_cast<uint64>(part0) |
(static_cast<uint64>(part1) << 32);
return buffer + sizeof(*value);
#endif
}

inline bool CodedInputStream::ReadLittleEndian32(uint32* value) {
#if defined(PROTOBUF_LITTLE_ENDIAN)
if (GOOGLE_PREDICT_TRUE(BufferSize() >= static_cast<int>(sizeof(*value)))) {
buffer_ = ReadLittleEndian32FromArray(buffer_, value);
return true;
} else {
return ReadLittleEndian32Fallback(value);
}
#else
return ReadLittleEndian32Fallback(value);
#endif
}

inline bool CodedInputStream::ReadLittleEndian64(uint64* value) {
#if defined(PROTOBUF_LITTLE_ENDIAN)
if (GOOGLE_PREDICT_TRUE(BufferSize() >= static_cast<int>(sizeof(*value)))) {
buffer_ = ReadLittleEndian64FromArray(buffer_, value);
return true;
} else {
return ReadLittleEndian64Fallback(value);
}
#else
return ReadLittleEndian64Fallback(value);
#endif
}

inline uint32 CodedInputStream::ReadTagNoLastTag() {
uint32 v = 0;
if (GOOGLE_PREDICT_TRUE(buffer_ < buffer_end_)) {
v = *buffer_;
if (v < 0x80) {
Advance(1);
return v;
}
}
v = ReadTagFallback(v);
return v;
}

inline std::pair<uint32, bool> CodedInputStream::ReadTagWithCutoffNoLastTag(
uint32 cutoff) {
uint32 first_byte_or_zero = 0;
if (GOOGLE_PREDICT_TRUE(buffer_ < buffer_end_)) {
first_byte_or_zero = buffer_[0];
if (static_cast<int8>(buffer_[0]) > 0) {
const uint32 kMax1ByteVarint = 0x7f;
uint32 tag = buffer_[0];
Advance(1);
return std::make_pair(tag, cutoff >= kMax1ByteVarint || tag <= cutoff);
}
if (cutoff >= 0x80 && GOOGLE_PREDICT_TRUE(buffer_ + 1 < buffer_end_) &&
GOOGLE_PREDICT_TRUE((buffer_[0] & ~buffer_[1]) >= 0x80)) {
const uint32 kMax2ByteVarint = (0x7f << 7) + 0x7f;
uint32 tag = (1u << 7) * buffer_[1] + (buffer_[0] - 0x80);
Advance(2);
bool at_or_below_cutoff = cutoff >= kMax2ByteVarint || tag <= cutoff;
return std::make_pair(tag, at_or_below_cutoff);
}
}
const uint32 tag = ReadTagFallback(first_byte_or_zero);
return std::make_pair(tag, static_cast<uint32>(tag - 1) < cutoff);
}

inline bool CodedInputStream::LastTagWas(uint32 expected) {
return last_tag_ == expected;
}

inline bool CodedInputStream::ConsumedEntireMessage() {
return legitimate_message_end_;
}

inline bool CodedInputStream::ExpectTag(uint32 expected) {
if (expected < (1 << 7)) {
if (GOOGLE_PREDICT_TRUE(buffer_ < buffer_end_) && buffer_[0] == expected) {
Advance(1);
return true;
} else {
return false;
}
} else if (expected < (1 << 14)) {
if (GOOGLE_PREDICT_TRUE(BufferSize() >= 2) &&
buffer_[0] == static_cast<uint8>(expected | 0x80) &&
buffer_[1] == static_cast<uint8>(expected >> 7)) {
Advance(2);
return true;
} else {
return false;
}
} else {
return false;
}
}

inline const uint8* CodedInputStream::ExpectTagFromArray(
const uint8* buffer, uint32 expected) {
if (expected < (1 << 7)) {
if (buffer[0] == expected) {
return buffer + 1;
}
} else if (expected < (1 << 14)) {
if (buffer[0] == static_cast<uint8>(expected | 0x80) &&
buffer[1] == static_cast<uint8>(expected >> 7)) {
return buffer + 2;
}
}
return NULL;
}

inline void CodedInputStream::GetDirectBufferPointerInline(const void** data,
int* size) {
*data = buffer_;
*size = static_cast<int>(buffer_end_ - buffer_);
}

inline bool CodedInputStream::ExpectAtEnd() {

if (buffer_ == buffer_end_ &&
((buffer_size_after_limit_ != 0) ||
(total_bytes_read_ == current_limit_))) {
last_tag_ = 0;                   
legitimate_message_end_ = true;  
return true;
} else {
return false;
}
}

inline int CodedInputStream::CurrentPosition() const {
return total_bytes_read_ - (BufferSize() + buffer_size_after_limit_);
}

inline uint8* CodedOutputStream::GetDirectBufferForNBytesAndAdvance(int size) {
if (buffer_size_ < size) {
return NULL;
} else {
uint8* result = buffer_;
Advance(size);
return result;
}
}

inline uint8* CodedOutputStream::WriteVarint32ToArray(uint32 value,
uint8* target) {
while (value >= 0x80) {
*target = static_cast<uint8>(value | 0x80);
value >>= 7;
++target;
}
*target = static_cast<uint8>(value);
return target + 1;
}

inline uint8* CodedOutputStream::WriteVarint64ToArray(uint64 value,
uint8* target) {
while (value >= 0x80) {
*target = static_cast<uint8>(value | 0x80);
value >>= 7;
++target;
}
*target = static_cast<uint8>(value);
return target + 1;
}

inline void CodedOutputStream::WriteVarint32SignExtended(int32 value) {
WriteVarint64(static_cast<uint64>(value));
}

inline uint8* CodedOutputStream::WriteVarint32SignExtendedToArray(
int32 value, uint8* target) {
return WriteVarint64ToArray(static_cast<uint64>(value), target);
}

inline uint8* CodedOutputStream::WriteLittleEndian32ToArray(uint32 value,
uint8* target) {
#if defined(PROTOBUF_LITTLE_ENDIAN)
memcpy(target, &value, sizeof(value));
#else
target[0] = static_cast<uint8>(value);
target[1] = static_cast<uint8>(value >>  8);
target[2] = static_cast<uint8>(value >> 16);
target[3] = static_cast<uint8>(value >> 24);
#endif
return target + sizeof(value);
}

inline uint8* CodedOutputStream::WriteLittleEndian64ToArray(uint64 value,
uint8* target) {
#if defined(PROTOBUF_LITTLE_ENDIAN)
memcpy(target, &value, sizeof(value));
#else
uint32 part0 = static_cast<uint32>(value);
uint32 part1 = static_cast<uint32>(value >> 32);

target[0] = static_cast<uint8>(part0);
target[1] = static_cast<uint8>(part0 >>  8);
target[2] = static_cast<uint8>(part0 >> 16);
target[3] = static_cast<uint8>(part0 >> 24);
target[4] = static_cast<uint8>(part1);
target[5] = static_cast<uint8>(part1 >>  8);
target[6] = static_cast<uint8>(part1 >> 16);
target[7] = static_cast<uint8>(part1 >> 24);
#endif
return target + sizeof(value);
}

inline void CodedOutputStream::WriteVarint32(uint32 value) {
if (buffer_size_ >= 5) {
uint8* target = buffer_;
uint8* end = WriteVarint32ToArray(value, target);
int size = static_cast<int>(end - target);
Advance(size);
} else {
WriteVarint32SlowPath(value);
}
}

inline void CodedOutputStream::WriteVarint64(uint64 value) {
if (buffer_size_ >= 10) {
uint8* target = buffer_;
uint8* end = WriteVarint64ToArray(value, target);
int size = static_cast<int>(end - target);
Advance(size);
} else {
WriteVarint64SlowPath(value);
}
}

inline void CodedOutputStream::WriteTag(uint32 value) {
WriteVarint32(value);
}

inline uint8* CodedOutputStream::WriteTagToArray(
uint32 value, uint8* target) {
return WriteVarint32ToArray(value, target);
}

inline size_t CodedOutputStream::VarintSize32(uint32 value) {
uint32 log2value = Bits::Log2FloorNonZero(value | 0x1);
return static_cast<size_t>((log2value * 9 + 73) / 64);
}

inline size_t CodedOutputStream::VarintSize64(uint64 value) {
uint32 log2value = Bits::Log2FloorNonZero64(value | 0x1);
return static_cast<size_t>((log2value * 9 + 73) / 64);
}

inline size_t CodedOutputStream::VarintSize32SignExtended(int32 value) {
if (value < 0) {
return 10;     
} else {
return VarintSize32(static_cast<uint32>(value));
}
}

inline void CodedOutputStream::WriteString(const string& str) {
WriteRaw(str.data(), static_cast<int>(str.size()));
}

inline void CodedOutputStream::WriteRawMaybeAliased(
const void* data, int size) {
if (aliasing_enabled_) {
WriteAliasedRaw(data, size);
} else {
WriteRaw(data, size);
}
}

inline uint8* CodedOutputStream::WriteStringToArray(
const string& str, uint8* target) {
return WriteRawToArray(str.data(), static_cast<int>(str.size()), target);
}

inline int CodedOutputStream::ByteCount() const {
return total_bytes_ - buffer_size_;
}

inline void CodedInputStream::Advance(int amount) {
buffer_ += amount;
}

inline void CodedOutputStream::Advance(int amount) {
buffer_ += amount;
buffer_size_ -= amount;
}

inline void CodedInputStream::SetRecursionLimit(int limit) {
recursion_budget_ += limit - recursion_limit_;
recursion_limit_ = limit;
}

inline bool CodedInputStream::IncrementRecursionDepth() {
--recursion_budget_;
return recursion_budget_ >= 0;
}

inline void CodedInputStream::DecrementRecursionDepth() {
if (recursion_budget_ < recursion_limit_) ++recursion_budget_;
}

inline void CodedInputStream::UnsafeDecrementRecursionDepth() {
assert(recursion_budget_ < recursion_limit_);
++recursion_budget_;
}

inline void CodedInputStream::SetExtensionRegistry(const DescriptorPool* pool,
MessageFactory* factory) {
extension_pool_ = pool;
extension_factory_ = factory;
}

inline const DescriptorPool* CodedInputStream::GetExtensionPool() {
return extension_pool_;
}

inline MessageFactory* CodedInputStream::GetExtensionFactory() {
return extension_factory_;
}

inline int CodedInputStream::BufferSize() const {
return static_cast<int>(buffer_end_ - buffer_);
}

inline CodedInputStream::CodedInputStream(ZeroCopyInputStream* input)
: buffer_(NULL),
buffer_end_(NULL),
input_(input),
total_bytes_read_(0),
overflow_bytes_(0),
last_tag_(0),
legitimate_message_end_(false),
aliasing_enabled_(false),
current_limit_(kint32max),
buffer_size_after_limit_(0),
total_bytes_limit_(kDefaultTotalBytesLimit),
recursion_budget_(default_recursion_limit_),
recursion_limit_(default_recursion_limit_),
extension_pool_(NULL),
extension_factory_(NULL) {
Refresh();
}

inline CodedInputStream::CodedInputStream(const uint8* buffer, int size)
: buffer_(buffer),
buffer_end_(buffer + size),
input_(NULL),
total_bytes_read_(size),
overflow_bytes_(0),
last_tag_(0),
legitimate_message_end_(false),
aliasing_enabled_(false),
current_limit_(size),
buffer_size_after_limit_(0),
total_bytes_limit_(kDefaultTotalBytesLimit),
recursion_budget_(default_recursion_limit_),
recursion_limit_(default_recursion_limit_),
extension_pool_(NULL),
extension_factory_(NULL) {
}

inline bool CodedInputStream::IsFlat() const {
return input_ == NULL;
}

inline bool CodedInputStream::Skip(int count) {
if (count < 0) return false;  

const int original_buffer_size = BufferSize();

if (count <= original_buffer_size) {
Advance(count);
return true;
}

return SkipFallback(count, original_buffer_size);
}

}  
}  


#if defined(_MSC_VER) && _MSC_VER >= 1300 && !defined(__INTEL_COMPILER)
#pragma runtime_checks("c", restore)
#endif  

}  
#endif  
