

#ifndef GOOGLE_PROTOBUF_WIRE_FORMAT_LITE_H__
#define GOOGLE_PROTOBUF_WIRE_FORMAT_LITE_H__

#include <string>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/repeated_field.h>

#ifndef NDEBUG
#define GOOGLE_PROTOBUF_UTF8_VALIDATION_ENABLED
#endif

#undef TYPE_BOOL

namespace google {

namespace protobuf {
template <typename T> class RepeatedField;  
}

namespace protobuf {
namespace internal {

class StringPieceField;

class LIBPROTOBUF_EXPORT WireFormatLite {
public:



enum WireType {
WIRETYPE_VARINT           = 0,
WIRETYPE_FIXED64          = 1,
WIRETYPE_LENGTH_DELIMITED = 2,
WIRETYPE_START_GROUP      = 3,
WIRETYPE_END_GROUP        = 4,
WIRETYPE_FIXED32          = 5,
};

enum FieldType {
TYPE_DOUBLE         = 1,
TYPE_FLOAT          = 2,
TYPE_INT64          = 3,
TYPE_UINT64         = 4,
TYPE_INT32          = 5,
TYPE_FIXED64        = 6,
TYPE_FIXED32        = 7,
TYPE_BOOL           = 8,
TYPE_STRING         = 9,
TYPE_GROUP          = 10,
TYPE_MESSAGE        = 11,
TYPE_BYTES          = 12,
TYPE_UINT32         = 13,
TYPE_ENUM           = 14,
TYPE_SFIXED32       = 15,
TYPE_SFIXED64       = 16,
TYPE_SINT32         = 17,
TYPE_SINT64         = 18,
MAX_FIELD_TYPE      = 18,
};

enum CppType {
CPPTYPE_INT32       = 1,
CPPTYPE_INT64       = 2,
CPPTYPE_UINT32      = 3,
CPPTYPE_UINT64      = 4,
CPPTYPE_DOUBLE      = 5,
CPPTYPE_FLOAT       = 6,
CPPTYPE_BOOL        = 7,
CPPTYPE_ENUM        = 8,
CPPTYPE_STRING      = 9,
CPPTYPE_MESSAGE     = 10,
MAX_CPPTYPE         = 10,
};

static CppType FieldTypeToCppType(FieldType type);

static inline WireFormatLite::WireType WireTypeForFieldType(
WireFormatLite::FieldType type) {
return kWireTypeForFieldType[type];
}

static const int kTagTypeBits = 3;
static const uint32 kTagTypeMask = (1 << kTagTypeBits) - 1;

static uint32 MakeTag(int field_number, WireType type);
static WireType GetTagWireType(uint32 tag);
static int GetTagFieldNumber(uint32 tag);

static inline size_t TagSize(int field_number,
WireFormatLite::FieldType type);

static bool SkipField(io::CodedInputStream* input, uint32 tag);

static bool SkipField(io::CodedInputStream* input, uint32 tag,
io::CodedOutputStream* output);

static bool SkipMessage(io::CodedInputStream* input);

static bool SkipMessage(io::CodedInputStream* input,
io::CodedOutputStream* output);

#define GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(FIELD_NUMBER, TYPE)                  \
static_cast<uint32>(                                                   \
(static_cast<uint32>(FIELD_NUMBER) << ::google::protobuf::internal::WireFormatLite::kTagTypeBits) \
| (TYPE))

static const int kMessageSetItemNumber = 1;
static const int kMessageSetTypeIdNumber = 2;
static const int kMessageSetMessageNumber = 3;
static const int kMessageSetItemStartTag =
GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetItemNumber,
WireFormatLite::WIRETYPE_START_GROUP);
static const int kMessageSetItemEndTag =
GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetItemNumber,
WireFormatLite::WIRETYPE_END_GROUP);
static const int kMessageSetTypeIdTag =
GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetTypeIdNumber,
WireFormatLite::WIRETYPE_VARINT);
static const int kMessageSetMessageTag =
GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(kMessageSetMessageNumber,
WireFormatLite::WIRETYPE_LENGTH_DELIMITED);

static const size_t kMessageSetItemTagsSize;

static uint32 EncodeFloat(float value);
static float DecodeFloat(uint32 value);
static uint64 EncodeDouble(double value);
static double DecodeDouble(uint64 value);

static uint32 ZigZagEncode32(int32 n);
static int32  ZigZagDecode32(uint32 n);
static uint64 ZigZagEncode64(int64 n);
static int64  ZigZagDecode64(uint64 n);


#ifdef NDEBUG
#define INL GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
#else
#define INL
#endif


template <typename CType, enum FieldType DeclaredType>
INL static bool ReadPrimitive(io::CodedInputStream* input, CType* value);

template <typename CType, enum FieldType DeclaredType>
INL static bool ReadRepeatedPrimitive(int tag_size, uint32 tag,
io::CodedInputStream* input,
RepeatedField<CType>* value);

template <typename CType, enum FieldType DeclaredType>
static bool ReadRepeatedPrimitiveNoInline(int tag_size, uint32 tag,
io::CodedInputStream* input,
RepeatedField<CType>* value);

template <typename CType, enum FieldType DeclaredType> INL
static const uint8* ReadPrimitiveFromArray(const uint8* buffer, CType* value);

template <typename CType, enum FieldType DeclaredType>
INL static bool ReadPackedPrimitive(io::CodedInputStream* input,
RepeatedField<CType>* value);

template <typename CType, enum FieldType DeclaredType>
static bool ReadPackedPrimitiveNoInline(io::CodedInputStream* input,
RepeatedField<CType>* value);

static bool ReadPackedEnumNoInline(io::CodedInputStream* input,
bool (*is_valid)(int),
RepeatedField<int>* values);

static bool ReadPackedEnumPreserveUnknowns(
io::CodedInputStream* input, int field_number, bool (*is_valid)(int),
io::CodedOutputStream* unknown_fields_stream, RepeatedField<int>* values);

static inline bool ReadString(io::CodedInputStream* input, string* value);
static inline bool ReadString(io::CodedInputStream* input, string** p);
static bool ReadBytes(io::CodedInputStream* input, string* value);
static bool ReadBytes(io::CodedInputStream* input, string** p);

enum Operation {
PARSE = 0,
SERIALIZE = 1,
};

static bool VerifyUtf8String(const char* data, int size,
Operation op,
const char* field_name);

template <typename MessageType>
static inline bool ReadGroup(int field_number, io::CodedInputStream* input,
MessageType* value);

template <typename MessageType>
static inline bool ReadMessage(io::CodedInputStream* input,
MessageType* value);

template <typename MessageType>
static inline bool ReadGroupNoVirtual(int field_number,
io::CodedInputStream* input,
MessageType* value) {
return ReadGroup(field_number, input, value);
}

template<typename MessageType>
static inline bool ReadMessageNoVirtual(io::CodedInputStream* input,
MessageType* value) {
return ReadMessage(input, value);
}

INL static void WriteTag(int field_number, WireType type,
io::CodedOutputStream* output);

INL static void WriteInt32NoTag(int32 value, io::CodedOutputStream* output);
INL static void WriteInt64NoTag(int64 value, io::CodedOutputStream* output);
INL static void WriteUInt32NoTag(uint32 value, io::CodedOutputStream* output);
INL static void WriteUInt64NoTag(uint64 value, io::CodedOutputStream* output);
INL static void WriteSInt32NoTag(int32 value, io::CodedOutputStream* output);
INL static void WriteSInt64NoTag(int64 value, io::CodedOutputStream* output);
INL static void WriteFixed32NoTag(uint32 value,
io::CodedOutputStream* output);
INL static void WriteFixed64NoTag(uint64 value,
io::CodedOutputStream* output);
INL static void WriteSFixed32NoTag(int32 value,
io::CodedOutputStream* output);
INL static void WriteSFixed64NoTag(int64 value,
io::CodedOutputStream* output);
INL static void WriteFloatNoTag(float value, io::CodedOutputStream* output);
INL static void WriteDoubleNoTag(double value, io::CodedOutputStream* output);
INL static void WriteBoolNoTag(bool value, io::CodedOutputStream* output);
INL static void WriteEnumNoTag(int value, io::CodedOutputStream* output);

static void WriteFloatArray(const float* a, int n,
io::CodedOutputStream* output);
static void WriteDoubleArray(const double* a, int n,
io::CodedOutputStream* output);
static void WriteFixed32Array(const uint32* a, int n,
io::CodedOutputStream* output);
static void WriteFixed64Array(const uint64* a, int n,
io::CodedOutputStream* output);
static void WriteSFixed32Array(const int32* a, int n,
io::CodedOutputStream* output);
static void WriteSFixed64Array(const int64* a, int n,
io::CodedOutputStream* output);
static void WriteBoolArray(const bool* a, int n,
io::CodedOutputStream* output);

static void WriteInt32(int field_number, int32 value,
io::CodedOutputStream* output);
static void WriteInt64(int field_number, int64 value,
io::CodedOutputStream* output);
static void WriteUInt32(int field_number, uint32 value,
io::CodedOutputStream* output);
static void WriteUInt64(int field_number, uint64 value,
io::CodedOutputStream* output);
static void WriteSInt32(int field_number, int32 value,
io::CodedOutputStream* output);
static void WriteSInt64(int field_number, int64 value,
io::CodedOutputStream* output);
static void WriteFixed32(int field_number, uint32 value,
io::CodedOutputStream* output);
static void WriteFixed64(int field_number, uint64 value,
io::CodedOutputStream* output);
static void WriteSFixed32(int field_number, int32 value,
io::CodedOutputStream* output);
static void WriteSFixed64(int field_number, int64 value,
io::CodedOutputStream* output);
static void WriteFloat(int field_number, float value,
io::CodedOutputStream* output);
static void WriteDouble(int field_number, double value,
io::CodedOutputStream* output);
static void WriteBool(int field_number, bool value,
io::CodedOutputStream* output);
static void WriteEnum(int field_number, int value,
io::CodedOutputStream* output);

static void WriteString(int field_number, const string& value,
io::CodedOutputStream* output);
static void WriteBytes(int field_number, const string& value,
io::CodedOutputStream* output);
static void WriteStringMaybeAliased(int field_number, const string& value,
io::CodedOutputStream* output);
static void WriteBytesMaybeAliased(int field_number, const string& value,
io::CodedOutputStream* output);

static void WriteGroup(int field_number, const MessageLite& value,
io::CodedOutputStream* output);
static void WriteMessage(int field_number, const MessageLite& value,
io::CodedOutputStream* output);
static void WriteGroupMaybeToArray(int field_number, const MessageLite& value,
io::CodedOutputStream* output);
static void WriteMessageMaybeToArray(int field_number,
const MessageLite& value,
io::CodedOutputStream* output);

template <typename MessageType>
static inline void WriteGroupNoVirtual(int field_number,
const MessageType& value,
io::CodedOutputStream* output);
template <typename MessageType>
static inline void WriteMessageNoVirtual(int field_number,
const MessageType& value,
io::CodedOutputStream* output);

INL static uint8* WriteTagToArray(int field_number, WireType type,
uint8* target);

INL static uint8* WriteInt32NoTagToArray(int32 value, uint8* target);
INL static uint8* WriteInt64NoTagToArray(int64 value, uint8* target);
INL static uint8* WriteUInt32NoTagToArray(uint32 value, uint8* target);
INL static uint8* WriteUInt64NoTagToArray(uint64 value, uint8* target);
INL static uint8* WriteSInt32NoTagToArray(int32 value, uint8* target);
INL static uint8* WriteSInt64NoTagToArray(int64 value, uint8* target);
INL static uint8* WriteFixed32NoTagToArray(uint32 value, uint8* target);
INL static uint8* WriteFixed64NoTagToArray(uint64 value, uint8* target);
INL static uint8* WriteSFixed32NoTagToArray(int32 value, uint8* target);
INL static uint8* WriteSFixed64NoTagToArray(int64 value, uint8* target);
INL static uint8* WriteFloatNoTagToArray(float value, uint8* target);
INL static uint8* WriteDoubleNoTagToArray(double value, uint8* target);
INL static uint8* WriteBoolNoTagToArray(bool value, uint8* target);
INL static uint8* WriteEnumNoTagToArray(int value, uint8* target);

template<typename T>
INL static uint8* WritePrimitiveNoTagToArray(
const RepeatedField<T>& value,
uint8* (*Writer)(T, uint8*), uint8* target);
template<typename T>
INL static uint8* WriteFixedNoTagToArray(
const RepeatedField<T>& value,
uint8* (*Writer)(T, uint8*), uint8* target);

INL static uint8* WriteInt32NoTagToArray(
const RepeatedField< int32>& value, uint8* output);
INL static uint8* WriteInt64NoTagToArray(
const RepeatedField< int64>& value, uint8* output);
INL static uint8* WriteUInt32NoTagToArray(
const RepeatedField<uint32>& value, uint8* output);
INL static uint8* WriteUInt64NoTagToArray(
const RepeatedField<uint64>& value, uint8* output);
INL static uint8* WriteSInt32NoTagToArray(
const RepeatedField< int32>& value, uint8* output);
INL static uint8* WriteSInt64NoTagToArray(
const RepeatedField< int64>& value, uint8* output);
INL static uint8* WriteFixed32NoTagToArray(
const RepeatedField<uint32>& value, uint8* output);
INL static uint8* WriteFixed64NoTagToArray(
const RepeatedField<uint64>& value, uint8* output);
INL static uint8* WriteSFixed32NoTagToArray(
const RepeatedField< int32>& value, uint8* output);
INL static uint8* WriteSFixed64NoTagToArray(
const RepeatedField< int64>& value, uint8* output);
INL static uint8* WriteFloatNoTagToArray(
const RepeatedField< float>& value, uint8* output);
INL static uint8* WriteDoubleNoTagToArray(
const RepeatedField<double>& value, uint8* output);
INL static uint8* WriteBoolNoTagToArray(
const RepeatedField<  bool>& value, uint8* output);
INL static uint8* WriteEnumNoTagToArray(
const RepeatedField<   int>& value, uint8* output);

INL static uint8* WriteInt32ToArray(int field_number, int32 value,
uint8* target);
INL static uint8* WriteInt64ToArray(int field_number, int64 value,
uint8* target);
INL static uint8* WriteUInt32ToArray(int field_number, uint32 value,
uint8* target);
INL static uint8* WriteUInt64ToArray(int field_number, uint64 value,
uint8* target);
INL static uint8* WriteSInt32ToArray(int field_number, int32 value,
uint8* target);
INL static uint8* WriteSInt64ToArray(int field_number, int64 value,
uint8* target);
INL static uint8* WriteFixed32ToArray(int field_number, uint32 value,
uint8* target);
INL static uint8* WriteFixed64ToArray(int field_number, uint64 value,
uint8* target);
INL static uint8* WriteSFixed32ToArray(int field_number, int32 value,
uint8* target);
INL static uint8* WriteSFixed64ToArray(int field_number, int64 value,
uint8* target);
INL static uint8* WriteFloatToArray(int field_number, float value,
uint8* target);
INL static uint8* WriteDoubleToArray(int field_number, double value,
uint8* target);
INL static uint8* WriteBoolToArray(int field_number, bool value,
uint8* target);
INL static uint8* WriteEnumToArray(int field_number, int value,
uint8* target);

template<typename T>
INL static uint8* WritePrimitiveToArray(
int field_number,
const RepeatedField<T>& value,
uint8* (*Writer)(int, T, uint8*), uint8* target);

INL static uint8* WriteInt32ToArray(
int field_number, const RepeatedField< int32>& value, uint8* output);
INL static uint8* WriteInt64ToArray(
int field_number, const RepeatedField< int64>& value, uint8* output);
INL static uint8* WriteUInt32ToArray(
int field_number, const RepeatedField<uint32>& value, uint8* output);
INL static uint8* WriteUInt64ToArray(
int field_number, const RepeatedField<uint64>& value, uint8* output);
INL static uint8* WriteSInt32ToArray(
int field_number, const RepeatedField< int32>& value, uint8* output);
INL static uint8* WriteSInt64ToArray(
int field_number, const RepeatedField< int64>& value, uint8* output);
INL static uint8* WriteFixed32ToArray(
int field_number, const RepeatedField<uint32>& value, uint8* output);
INL static uint8* WriteFixed64ToArray(
int field_number, const RepeatedField<uint64>& value, uint8* output);
INL static uint8* WriteSFixed32ToArray(
int field_number, const RepeatedField< int32>& value, uint8* output);
INL static uint8* WriteSFixed64ToArray(
int field_number, const RepeatedField< int64>& value, uint8* output);
INL static uint8* WriteFloatToArray(
int field_number, const RepeatedField< float>& value, uint8* output);
INL static uint8* WriteDoubleToArray(
int field_number, const RepeatedField<double>& value, uint8* output);
INL static uint8* WriteBoolToArray(
int field_number, const RepeatedField<  bool>& value, uint8* output);
INL static uint8* WriteEnumToArray(
int field_number, const RepeatedField<   int>& value, uint8* output);

INL static uint8* WriteStringToArray(int field_number, const string& value,
uint8* target);
INL static uint8* WriteBytesToArray(int field_number, const string& value,
uint8* target);

template<typename MessageType>
INL static uint8* InternalWriteGroupToArray(int field_number,
const MessageType& value,
bool deterministic,
uint8* target);
template<typename MessageType>
INL static uint8* InternalWriteMessageToArray(int field_number,
const MessageType& value,
bool deterministic,
uint8* target);

template <typename MessageType>
INL static uint8* InternalWriteGroupNoVirtualToArray(int field_number,
const MessageType& value,
bool deterministic,
uint8* target);
template <typename MessageType>
INL static uint8* InternalWriteMessageNoVirtualToArray(
int field_number, const MessageType& value, bool deterministic,
uint8* target);

INL static uint8* WriteGroupToArray(int field_number,
const MessageLite& value, uint8* target) {
return InternalWriteGroupToArray(field_number, value, false, target);
}
INL static uint8* WriteMessageToArray(int field_number,
const MessageLite& value,
uint8* target) {
return InternalWriteMessageToArray(field_number, value, false, target);
}
template <typename MessageType>
INL static uint8* WriteGroupNoVirtualToArray(int field_number,
const MessageType& value,
uint8* target) {
return InternalWriteGroupNoVirtualToArray(field_number, value, false,
target);
}
template <typename MessageType>
INL static uint8* WriteMessageNoVirtualToArray(int field_number,
const MessageType& value,
uint8* target) {
return InternalWriteMessageNoVirtualToArray(field_number, value, false,
target);
}

#undef INL

static inline size_t Int32Size   ( int32 value);
static inline size_t Int64Size   ( int64 value);
static inline size_t UInt32Size  (uint32 value);
static inline size_t UInt64Size  (uint64 value);
static inline size_t SInt32Size  ( int32 value);
static inline size_t SInt64Size  ( int64 value);
static inline size_t EnumSize    (   int value);

static size_t Int32Size (const RepeatedField< int32>& value);
static size_t Int64Size (const RepeatedField< int64>& value);
static size_t UInt32Size(const RepeatedField<uint32>& value);
static size_t UInt64Size(const RepeatedField<uint64>& value);
static size_t SInt32Size(const RepeatedField< int32>& value);
static size_t SInt64Size(const RepeatedField< int64>& value);
static size_t EnumSize  (const RepeatedField<   int>& value);

static const size_t kFixed32Size  = 4;
static const size_t kFixed64Size  = 8;
static const size_t kSFixed32Size = 4;
static const size_t kSFixed64Size = 8;
static const size_t kFloatSize    = 4;
static const size_t kDoubleSize   = 8;
static const size_t kBoolSize     = 1;

static inline size_t StringSize(const string& value);
static inline size_t BytesSize (const string& value);

template<typename MessageType>
static inline size_t GroupSize  (const MessageType& value);
template<typename MessageType>
static inline size_t MessageSize(const MessageType& value);

template<typename MessageType>
static inline size_t GroupSizeNoVirtual  (const MessageType& value);
template<typename MessageType>
static inline size_t MessageSizeNoVirtual(const MessageType& value);

static inline size_t LengthDelimitedSize(size_t length);

private:
template <typename CType, enum FieldType DeclaredType>
GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
static bool ReadRepeatedFixedSizePrimitive(
int tag_size,
uint32 tag,
google::protobuf::io::CodedInputStream* input,
RepeatedField<CType>* value);

template <typename CType, enum FieldType DeclaredType>
GOOGLE_PROTOBUF_ATTRIBUTE_ALWAYS_INLINE
static bool ReadPackedFixedSizePrimitive(
google::protobuf::io::CodedInputStream* input, RepeatedField<CType>* value);

static const CppType kFieldTypeToCppTypeMap[];
static const WireFormatLite::WireType kWireTypeForFieldType[];

GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(WireFormatLite);
};

class LIBPROTOBUF_EXPORT FieldSkipper {
public:
FieldSkipper() {}
virtual ~FieldSkipper() {}

virtual bool SkipField(io::CodedInputStream* input, uint32 tag);

virtual bool SkipMessage(io::CodedInputStream* input);

virtual void SkipUnknownEnum(int field_number, int value);
};


class LIBPROTOBUF_EXPORT CodedOutputStreamFieldSkipper : public FieldSkipper {
public:
explicit CodedOutputStreamFieldSkipper(io::CodedOutputStream* unknown_fields)
: unknown_fields_(unknown_fields) {}
virtual ~CodedOutputStreamFieldSkipper() {}

virtual bool SkipField(io::CodedInputStream* input, uint32 tag);
virtual bool SkipMessage(io::CodedInputStream* input);
virtual void SkipUnknownEnum(int field_number, int value);

protected:
io::CodedOutputStream* unknown_fields_;
};



inline WireFormatLite::CppType
WireFormatLite::FieldTypeToCppType(FieldType type) {
return kFieldTypeToCppTypeMap[type];
}

inline uint32 WireFormatLite::MakeTag(int field_number, WireType type) {
return GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(field_number, type);
}

inline WireFormatLite::WireType WireFormatLite::GetTagWireType(uint32 tag) {
return static_cast<WireType>(tag & kTagTypeMask);
}

inline int WireFormatLite::GetTagFieldNumber(uint32 tag) {
return static_cast<int>(tag >> kTagTypeBits);
}

inline size_t WireFormatLite::TagSize(int field_number,
WireFormatLite::FieldType type) {
size_t result = io::CodedOutputStream::VarintSize32(
static_cast<uint32>(field_number << kTagTypeBits));
if (type == TYPE_GROUP) {
return result * 2;
} else {
return result;
}
}

inline uint32 WireFormatLite::EncodeFloat(float value) {
union {float f; uint32 i;};
f = value;
return i;
}

inline float WireFormatLite::DecodeFloat(uint32 value) {
union {float f; uint32 i;};
i = value;
return f;
}

inline uint64 WireFormatLite::EncodeDouble(double value) {
union {double f; uint64 i;};
f = value;
return i;
}

inline double WireFormatLite::DecodeDouble(uint64 value) {
union {double f; uint64 i;};
i = value;
return f;
}


inline uint32 WireFormatLite::ZigZagEncode32(int32 n) {
return (static_cast<uint32>(n) << 1) ^ static_cast<uint32>(n >> 31);
}

inline int32 WireFormatLite::ZigZagDecode32(uint32 n) {
return static_cast<int32>((n >> 1) ^ (~(n & 1) + 1));
}

inline uint64 WireFormatLite::ZigZagEncode64(int64 n) {
return (static_cast<uint64>(n) << 1) ^ static_cast<uint64>(n >> 63);
}

inline int64 WireFormatLite::ZigZagDecode64(uint64 n) {
return static_cast<int64>((n >> 1) ^ (~(n & 1) + 1));
}


inline bool WireFormatLite::ReadString(io::CodedInputStream* input,
string* value) {
return ReadBytes(input, value);
}

inline bool WireFormatLite::ReadString(io::CodedInputStream* input,
string** p) {
return ReadBytes(input, p);
}

}  
}  

}  
#endif  
