
#ifndef PROTOBUF_INCLUDED_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto
#define PROTOBUF_INCLUDED_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/repeated_field.h>  
#include <google/protobuf/extension_set.h>  
#include <google/protobuf/generated_enum_reflection.h>
#define PROTOBUF_INTERNAL_EXPORT_protobuf_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto 

namespace protobuf_tensorflow_2fcore_2flib_2fcore_2ferror_5fcodes_2eproto {
struct TableStruct {
static const ::google::protobuf::internal::ParseTableField entries[];
static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
static const ::google::protobuf::internal::ParseTable schema[1];
static const ::google::protobuf::internal::FieldMetadata field_metadata[];
static const ::google::protobuf::internal::SerializationTable serialization_table[];
static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  
namespace tensorflow {
namespace error {
}  
}  
namespace tensorflow {
namespace error {

enum Code {
OK = 0,
CANCELLED = 1,
UNKNOWN = 2,
INVALID_ARGUMENT = 3,
DEADLINE_EXCEEDED = 4,
NOT_FOUND = 5,
ALREADY_EXISTS = 6,
PERMISSION_DENIED = 7,
UNAUTHENTICATED = 16,
RESOURCE_EXHAUSTED = 8,
FAILED_PRECONDITION = 9,
ABORTED = 10,
OUT_OF_RANGE = 11,
UNIMPLEMENTED = 12,
INTERNAL = 13,
UNAVAILABLE = 14,
DATA_LOSS = 15,
DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ = 20,
Code_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
Code_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool Code_IsValid(int value);
const Code Code_MIN = OK;
const Code Code_MAX = DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_;
const int Code_ARRAYSIZE = Code_MAX + 1;

const ::google::protobuf::EnumDescriptor* Code_descriptor();
inline const ::std::string& Code_Name(Code value) {
return ::google::protobuf::internal::NameOfEnum(
Code_descriptor(), value);
}
inline bool Code_Parse(
const ::std::string& name, Code* value) {
return ::google::protobuf::internal::ParseNamedEnum<Code>(
Code_descriptor(), name, value);
}





#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  


}  
}  

namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::tensorflow::error::Code> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::tensorflow::error::Code>() {
return ::tensorflow::error::Code_descriptor();
}

}  
}  


#endif  
