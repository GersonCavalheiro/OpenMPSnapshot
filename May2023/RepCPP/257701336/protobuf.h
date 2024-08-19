

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_PROTOBUF_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_PROTOBUF_H_


#ifndef TENSORFLOW_LITE_PROTOS
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/type_resolver_util.h"
#endif

#include "google/protobuf/arena.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_field.h"

namespace tensorflow {
namespace protobuf = ::google::protobuf;
using protobuf_int64 = ::google::protobuf::int64;
using protobuf_uint64 = ::google::protobuf::uint64;
extern const char* kProtobufInt64Typename;
extern const char* kProtobufUint64Typename;
}  

#endif  
