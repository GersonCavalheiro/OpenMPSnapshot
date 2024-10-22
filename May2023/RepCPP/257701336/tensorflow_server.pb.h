
#ifndef PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto
#define PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto

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
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  
#include <google/protobuf/extension_set.h>  
#include <google/protobuf/unknown_field_set.h>
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#define PROTOBUF_INTERNAL_EXPORT_protobuf_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto 

namespace protobuf_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto {
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
class ServerDef;
class ServerDefDefaultTypeInternal;
extern ServerDefDefaultTypeInternal _ServerDef_default_instance_;
}  
namespace google {
namespace protobuf {
template<> ::tensorflow::ServerDef* Arena::CreateMaybeMessage<::tensorflow::ServerDef>(Arena*);
}  
}  
namespace tensorflow {


class ServerDef : public ::google::protobuf::Message  {
public:
ServerDef();
virtual ~ServerDef();

ServerDef(const ServerDef& from);

inline ServerDef& operator=(const ServerDef& from) {
CopyFrom(from);
return *this;
}
#if LANG_CXX11
ServerDef(ServerDef&& from) noexcept
: ServerDef() {
*this = ::std::move(from);
}

inline ServerDef& operator=(ServerDef&& from) noexcept {
if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
if (this != &from) InternalSwap(&from);
} else {
CopyFrom(from);
}
return *this;
}
#endif
inline ::google::protobuf::Arena* GetArena() const final {
return GetArenaNoVirtual();
}
inline void* GetMaybeArenaPointer() const final {
return MaybeArenaPtr();
}
static const ::google::protobuf::Descriptor* descriptor();
static const ServerDef& default_instance();

static void InitAsDefaultInstance();  
static inline const ServerDef* internal_default_instance() {
return reinterpret_cast<const ServerDef*>(
&_ServerDef_default_instance_);
}
static constexpr int kIndexInFileMessages =
0;

void UnsafeArenaSwap(ServerDef* other);
void Swap(ServerDef* other);
friend void swap(ServerDef& a, ServerDef& b) {
a.Swap(&b);
}


inline ServerDef* New() const final {
return CreateMaybeMessage<ServerDef>(NULL);
}

ServerDef* New(::google::protobuf::Arena* arena) const final {
return CreateMaybeMessage<ServerDef>(arena);
}
void CopyFrom(const ::google::protobuf::Message& from) final;
void MergeFrom(const ::google::protobuf::Message& from) final;
void CopyFrom(const ServerDef& from);
void MergeFrom(const ServerDef& from);
void Clear() final;
bool IsInitialized() const final;

size_t ByteSizeLong() const final;
bool MergePartialFromCodedStream(
::google::protobuf::io::CodedInputStream* input) final;
void SerializeWithCachedSizes(
::google::protobuf::io::CodedOutputStream* output) const final;
::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
bool deterministic, ::google::protobuf::uint8* target) const final;
int GetCachedSize() const final { return _cached_size_.Get(); }

private:
void SharedCtor();
void SharedDtor();
void SetCachedSize(int size) const final;
void InternalSwap(ServerDef* other);
protected:
explicit ServerDef(::google::protobuf::Arena* arena);
private:
static void ArenaDtor(void* object);
inline void RegisterArenaDtor(::google::protobuf::Arena* arena);
private:
inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
return _internal_metadata_.arena();
}
inline void* MaybeArenaPtr() const {
return _internal_metadata_.raw_arena_ptr();
}
public:

::google::protobuf::Metadata GetMetadata() const final;



void clear_job_name();
static const int kJobNameFieldNumber = 2;
const ::std::string& job_name() const;
void set_job_name(const ::std::string& value);
#if LANG_CXX11
void set_job_name(::std::string&& value);
#endif
void set_job_name(const char* value);
void set_job_name(const char* value, size_t size);
::std::string* mutable_job_name();
::std::string* release_job_name();
void set_allocated_job_name(::std::string* job_name);
PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
"    string fields are deprecated and will be removed in a"
"    future release.")
::std::string* unsafe_arena_release_job_name();
PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
"    string fields are deprecated and will be removed in a"
"    future release.")
void unsafe_arena_set_allocated_job_name(
::std::string* job_name);

void clear_protocol();
static const int kProtocolFieldNumber = 5;
const ::std::string& protocol() const;
void set_protocol(const ::std::string& value);
#if LANG_CXX11
void set_protocol(::std::string&& value);
#endif
void set_protocol(const char* value);
void set_protocol(const char* value, size_t size);
::std::string* mutable_protocol();
::std::string* release_protocol();
void set_allocated_protocol(::std::string* protocol);
PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
"    string fields are deprecated and will be removed in a"
"    future release.")
::std::string* unsafe_arena_release_protocol();
PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
"    string fields are deprecated and will be removed in a"
"    future release.")
void unsafe_arena_set_allocated_protocol(
::std::string* protocol);

bool has_cluster() const;
void clear_cluster();
static const int kClusterFieldNumber = 1;
private:
const ::tensorflow::ClusterDef& _internal_cluster() const;
public:
const ::tensorflow::ClusterDef& cluster() const;
::tensorflow::ClusterDef* release_cluster();
::tensorflow::ClusterDef* mutable_cluster();
void set_allocated_cluster(::tensorflow::ClusterDef* cluster);
void unsafe_arena_set_allocated_cluster(
::tensorflow::ClusterDef* cluster);
::tensorflow::ClusterDef* unsafe_arena_release_cluster();

bool has_default_session_config() const;
void clear_default_session_config();
static const int kDefaultSessionConfigFieldNumber = 4;
private:
const ::tensorflow::ConfigProto& _internal_default_session_config() const;
public:
const ::tensorflow::ConfigProto& default_session_config() const;
::tensorflow::ConfigProto* release_default_session_config();
::tensorflow::ConfigProto* mutable_default_session_config();
void set_allocated_default_session_config(::tensorflow::ConfigProto* default_session_config);
void unsafe_arena_set_allocated_default_session_config(
::tensorflow::ConfigProto* default_session_config);
::tensorflow::ConfigProto* unsafe_arena_release_default_session_config();

void clear_task_index();
static const int kTaskIndexFieldNumber = 3;
::google::protobuf::int32 task_index() const;
void set_task_index(::google::protobuf::int32 value);

private:

::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
typedef void InternalArenaConstructable_;
typedef void DestructorSkippable_;
::google::protobuf::internal::ArenaStringPtr job_name_;
::google::protobuf::internal::ArenaStringPtr protocol_;
::tensorflow::ClusterDef* cluster_;
::tensorflow::ConfigProto* default_session_config_;
::google::protobuf::int32 task_index_;
mutable ::google::protobuf::internal::CachedSize _cached_size_;
friend struct ::protobuf_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto::TableStruct;
};



#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  

inline bool ServerDef::has_cluster() const {
return this != internal_default_instance() && cluster_ != NULL;
}
inline const ::tensorflow::ClusterDef& ServerDef::_internal_cluster() const {
return *cluster_;
}
inline const ::tensorflow::ClusterDef& ServerDef::cluster() const {
const ::tensorflow::ClusterDef* p = cluster_;
return p != NULL ? *p : *reinterpret_cast<const ::tensorflow::ClusterDef*>(
&::tensorflow::_ClusterDef_default_instance_);
}
inline ::tensorflow::ClusterDef* ServerDef::release_cluster() {

::tensorflow::ClusterDef* temp = cluster_;
if (GetArenaNoVirtual() != NULL) {
temp = ::google::protobuf::internal::DuplicateIfNonNull(temp);
}
cluster_ = NULL;
return temp;
}
inline ::tensorflow::ClusterDef* ServerDef::unsafe_arena_release_cluster() {

::tensorflow::ClusterDef* temp = cluster_;
cluster_ = NULL;
return temp;
}
inline ::tensorflow::ClusterDef* ServerDef::mutable_cluster() {

if (cluster_ == NULL) {
auto* p = CreateMaybeMessage<::tensorflow::ClusterDef>(GetArenaNoVirtual());
cluster_ = p;
}
return cluster_;
}
inline void ServerDef::set_allocated_cluster(::tensorflow::ClusterDef* cluster) {
::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
if (message_arena == NULL) {
delete reinterpret_cast< ::google::protobuf::MessageLite*>(cluster_);
}
if (cluster) {
::google::protobuf::Arena* submessage_arena =
reinterpret_cast<::google::protobuf::MessageLite*>(cluster)->GetArena();
if (message_arena != submessage_arena) {
cluster = ::google::protobuf::internal::GetOwnedMessage(
message_arena, cluster, submessage_arena);
}

} else {

}
cluster_ = cluster;
}

inline void ServerDef::clear_job_name() {
job_name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& ServerDef::job_name() const {
return job_name_.Get();
}
inline void ServerDef::set_job_name(const ::std::string& value) {

job_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
}
#if LANG_CXX11
inline void ServerDef::set_job_name(::std::string&& value) {

job_name_.Set(
&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
}
#endif
inline void ServerDef::set_job_name(const char* value) {
GOOGLE_DCHECK(value != NULL);

job_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
GetArenaNoVirtual());
}
inline void ServerDef::set_job_name(const char* value,
size_t size) {

job_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
}
inline ::std::string* ServerDef::mutable_job_name() {

return job_name_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* ServerDef::release_job_name() {

return job_name_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void ServerDef::set_allocated_job_name(::std::string* job_name) {
if (job_name != NULL) {

} else {

}
job_name_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), job_name,
GetArenaNoVirtual());
}
inline ::std::string* ServerDef::unsafe_arena_release_job_name() {
GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);

return job_name_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
GetArenaNoVirtual());
}
inline void ServerDef::unsafe_arena_set_allocated_job_name(
::std::string* job_name) {
GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
if (job_name != NULL) {

} else {

}
job_name_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
job_name, GetArenaNoVirtual());
}

inline void ServerDef::clear_task_index() {
task_index_ = 0;
}
inline ::google::protobuf::int32 ServerDef::task_index() const {
return task_index_;
}
inline void ServerDef::set_task_index(::google::protobuf::int32 value) {

task_index_ = value;
}

inline bool ServerDef::has_default_session_config() const {
return this != internal_default_instance() && default_session_config_ != NULL;
}
inline const ::tensorflow::ConfigProto& ServerDef::_internal_default_session_config() const {
return *default_session_config_;
}
inline const ::tensorflow::ConfigProto& ServerDef::default_session_config() const {
const ::tensorflow::ConfigProto* p = default_session_config_;
return p != NULL ? *p : *reinterpret_cast<const ::tensorflow::ConfigProto*>(
&::tensorflow::_ConfigProto_default_instance_);
}
inline ::tensorflow::ConfigProto* ServerDef::release_default_session_config() {

::tensorflow::ConfigProto* temp = default_session_config_;
if (GetArenaNoVirtual() != NULL) {
temp = ::google::protobuf::internal::DuplicateIfNonNull(temp);
}
default_session_config_ = NULL;
return temp;
}
inline ::tensorflow::ConfigProto* ServerDef::unsafe_arena_release_default_session_config() {

::tensorflow::ConfigProto* temp = default_session_config_;
default_session_config_ = NULL;
return temp;
}
inline ::tensorflow::ConfigProto* ServerDef::mutable_default_session_config() {

if (default_session_config_ == NULL) {
auto* p = CreateMaybeMessage<::tensorflow::ConfigProto>(GetArenaNoVirtual());
default_session_config_ = p;
}
return default_session_config_;
}
inline void ServerDef::set_allocated_default_session_config(::tensorflow::ConfigProto* default_session_config) {
::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
if (message_arena == NULL) {
delete reinterpret_cast< ::google::protobuf::MessageLite*>(default_session_config_);
}
if (default_session_config) {
::google::protobuf::Arena* submessage_arena =
reinterpret_cast<::google::protobuf::MessageLite*>(default_session_config)->GetArena();
if (message_arena != submessage_arena) {
default_session_config = ::google::protobuf::internal::GetOwnedMessage(
message_arena, default_session_config, submessage_arena);
}

} else {

}
default_session_config_ = default_session_config;
}

inline void ServerDef::clear_protocol() {
protocol_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& ServerDef::protocol() const {
return protocol_.Get();
}
inline void ServerDef::set_protocol(const ::std::string& value) {

protocol_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
}
#if LANG_CXX11
inline void ServerDef::set_protocol(::std::string&& value) {

protocol_.Set(
&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
}
#endif
inline void ServerDef::set_protocol(const char* value) {
GOOGLE_DCHECK(value != NULL);

protocol_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
GetArenaNoVirtual());
}
inline void ServerDef::set_protocol(const char* value,
size_t size) {

protocol_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
}
inline ::std::string* ServerDef::mutable_protocol() {

return protocol_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* ServerDef::release_protocol() {

return protocol_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void ServerDef::set_allocated_protocol(::std::string* protocol) {
if (protocol != NULL) {

} else {

}
protocol_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), protocol,
GetArenaNoVirtual());
}
inline ::std::string* ServerDef::unsafe_arena_release_protocol() {
GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);

return protocol_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
GetArenaNoVirtual());
}
inline void ServerDef::unsafe_arena_set_allocated_protocol(
::std::string* protocol) {
GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
if (protocol != NULL) {

} else {

}
protocol_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
protocol, GetArenaNoVirtual());
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  


}  


#endif  
