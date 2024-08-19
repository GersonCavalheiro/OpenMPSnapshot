
#ifndef PROTOBUF_INCLUDED_google_2fprotobuf_2ftimestamp_2eproto
#define PROTOBUF_INCLUDED_google_2fprotobuf_2ftimestamp_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_protobuf_google_2fprotobuf_2ftimestamp_2eproto LIBPROTOBUF_EXPORT

namespace protobuf_google_2fprotobuf_2ftimestamp_2eproto {
struct LIBPROTOBUF_EXPORT TableStruct {
static const ::google::protobuf::internal::ParseTableField entries[];
static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
static const ::google::protobuf::internal::ParseTable schema[1];
static const ::google::protobuf::internal::FieldMetadata field_metadata[];
static const ::google::protobuf::internal::SerializationTable serialization_table[];
static const ::google::protobuf::uint32 offsets[];
};
void LIBPROTOBUF_EXPORT AddDescriptors();
}  
namespace google {
namespace protobuf {
class Timestamp;
class TimestampDefaultTypeInternal;
LIBPROTOBUF_EXPORT extern TimestampDefaultTypeInternal _Timestamp_default_instance_;
}  
}  
namespace google {
namespace protobuf {
template<> LIBPROTOBUF_EXPORT ::google::protobuf::Timestamp* Arena::CreateMaybeMessage<::google::protobuf::Timestamp>(Arena*);
}  
}  
namespace google {
namespace protobuf {


class LIBPROTOBUF_EXPORT Timestamp : public ::google::protobuf::Message  {
public:
Timestamp();
virtual ~Timestamp();

Timestamp(const Timestamp& from);

inline Timestamp& operator=(const Timestamp& from) {
CopyFrom(from);
return *this;
}
#if LANG_CXX11
Timestamp(Timestamp&& from) noexcept
: Timestamp() {
*this = ::std::move(from);
}

inline Timestamp& operator=(Timestamp&& from) noexcept {
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
static const Timestamp& default_instance();

static void InitAsDefaultInstance();  
static inline const Timestamp* internal_default_instance() {
return reinterpret_cast<const Timestamp*>(
&_Timestamp_default_instance_);
}
static constexpr int kIndexInFileMessages =
0;

void UnsafeArenaSwap(Timestamp* other);
void Swap(Timestamp* other);
friend void swap(Timestamp& a, Timestamp& b) {
a.Swap(&b);
}


inline Timestamp* New() const final {
return CreateMaybeMessage<Timestamp>(NULL);
}

Timestamp* New(::google::protobuf::Arena* arena) const final {
return CreateMaybeMessage<Timestamp>(arena);
}
void CopyFrom(const ::google::protobuf::Message& from) final;
void MergeFrom(const ::google::protobuf::Message& from) final;
void CopyFrom(const Timestamp& from);
void MergeFrom(const Timestamp& from);
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
void InternalSwap(Timestamp* other);
protected:
explicit Timestamp(::google::protobuf::Arena* arena);
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



void clear_seconds();
static const int kSecondsFieldNumber = 1;
::google::protobuf::int64 seconds() const;
void set_seconds(::google::protobuf::int64 value);

void clear_nanos();
static const int kNanosFieldNumber = 2;
::google::protobuf::int32 nanos() const;
void set_nanos(::google::protobuf::int32 value);

private:

::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
typedef void InternalArenaConstructable_;
typedef void DestructorSkippable_;
::google::protobuf::int64 seconds_;
::google::protobuf::int32 nanos_;
mutable ::google::protobuf::internal::CachedSize _cached_size_;
friend struct ::protobuf_google_2fprotobuf_2ftimestamp_2eproto::TableStruct;
};



#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  

inline void Timestamp::clear_seconds() {
seconds_ = GOOGLE_LONGLONG(0);
}
inline ::google::protobuf::int64 Timestamp::seconds() const {
return seconds_;
}
inline void Timestamp::set_seconds(::google::protobuf::int64 value) {

seconds_ = value;
}

inline void Timestamp::clear_nanos() {
nanos_ = 0;
}
inline ::google::protobuf::int32 Timestamp::nanos() const {
return nanos_;
}
inline void Timestamp::set_nanos(::google::protobuf::int32 value) {

nanos_ = value;
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  


}  
}  


#endif  
