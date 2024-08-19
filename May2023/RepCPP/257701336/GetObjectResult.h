

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/core/utils/Array.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
#include <aws/s3/model/ServerSideEncryption.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/s3/model/StorageClass.h>
#include <aws/s3/model/RequestCharged.h>
#include <aws/s3/model/ReplicationStatus.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace S3
{
namespace Model
{
class AWS_S3_API GetObjectResult
{
public:
GetObjectResult();
GetObjectResult(GetObjectResult&&);
GetObjectResult& operator=(GetObjectResult&&);
GetObjectResult(const GetObjectResult&) = delete;
GetObjectResult& operator=(const GetObjectResult&) = delete;


GetObjectResult(Aws::AmazonWebServiceResult<Aws::Utils::Stream::ResponseStream>&& result);
GetObjectResult& operator=(Aws::AmazonWebServiceResult<Aws::Utils::Stream::ResponseStream>&& result);




inline Aws::IOStream& GetBody() { return m_body.GetUnderlyingStream(); }


inline void ReplaceBody(Aws::IOStream* body) { m_body = Aws::Utils::Stream::ResponseStream(body); }



inline bool GetDeleteMarker() const{ return m_deleteMarker; }


inline void SetDeleteMarker(bool value) { m_deleteMarker = value; }


inline GetObjectResult& WithDeleteMarker(bool value) { SetDeleteMarker(value); return *this;}



inline const Aws::String& GetAcceptRanges() const{ return m_acceptRanges; }


inline void SetAcceptRanges(const Aws::String& value) { m_acceptRanges = value; }


inline void SetAcceptRanges(Aws::String&& value) { m_acceptRanges = std::move(value); }


inline void SetAcceptRanges(const char* value) { m_acceptRanges.assign(value); }


inline GetObjectResult& WithAcceptRanges(const Aws::String& value) { SetAcceptRanges(value); return *this;}


inline GetObjectResult& WithAcceptRanges(Aws::String&& value) { SetAcceptRanges(std::move(value)); return *this;}


inline GetObjectResult& WithAcceptRanges(const char* value) { SetAcceptRanges(value); return *this;}



inline const Aws::String& GetExpiration() const{ return m_expiration; }


inline void SetExpiration(const Aws::String& value) { m_expiration = value; }


inline void SetExpiration(Aws::String&& value) { m_expiration = std::move(value); }


inline void SetExpiration(const char* value) { m_expiration.assign(value); }


inline GetObjectResult& WithExpiration(const Aws::String& value) { SetExpiration(value); return *this;}


inline GetObjectResult& WithExpiration(Aws::String&& value) { SetExpiration(std::move(value)); return *this;}


inline GetObjectResult& WithExpiration(const char* value) { SetExpiration(value); return *this;}



inline const Aws::String& GetRestore() const{ return m_restore; }


inline void SetRestore(const Aws::String& value) { m_restore = value; }


inline void SetRestore(Aws::String&& value) { m_restore = std::move(value); }


inline void SetRestore(const char* value) { m_restore.assign(value); }


inline GetObjectResult& WithRestore(const Aws::String& value) { SetRestore(value); return *this;}


inline GetObjectResult& WithRestore(Aws::String&& value) { SetRestore(std::move(value)); return *this;}


inline GetObjectResult& WithRestore(const char* value) { SetRestore(value); return *this;}



inline const Aws::Utils::DateTime& GetLastModified() const{ return m_lastModified; }


inline void SetLastModified(const Aws::Utils::DateTime& value) { m_lastModified = value; }


inline void SetLastModified(Aws::Utils::DateTime&& value) { m_lastModified = std::move(value); }


inline GetObjectResult& WithLastModified(const Aws::Utils::DateTime& value) { SetLastModified(value); return *this;}


inline GetObjectResult& WithLastModified(Aws::Utils::DateTime&& value) { SetLastModified(std::move(value)); return *this;}



inline long long GetContentLength() const{ return m_contentLength; }


inline void SetContentLength(long long value) { m_contentLength = value; }


inline GetObjectResult& WithContentLength(long long value) { SetContentLength(value); return *this;}



inline const Aws::String& GetETag() const{ return m_eTag; }


inline void SetETag(const Aws::String& value) { m_eTag = value; }


inline void SetETag(Aws::String&& value) { m_eTag = std::move(value); }


inline void SetETag(const char* value) { m_eTag.assign(value); }


inline GetObjectResult& WithETag(const Aws::String& value) { SetETag(value); return *this;}


inline GetObjectResult& WithETag(Aws::String&& value) { SetETag(std::move(value)); return *this;}


inline GetObjectResult& WithETag(const char* value) { SetETag(value); return *this;}



inline int GetMissingMeta() const{ return m_missingMeta; }


inline void SetMissingMeta(int value) { m_missingMeta = value; }


inline GetObjectResult& WithMissingMeta(int value) { SetMissingMeta(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionId.assign(value); }


inline GetObjectResult& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline GetObjectResult& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline GetObjectResult& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline const Aws::String& GetCacheControl() const{ return m_cacheControl; }


inline void SetCacheControl(const Aws::String& value) { m_cacheControl = value; }


inline void SetCacheControl(Aws::String&& value) { m_cacheControl = std::move(value); }


inline void SetCacheControl(const char* value) { m_cacheControl.assign(value); }


inline GetObjectResult& WithCacheControl(const Aws::String& value) { SetCacheControl(value); return *this;}


inline GetObjectResult& WithCacheControl(Aws::String&& value) { SetCacheControl(std::move(value)); return *this;}


inline GetObjectResult& WithCacheControl(const char* value) { SetCacheControl(value); return *this;}



inline const Aws::String& GetContentDisposition() const{ return m_contentDisposition; }


inline void SetContentDisposition(const Aws::String& value) { m_contentDisposition = value; }


inline void SetContentDisposition(Aws::String&& value) { m_contentDisposition = std::move(value); }


inline void SetContentDisposition(const char* value) { m_contentDisposition.assign(value); }


inline GetObjectResult& WithContentDisposition(const Aws::String& value) { SetContentDisposition(value); return *this;}


inline GetObjectResult& WithContentDisposition(Aws::String&& value) { SetContentDisposition(std::move(value)); return *this;}


inline GetObjectResult& WithContentDisposition(const char* value) { SetContentDisposition(value); return *this;}



inline const Aws::String& GetContentEncoding() const{ return m_contentEncoding; }


inline void SetContentEncoding(const Aws::String& value) { m_contentEncoding = value; }


inline void SetContentEncoding(Aws::String&& value) { m_contentEncoding = std::move(value); }


inline void SetContentEncoding(const char* value) { m_contentEncoding.assign(value); }


inline GetObjectResult& WithContentEncoding(const Aws::String& value) { SetContentEncoding(value); return *this;}


inline GetObjectResult& WithContentEncoding(Aws::String&& value) { SetContentEncoding(std::move(value)); return *this;}


inline GetObjectResult& WithContentEncoding(const char* value) { SetContentEncoding(value); return *this;}



inline const Aws::String& GetContentLanguage() const{ return m_contentLanguage; }


inline void SetContentLanguage(const Aws::String& value) { m_contentLanguage = value; }


inline void SetContentLanguage(Aws::String&& value) { m_contentLanguage = std::move(value); }


inline void SetContentLanguage(const char* value) { m_contentLanguage.assign(value); }


inline GetObjectResult& WithContentLanguage(const Aws::String& value) { SetContentLanguage(value); return *this;}


inline GetObjectResult& WithContentLanguage(Aws::String&& value) { SetContentLanguage(std::move(value)); return *this;}


inline GetObjectResult& WithContentLanguage(const char* value) { SetContentLanguage(value); return *this;}



inline const Aws::String& GetContentRange() const{ return m_contentRange; }


inline void SetContentRange(const Aws::String& value) { m_contentRange = value; }


inline void SetContentRange(Aws::String&& value) { m_contentRange = std::move(value); }


inline void SetContentRange(const char* value) { m_contentRange.assign(value); }


inline GetObjectResult& WithContentRange(const Aws::String& value) { SetContentRange(value); return *this;}


inline GetObjectResult& WithContentRange(Aws::String&& value) { SetContentRange(std::move(value)); return *this;}


inline GetObjectResult& WithContentRange(const char* value) { SetContentRange(value); return *this;}



inline const Aws::String& GetContentType() const{ return m_contentType; }


inline void SetContentType(const Aws::String& value) { m_contentType = value; }


inline void SetContentType(Aws::String&& value) { m_contentType = std::move(value); }


inline void SetContentType(const char* value) { m_contentType.assign(value); }


inline GetObjectResult& WithContentType(const Aws::String& value) { SetContentType(value); return *this;}


inline GetObjectResult& WithContentType(Aws::String&& value) { SetContentType(std::move(value)); return *this;}


inline GetObjectResult& WithContentType(const char* value) { SetContentType(value); return *this;}



inline const Aws::Utils::DateTime& GetExpires() const{ return m_expires; }


inline void SetExpires(const Aws::Utils::DateTime& value) { m_expires = value; }


inline void SetExpires(Aws::Utils::DateTime&& value) { m_expires = std::move(value); }


inline GetObjectResult& WithExpires(const Aws::Utils::DateTime& value) { SetExpires(value); return *this;}


inline GetObjectResult& WithExpires(Aws::Utils::DateTime&& value) { SetExpires(std::move(value)); return *this;}



inline const Aws::String& GetWebsiteRedirectLocation() const{ return m_websiteRedirectLocation; }


inline void SetWebsiteRedirectLocation(const Aws::String& value) { m_websiteRedirectLocation = value; }


inline void SetWebsiteRedirectLocation(Aws::String&& value) { m_websiteRedirectLocation = std::move(value); }


inline void SetWebsiteRedirectLocation(const char* value) { m_websiteRedirectLocation.assign(value); }


inline GetObjectResult& WithWebsiteRedirectLocation(const Aws::String& value) { SetWebsiteRedirectLocation(value); return *this;}


inline GetObjectResult& WithWebsiteRedirectLocation(Aws::String&& value) { SetWebsiteRedirectLocation(std::move(value)); return *this;}


inline GetObjectResult& WithWebsiteRedirectLocation(const char* value) { SetWebsiteRedirectLocation(value); return *this;}



inline const ServerSideEncryption& GetServerSideEncryption() const{ return m_serverSideEncryption; }


inline void SetServerSideEncryption(const ServerSideEncryption& value) { m_serverSideEncryption = value; }


inline void SetServerSideEncryption(ServerSideEncryption&& value) { m_serverSideEncryption = std::move(value); }


inline GetObjectResult& WithServerSideEncryption(const ServerSideEncryption& value) { SetServerSideEncryption(value); return *this;}


inline GetObjectResult& WithServerSideEncryption(ServerSideEncryption&& value) { SetServerSideEncryption(std::move(value)); return *this;}



inline const Aws::Map<Aws::String, Aws::String>& GetMetadata() const{ return m_metadata; }


inline void SetMetadata(const Aws::Map<Aws::String, Aws::String>& value) { m_metadata = value; }


inline void SetMetadata(Aws::Map<Aws::String, Aws::String>&& value) { m_metadata = std::move(value); }


inline GetObjectResult& WithMetadata(const Aws::Map<Aws::String, Aws::String>& value) { SetMetadata(value); return *this;}


inline GetObjectResult& WithMetadata(Aws::Map<Aws::String, Aws::String>&& value) { SetMetadata(std::move(value)); return *this;}


inline GetObjectResult& AddMetadata(const Aws::String& key, const Aws::String& value) { m_metadata.emplace(key, value); return *this; }


inline GetObjectResult& AddMetadata(Aws::String&& key, const Aws::String& value) { m_metadata.emplace(std::move(key), value); return *this; }


inline GetObjectResult& AddMetadata(const Aws::String& key, Aws::String&& value) { m_metadata.emplace(key, std::move(value)); return *this; }


inline GetObjectResult& AddMetadata(Aws::String&& key, Aws::String&& value) { m_metadata.emplace(std::move(key), std::move(value)); return *this; }


inline GetObjectResult& AddMetadata(const char* key, Aws::String&& value) { m_metadata.emplace(key, std::move(value)); return *this; }


inline GetObjectResult& AddMetadata(Aws::String&& key, const char* value) { m_metadata.emplace(std::move(key), value); return *this; }


inline GetObjectResult& AddMetadata(const char* key, const char* value) { m_metadata.emplace(key, value); return *this; }



inline const Aws::String& GetSSECustomerAlgorithm() const{ return m_sSECustomerAlgorithm; }


inline void SetSSECustomerAlgorithm(const Aws::String& value) { m_sSECustomerAlgorithm = value; }


inline void SetSSECustomerAlgorithm(Aws::String&& value) { m_sSECustomerAlgorithm = std::move(value); }


inline void SetSSECustomerAlgorithm(const char* value) { m_sSECustomerAlgorithm.assign(value); }


inline GetObjectResult& WithSSECustomerAlgorithm(const Aws::String& value) { SetSSECustomerAlgorithm(value); return *this;}


inline GetObjectResult& WithSSECustomerAlgorithm(Aws::String&& value) { SetSSECustomerAlgorithm(std::move(value)); return *this;}


inline GetObjectResult& WithSSECustomerAlgorithm(const char* value) { SetSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetSSECustomerKeyMD5() const{ return m_sSECustomerKeyMD5; }


inline void SetSSECustomerKeyMD5(const Aws::String& value) { m_sSECustomerKeyMD5 = value; }


inline void SetSSECustomerKeyMD5(Aws::String&& value) { m_sSECustomerKeyMD5 = std::move(value); }


inline void SetSSECustomerKeyMD5(const char* value) { m_sSECustomerKeyMD5.assign(value); }


inline GetObjectResult& WithSSECustomerKeyMD5(const Aws::String& value) { SetSSECustomerKeyMD5(value); return *this;}


inline GetObjectResult& WithSSECustomerKeyMD5(Aws::String&& value) { SetSSECustomerKeyMD5(std::move(value)); return *this;}


inline GetObjectResult& WithSSECustomerKeyMD5(const char* value) { SetSSECustomerKeyMD5(value); return *this;}



inline const Aws::String& GetSSEKMSKeyId() const{ return m_sSEKMSKeyId; }


inline void SetSSEKMSKeyId(const Aws::String& value) { m_sSEKMSKeyId = value; }


inline void SetSSEKMSKeyId(Aws::String&& value) { m_sSEKMSKeyId = std::move(value); }


inline void SetSSEKMSKeyId(const char* value) { m_sSEKMSKeyId.assign(value); }


inline GetObjectResult& WithSSEKMSKeyId(const Aws::String& value) { SetSSEKMSKeyId(value); return *this;}


inline GetObjectResult& WithSSEKMSKeyId(Aws::String&& value) { SetSSEKMSKeyId(std::move(value)); return *this;}


inline GetObjectResult& WithSSEKMSKeyId(const char* value) { SetSSEKMSKeyId(value); return *this;}



inline const StorageClass& GetStorageClass() const{ return m_storageClass; }


inline void SetStorageClass(const StorageClass& value) { m_storageClass = value; }


inline void SetStorageClass(StorageClass&& value) { m_storageClass = std::move(value); }


inline GetObjectResult& WithStorageClass(const StorageClass& value) { SetStorageClass(value); return *this;}


inline GetObjectResult& WithStorageClass(StorageClass&& value) { SetStorageClass(std::move(value)); return *this;}



inline const RequestCharged& GetRequestCharged() const{ return m_requestCharged; }


inline void SetRequestCharged(const RequestCharged& value) { m_requestCharged = value; }


inline void SetRequestCharged(RequestCharged&& value) { m_requestCharged = std::move(value); }


inline GetObjectResult& WithRequestCharged(const RequestCharged& value) { SetRequestCharged(value); return *this;}


inline GetObjectResult& WithRequestCharged(RequestCharged&& value) { SetRequestCharged(std::move(value)); return *this;}



inline const ReplicationStatus& GetReplicationStatus() const{ return m_replicationStatus; }


inline void SetReplicationStatus(const ReplicationStatus& value) { m_replicationStatus = value; }


inline void SetReplicationStatus(ReplicationStatus&& value) { m_replicationStatus = std::move(value); }


inline GetObjectResult& WithReplicationStatus(const ReplicationStatus& value) { SetReplicationStatus(value); return *this;}


inline GetObjectResult& WithReplicationStatus(ReplicationStatus&& value) { SetReplicationStatus(std::move(value)); return *this;}



inline int GetPartsCount() const{ return m_partsCount; }


inline void SetPartsCount(int value) { m_partsCount = value; }


inline GetObjectResult& WithPartsCount(int value) { SetPartsCount(value); return *this;}



inline int GetTagCount() const{ return m_tagCount; }


inline void SetTagCount(int value) { m_tagCount = value; }


inline GetObjectResult& WithTagCount(int value) { SetTagCount(value); return *this;}



inline const Aws::String& GetId2() const{ return m_id2; }


inline void SetId2(const Aws::String& value) { m_id2 = value; }


inline void SetId2(Aws::String&& value) { m_id2 = std::move(value); }


inline void SetId2(const char* value) { m_id2.assign(value); }


inline GetObjectResult& WithId2(const Aws::String& value) { SetId2(value); return *this;}


inline GetObjectResult& WithId2(Aws::String&& value) { SetId2(std::move(value)); return *this;}


inline GetObjectResult& WithId2(const char* value) { SetId2(value); return *this;}



inline const Aws::String& GetRequestId() const{ return m_requestId; }


inline void SetRequestId(const Aws::String& value) { m_requestId = value; }


inline void SetRequestId(Aws::String&& value) { m_requestId = std::move(value); }


inline void SetRequestId(const char* value) { m_requestId.assign(value); }


inline GetObjectResult& WithRequestId(const Aws::String& value) { SetRequestId(value); return *this;}


inline GetObjectResult& WithRequestId(Aws::String&& value) { SetRequestId(std::move(value)); return *this;}


inline GetObjectResult& WithRequestId(const char* value) { SetRequestId(value); return *this;}

private:

Aws::Utils::Stream::ResponseStream m_body;

bool m_deleteMarker;

Aws::String m_acceptRanges;

Aws::String m_expiration;

Aws::String m_restore;

Aws::Utils::DateTime m_lastModified;

long long m_contentLength;

Aws::String m_eTag;

int m_missingMeta;

Aws::String m_versionId;

Aws::String m_cacheControl;

Aws::String m_contentDisposition;

Aws::String m_contentEncoding;

Aws::String m_contentLanguage;

Aws::String m_contentRange;

Aws::String m_contentType;

Aws::Utils::DateTime m_expires;

Aws::String m_websiteRedirectLocation;

ServerSideEncryption m_serverSideEncryption;

Aws::Map<Aws::String, Aws::String> m_metadata;

Aws::String m_sSECustomerAlgorithm;

Aws::String m_sSECustomerKeyMD5;

Aws::String m_sSEKMSKeyId;

StorageClass m_storageClass;

RequestCharged m_requestCharged;

ReplicationStatus m_replicationStatus;

int m_partsCount;

int m_tagCount;

Aws::String m_id2;

Aws::String m_requestId;
};

} 
} 
} 
