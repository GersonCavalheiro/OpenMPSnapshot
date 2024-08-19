

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/s3/model/ObjectCannedACL.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/s3/model/MetadataDirective.h>
#include <aws/s3/model/TaggingDirective.h>
#include <aws/s3/model/ServerSideEncryption.h>
#include <aws/s3/model/StorageClass.h>
#include <aws/s3/model/RequestPayer.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API CopyObjectRequest : public S3Request
{
public:
CopyObjectRequest();

inline virtual const char* GetServiceRequestName() const override { return "CopyObject"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const ObjectCannedACL& GetACL() const{ return m_aCL; }


inline void SetACL(const ObjectCannedACL& value) { m_aCLHasBeenSet = true; m_aCL = value; }


inline void SetACL(ObjectCannedACL&& value) { m_aCLHasBeenSet = true; m_aCL = std::move(value); }


inline CopyObjectRequest& WithACL(const ObjectCannedACL& value) { SetACL(value); return *this;}


inline CopyObjectRequest& WithACL(ObjectCannedACL&& value) { SetACL(std::move(value)); return *this;}



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline CopyObjectRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline CopyObjectRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline CopyObjectRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetCacheControl() const{ return m_cacheControl; }


inline void SetCacheControl(const Aws::String& value) { m_cacheControlHasBeenSet = true; m_cacheControl = value; }


inline void SetCacheControl(Aws::String&& value) { m_cacheControlHasBeenSet = true; m_cacheControl = std::move(value); }


inline void SetCacheControl(const char* value) { m_cacheControlHasBeenSet = true; m_cacheControl.assign(value); }


inline CopyObjectRequest& WithCacheControl(const Aws::String& value) { SetCacheControl(value); return *this;}


inline CopyObjectRequest& WithCacheControl(Aws::String&& value) { SetCacheControl(std::move(value)); return *this;}


inline CopyObjectRequest& WithCacheControl(const char* value) { SetCacheControl(value); return *this;}



inline const Aws::String& GetContentDisposition() const{ return m_contentDisposition; }


inline void SetContentDisposition(const Aws::String& value) { m_contentDispositionHasBeenSet = true; m_contentDisposition = value; }


inline void SetContentDisposition(Aws::String&& value) { m_contentDispositionHasBeenSet = true; m_contentDisposition = std::move(value); }


inline void SetContentDisposition(const char* value) { m_contentDispositionHasBeenSet = true; m_contentDisposition.assign(value); }


inline CopyObjectRequest& WithContentDisposition(const Aws::String& value) { SetContentDisposition(value); return *this;}


inline CopyObjectRequest& WithContentDisposition(Aws::String&& value) { SetContentDisposition(std::move(value)); return *this;}


inline CopyObjectRequest& WithContentDisposition(const char* value) { SetContentDisposition(value); return *this;}



inline const Aws::String& GetContentEncoding() const{ return m_contentEncoding; }


inline void SetContentEncoding(const Aws::String& value) { m_contentEncodingHasBeenSet = true; m_contentEncoding = value; }


inline void SetContentEncoding(Aws::String&& value) { m_contentEncodingHasBeenSet = true; m_contentEncoding = std::move(value); }


inline void SetContentEncoding(const char* value) { m_contentEncodingHasBeenSet = true; m_contentEncoding.assign(value); }


inline CopyObjectRequest& WithContentEncoding(const Aws::String& value) { SetContentEncoding(value); return *this;}


inline CopyObjectRequest& WithContentEncoding(Aws::String&& value) { SetContentEncoding(std::move(value)); return *this;}


inline CopyObjectRequest& WithContentEncoding(const char* value) { SetContentEncoding(value); return *this;}



inline const Aws::String& GetContentLanguage() const{ return m_contentLanguage; }


inline void SetContentLanguage(const Aws::String& value) { m_contentLanguageHasBeenSet = true; m_contentLanguage = value; }


inline void SetContentLanguage(Aws::String&& value) { m_contentLanguageHasBeenSet = true; m_contentLanguage = std::move(value); }


inline void SetContentLanguage(const char* value) { m_contentLanguageHasBeenSet = true; m_contentLanguage.assign(value); }


inline CopyObjectRequest& WithContentLanguage(const Aws::String& value) { SetContentLanguage(value); return *this;}


inline CopyObjectRequest& WithContentLanguage(Aws::String&& value) { SetContentLanguage(std::move(value)); return *this;}


inline CopyObjectRequest& WithContentLanguage(const char* value) { SetContentLanguage(value); return *this;}



inline const Aws::String& GetContentType() const{ return m_contentType; }


inline void SetContentType(const Aws::String& value) { m_contentTypeHasBeenSet = true; m_contentType = value; }


inline void SetContentType(Aws::String&& value) { m_contentTypeHasBeenSet = true; m_contentType = std::move(value); }


inline void SetContentType(const char* value) { m_contentTypeHasBeenSet = true; m_contentType.assign(value); }


inline CopyObjectRequest& WithContentType(const Aws::String& value) { SetContentType(value); return *this;}


inline CopyObjectRequest& WithContentType(Aws::String&& value) { SetContentType(std::move(value)); return *this;}


inline CopyObjectRequest& WithContentType(const char* value) { SetContentType(value); return *this;}



inline const Aws::String& GetCopySource() const{ return m_copySource; }


inline void SetCopySource(const Aws::String& value) { m_copySourceHasBeenSet = true; m_copySource = value; }


inline void SetCopySource(Aws::String&& value) { m_copySourceHasBeenSet = true; m_copySource = std::move(value); }


inline void SetCopySource(const char* value) { m_copySourceHasBeenSet = true; m_copySource.assign(value); }


inline CopyObjectRequest& WithCopySource(const Aws::String& value) { SetCopySource(value); return *this;}


inline CopyObjectRequest& WithCopySource(Aws::String&& value) { SetCopySource(std::move(value)); return *this;}


inline CopyObjectRequest& WithCopySource(const char* value) { SetCopySource(value); return *this;}



inline const Aws::String& GetCopySourceIfMatch() const{ return m_copySourceIfMatch; }


inline void SetCopySourceIfMatch(const Aws::String& value) { m_copySourceIfMatchHasBeenSet = true; m_copySourceIfMatch = value; }


inline void SetCopySourceIfMatch(Aws::String&& value) { m_copySourceIfMatchHasBeenSet = true; m_copySourceIfMatch = std::move(value); }


inline void SetCopySourceIfMatch(const char* value) { m_copySourceIfMatchHasBeenSet = true; m_copySourceIfMatch.assign(value); }


inline CopyObjectRequest& WithCopySourceIfMatch(const Aws::String& value) { SetCopySourceIfMatch(value); return *this;}


inline CopyObjectRequest& WithCopySourceIfMatch(Aws::String&& value) { SetCopySourceIfMatch(std::move(value)); return *this;}


inline CopyObjectRequest& WithCopySourceIfMatch(const char* value) { SetCopySourceIfMatch(value); return *this;}



inline const Aws::Utils::DateTime& GetCopySourceIfModifiedSince() const{ return m_copySourceIfModifiedSince; }


inline void SetCopySourceIfModifiedSince(const Aws::Utils::DateTime& value) { m_copySourceIfModifiedSinceHasBeenSet = true; m_copySourceIfModifiedSince = value; }


inline void SetCopySourceIfModifiedSince(Aws::Utils::DateTime&& value) { m_copySourceIfModifiedSinceHasBeenSet = true; m_copySourceIfModifiedSince = std::move(value); }


inline CopyObjectRequest& WithCopySourceIfModifiedSince(const Aws::Utils::DateTime& value) { SetCopySourceIfModifiedSince(value); return *this;}


inline CopyObjectRequest& WithCopySourceIfModifiedSince(Aws::Utils::DateTime&& value) { SetCopySourceIfModifiedSince(std::move(value)); return *this;}



inline const Aws::String& GetCopySourceIfNoneMatch() const{ return m_copySourceIfNoneMatch; }


inline void SetCopySourceIfNoneMatch(const Aws::String& value) { m_copySourceIfNoneMatchHasBeenSet = true; m_copySourceIfNoneMatch = value; }


inline void SetCopySourceIfNoneMatch(Aws::String&& value) { m_copySourceIfNoneMatchHasBeenSet = true; m_copySourceIfNoneMatch = std::move(value); }


inline void SetCopySourceIfNoneMatch(const char* value) { m_copySourceIfNoneMatchHasBeenSet = true; m_copySourceIfNoneMatch.assign(value); }


inline CopyObjectRequest& WithCopySourceIfNoneMatch(const Aws::String& value) { SetCopySourceIfNoneMatch(value); return *this;}


inline CopyObjectRequest& WithCopySourceIfNoneMatch(Aws::String&& value) { SetCopySourceIfNoneMatch(std::move(value)); return *this;}


inline CopyObjectRequest& WithCopySourceIfNoneMatch(const char* value) { SetCopySourceIfNoneMatch(value); return *this;}



inline const Aws::Utils::DateTime& GetCopySourceIfUnmodifiedSince() const{ return m_copySourceIfUnmodifiedSince; }


inline void SetCopySourceIfUnmodifiedSince(const Aws::Utils::DateTime& value) { m_copySourceIfUnmodifiedSinceHasBeenSet = true; m_copySourceIfUnmodifiedSince = value; }


inline void SetCopySourceIfUnmodifiedSince(Aws::Utils::DateTime&& value) { m_copySourceIfUnmodifiedSinceHasBeenSet = true; m_copySourceIfUnmodifiedSince = std::move(value); }


inline CopyObjectRequest& WithCopySourceIfUnmodifiedSince(const Aws::Utils::DateTime& value) { SetCopySourceIfUnmodifiedSince(value); return *this;}


inline CopyObjectRequest& WithCopySourceIfUnmodifiedSince(Aws::Utils::DateTime&& value) { SetCopySourceIfUnmodifiedSince(std::move(value)); return *this;}



inline const Aws::Utils::DateTime& GetExpires() const{ return m_expires; }


inline void SetExpires(const Aws::Utils::DateTime& value) { m_expiresHasBeenSet = true; m_expires = value; }


inline void SetExpires(Aws::Utils::DateTime&& value) { m_expiresHasBeenSet = true; m_expires = std::move(value); }


inline CopyObjectRequest& WithExpires(const Aws::Utils::DateTime& value) { SetExpires(value); return *this;}


inline CopyObjectRequest& WithExpires(Aws::Utils::DateTime&& value) { SetExpires(std::move(value)); return *this;}



inline const Aws::String& GetGrantFullControl() const{ return m_grantFullControl; }


inline void SetGrantFullControl(const Aws::String& value) { m_grantFullControlHasBeenSet = true; m_grantFullControl = value; }


inline void SetGrantFullControl(Aws::String&& value) { m_grantFullControlHasBeenSet = true; m_grantFullControl = std::move(value); }


inline void SetGrantFullControl(const char* value) { m_grantFullControlHasBeenSet = true; m_grantFullControl.assign(value); }


inline CopyObjectRequest& WithGrantFullControl(const Aws::String& value) { SetGrantFullControl(value); return *this;}


inline CopyObjectRequest& WithGrantFullControl(Aws::String&& value) { SetGrantFullControl(std::move(value)); return *this;}


inline CopyObjectRequest& WithGrantFullControl(const char* value) { SetGrantFullControl(value); return *this;}



inline const Aws::String& GetGrantRead() const{ return m_grantRead; }


inline void SetGrantRead(const Aws::String& value) { m_grantReadHasBeenSet = true; m_grantRead = value; }


inline void SetGrantRead(Aws::String&& value) { m_grantReadHasBeenSet = true; m_grantRead = std::move(value); }


inline void SetGrantRead(const char* value) { m_grantReadHasBeenSet = true; m_grantRead.assign(value); }


inline CopyObjectRequest& WithGrantRead(const Aws::String& value) { SetGrantRead(value); return *this;}


inline CopyObjectRequest& WithGrantRead(Aws::String&& value) { SetGrantRead(std::move(value)); return *this;}


inline CopyObjectRequest& WithGrantRead(const char* value) { SetGrantRead(value); return *this;}



inline const Aws::String& GetGrantReadACP() const{ return m_grantReadACP; }


inline void SetGrantReadACP(const Aws::String& value) { m_grantReadACPHasBeenSet = true; m_grantReadACP = value; }


inline void SetGrantReadACP(Aws::String&& value) { m_grantReadACPHasBeenSet = true; m_grantReadACP = std::move(value); }


inline void SetGrantReadACP(const char* value) { m_grantReadACPHasBeenSet = true; m_grantReadACP.assign(value); }


inline CopyObjectRequest& WithGrantReadACP(const Aws::String& value) { SetGrantReadACP(value); return *this;}


inline CopyObjectRequest& WithGrantReadACP(Aws::String&& value) { SetGrantReadACP(std::move(value)); return *this;}


inline CopyObjectRequest& WithGrantReadACP(const char* value) { SetGrantReadACP(value); return *this;}



inline const Aws::String& GetGrantWriteACP() const{ return m_grantWriteACP; }


inline void SetGrantWriteACP(const Aws::String& value) { m_grantWriteACPHasBeenSet = true; m_grantWriteACP = value; }


inline void SetGrantWriteACP(Aws::String&& value) { m_grantWriteACPHasBeenSet = true; m_grantWriteACP = std::move(value); }


inline void SetGrantWriteACP(const char* value) { m_grantWriteACPHasBeenSet = true; m_grantWriteACP.assign(value); }


inline CopyObjectRequest& WithGrantWriteACP(const Aws::String& value) { SetGrantWriteACP(value); return *this;}


inline CopyObjectRequest& WithGrantWriteACP(Aws::String&& value) { SetGrantWriteACP(std::move(value)); return *this;}


inline CopyObjectRequest& WithGrantWriteACP(const char* value) { SetGrantWriteACP(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline CopyObjectRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline CopyObjectRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline CopyObjectRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::Map<Aws::String, Aws::String>& GetMetadata() const{ return m_metadata; }


inline void SetMetadata(const Aws::Map<Aws::String, Aws::String>& value) { m_metadataHasBeenSet = true; m_metadata = value; }


inline void SetMetadata(Aws::Map<Aws::String, Aws::String>&& value) { m_metadataHasBeenSet = true; m_metadata = std::move(value); }


inline CopyObjectRequest& WithMetadata(const Aws::Map<Aws::String, Aws::String>& value) { SetMetadata(value); return *this;}


inline CopyObjectRequest& WithMetadata(Aws::Map<Aws::String, Aws::String>&& value) { SetMetadata(std::move(value)); return *this;}


inline CopyObjectRequest& AddMetadata(const Aws::String& key, const Aws::String& value) { m_metadataHasBeenSet = true; m_metadata.emplace(key, value); return *this; }


inline CopyObjectRequest& AddMetadata(Aws::String&& key, const Aws::String& value) { m_metadataHasBeenSet = true; m_metadata.emplace(std::move(key), value); return *this; }


inline CopyObjectRequest& AddMetadata(const Aws::String& key, Aws::String&& value) { m_metadataHasBeenSet = true; m_metadata.emplace(key, std::move(value)); return *this; }


inline CopyObjectRequest& AddMetadata(Aws::String&& key, Aws::String&& value) { m_metadataHasBeenSet = true; m_metadata.emplace(std::move(key), std::move(value)); return *this; }


inline CopyObjectRequest& AddMetadata(const char* key, Aws::String&& value) { m_metadataHasBeenSet = true; m_metadata.emplace(key, std::move(value)); return *this; }


inline CopyObjectRequest& AddMetadata(Aws::String&& key, const char* value) { m_metadataHasBeenSet = true; m_metadata.emplace(std::move(key), value); return *this; }


inline CopyObjectRequest& AddMetadata(const char* key, const char* value) { m_metadataHasBeenSet = true; m_metadata.emplace(key, value); return *this; }



inline const MetadataDirective& GetMetadataDirective() const{ return m_metadataDirective; }


inline void SetMetadataDirective(const MetadataDirective& value) { m_metadataDirectiveHasBeenSet = true; m_metadataDirective = value; }


inline void SetMetadataDirective(MetadataDirective&& value) { m_metadataDirectiveHasBeenSet = true; m_metadataDirective = std::move(value); }


inline CopyObjectRequest& WithMetadataDirective(const MetadataDirective& value) { SetMetadataDirective(value); return *this;}


inline CopyObjectRequest& WithMetadataDirective(MetadataDirective&& value) { SetMetadataDirective(std::move(value)); return *this;}



inline const TaggingDirective& GetTaggingDirective() const{ return m_taggingDirective; }


inline void SetTaggingDirective(const TaggingDirective& value) { m_taggingDirectiveHasBeenSet = true; m_taggingDirective = value; }


inline void SetTaggingDirective(TaggingDirective&& value) { m_taggingDirectiveHasBeenSet = true; m_taggingDirective = std::move(value); }


inline CopyObjectRequest& WithTaggingDirective(const TaggingDirective& value) { SetTaggingDirective(value); return *this;}


inline CopyObjectRequest& WithTaggingDirective(TaggingDirective&& value) { SetTaggingDirective(std::move(value)); return *this;}



inline const ServerSideEncryption& GetServerSideEncryption() const{ return m_serverSideEncryption; }


inline void SetServerSideEncryption(const ServerSideEncryption& value) { m_serverSideEncryptionHasBeenSet = true; m_serverSideEncryption = value; }


inline void SetServerSideEncryption(ServerSideEncryption&& value) { m_serverSideEncryptionHasBeenSet = true; m_serverSideEncryption = std::move(value); }


inline CopyObjectRequest& WithServerSideEncryption(const ServerSideEncryption& value) { SetServerSideEncryption(value); return *this;}


inline CopyObjectRequest& WithServerSideEncryption(ServerSideEncryption&& value) { SetServerSideEncryption(std::move(value)); return *this;}



inline const StorageClass& GetStorageClass() const{ return m_storageClass; }


inline void SetStorageClass(const StorageClass& value) { m_storageClassHasBeenSet = true; m_storageClass = value; }


inline void SetStorageClass(StorageClass&& value) { m_storageClassHasBeenSet = true; m_storageClass = std::move(value); }


inline CopyObjectRequest& WithStorageClass(const StorageClass& value) { SetStorageClass(value); return *this;}


inline CopyObjectRequest& WithStorageClass(StorageClass&& value) { SetStorageClass(std::move(value)); return *this;}



inline const Aws::String& GetWebsiteRedirectLocation() const{ return m_websiteRedirectLocation; }


inline void SetWebsiteRedirectLocation(const Aws::String& value) { m_websiteRedirectLocationHasBeenSet = true; m_websiteRedirectLocation = value; }


inline void SetWebsiteRedirectLocation(Aws::String&& value) { m_websiteRedirectLocationHasBeenSet = true; m_websiteRedirectLocation = std::move(value); }


inline void SetWebsiteRedirectLocation(const char* value) { m_websiteRedirectLocationHasBeenSet = true; m_websiteRedirectLocation.assign(value); }


inline CopyObjectRequest& WithWebsiteRedirectLocation(const Aws::String& value) { SetWebsiteRedirectLocation(value); return *this;}


inline CopyObjectRequest& WithWebsiteRedirectLocation(Aws::String&& value) { SetWebsiteRedirectLocation(std::move(value)); return *this;}


inline CopyObjectRequest& WithWebsiteRedirectLocation(const char* value) { SetWebsiteRedirectLocation(value); return *this;}



inline const Aws::String& GetSSECustomerAlgorithm() const{ return m_sSECustomerAlgorithm; }


inline void SetSSECustomerAlgorithm(const Aws::String& value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm = value; }


inline void SetSSECustomerAlgorithm(Aws::String&& value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm = std::move(value); }


inline void SetSSECustomerAlgorithm(const char* value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm.assign(value); }


inline CopyObjectRequest& WithSSECustomerAlgorithm(const Aws::String& value) { SetSSECustomerAlgorithm(value); return *this;}


inline CopyObjectRequest& WithSSECustomerAlgorithm(Aws::String&& value) { SetSSECustomerAlgorithm(std::move(value)); return *this;}


inline CopyObjectRequest& WithSSECustomerAlgorithm(const char* value) { SetSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetSSECustomerKey() const{ return m_sSECustomerKey; }


inline void SetSSECustomerKey(const Aws::String& value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey = value; }


inline void SetSSECustomerKey(Aws::String&& value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey = std::move(value); }


inline void SetSSECustomerKey(const char* value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey.assign(value); }


inline CopyObjectRequest& WithSSECustomerKey(const Aws::String& value) { SetSSECustomerKey(value); return *this;}


inline CopyObjectRequest& WithSSECustomerKey(Aws::String&& value) { SetSSECustomerKey(std::move(value)); return *this;}


inline CopyObjectRequest& WithSSECustomerKey(const char* value) { SetSSECustomerKey(value); return *this;}



inline const Aws::String& GetSSECustomerKeyMD5() const{ return m_sSECustomerKeyMD5; }


inline void SetSSECustomerKeyMD5(const Aws::String& value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5 = value; }


inline void SetSSECustomerKeyMD5(Aws::String&& value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5 = std::move(value); }


inline void SetSSECustomerKeyMD5(const char* value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5.assign(value); }


inline CopyObjectRequest& WithSSECustomerKeyMD5(const Aws::String& value) { SetSSECustomerKeyMD5(value); return *this;}


inline CopyObjectRequest& WithSSECustomerKeyMD5(Aws::String&& value) { SetSSECustomerKeyMD5(std::move(value)); return *this;}


inline CopyObjectRequest& WithSSECustomerKeyMD5(const char* value) { SetSSECustomerKeyMD5(value); return *this;}



inline const Aws::String& GetSSEKMSKeyId() const{ return m_sSEKMSKeyId; }


inline void SetSSEKMSKeyId(const Aws::String& value) { m_sSEKMSKeyIdHasBeenSet = true; m_sSEKMSKeyId = value; }


inline void SetSSEKMSKeyId(Aws::String&& value) { m_sSEKMSKeyIdHasBeenSet = true; m_sSEKMSKeyId = std::move(value); }


inline void SetSSEKMSKeyId(const char* value) { m_sSEKMSKeyIdHasBeenSet = true; m_sSEKMSKeyId.assign(value); }


inline CopyObjectRequest& WithSSEKMSKeyId(const Aws::String& value) { SetSSEKMSKeyId(value); return *this;}


inline CopyObjectRequest& WithSSEKMSKeyId(Aws::String&& value) { SetSSEKMSKeyId(std::move(value)); return *this;}


inline CopyObjectRequest& WithSSEKMSKeyId(const char* value) { SetSSEKMSKeyId(value); return *this;}



inline const Aws::String& GetCopySourceSSECustomerAlgorithm() const{ return m_copySourceSSECustomerAlgorithm; }


inline void SetCopySourceSSECustomerAlgorithm(const Aws::String& value) { m_copySourceSSECustomerAlgorithmHasBeenSet = true; m_copySourceSSECustomerAlgorithm = value; }


inline void SetCopySourceSSECustomerAlgorithm(Aws::String&& value) { m_copySourceSSECustomerAlgorithmHasBeenSet = true; m_copySourceSSECustomerAlgorithm = std::move(value); }


inline void SetCopySourceSSECustomerAlgorithm(const char* value) { m_copySourceSSECustomerAlgorithmHasBeenSet = true; m_copySourceSSECustomerAlgorithm.assign(value); }


inline CopyObjectRequest& WithCopySourceSSECustomerAlgorithm(const Aws::String& value) { SetCopySourceSSECustomerAlgorithm(value); return *this;}


inline CopyObjectRequest& WithCopySourceSSECustomerAlgorithm(Aws::String&& value) { SetCopySourceSSECustomerAlgorithm(std::move(value)); return *this;}


inline CopyObjectRequest& WithCopySourceSSECustomerAlgorithm(const char* value) { SetCopySourceSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetCopySourceSSECustomerKey() const{ return m_copySourceSSECustomerKey; }


inline void SetCopySourceSSECustomerKey(const Aws::String& value) { m_copySourceSSECustomerKeyHasBeenSet = true; m_copySourceSSECustomerKey = value; }


inline void SetCopySourceSSECustomerKey(Aws::String&& value) { m_copySourceSSECustomerKeyHasBeenSet = true; m_copySourceSSECustomerKey = std::move(value); }


inline void SetCopySourceSSECustomerKey(const char* value) { m_copySourceSSECustomerKeyHasBeenSet = true; m_copySourceSSECustomerKey.assign(value); }


inline CopyObjectRequest& WithCopySourceSSECustomerKey(const Aws::String& value) { SetCopySourceSSECustomerKey(value); return *this;}


inline CopyObjectRequest& WithCopySourceSSECustomerKey(Aws::String&& value) { SetCopySourceSSECustomerKey(std::move(value)); return *this;}


inline CopyObjectRequest& WithCopySourceSSECustomerKey(const char* value) { SetCopySourceSSECustomerKey(value); return *this;}



inline const Aws::String& GetCopySourceSSECustomerKeyMD5() const{ return m_copySourceSSECustomerKeyMD5; }


inline void SetCopySourceSSECustomerKeyMD5(const Aws::String& value) { m_copySourceSSECustomerKeyMD5HasBeenSet = true; m_copySourceSSECustomerKeyMD5 = value; }


inline void SetCopySourceSSECustomerKeyMD5(Aws::String&& value) { m_copySourceSSECustomerKeyMD5HasBeenSet = true; m_copySourceSSECustomerKeyMD5 = std::move(value); }


inline void SetCopySourceSSECustomerKeyMD5(const char* value) { m_copySourceSSECustomerKeyMD5HasBeenSet = true; m_copySourceSSECustomerKeyMD5.assign(value); }


inline CopyObjectRequest& WithCopySourceSSECustomerKeyMD5(const Aws::String& value) { SetCopySourceSSECustomerKeyMD5(value); return *this;}


inline CopyObjectRequest& WithCopySourceSSECustomerKeyMD5(Aws::String&& value) { SetCopySourceSSECustomerKeyMD5(std::move(value)); return *this;}


inline CopyObjectRequest& WithCopySourceSSECustomerKeyMD5(const char* value) { SetCopySourceSSECustomerKeyMD5(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline CopyObjectRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline CopyObjectRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}



inline const Aws::String& GetTagging() const{ return m_tagging; }


inline void SetTagging(const Aws::String& value) { m_taggingHasBeenSet = true; m_tagging = value; }


inline void SetTagging(Aws::String&& value) { m_taggingHasBeenSet = true; m_tagging = std::move(value); }


inline void SetTagging(const char* value) { m_taggingHasBeenSet = true; m_tagging.assign(value); }


inline CopyObjectRequest& WithTagging(const Aws::String& value) { SetTagging(value); return *this;}


inline CopyObjectRequest& WithTagging(Aws::String&& value) { SetTagging(std::move(value)); return *this;}


inline CopyObjectRequest& WithTagging(const char* value) { SetTagging(value); return *this;}

private:

ObjectCannedACL m_aCL;
bool m_aCLHasBeenSet;

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_cacheControl;
bool m_cacheControlHasBeenSet;

Aws::String m_contentDisposition;
bool m_contentDispositionHasBeenSet;

Aws::String m_contentEncoding;
bool m_contentEncodingHasBeenSet;

Aws::String m_contentLanguage;
bool m_contentLanguageHasBeenSet;

Aws::String m_contentType;
bool m_contentTypeHasBeenSet;

Aws::String m_copySource;
bool m_copySourceHasBeenSet;

Aws::String m_copySourceIfMatch;
bool m_copySourceIfMatchHasBeenSet;

Aws::Utils::DateTime m_copySourceIfModifiedSince;
bool m_copySourceIfModifiedSinceHasBeenSet;

Aws::String m_copySourceIfNoneMatch;
bool m_copySourceIfNoneMatchHasBeenSet;

Aws::Utils::DateTime m_copySourceIfUnmodifiedSince;
bool m_copySourceIfUnmodifiedSinceHasBeenSet;

Aws::Utils::DateTime m_expires;
bool m_expiresHasBeenSet;

Aws::String m_grantFullControl;
bool m_grantFullControlHasBeenSet;

Aws::String m_grantRead;
bool m_grantReadHasBeenSet;

Aws::String m_grantReadACP;
bool m_grantReadACPHasBeenSet;

Aws::String m_grantWriteACP;
bool m_grantWriteACPHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::Map<Aws::String, Aws::String> m_metadata;
bool m_metadataHasBeenSet;

MetadataDirective m_metadataDirective;
bool m_metadataDirectiveHasBeenSet;

TaggingDirective m_taggingDirective;
bool m_taggingDirectiveHasBeenSet;

ServerSideEncryption m_serverSideEncryption;
bool m_serverSideEncryptionHasBeenSet;

StorageClass m_storageClass;
bool m_storageClassHasBeenSet;

Aws::String m_websiteRedirectLocation;
bool m_websiteRedirectLocationHasBeenSet;

Aws::String m_sSECustomerAlgorithm;
bool m_sSECustomerAlgorithmHasBeenSet;

Aws::String m_sSECustomerKey;
bool m_sSECustomerKeyHasBeenSet;

Aws::String m_sSECustomerKeyMD5;
bool m_sSECustomerKeyMD5HasBeenSet;

Aws::String m_sSEKMSKeyId;
bool m_sSEKMSKeyIdHasBeenSet;

Aws::String m_copySourceSSECustomerAlgorithm;
bool m_copySourceSSECustomerAlgorithmHasBeenSet;

Aws::String m_copySourceSSECustomerKey;
bool m_copySourceSSECustomerKeyHasBeenSet;

Aws::String m_copySourceSSECustomerKeyMD5;
bool m_copySourceSSECustomerKeyMD5HasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;

Aws::String m_tagging;
bool m_taggingHasBeenSet;
};

} 
} 
} 
