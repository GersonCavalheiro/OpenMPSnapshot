

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/ServerSideEncryption.h>
#include <aws/s3/model/RequestCharged.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Xml
{
class XmlDocument;
} 
} 
namespace S3
{
namespace Model
{
class AWS_S3_API CompleteMultipartUploadResult
{
public:
CompleteMultipartUploadResult();
CompleteMultipartUploadResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
CompleteMultipartUploadResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::String& GetLocation() const{ return m_location; }


inline void SetLocation(const Aws::String& value) { m_location = value; }


inline void SetLocation(Aws::String&& value) { m_location = std::move(value); }


inline void SetLocation(const char* value) { m_location.assign(value); }


inline CompleteMultipartUploadResult& WithLocation(const Aws::String& value) { SetLocation(value); return *this;}


inline CompleteMultipartUploadResult& WithLocation(Aws::String&& value) { SetLocation(std::move(value)); return *this;}


inline CompleteMultipartUploadResult& WithLocation(const char* value) { SetLocation(value); return *this;}



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucket.assign(value); }


inline CompleteMultipartUploadResult& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline CompleteMultipartUploadResult& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline CompleteMultipartUploadResult& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_key = value; }


inline void SetKey(Aws::String&& value) { m_key = std::move(value); }


inline void SetKey(const char* value) { m_key.assign(value); }


inline CompleteMultipartUploadResult& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline CompleteMultipartUploadResult& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline CompleteMultipartUploadResult& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetExpiration() const{ return m_expiration; }


inline void SetExpiration(const Aws::String& value) { m_expiration = value; }


inline void SetExpiration(Aws::String&& value) { m_expiration = std::move(value); }


inline void SetExpiration(const char* value) { m_expiration.assign(value); }


inline CompleteMultipartUploadResult& WithExpiration(const Aws::String& value) { SetExpiration(value); return *this;}


inline CompleteMultipartUploadResult& WithExpiration(Aws::String&& value) { SetExpiration(std::move(value)); return *this;}


inline CompleteMultipartUploadResult& WithExpiration(const char* value) { SetExpiration(value); return *this;}



inline const Aws::String& GetETag() const{ return m_eTag; }


inline void SetETag(const Aws::String& value) { m_eTag = value; }


inline void SetETag(Aws::String&& value) { m_eTag = std::move(value); }


inline void SetETag(const char* value) { m_eTag.assign(value); }


inline CompleteMultipartUploadResult& WithETag(const Aws::String& value) { SetETag(value); return *this;}


inline CompleteMultipartUploadResult& WithETag(Aws::String&& value) { SetETag(std::move(value)); return *this;}


inline CompleteMultipartUploadResult& WithETag(const char* value) { SetETag(value); return *this;}



inline const ServerSideEncryption& GetServerSideEncryption() const{ return m_serverSideEncryption; }


inline void SetServerSideEncryption(const ServerSideEncryption& value) { m_serverSideEncryption = value; }


inline void SetServerSideEncryption(ServerSideEncryption&& value) { m_serverSideEncryption = std::move(value); }


inline CompleteMultipartUploadResult& WithServerSideEncryption(const ServerSideEncryption& value) { SetServerSideEncryption(value); return *this;}


inline CompleteMultipartUploadResult& WithServerSideEncryption(ServerSideEncryption&& value) { SetServerSideEncryption(std::move(value)); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionId.assign(value); }


inline CompleteMultipartUploadResult& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline CompleteMultipartUploadResult& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline CompleteMultipartUploadResult& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline const Aws::String& GetSSEKMSKeyId() const{ return m_sSEKMSKeyId; }


inline void SetSSEKMSKeyId(const Aws::String& value) { m_sSEKMSKeyId = value; }


inline void SetSSEKMSKeyId(Aws::String&& value) { m_sSEKMSKeyId = std::move(value); }


inline void SetSSEKMSKeyId(const char* value) { m_sSEKMSKeyId.assign(value); }


inline CompleteMultipartUploadResult& WithSSEKMSKeyId(const Aws::String& value) { SetSSEKMSKeyId(value); return *this;}


inline CompleteMultipartUploadResult& WithSSEKMSKeyId(Aws::String&& value) { SetSSEKMSKeyId(std::move(value)); return *this;}


inline CompleteMultipartUploadResult& WithSSEKMSKeyId(const char* value) { SetSSEKMSKeyId(value); return *this;}



inline const RequestCharged& GetRequestCharged() const{ return m_requestCharged; }


inline void SetRequestCharged(const RequestCharged& value) { m_requestCharged = value; }


inline void SetRequestCharged(RequestCharged&& value) { m_requestCharged = std::move(value); }


inline CompleteMultipartUploadResult& WithRequestCharged(const RequestCharged& value) { SetRequestCharged(value); return *this;}


inline CompleteMultipartUploadResult& WithRequestCharged(RequestCharged&& value) { SetRequestCharged(std::move(value)); return *this;}

private:

Aws::String m_location;

Aws::String m_bucket;

Aws::String m_key;

Aws::String m_expiration;

Aws::String m_eTag;

ServerSideEncryption m_serverSideEncryption;

Aws::String m_versionId;

Aws::String m_sSEKMSKeyId;

RequestCharged m_requestCharged;
};

} 
} 
} 
