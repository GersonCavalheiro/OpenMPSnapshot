

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
class AWS_S3_API PutObjectResult
{
public:
PutObjectResult();
PutObjectResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
PutObjectResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::String& GetExpiration() const{ return m_expiration; }


inline void SetExpiration(const Aws::String& value) { m_expiration = value; }


inline void SetExpiration(Aws::String&& value) { m_expiration = std::move(value); }


inline void SetExpiration(const char* value) { m_expiration.assign(value); }


inline PutObjectResult& WithExpiration(const Aws::String& value) { SetExpiration(value); return *this;}


inline PutObjectResult& WithExpiration(Aws::String&& value) { SetExpiration(std::move(value)); return *this;}


inline PutObjectResult& WithExpiration(const char* value) { SetExpiration(value); return *this;}



inline const Aws::String& GetETag() const{ return m_eTag; }


inline void SetETag(const Aws::String& value) { m_eTag = value; }


inline void SetETag(Aws::String&& value) { m_eTag = std::move(value); }


inline void SetETag(const char* value) { m_eTag.assign(value); }


inline PutObjectResult& WithETag(const Aws::String& value) { SetETag(value); return *this;}


inline PutObjectResult& WithETag(Aws::String&& value) { SetETag(std::move(value)); return *this;}


inline PutObjectResult& WithETag(const char* value) { SetETag(value); return *this;}



inline const ServerSideEncryption& GetServerSideEncryption() const{ return m_serverSideEncryption; }


inline void SetServerSideEncryption(const ServerSideEncryption& value) { m_serverSideEncryption = value; }


inline void SetServerSideEncryption(ServerSideEncryption&& value) { m_serverSideEncryption = std::move(value); }


inline PutObjectResult& WithServerSideEncryption(const ServerSideEncryption& value) { SetServerSideEncryption(value); return *this;}


inline PutObjectResult& WithServerSideEncryption(ServerSideEncryption&& value) { SetServerSideEncryption(std::move(value)); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionId.assign(value); }


inline PutObjectResult& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline PutObjectResult& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline PutObjectResult& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline const Aws::String& GetSSECustomerAlgorithm() const{ return m_sSECustomerAlgorithm; }


inline void SetSSECustomerAlgorithm(const Aws::String& value) { m_sSECustomerAlgorithm = value; }


inline void SetSSECustomerAlgorithm(Aws::String&& value) { m_sSECustomerAlgorithm = std::move(value); }


inline void SetSSECustomerAlgorithm(const char* value) { m_sSECustomerAlgorithm.assign(value); }


inline PutObjectResult& WithSSECustomerAlgorithm(const Aws::String& value) { SetSSECustomerAlgorithm(value); return *this;}


inline PutObjectResult& WithSSECustomerAlgorithm(Aws::String&& value) { SetSSECustomerAlgorithm(std::move(value)); return *this;}


inline PutObjectResult& WithSSECustomerAlgorithm(const char* value) { SetSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetSSECustomerKeyMD5() const{ return m_sSECustomerKeyMD5; }


inline void SetSSECustomerKeyMD5(const Aws::String& value) { m_sSECustomerKeyMD5 = value; }


inline void SetSSECustomerKeyMD5(Aws::String&& value) { m_sSECustomerKeyMD5 = std::move(value); }


inline void SetSSECustomerKeyMD5(const char* value) { m_sSECustomerKeyMD5.assign(value); }


inline PutObjectResult& WithSSECustomerKeyMD5(const Aws::String& value) { SetSSECustomerKeyMD5(value); return *this;}


inline PutObjectResult& WithSSECustomerKeyMD5(Aws::String&& value) { SetSSECustomerKeyMD5(std::move(value)); return *this;}


inline PutObjectResult& WithSSECustomerKeyMD5(const char* value) { SetSSECustomerKeyMD5(value); return *this;}



inline const Aws::String& GetSSEKMSKeyId() const{ return m_sSEKMSKeyId; }


inline void SetSSEKMSKeyId(const Aws::String& value) { m_sSEKMSKeyId = value; }


inline void SetSSEKMSKeyId(Aws::String&& value) { m_sSEKMSKeyId = std::move(value); }


inline void SetSSEKMSKeyId(const char* value) { m_sSEKMSKeyId.assign(value); }


inline PutObjectResult& WithSSEKMSKeyId(const Aws::String& value) { SetSSEKMSKeyId(value); return *this;}


inline PutObjectResult& WithSSEKMSKeyId(Aws::String&& value) { SetSSEKMSKeyId(std::move(value)); return *this;}


inline PutObjectResult& WithSSEKMSKeyId(const char* value) { SetSSEKMSKeyId(value); return *this;}



inline const RequestCharged& GetRequestCharged() const{ return m_requestCharged; }


inline void SetRequestCharged(const RequestCharged& value) { m_requestCharged = value; }


inline void SetRequestCharged(RequestCharged&& value) { m_requestCharged = std::move(value); }


inline PutObjectResult& WithRequestCharged(const RequestCharged& value) { SetRequestCharged(value); return *this;}


inline PutObjectResult& WithRequestCharged(RequestCharged&& value) { SetRequestCharged(std::move(value)); return *this;}

private:

Aws::String m_expiration;

Aws::String m_eTag;

ServerSideEncryption m_serverSideEncryption;

Aws::String m_versionId;

Aws::String m_sSECustomerAlgorithm;

Aws::String m_sSECustomerKeyMD5;

Aws::String m_sSEKMSKeyId;

RequestCharged m_requestCharged;
};

} 
} 
} 
