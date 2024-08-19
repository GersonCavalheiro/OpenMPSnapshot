

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/DateTime.h>
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
class AWS_S3_API CreateMultipartUploadResult
{
public:
CreateMultipartUploadResult();
CreateMultipartUploadResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
CreateMultipartUploadResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::Utils::DateTime& GetAbortDate() const{ return m_abortDate; }


inline void SetAbortDate(const Aws::Utils::DateTime& value) { m_abortDate = value; }


inline void SetAbortDate(Aws::Utils::DateTime&& value) { m_abortDate = std::move(value); }


inline CreateMultipartUploadResult& WithAbortDate(const Aws::Utils::DateTime& value) { SetAbortDate(value); return *this;}


inline CreateMultipartUploadResult& WithAbortDate(Aws::Utils::DateTime&& value) { SetAbortDate(std::move(value)); return *this;}



inline const Aws::String& GetAbortRuleId() const{ return m_abortRuleId; }


inline void SetAbortRuleId(const Aws::String& value) { m_abortRuleId = value; }


inline void SetAbortRuleId(Aws::String&& value) { m_abortRuleId = std::move(value); }


inline void SetAbortRuleId(const char* value) { m_abortRuleId.assign(value); }


inline CreateMultipartUploadResult& WithAbortRuleId(const Aws::String& value) { SetAbortRuleId(value); return *this;}


inline CreateMultipartUploadResult& WithAbortRuleId(Aws::String&& value) { SetAbortRuleId(std::move(value)); return *this;}


inline CreateMultipartUploadResult& WithAbortRuleId(const char* value) { SetAbortRuleId(value); return *this;}



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucket.assign(value); }


inline CreateMultipartUploadResult& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline CreateMultipartUploadResult& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline CreateMultipartUploadResult& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_key = value; }


inline void SetKey(Aws::String&& value) { m_key = std::move(value); }


inline void SetKey(const char* value) { m_key.assign(value); }


inline CreateMultipartUploadResult& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline CreateMultipartUploadResult& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline CreateMultipartUploadResult& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetUploadId() const{ return m_uploadId; }


inline void SetUploadId(const Aws::String& value) { m_uploadId = value; }


inline void SetUploadId(Aws::String&& value) { m_uploadId = std::move(value); }


inline void SetUploadId(const char* value) { m_uploadId.assign(value); }


inline CreateMultipartUploadResult& WithUploadId(const Aws::String& value) { SetUploadId(value); return *this;}


inline CreateMultipartUploadResult& WithUploadId(Aws::String&& value) { SetUploadId(std::move(value)); return *this;}


inline CreateMultipartUploadResult& WithUploadId(const char* value) { SetUploadId(value); return *this;}



inline const ServerSideEncryption& GetServerSideEncryption() const{ return m_serverSideEncryption; }


inline void SetServerSideEncryption(const ServerSideEncryption& value) { m_serverSideEncryption = value; }


inline void SetServerSideEncryption(ServerSideEncryption&& value) { m_serverSideEncryption = std::move(value); }


inline CreateMultipartUploadResult& WithServerSideEncryption(const ServerSideEncryption& value) { SetServerSideEncryption(value); return *this;}


inline CreateMultipartUploadResult& WithServerSideEncryption(ServerSideEncryption&& value) { SetServerSideEncryption(std::move(value)); return *this;}



inline const Aws::String& GetSSECustomerAlgorithm() const{ return m_sSECustomerAlgorithm; }


inline void SetSSECustomerAlgorithm(const Aws::String& value) { m_sSECustomerAlgorithm = value; }


inline void SetSSECustomerAlgorithm(Aws::String&& value) { m_sSECustomerAlgorithm = std::move(value); }


inline void SetSSECustomerAlgorithm(const char* value) { m_sSECustomerAlgorithm.assign(value); }


inline CreateMultipartUploadResult& WithSSECustomerAlgorithm(const Aws::String& value) { SetSSECustomerAlgorithm(value); return *this;}


inline CreateMultipartUploadResult& WithSSECustomerAlgorithm(Aws::String&& value) { SetSSECustomerAlgorithm(std::move(value)); return *this;}


inline CreateMultipartUploadResult& WithSSECustomerAlgorithm(const char* value) { SetSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetSSECustomerKeyMD5() const{ return m_sSECustomerKeyMD5; }


inline void SetSSECustomerKeyMD5(const Aws::String& value) { m_sSECustomerKeyMD5 = value; }


inline void SetSSECustomerKeyMD5(Aws::String&& value) { m_sSECustomerKeyMD5 = std::move(value); }


inline void SetSSECustomerKeyMD5(const char* value) { m_sSECustomerKeyMD5.assign(value); }


inline CreateMultipartUploadResult& WithSSECustomerKeyMD5(const Aws::String& value) { SetSSECustomerKeyMD5(value); return *this;}


inline CreateMultipartUploadResult& WithSSECustomerKeyMD5(Aws::String&& value) { SetSSECustomerKeyMD5(std::move(value)); return *this;}


inline CreateMultipartUploadResult& WithSSECustomerKeyMD5(const char* value) { SetSSECustomerKeyMD5(value); return *this;}



inline const Aws::String& GetSSEKMSKeyId() const{ return m_sSEKMSKeyId; }


inline void SetSSEKMSKeyId(const Aws::String& value) { m_sSEKMSKeyId = value; }


inline void SetSSEKMSKeyId(Aws::String&& value) { m_sSEKMSKeyId = std::move(value); }


inline void SetSSEKMSKeyId(const char* value) { m_sSEKMSKeyId.assign(value); }


inline CreateMultipartUploadResult& WithSSEKMSKeyId(const Aws::String& value) { SetSSEKMSKeyId(value); return *this;}


inline CreateMultipartUploadResult& WithSSEKMSKeyId(Aws::String&& value) { SetSSEKMSKeyId(std::move(value)); return *this;}


inline CreateMultipartUploadResult& WithSSEKMSKeyId(const char* value) { SetSSEKMSKeyId(value); return *this;}



inline const RequestCharged& GetRequestCharged() const{ return m_requestCharged; }


inline void SetRequestCharged(const RequestCharged& value) { m_requestCharged = value; }


inline void SetRequestCharged(RequestCharged&& value) { m_requestCharged = std::move(value); }


inline CreateMultipartUploadResult& WithRequestCharged(const RequestCharged& value) { SetRequestCharged(value); return *this;}


inline CreateMultipartUploadResult& WithRequestCharged(RequestCharged&& value) { SetRequestCharged(std::move(value)); return *this;}

private:

Aws::Utils::DateTime m_abortDate;

Aws::String m_abortRuleId;

Aws::String m_bucket;

Aws::String m_key;

Aws::String m_uploadId;

ServerSideEncryption m_serverSideEncryption;

Aws::String m_sSECustomerAlgorithm;

Aws::String m_sSECustomerKeyMD5;

Aws::String m_sSEKMSKeyId;

RequestCharged m_requestCharged;
};

} 
} 
} 
