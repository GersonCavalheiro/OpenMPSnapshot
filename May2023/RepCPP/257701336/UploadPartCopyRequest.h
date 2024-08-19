

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
#include <aws/s3/model/RequestPayer.h>
#include <utility>

namespace Aws
{
namespace Http
{
class URI;
} 
namespace S3
{
namespace Model
{


class AWS_S3_API UploadPartCopyRequest : public S3Request
{
public:
UploadPartCopyRequest();

inline virtual const char* GetServiceRequestName() const override { return "UploadPartCopy"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline UploadPartCopyRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline UploadPartCopyRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetCopySource() const{ return m_copySource; }


inline void SetCopySource(const Aws::String& value) { m_copySourceHasBeenSet = true; m_copySource = value; }


inline void SetCopySource(Aws::String&& value) { m_copySourceHasBeenSet = true; m_copySource = std::move(value); }


inline void SetCopySource(const char* value) { m_copySourceHasBeenSet = true; m_copySource.assign(value); }


inline UploadPartCopyRequest& WithCopySource(const Aws::String& value) { SetCopySource(value); return *this;}


inline UploadPartCopyRequest& WithCopySource(Aws::String&& value) { SetCopySource(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithCopySource(const char* value) { SetCopySource(value); return *this;}



inline const Aws::String& GetCopySourceIfMatch() const{ return m_copySourceIfMatch; }


inline void SetCopySourceIfMatch(const Aws::String& value) { m_copySourceIfMatchHasBeenSet = true; m_copySourceIfMatch = value; }


inline void SetCopySourceIfMatch(Aws::String&& value) { m_copySourceIfMatchHasBeenSet = true; m_copySourceIfMatch = std::move(value); }


inline void SetCopySourceIfMatch(const char* value) { m_copySourceIfMatchHasBeenSet = true; m_copySourceIfMatch.assign(value); }


inline UploadPartCopyRequest& WithCopySourceIfMatch(const Aws::String& value) { SetCopySourceIfMatch(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceIfMatch(Aws::String&& value) { SetCopySourceIfMatch(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithCopySourceIfMatch(const char* value) { SetCopySourceIfMatch(value); return *this;}



inline const Aws::Utils::DateTime& GetCopySourceIfModifiedSince() const{ return m_copySourceIfModifiedSince; }


inline void SetCopySourceIfModifiedSince(const Aws::Utils::DateTime& value) { m_copySourceIfModifiedSinceHasBeenSet = true; m_copySourceIfModifiedSince = value; }


inline void SetCopySourceIfModifiedSince(Aws::Utils::DateTime&& value) { m_copySourceIfModifiedSinceHasBeenSet = true; m_copySourceIfModifiedSince = std::move(value); }


inline UploadPartCopyRequest& WithCopySourceIfModifiedSince(const Aws::Utils::DateTime& value) { SetCopySourceIfModifiedSince(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceIfModifiedSince(Aws::Utils::DateTime&& value) { SetCopySourceIfModifiedSince(std::move(value)); return *this;}



inline const Aws::String& GetCopySourceIfNoneMatch() const{ return m_copySourceIfNoneMatch; }


inline void SetCopySourceIfNoneMatch(const Aws::String& value) { m_copySourceIfNoneMatchHasBeenSet = true; m_copySourceIfNoneMatch = value; }


inline void SetCopySourceIfNoneMatch(Aws::String&& value) { m_copySourceIfNoneMatchHasBeenSet = true; m_copySourceIfNoneMatch = std::move(value); }


inline void SetCopySourceIfNoneMatch(const char* value) { m_copySourceIfNoneMatchHasBeenSet = true; m_copySourceIfNoneMatch.assign(value); }


inline UploadPartCopyRequest& WithCopySourceIfNoneMatch(const Aws::String& value) { SetCopySourceIfNoneMatch(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceIfNoneMatch(Aws::String&& value) { SetCopySourceIfNoneMatch(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithCopySourceIfNoneMatch(const char* value) { SetCopySourceIfNoneMatch(value); return *this;}



inline const Aws::Utils::DateTime& GetCopySourceIfUnmodifiedSince() const{ return m_copySourceIfUnmodifiedSince; }


inline void SetCopySourceIfUnmodifiedSince(const Aws::Utils::DateTime& value) { m_copySourceIfUnmodifiedSinceHasBeenSet = true; m_copySourceIfUnmodifiedSince = value; }


inline void SetCopySourceIfUnmodifiedSince(Aws::Utils::DateTime&& value) { m_copySourceIfUnmodifiedSinceHasBeenSet = true; m_copySourceIfUnmodifiedSince = std::move(value); }


inline UploadPartCopyRequest& WithCopySourceIfUnmodifiedSince(const Aws::Utils::DateTime& value) { SetCopySourceIfUnmodifiedSince(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceIfUnmodifiedSince(Aws::Utils::DateTime&& value) { SetCopySourceIfUnmodifiedSince(std::move(value)); return *this;}



inline const Aws::String& GetCopySourceRange() const{ return m_copySourceRange; }


inline void SetCopySourceRange(const Aws::String& value) { m_copySourceRangeHasBeenSet = true; m_copySourceRange = value; }


inline void SetCopySourceRange(Aws::String&& value) { m_copySourceRangeHasBeenSet = true; m_copySourceRange = std::move(value); }


inline void SetCopySourceRange(const char* value) { m_copySourceRangeHasBeenSet = true; m_copySourceRange.assign(value); }


inline UploadPartCopyRequest& WithCopySourceRange(const Aws::String& value) { SetCopySourceRange(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceRange(Aws::String&& value) { SetCopySourceRange(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithCopySourceRange(const char* value) { SetCopySourceRange(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline UploadPartCopyRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline UploadPartCopyRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline int GetPartNumber() const{ return m_partNumber; }


inline void SetPartNumber(int value) { m_partNumberHasBeenSet = true; m_partNumber = value; }


inline UploadPartCopyRequest& WithPartNumber(int value) { SetPartNumber(value); return *this;}



inline const Aws::String& GetUploadId() const{ return m_uploadId; }


inline void SetUploadId(const Aws::String& value) { m_uploadIdHasBeenSet = true; m_uploadId = value; }


inline void SetUploadId(Aws::String&& value) { m_uploadIdHasBeenSet = true; m_uploadId = std::move(value); }


inline void SetUploadId(const char* value) { m_uploadIdHasBeenSet = true; m_uploadId.assign(value); }


inline UploadPartCopyRequest& WithUploadId(const Aws::String& value) { SetUploadId(value); return *this;}


inline UploadPartCopyRequest& WithUploadId(Aws::String&& value) { SetUploadId(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithUploadId(const char* value) { SetUploadId(value); return *this;}



inline const Aws::String& GetSSECustomerAlgorithm() const{ return m_sSECustomerAlgorithm; }


inline void SetSSECustomerAlgorithm(const Aws::String& value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm = value; }


inline void SetSSECustomerAlgorithm(Aws::String&& value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm = std::move(value); }


inline void SetSSECustomerAlgorithm(const char* value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm.assign(value); }


inline UploadPartCopyRequest& WithSSECustomerAlgorithm(const Aws::String& value) { SetSSECustomerAlgorithm(value); return *this;}


inline UploadPartCopyRequest& WithSSECustomerAlgorithm(Aws::String&& value) { SetSSECustomerAlgorithm(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithSSECustomerAlgorithm(const char* value) { SetSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetSSECustomerKey() const{ return m_sSECustomerKey; }


inline void SetSSECustomerKey(const Aws::String& value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey = value; }


inline void SetSSECustomerKey(Aws::String&& value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey = std::move(value); }


inline void SetSSECustomerKey(const char* value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey.assign(value); }


inline UploadPartCopyRequest& WithSSECustomerKey(const Aws::String& value) { SetSSECustomerKey(value); return *this;}


inline UploadPartCopyRequest& WithSSECustomerKey(Aws::String&& value) { SetSSECustomerKey(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithSSECustomerKey(const char* value) { SetSSECustomerKey(value); return *this;}



inline const Aws::String& GetSSECustomerKeyMD5() const{ return m_sSECustomerKeyMD5; }


inline void SetSSECustomerKeyMD5(const Aws::String& value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5 = value; }


inline void SetSSECustomerKeyMD5(Aws::String&& value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5 = std::move(value); }


inline void SetSSECustomerKeyMD5(const char* value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5.assign(value); }


inline UploadPartCopyRequest& WithSSECustomerKeyMD5(const Aws::String& value) { SetSSECustomerKeyMD5(value); return *this;}


inline UploadPartCopyRequest& WithSSECustomerKeyMD5(Aws::String&& value) { SetSSECustomerKeyMD5(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithSSECustomerKeyMD5(const char* value) { SetSSECustomerKeyMD5(value); return *this;}



inline const Aws::String& GetCopySourceSSECustomerAlgorithm() const{ return m_copySourceSSECustomerAlgorithm; }


inline void SetCopySourceSSECustomerAlgorithm(const Aws::String& value) { m_copySourceSSECustomerAlgorithmHasBeenSet = true; m_copySourceSSECustomerAlgorithm = value; }


inline void SetCopySourceSSECustomerAlgorithm(Aws::String&& value) { m_copySourceSSECustomerAlgorithmHasBeenSet = true; m_copySourceSSECustomerAlgorithm = std::move(value); }


inline void SetCopySourceSSECustomerAlgorithm(const char* value) { m_copySourceSSECustomerAlgorithmHasBeenSet = true; m_copySourceSSECustomerAlgorithm.assign(value); }


inline UploadPartCopyRequest& WithCopySourceSSECustomerAlgorithm(const Aws::String& value) { SetCopySourceSSECustomerAlgorithm(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceSSECustomerAlgorithm(Aws::String&& value) { SetCopySourceSSECustomerAlgorithm(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithCopySourceSSECustomerAlgorithm(const char* value) { SetCopySourceSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetCopySourceSSECustomerKey() const{ return m_copySourceSSECustomerKey; }


inline void SetCopySourceSSECustomerKey(const Aws::String& value) { m_copySourceSSECustomerKeyHasBeenSet = true; m_copySourceSSECustomerKey = value; }


inline void SetCopySourceSSECustomerKey(Aws::String&& value) { m_copySourceSSECustomerKeyHasBeenSet = true; m_copySourceSSECustomerKey = std::move(value); }


inline void SetCopySourceSSECustomerKey(const char* value) { m_copySourceSSECustomerKeyHasBeenSet = true; m_copySourceSSECustomerKey.assign(value); }


inline UploadPartCopyRequest& WithCopySourceSSECustomerKey(const Aws::String& value) { SetCopySourceSSECustomerKey(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceSSECustomerKey(Aws::String&& value) { SetCopySourceSSECustomerKey(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithCopySourceSSECustomerKey(const char* value) { SetCopySourceSSECustomerKey(value); return *this;}



inline const Aws::String& GetCopySourceSSECustomerKeyMD5() const{ return m_copySourceSSECustomerKeyMD5; }


inline void SetCopySourceSSECustomerKeyMD5(const Aws::String& value) { m_copySourceSSECustomerKeyMD5HasBeenSet = true; m_copySourceSSECustomerKeyMD5 = value; }


inline void SetCopySourceSSECustomerKeyMD5(Aws::String&& value) { m_copySourceSSECustomerKeyMD5HasBeenSet = true; m_copySourceSSECustomerKeyMD5 = std::move(value); }


inline void SetCopySourceSSECustomerKeyMD5(const char* value) { m_copySourceSSECustomerKeyMD5HasBeenSet = true; m_copySourceSSECustomerKeyMD5.assign(value); }


inline UploadPartCopyRequest& WithCopySourceSSECustomerKeyMD5(const Aws::String& value) { SetCopySourceSSECustomerKeyMD5(value); return *this;}


inline UploadPartCopyRequest& WithCopySourceSSECustomerKeyMD5(Aws::String&& value) { SetCopySourceSSECustomerKeyMD5(std::move(value)); return *this;}


inline UploadPartCopyRequest& WithCopySourceSSECustomerKeyMD5(const char* value) { SetCopySourceSSECustomerKeyMD5(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline UploadPartCopyRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline UploadPartCopyRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

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

Aws::String m_copySourceRange;
bool m_copySourceRangeHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

int m_partNumber;
bool m_partNumberHasBeenSet;

Aws::String m_uploadId;
bool m_uploadIdHasBeenSet;

Aws::String m_sSECustomerAlgorithm;
bool m_sSECustomerAlgorithmHasBeenSet;

Aws::String m_sSECustomerKey;
bool m_sSECustomerKeyHasBeenSet;

Aws::String m_sSECustomerKeyMD5;
bool m_sSECustomerKeyMD5HasBeenSet;

Aws::String m_copySourceSSECustomerAlgorithm;
bool m_copySourceSSECustomerAlgorithmHasBeenSet;

Aws::String m_copySourceSSECustomerKey;
bool m_copySourceSSECustomerKeyHasBeenSet;

Aws::String m_copySourceSSECustomerKeyMD5;
bool m_copySourceSSECustomerKeyMD5HasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;
};

} 
} 
} 
