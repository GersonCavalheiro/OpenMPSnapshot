

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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


class AWS_S3_API AbortMultipartUploadRequest : public S3Request
{
public:
AbortMultipartUploadRequest();

inline virtual const char* GetServiceRequestName() const override { return "AbortMultipartUpload"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline AbortMultipartUploadRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline AbortMultipartUploadRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline AbortMultipartUploadRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline AbortMultipartUploadRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline AbortMultipartUploadRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline AbortMultipartUploadRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetUploadId() const{ return m_uploadId; }


inline void SetUploadId(const Aws::String& value) { m_uploadIdHasBeenSet = true; m_uploadId = value; }


inline void SetUploadId(Aws::String&& value) { m_uploadIdHasBeenSet = true; m_uploadId = std::move(value); }


inline void SetUploadId(const char* value) { m_uploadIdHasBeenSet = true; m_uploadId.assign(value); }


inline AbortMultipartUploadRequest& WithUploadId(const Aws::String& value) { SetUploadId(value); return *this;}


inline AbortMultipartUploadRequest& WithUploadId(Aws::String&& value) { SetUploadId(std::move(value)); return *this;}


inline AbortMultipartUploadRequest& WithUploadId(const char* value) { SetUploadId(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline AbortMultipartUploadRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline AbortMultipartUploadRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_uploadId;
bool m_uploadIdHasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;
};

} 
} 
} 
