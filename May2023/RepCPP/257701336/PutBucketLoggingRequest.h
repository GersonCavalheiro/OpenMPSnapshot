

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/BucketLoggingStatus.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketLoggingRequest : public S3Request
{
public:
PutBucketLoggingRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketLogging"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketLoggingRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketLoggingRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketLoggingRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const BucketLoggingStatus& GetBucketLoggingStatus() const{ return m_bucketLoggingStatus; }


inline void SetBucketLoggingStatus(const BucketLoggingStatus& value) { m_bucketLoggingStatusHasBeenSet = true; m_bucketLoggingStatus = value; }


inline void SetBucketLoggingStatus(BucketLoggingStatus&& value) { m_bucketLoggingStatusHasBeenSet = true; m_bucketLoggingStatus = std::move(value); }


inline PutBucketLoggingRequest& WithBucketLoggingStatus(const BucketLoggingStatus& value) { SetBucketLoggingStatus(value); return *this;}


inline PutBucketLoggingRequest& WithBucketLoggingStatus(BucketLoggingStatus&& value) { SetBucketLoggingStatus(std::move(value)); return *this;}



inline const Aws::String& GetContentMD5() const{ return m_contentMD5; }


inline void SetContentMD5(const Aws::String& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = value; }


inline void SetContentMD5(Aws::String&& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = std::move(value); }


inline void SetContentMD5(const char* value) { m_contentMD5HasBeenSet = true; m_contentMD5.assign(value); }


inline PutBucketLoggingRequest& WithContentMD5(const Aws::String& value) { SetContentMD5(value); return *this;}


inline PutBucketLoggingRequest& WithContentMD5(Aws::String&& value) { SetContentMD5(std::move(value)); return *this;}


inline PutBucketLoggingRequest& WithContentMD5(const char* value) { SetContentMD5(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

BucketLoggingStatus m_bucketLoggingStatus;
bool m_bucketLoggingStatusHasBeenSet;

Aws::String m_contentMD5;
bool m_contentMD5HasBeenSet;
};

} 
} 
} 
