

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketPolicyRequest : public StreamingS3Request
{
public:
PutBucketPolicyRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketPolicy"; }

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;

inline bool ShouldComputeContentMd5() const override { return true; }



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketPolicyRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketPolicyRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketPolicyRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetContentMD5() const{ return m_contentMD5; }


inline void SetContentMD5(const Aws::String& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = value; }


inline void SetContentMD5(Aws::String&& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = std::move(value); }


inline void SetContentMD5(const char* value) { m_contentMD5HasBeenSet = true; m_contentMD5.assign(value); }


inline PutBucketPolicyRequest& WithContentMD5(const Aws::String& value) { SetContentMD5(value); return *this;}


inline PutBucketPolicyRequest& WithContentMD5(Aws::String&& value) { SetContentMD5(std::move(value)); return *this;}


inline PutBucketPolicyRequest& WithContentMD5(const char* value) { SetContentMD5(value); return *this;}



inline bool GetConfirmRemoveSelfBucketAccess() const{ return m_confirmRemoveSelfBucketAccess; }


inline void SetConfirmRemoveSelfBucketAccess(bool value) { m_confirmRemoveSelfBucketAccessHasBeenSet = true; m_confirmRemoveSelfBucketAccess = value; }


inline PutBucketPolicyRequest& WithConfirmRemoveSelfBucketAccess(bool value) { SetConfirmRemoveSelfBucketAccess(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_contentMD5;
bool m_contentMD5HasBeenSet;

bool m_confirmRemoveSelfBucketAccess;
bool m_confirmRemoveSelfBucketAccessHasBeenSet;

};

} 
} 
} 
