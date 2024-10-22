

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/Tagging.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketTaggingRequest : public S3Request
{
public:
PutBucketTaggingRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketTagging"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;

inline bool ShouldComputeContentMd5() const override { return true; }



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketTaggingRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketTaggingRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketTaggingRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetContentMD5() const{ return m_contentMD5; }


inline void SetContentMD5(const Aws::String& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = value; }


inline void SetContentMD5(Aws::String&& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = std::move(value); }


inline void SetContentMD5(const char* value) { m_contentMD5HasBeenSet = true; m_contentMD5.assign(value); }


inline PutBucketTaggingRequest& WithContentMD5(const Aws::String& value) { SetContentMD5(value); return *this;}


inline PutBucketTaggingRequest& WithContentMD5(Aws::String&& value) { SetContentMD5(std::move(value)); return *this;}


inline PutBucketTaggingRequest& WithContentMD5(const char* value) { SetContentMD5(value); return *this;}



inline const Tagging& GetTagging() const{ return m_tagging; }


inline void SetTagging(const Tagging& value) { m_taggingHasBeenSet = true; m_tagging = value; }


inline void SetTagging(Tagging&& value) { m_taggingHasBeenSet = true; m_tagging = std::move(value); }


inline PutBucketTaggingRequest& WithTagging(const Tagging& value) { SetTagging(value); return *this;}


inline PutBucketTaggingRequest& WithTagging(Tagging&& value) { SetTagging(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_contentMD5;
bool m_contentMD5HasBeenSet;

Tagging m_tagging;
bool m_taggingHasBeenSet;
};

} 
} 
} 
