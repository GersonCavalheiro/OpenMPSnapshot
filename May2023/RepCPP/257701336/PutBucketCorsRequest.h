

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/CORSConfiguration.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketCorsRequest : public S3Request
{
public:
PutBucketCorsRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketCors"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;

inline bool ShouldComputeContentMd5() const override { return true; }



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketCorsRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketCorsRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketCorsRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const CORSConfiguration& GetCORSConfiguration() const{ return m_cORSConfiguration; }


inline void SetCORSConfiguration(const CORSConfiguration& value) { m_cORSConfigurationHasBeenSet = true; m_cORSConfiguration = value; }


inline void SetCORSConfiguration(CORSConfiguration&& value) { m_cORSConfigurationHasBeenSet = true; m_cORSConfiguration = std::move(value); }


inline PutBucketCorsRequest& WithCORSConfiguration(const CORSConfiguration& value) { SetCORSConfiguration(value); return *this;}


inline PutBucketCorsRequest& WithCORSConfiguration(CORSConfiguration&& value) { SetCORSConfiguration(std::move(value)); return *this;}



inline const Aws::String& GetContentMD5() const{ return m_contentMD5; }


inline void SetContentMD5(const Aws::String& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = value; }


inline void SetContentMD5(Aws::String&& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = std::move(value); }


inline void SetContentMD5(const char* value) { m_contentMD5HasBeenSet = true; m_contentMD5.assign(value); }


inline PutBucketCorsRequest& WithContentMD5(const Aws::String& value) { SetContentMD5(value); return *this;}


inline PutBucketCorsRequest& WithContentMD5(Aws::String&& value) { SetContentMD5(std::move(value)); return *this;}


inline PutBucketCorsRequest& WithContentMD5(const char* value) { SetContentMD5(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

CORSConfiguration m_cORSConfiguration;
bool m_cORSConfigurationHasBeenSet;

Aws::String m_contentMD5;
bool m_contentMD5HasBeenSet;
};

} 
} 
} 
