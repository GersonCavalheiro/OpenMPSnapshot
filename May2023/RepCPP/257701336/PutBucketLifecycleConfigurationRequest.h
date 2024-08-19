

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/BucketLifecycleConfiguration.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketLifecycleConfigurationRequest : public S3Request
{
public:
PutBucketLifecycleConfigurationRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketLifecycleConfiguration"; }

Aws::String SerializePayload() const override;

inline bool ShouldComputeContentMd5() const override { return true; }



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketLifecycleConfigurationRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketLifecycleConfigurationRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketLifecycleConfigurationRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const BucketLifecycleConfiguration& GetLifecycleConfiguration() const{ return m_lifecycleConfiguration; }


inline void SetLifecycleConfiguration(const BucketLifecycleConfiguration& value) { m_lifecycleConfigurationHasBeenSet = true; m_lifecycleConfiguration = value; }


inline void SetLifecycleConfiguration(BucketLifecycleConfiguration&& value) { m_lifecycleConfigurationHasBeenSet = true; m_lifecycleConfiguration = std::move(value); }


inline PutBucketLifecycleConfigurationRequest& WithLifecycleConfiguration(const BucketLifecycleConfiguration& value) { SetLifecycleConfiguration(value); return *this;}


inline PutBucketLifecycleConfigurationRequest& WithLifecycleConfiguration(BucketLifecycleConfiguration&& value) { SetLifecycleConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

BucketLifecycleConfiguration m_lifecycleConfiguration;
bool m_lifecycleConfigurationHasBeenSet;
};

} 
} 
} 
