

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/AccelerateConfiguration.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketAccelerateConfigurationRequest : public S3Request
{
public:
PutBucketAccelerateConfigurationRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketAccelerateConfiguration"; }

Aws::String SerializePayload() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketAccelerateConfigurationRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketAccelerateConfigurationRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketAccelerateConfigurationRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const AccelerateConfiguration& GetAccelerateConfiguration() const{ return m_accelerateConfiguration; }


inline void SetAccelerateConfiguration(const AccelerateConfiguration& value) { m_accelerateConfigurationHasBeenSet = true; m_accelerateConfiguration = value; }


inline void SetAccelerateConfiguration(AccelerateConfiguration&& value) { m_accelerateConfigurationHasBeenSet = true; m_accelerateConfiguration = std::move(value); }


inline PutBucketAccelerateConfigurationRequest& WithAccelerateConfiguration(const AccelerateConfiguration& value) { SetAccelerateConfiguration(value); return *this;}


inline PutBucketAccelerateConfigurationRequest& WithAccelerateConfiguration(AccelerateConfiguration&& value) { SetAccelerateConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

AccelerateConfiguration m_accelerateConfiguration;
bool m_accelerateConfigurationHasBeenSet;
};

} 
} 
} 
