

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/NotificationConfiguration.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketNotificationConfigurationRequest : public S3Request
{
public:
PutBucketNotificationConfigurationRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketNotificationConfiguration"; }

Aws::String SerializePayload() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketNotificationConfigurationRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketNotificationConfigurationRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketNotificationConfigurationRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const NotificationConfiguration& GetNotificationConfiguration() const{ return m_notificationConfiguration; }


inline void SetNotificationConfiguration(const NotificationConfiguration& value) { m_notificationConfigurationHasBeenSet = true; m_notificationConfiguration = value; }


inline void SetNotificationConfiguration(NotificationConfiguration&& value) { m_notificationConfigurationHasBeenSet = true; m_notificationConfiguration = std::move(value); }


inline PutBucketNotificationConfigurationRequest& WithNotificationConfiguration(const NotificationConfiguration& value) { SetNotificationConfiguration(value); return *this;}


inline PutBucketNotificationConfigurationRequest& WithNotificationConfiguration(NotificationConfiguration&& value) { SetNotificationConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

NotificationConfiguration m_notificationConfiguration;
bool m_notificationConfigurationHasBeenSet;
};

} 
} 
} 
