

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/AnalyticsConfiguration.h>
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


class AWS_S3_API PutBucketAnalyticsConfigurationRequest : public S3Request
{
public:
PutBucketAnalyticsConfigurationRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketAnalyticsConfiguration"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketAnalyticsConfigurationRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketAnalyticsConfigurationRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketAnalyticsConfigurationRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline PutBucketAnalyticsConfigurationRequest& WithId(const Aws::String& value) { SetId(value); return *this;}


inline PutBucketAnalyticsConfigurationRequest& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline PutBucketAnalyticsConfigurationRequest& WithId(const char* value) { SetId(value); return *this;}



inline const AnalyticsConfiguration& GetAnalyticsConfiguration() const{ return m_analyticsConfiguration; }


inline void SetAnalyticsConfiguration(const AnalyticsConfiguration& value) { m_analyticsConfigurationHasBeenSet = true; m_analyticsConfiguration = value; }


inline void SetAnalyticsConfiguration(AnalyticsConfiguration&& value) { m_analyticsConfigurationHasBeenSet = true; m_analyticsConfiguration = std::move(value); }


inline PutBucketAnalyticsConfigurationRequest& WithAnalyticsConfiguration(const AnalyticsConfiguration& value) { SetAnalyticsConfiguration(value); return *this;}


inline PutBucketAnalyticsConfigurationRequest& WithAnalyticsConfiguration(AnalyticsConfiguration&& value) { SetAnalyticsConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_id;
bool m_idHasBeenSet;

AnalyticsConfiguration m_analyticsConfiguration;
bool m_analyticsConfigurationHasBeenSet;
};

} 
} 
} 
