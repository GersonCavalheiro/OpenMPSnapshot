

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/MetricsConfiguration.h>
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


class AWS_S3_API PutBucketMetricsConfigurationRequest : public S3Request
{
public:
PutBucketMetricsConfigurationRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketMetricsConfiguration"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketMetricsConfigurationRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketMetricsConfigurationRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketMetricsConfigurationRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline PutBucketMetricsConfigurationRequest& WithId(const Aws::String& value) { SetId(value); return *this;}


inline PutBucketMetricsConfigurationRequest& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline PutBucketMetricsConfigurationRequest& WithId(const char* value) { SetId(value); return *this;}



inline const MetricsConfiguration& GetMetricsConfiguration() const{ return m_metricsConfiguration; }


inline void SetMetricsConfiguration(const MetricsConfiguration& value) { m_metricsConfigurationHasBeenSet = true; m_metricsConfiguration = value; }


inline void SetMetricsConfiguration(MetricsConfiguration&& value) { m_metricsConfigurationHasBeenSet = true; m_metricsConfiguration = std::move(value); }


inline PutBucketMetricsConfigurationRequest& WithMetricsConfiguration(const MetricsConfiguration& value) { SetMetricsConfiguration(value); return *this;}


inline PutBucketMetricsConfigurationRequest& WithMetricsConfiguration(MetricsConfiguration&& value) { SetMetricsConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_id;
bool m_idHasBeenSet;

MetricsConfiguration m_metricsConfiguration;
bool m_metricsConfigurationHasBeenSet;
};

} 
} 
} 
