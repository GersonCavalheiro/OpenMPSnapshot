

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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


class AWS_S3_API DeleteBucketMetricsConfigurationRequest : public S3Request
{
public:
DeleteBucketMetricsConfigurationRequest();

inline virtual const char* GetServiceRequestName() const override { return "DeleteBucketMetricsConfiguration"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline DeleteBucketMetricsConfigurationRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline DeleteBucketMetricsConfigurationRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline DeleteBucketMetricsConfigurationRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline DeleteBucketMetricsConfigurationRequest& WithId(const Aws::String& value) { SetId(value); return *this;}


inline DeleteBucketMetricsConfigurationRequest& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline DeleteBucketMetricsConfigurationRequest& WithId(const char* value) { SetId(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_id;
bool m_idHasBeenSet;
};

} 
} 
} 
