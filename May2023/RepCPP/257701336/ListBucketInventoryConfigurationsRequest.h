

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


class AWS_S3_API ListBucketInventoryConfigurationsRequest : public S3Request
{
public:
ListBucketInventoryConfigurationsRequest();

inline virtual const char* GetServiceRequestName() const override { return "ListBucketInventoryConfigurations"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline ListBucketInventoryConfigurationsRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline ListBucketInventoryConfigurationsRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline ListBucketInventoryConfigurationsRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetContinuationToken() const{ return m_continuationToken; }


inline void SetContinuationToken(const Aws::String& value) { m_continuationTokenHasBeenSet = true; m_continuationToken = value; }


inline void SetContinuationToken(Aws::String&& value) { m_continuationTokenHasBeenSet = true; m_continuationToken = std::move(value); }


inline void SetContinuationToken(const char* value) { m_continuationTokenHasBeenSet = true; m_continuationToken.assign(value); }


inline ListBucketInventoryConfigurationsRequest& WithContinuationToken(const Aws::String& value) { SetContinuationToken(value); return *this;}


inline ListBucketInventoryConfigurationsRequest& WithContinuationToken(Aws::String&& value) { SetContinuationToken(std::move(value)); return *this;}


inline ListBucketInventoryConfigurationsRequest& WithContinuationToken(const char* value) { SetContinuationToken(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_continuationToken;
bool m_continuationTokenHasBeenSet;
};

} 
} 
} 
