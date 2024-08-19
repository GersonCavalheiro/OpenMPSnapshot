

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/InventoryConfiguration.h>
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


class AWS_S3_API PutBucketInventoryConfigurationRequest : public S3Request
{
public:
PutBucketInventoryConfigurationRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketInventoryConfiguration"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketInventoryConfigurationRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketInventoryConfigurationRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketInventoryConfigurationRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline PutBucketInventoryConfigurationRequest& WithId(const Aws::String& value) { SetId(value); return *this;}


inline PutBucketInventoryConfigurationRequest& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline PutBucketInventoryConfigurationRequest& WithId(const char* value) { SetId(value); return *this;}



inline const InventoryConfiguration& GetInventoryConfiguration() const{ return m_inventoryConfiguration; }


inline void SetInventoryConfiguration(const InventoryConfiguration& value) { m_inventoryConfigurationHasBeenSet = true; m_inventoryConfiguration = value; }


inline void SetInventoryConfiguration(InventoryConfiguration&& value) { m_inventoryConfigurationHasBeenSet = true; m_inventoryConfiguration = std::move(value); }


inline PutBucketInventoryConfigurationRequest& WithInventoryConfiguration(const InventoryConfiguration& value) { SetInventoryConfiguration(value); return *this;}


inline PutBucketInventoryConfigurationRequest& WithInventoryConfiguration(InventoryConfiguration&& value) { SetInventoryConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_id;
bool m_idHasBeenSet;

InventoryConfiguration m_inventoryConfiguration;
bool m_inventoryConfigurationHasBeenSet;
};

} 
} 
} 
