

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


class AWS_S3_API DeleteObjectTaggingRequest : public S3Request
{
public:
DeleteObjectTaggingRequest();

inline virtual const char* GetServiceRequestName() const override { return "DeleteObjectTagging"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline DeleteObjectTaggingRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline DeleteObjectTaggingRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline DeleteObjectTaggingRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline DeleteObjectTaggingRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline DeleteObjectTaggingRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline DeleteObjectTaggingRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline DeleteObjectTaggingRequest& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline DeleteObjectTaggingRequest& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline DeleteObjectTaggingRequest& WithVersionId(const char* value) { SetVersionId(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;
};

} 
} 
} 
