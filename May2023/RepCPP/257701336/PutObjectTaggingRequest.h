

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/Tagging.h>
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


class AWS_S3_API PutObjectTaggingRequest : public S3Request
{
public:
PutObjectTaggingRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutObjectTagging"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutObjectTaggingRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutObjectTaggingRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutObjectTaggingRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline PutObjectTaggingRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline PutObjectTaggingRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline PutObjectTaggingRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline PutObjectTaggingRequest& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline PutObjectTaggingRequest& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline PutObjectTaggingRequest& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline const Aws::String& GetContentMD5() const{ return m_contentMD5; }


inline void SetContentMD5(const Aws::String& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = value; }


inline void SetContentMD5(Aws::String&& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = std::move(value); }


inline void SetContentMD5(const char* value) { m_contentMD5HasBeenSet = true; m_contentMD5.assign(value); }


inline PutObjectTaggingRequest& WithContentMD5(const Aws::String& value) { SetContentMD5(value); return *this;}


inline PutObjectTaggingRequest& WithContentMD5(Aws::String&& value) { SetContentMD5(std::move(value)); return *this;}


inline PutObjectTaggingRequest& WithContentMD5(const char* value) { SetContentMD5(value); return *this;}



inline const Tagging& GetTagging() const{ return m_tagging; }


inline void SetTagging(const Tagging& value) { m_taggingHasBeenSet = true; m_tagging = value; }


inline void SetTagging(Tagging&& value) { m_taggingHasBeenSet = true; m_tagging = std::move(value); }


inline PutObjectTaggingRequest& WithTagging(const Tagging& value) { SetTagging(value); return *this;}


inline PutObjectTaggingRequest& WithTagging(Tagging&& value) { SetTagging(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;

Aws::String m_contentMD5;
bool m_contentMD5HasBeenSet;

Tagging m_tagging;
bool m_taggingHasBeenSet;
};

} 
} 
} 
