

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/VersioningConfiguration.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API PutBucketVersioningRequest : public S3Request
{
public:
PutBucketVersioningRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutBucketVersioning"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutBucketVersioningRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutBucketVersioningRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutBucketVersioningRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetContentMD5() const{ return m_contentMD5; }


inline void SetContentMD5(const Aws::String& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = value; }


inline void SetContentMD5(Aws::String&& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = std::move(value); }


inline void SetContentMD5(const char* value) { m_contentMD5HasBeenSet = true; m_contentMD5.assign(value); }


inline PutBucketVersioningRequest& WithContentMD5(const Aws::String& value) { SetContentMD5(value); return *this;}


inline PutBucketVersioningRequest& WithContentMD5(Aws::String&& value) { SetContentMD5(std::move(value)); return *this;}


inline PutBucketVersioningRequest& WithContentMD5(const char* value) { SetContentMD5(value); return *this;}



inline const Aws::String& GetMFA() const{ return m_mFA; }


inline void SetMFA(const Aws::String& value) { m_mFAHasBeenSet = true; m_mFA = value; }


inline void SetMFA(Aws::String&& value) { m_mFAHasBeenSet = true; m_mFA = std::move(value); }


inline void SetMFA(const char* value) { m_mFAHasBeenSet = true; m_mFA.assign(value); }


inline PutBucketVersioningRequest& WithMFA(const Aws::String& value) { SetMFA(value); return *this;}


inline PutBucketVersioningRequest& WithMFA(Aws::String&& value) { SetMFA(std::move(value)); return *this;}


inline PutBucketVersioningRequest& WithMFA(const char* value) { SetMFA(value); return *this;}



inline const VersioningConfiguration& GetVersioningConfiguration() const{ return m_versioningConfiguration; }


inline void SetVersioningConfiguration(const VersioningConfiguration& value) { m_versioningConfigurationHasBeenSet = true; m_versioningConfiguration = value; }


inline void SetVersioningConfiguration(VersioningConfiguration&& value) { m_versioningConfigurationHasBeenSet = true; m_versioningConfiguration = std::move(value); }


inline PutBucketVersioningRequest& WithVersioningConfiguration(const VersioningConfiguration& value) { SetVersioningConfiguration(value); return *this;}


inline PutBucketVersioningRequest& WithVersioningConfiguration(VersioningConfiguration&& value) { SetVersioningConfiguration(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_contentMD5;
bool m_contentMD5HasBeenSet;

Aws::String m_mFA;
bool m_mFAHasBeenSet;

VersioningConfiguration m_versioningConfiguration;
bool m_versioningConfigurationHasBeenSet;
};

} 
} 
} 
