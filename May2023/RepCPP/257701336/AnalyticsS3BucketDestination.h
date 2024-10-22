

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/AnalyticsS3ExportFileFormat.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
namespace Utils
{
namespace Xml
{
class XmlNode;
} 
} 
namespace S3
{
namespace Model
{

class AWS_S3_API AnalyticsS3BucketDestination
{
public:
AnalyticsS3BucketDestination();
AnalyticsS3BucketDestination(const Aws::Utils::Xml::XmlNode& xmlNode);
AnalyticsS3BucketDestination& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const AnalyticsS3ExportFileFormat& GetFormat() const{ return m_format; }


inline void SetFormat(const AnalyticsS3ExportFileFormat& value) { m_formatHasBeenSet = true; m_format = value; }


inline void SetFormat(AnalyticsS3ExportFileFormat&& value) { m_formatHasBeenSet = true; m_format = std::move(value); }


inline AnalyticsS3BucketDestination& WithFormat(const AnalyticsS3ExportFileFormat& value) { SetFormat(value); return *this;}


inline AnalyticsS3BucketDestination& WithFormat(AnalyticsS3ExportFileFormat&& value) { SetFormat(std::move(value)); return *this;}



inline const Aws::String& GetBucketAccountId() const{ return m_bucketAccountId; }


inline void SetBucketAccountId(const Aws::String& value) { m_bucketAccountIdHasBeenSet = true; m_bucketAccountId = value; }


inline void SetBucketAccountId(Aws::String&& value) { m_bucketAccountIdHasBeenSet = true; m_bucketAccountId = std::move(value); }


inline void SetBucketAccountId(const char* value) { m_bucketAccountIdHasBeenSet = true; m_bucketAccountId.assign(value); }


inline AnalyticsS3BucketDestination& WithBucketAccountId(const Aws::String& value) { SetBucketAccountId(value); return *this;}


inline AnalyticsS3BucketDestination& WithBucketAccountId(Aws::String&& value) { SetBucketAccountId(std::move(value)); return *this;}


inline AnalyticsS3BucketDestination& WithBucketAccountId(const char* value) { SetBucketAccountId(value); return *this;}



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline AnalyticsS3BucketDestination& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline AnalyticsS3BucketDestination& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline AnalyticsS3BucketDestination& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline AnalyticsS3BucketDestination& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline AnalyticsS3BucketDestination& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline AnalyticsS3BucketDestination& WithPrefix(const char* value) { SetPrefix(value); return *this;}

private:

AnalyticsS3ExportFileFormat m_format;
bool m_formatHasBeenSet;

Aws::String m_bucketAccountId;
bool m_bucketAccountIdHasBeenSet;

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_prefix;
bool m_prefixHasBeenSet;
};

} 
} 
} 
