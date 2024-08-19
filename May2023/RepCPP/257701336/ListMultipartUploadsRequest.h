

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/EncodingType.h>
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


class AWS_S3_API ListMultipartUploadsRequest : public S3Request
{
public:
ListMultipartUploadsRequest();

inline virtual const char* GetServiceRequestName() const override { return "ListMultipartUploads"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline ListMultipartUploadsRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline ListMultipartUploadsRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline ListMultipartUploadsRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetDelimiter() const{ return m_delimiter; }


inline void SetDelimiter(const Aws::String& value) { m_delimiterHasBeenSet = true; m_delimiter = value; }


inline void SetDelimiter(Aws::String&& value) { m_delimiterHasBeenSet = true; m_delimiter = std::move(value); }


inline void SetDelimiter(const char* value) { m_delimiterHasBeenSet = true; m_delimiter.assign(value); }


inline ListMultipartUploadsRequest& WithDelimiter(const Aws::String& value) { SetDelimiter(value); return *this;}


inline ListMultipartUploadsRequest& WithDelimiter(Aws::String&& value) { SetDelimiter(std::move(value)); return *this;}


inline ListMultipartUploadsRequest& WithDelimiter(const char* value) { SetDelimiter(value); return *this;}



inline const EncodingType& GetEncodingType() const{ return m_encodingType; }


inline void SetEncodingType(const EncodingType& value) { m_encodingTypeHasBeenSet = true; m_encodingType = value; }


inline void SetEncodingType(EncodingType&& value) { m_encodingTypeHasBeenSet = true; m_encodingType = std::move(value); }


inline ListMultipartUploadsRequest& WithEncodingType(const EncodingType& value) { SetEncodingType(value); return *this;}


inline ListMultipartUploadsRequest& WithEncodingType(EncodingType&& value) { SetEncodingType(std::move(value)); return *this;}



inline const Aws::String& GetKeyMarker() const{ return m_keyMarker; }


inline void SetKeyMarker(const Aws::String& value) { m_keyMarkerHasBeenSet = true; m_keyMarker = value; }


inline void SetKeyMarker(Aws::String&& value) { m_keyMarkerHasBeenSet = true; m_keyMarker = std::move(value); }


inline void SetKeyMarker(const char* value) { m_keyMarkerHasBeenSet = true; m_keyMarker.assign(value); }


inline ListMultipartUploadsRequest& WithKeyMarker(const Aws::String& value) { SetKeyMarker(value); return *this;}


inline ListMultipartUploadsRequest& WithKeyMarker(Aws::String&& value) { SetKeyMarker(std::move(value)); return *this;}


inline ListMultipartUploadsRequest& WithKeyMarker(const char* value) { SetKeyMarker(value); return *this;}



inline int GetMaxUploads() const{ return m_maxUploads; }


inline void SetMaxUploads(int value) { m_maxUploadsHasBeenSet = true; m_maxUploads = value; }


inline ListMultipartUploadsRequest& WithMaxUploads(int value) { SetMaxUploads(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline ListMultipartUploadsRequest& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ListMultipartUploadsRequest& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ListMultipartUploadsRequest& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Aws::String& GetUploadIdMarker() const{ return m_uploadIdMarker; }


inline void SetUploadIdMarker(const Aws::String& value) { m_uploadIdMarkerHasBeenSet = true; m_uploadIdMarker = value; }


inline void SetUploadIdMarker(Aws::String&& value) { m_uploadIdMarkerHasBeenSet = true; m_uploadIdMarker = std::move(value); }


inline void SetUploadIdMarker(const char* value) { m_uploadIdMarkerHasBeenSet = true; m_uploadIdMarker.assign(value); }


inline ListMultipartUploadsRequest& WithUploadIdMarker(const Aws::String& value) { SetUploadIdMarker(value); return *this;}


inline ListMultipartUploadsRequest& WithUploadIdMarker(Aws::String&& value) { SetUploadIdMarker(std::move(value)); return *this;}


inline ListMultipartUploadsRequest& WithUploadIdMarker(const char* value) { SetUploadIdMarker(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_delimiter;
bool m_delimiterHasBeenSet;

EncodingType m_encodingType;
bool m_encodingTypeHasBeenSet;

Aws::String m_keyMarker;
bool m_keyMarkerHasBeenSet;

int m_maxUploads;
bool m_maxUploadsHasBeenSet;

Aws::String m_prefix;
bool m_prefixHasBeenSet;

Aws::String m_uploadIdMarker;
bool m_uploadIdMarkerHasBeenSet;
};

} 
} 
} 
