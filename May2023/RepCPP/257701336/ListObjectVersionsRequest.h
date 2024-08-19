

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


class AWS_S3_API ListObjectVersionsRequest : public S3Request
{
public:
ListObjectVersionsRequest();

inline virtual const char* GetServiceRequestName() const override { return "ListObjectVersions"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline ListObjectVersionsRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline ListObjectVersionsRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline ListObjectVersionsRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetDelimiter() const{ return m_delimiter; }


inline void SetDelimiter(const Aws::String& value) { m_delimiterHasBeenSet = true; m_delimiter = value; }


inline void SetDelimiter(Aws::String&& value) { m_delimiterHasBeenSet = true; m_delimiter = std::move(value); }


inline void SetDelimiter(const char* value) { m_delimiterHasBeenSet = true; m_delimiter.assign(value); }


inline ListObjectVersionsRequest& WithDelimiter(const Aws::String& value) { SetDelimiter(value); return *this;}


inline ListObjectVersionsRequest& WithDelimiter(Aws::String&& value) { SetDelimiter(std::move(value)); return *this;}


inline ListObjectVersionsRequest& WithDelimiter(const char* value) { SetDelimiter(value); return *this;}



inline const EncodingType& GetEncodingType() const{ return m_encodingType; }


inline void SetEncodingType(const EncodingType& value) { m_encodingTypeHasBeenSet = true; m_encodingType = value; }


inline void SetEncodingType(EncodingType&& value) { m_encodingTypeHasBeenSet = true; m_encodingType = std::move(value); }


inline ListObjectVersionsRequest& WithEncodingType(const EncodingType& value) { SetEncodingType(value); return *this;}


inline ListObjectVersionsRequest& WithEncodingType(EncodingType&& value) { SetEncodingType(std::move(value)); return *this;}



inline const Aws::String& GetKeyMarker() const{ return m_keyMarker; }


inline void SetKeyMarker(const Aws::String& value) { m_keyMarkerHasBeenSet = true; m_keyMarker = value; }


inline void SetKeyMarker(Aws::String&& value) { m_keyMarkerHasBeenSet = true; m_keyMarker = std::move(value); }


inline void SetKeyMarker(const char* value) { m_keyMarkerHasBeenSet = true; m_keyMarker.assign(value); }


inline ListObjectVersionsRequest& WithKeyMarker(const Aws::String& value) { SetKeyMarker(value); return *this;}


inline ListObjectVersionsRequest& WithKeyMarker(Aws::String&& value) { SetKeyMarker(std::move(value)); return *this;}


inline ListObjectVersionsRequest& WithKeyMarker(const char* value) { SetKeyMarker(value); return *this;}



inline int GetMaxKeys() const{ return m_maxKeys; }


inline void SetMaxKeys(int value) { m_maxKeysHasBeenSet = true; m_maxKeys = value; }


inline ListObjectVersionsRequest& WithMaxKeys(int value) { SetMaxKeys(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline ListObjectVersionsRequest& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ListObjectVersionsRequest& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ListObjectVersionsRequest& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Aws::String& GetVersionIdMarker() const{ return m_versionIdMarker; }


inline void SetVersionIdMarker(const Aws::String& value) { m_versionIdMarkerHasBeenSet = true; m_versionIdMarker = value; }


inline void SetVersionIdMarker(Aws::String&& value) { m_versionIdMarkerHasBeenSet = true; m_versionIdMarker = std::move(value); }


inline void SetVersionIdMarker(const char* value) { m_versionIdMarkerHasBeenSet = true; m_versionIdMarker.assign(value); }


inline ListObjectVersionsRequest& WithVersionIdMarker(const Aws::String& value) { SetVersionIdMarker(value); return *this;}


inline ListObjectVersionsRequest& WithVersionIdMarker(Aws::String&& value) { SetVersionIdMarker(std::move(value)); return *this;}


inline ListObjectVersionsRequest& WithVersionIdMarker(const char* value) { SetVersionIdMarker(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_delimiter;
bool m_delimiterHasBeenSet;

EncodingType m_encodingType;
bool m_encodingTypeHasBeenSet;

Aws::String m_keyMarker;
bool m_keyMarkerHasBeenSet;

int m_maxKeys;
bool m_maxKeysHasBeenSet;

Aws::String m_prefix;
bool m_prefixHasBeenSet;

Aws::String m_versionIdMarker;
bool m_versionIdMarkerHasBeenSet;
};

} 
} 
} 
