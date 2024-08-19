

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/EncodingType.h>
#include <aws/s3/model/RequestPayer.h>
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


class AWS_S3_API ListObjectsRequest : public S3Request
{
public:
ListObjectsRequest();

inline virtual const char* GetServiceRequestName() const override { return "ListObjects"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline ListObjectsRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline ListObjectsRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline ListObjectsRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetDelimiter() const{ return m_delimiter; }


inline void SetDelimiter(const Aws::String& value) { m_delimiterHasBeenSet = true; m_delimiter = value; }


inline void SetDelimiter(Aws::String&& value) { m_delimiterHasBeenSet = true; m_delimiter = std::move(value); }


inline void SetDelimiter(const char* value) { m_delimiterHasBeenSet = true; m_delimiter.assign(value); }


inline ListObjectsRequest& WithDelimiter(const Aws::String& value) { SetDelimiter(value); return *this;}


inline ListObjectsRequest& WithDelimiter(Aws::String&& value) { SetDelimiter(std::move(value)); return *this;}


inline ListObjectsRequest& WithDelimiter(const char* value) { SetDelimiter(value); return *this;}



inline const EncodingType& GetEncodingType() const{ return m_encodingType; }


inline void SetEncodingType(const EncodingType& value) { m_encodingTypeHasBeenSet = true; m_encodingType = value; }


inline void SetEncodingType(EncodingType&& value) { m_encodingTypeHasBeenSet = true; m_encodingType = std::move(value); }


inline ListObjectsRequest& WithEncodingType(const EncodingType& value) { SetEncodingType(value); return *this;}


inline ListObjectsRequest& WithEncodingType(EncodingType&& value) { SetEncodingType(std::move(value)); return *this;}



inline const Aws::String& GetMarker() const{ return m_marker; }


inline void SetMarker(const Aws::String& value) { m_markerHasBeenSet = true; m_marker = value; }


inline void SetMarker(Aws::String&& value) { m_markerHasBeenSet = true; m_marker = std::move(value); }


inline void SetMarker(const char* value) { m_markerHasBeenSet = true; m_marker.assign(value); }


inline ListObjectsRequest& WithMarker(const Aws::String& value) { SetMarker(value); return *this;}


inline ListObjectsRequest& WithMarker(Aws::String&& value) { SetMarker(std::move(value)); return *this;}


inline ListObjectsRequest& WithMarker(const char* value) { SetMarker(value); return *this;}



inline int GetMaxKeys() const{ return m_maxKeys; }


inline void SetMaxKeys(int value) { m_maxKeysHasBeenSet = true; m_maxKeys = value; }


inline ListObjectsRequest& WithMaxKeys(int value) { SetMaxKeys(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline ListObjectsRequest& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ListObjectsRequest& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ListObjectsRequest& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline ListObjectsRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline ListObjectsRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_delimiter;
bool m_delimiterHasBeenSet;

EncodingType m_encodingType;
bool m_encodingTypeHasBeenSet;

Aws::String m_marker;
bool m_markerHasBeenSet;

int m_maxKeys;
bool m_maxKeysHasBeenSet;

Aws::String m_prefix;
bool m_prefixHasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;
};

} 
} 
} 
