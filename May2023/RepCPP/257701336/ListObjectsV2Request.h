

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


class AWS_S3_API ListObjectsV2Request : public S3Request
{
public:
ListObjectsV2Request();

inline virtual const char* GetServiceRequestName() const override { return "ListObjectsV2"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline ListObjectsV2Request& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline ListObjectsV2Request& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline ListObjectsV2Request& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetDelimiter() const{ return m_delimiter; }


inline void SetDelimiter(const Aws::String& value) { m_delimiterHasBeenSet = true; m_delimiter = value; }


inline void SetDelimiter(Aws::String&& value) { m_delimiterHasBeenSet = true; m_delimiter = std::move(value); }


inline void SetDelimiter(const char* value) { m_delimiterHasBeenSet = true; m_delimiter.assign(value); }


inline ListObjectsV2Request& WithDelimiter(const Aws::String& value) { SetDelimiter(value); return *this;}


inline ListObjectsV2Request& WithDelimiter(Aws::String&& value) { SetDelimiter(std::move(value)); return *this;}


inline ListObjectsV2Request& WithDelimiter(const char* value) { SetDelimiter(value); return *this;}



inline const EncodingType& GetEncodingType() const{ return m_encodingType; }


inline void SetEncodingType(const EncodingType& value) { m_encodingTypeHasBeenSet = true; m_encodingType = value; }


inline void SetEncodingType(EncodingType&& value) { m_encodingTypeHasBeenSet = true; m_encodingType = std::move(value); }


inline ListObjectsV2Request& WithEncodingType(const EncodingType& value) { SetEncodingType(value); return *this;}


inline ListObjectsV2Request& WithEncodingType(EncodingType&& value) { SetEncodingType(std::move(value)); return *this;}



inline int GetMaxKeys() const{ return m_maxKeys; }


inline void SetMaxKeys(int value) { m_maxKeysHasBeenSet = true; m_maxKeys = value; }


inline ListObjectsV2Request& WithMaxKeys(int value) { SetMaxKeys(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefixHasBeenSet = true; m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefixHasBeenSet = true; m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefixHasBeenSet = true; m_prefix.assign(value); }


inline ListObjectsV2Request& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ListObjectsV2Request& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ListObjectsV2Request& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Aws::String& GetContinuationToken() const{ return m_continuationToken; }


inline void SetContinuationToken(const Aws::String& value) { m_continuationTokenHasBeenSet = true; m_continuationToken = value; }


inline void SetContinuationToken(Aws::String&& value) { m_continuationTokenHasBeenSet = true; m_continuationToken = std::move(value); }


inline void SetContinuationToken(const char* value) { m_continuationTokenHasBeenSet = true; m_continuationToken.assign(value); }


inline ListObjectsV2Request& WithContinuationToken(const Aws::String& value) { SetContinuationToken(value); return *this;}


inline ListObjectsV2Request& WithContinuationToken(Aws::String&& value) { SetContinuationToken(std::move(value)); return *this;}


inline ListObjectsV2Request& WithContinuationToken(const char* value) { SetContinuationToken(value); return *this;}



inline bool GetFetchOwner() const{ return m_fetchOwner; }


inline void SetFetchOwner(bool value) { m_fetchOwnerHasBeenSet = true; m_fetchOwner = value; }


inline ListObjectsV2Request& WithFetchOwner(bool value) { SetFetchOwner(value); return *this;}



inline const Aws::String& GetStartAfter() const{ return m_startAfter; }


inline void SetStartAfter(const Aws::String& value) { m_startAfterHasBeenSet = true; m_startAfter = value; }


inline void SetStartAfter(Aws::String&& value) { m_startAfterHasBeenSet = true; m_startAfter = std::move(value); }


inline void SetStartAfter(const char* value) { m_startAfterHasBeenSet = true; m_startAfter.assign(value); }


inline ListObjectsV2Request& WithStartAfter(const Aws::String& value) { SetStartAfter(value); return *this;}


inline ListObjectsV2Request& WithStartAfter(Aws::String&& value) { SetStartAfter(std::move(value)); return *this;}


inline ListObjectsV2Request& WithStartAfter(const char* value) { SetStartAfter(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline ListObjectsV2Request& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline ListObjectsV2Request& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_delimiter;
bool m_delimiterHasBeenSet;

EncodingType m_encodingType;
bool m_encodingTypeHasBeenSet;

int m_maxKeys;
bool m_maxKeysHasBeenSet;

Aws::String m_prefix;
bool m_prefixHasBeenSet;

Aws::String m_continuationToken;
bool m_continuationTokenHasBeenSet;

bool m_fetchOwner;
bool m_fetchOwnerHasBeenSet;

Aws::String m_startAfter;
bool m_startAfterHasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;
};

} 
} 
} 
