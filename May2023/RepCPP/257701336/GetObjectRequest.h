

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
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


class AWS_S3_API GetObjectRequest : public S3Request
{
public:
GetObjectRequest();

inline virtual const char* GetServiceRequestName() const override { return "GetObject"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline GetObjectRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline GetObjectRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline GetObjectRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetIfMatch() const{ return m_ifMatch; }


inline void SetIfMatch(const Aws::String& value) { m_ifMatchHasBeenSet = true; m_ifMatch = value; }


inline void SetIfMatch(Aws::String&& value) { m_ifMatchHasBeenSet = true; m_ifMatch = std::move(value); }


inline void SetIfMatch(const char* value) { m_ifMatchHasBeenSet = true; m_ifMatch.assign(value); }


inline GetObjectRequest& WithIfMatch(const Aws::String& value) { SetIfMatch(value); return *this;}


inline GetObjectRequest& WithIfMatch(Aws::String&& value) { SetIfMatch(std::move(value)); return *this;}


inline GetObjectRequest& WithIfMatch(const char* value) { SetIfMatch(value); return *this;}



inline const Aws::Utils::DateTime& GetIfModifiedSince() const{ return m_ifModifiedSince; }


inline void SetIfModifiedSince(const Aws::Utils::DateTime& value) { m_ifModifiedSinceHasBeenSet = true; m_ifModifiedSince = value; }


inline void SetIfModifiedSince(Aws::Utils::DateTime&& value) { m_ifModifiedSinceHasBeenSet = true; m_ifModifiedSince = std::move(value); }


inline GetObjectRequest& WithIfModifiedSince(const Aws::Utils::DateTime& value) { SetIfModifiedSince(value); return *this;}


inline GetObjectRequest& WithIfModifiedSince(Aws::Utils::DateTime&& value) { SetIfModifiedSince(std::move(value)); return *this;}



inline const Aws::String& GetIfNoneMatch() const{ return m_ifNoneMatch; }


inline void SetIfNoneMatch(const Aws::String& value) { m_ifNoneMatchHasBeenSet = true; m_ifNoneMatch = value; }


inline void SetIfNoneMatch(Aws::String&& value) { m_ifNoneMatchHasBeenSet = true; m_ifNoneMatch = std::move(value); }


inline void SetIfNoneMatch(const char* value) { m_ifNoneMatchHasBeenSet = true; m_ifNoneMatch.assign(value); }


inline GetObjectRequest& WithIfNoneMatch(const Aws::String& value) { SetIfNoneMatch(value); return *this;}


inline GetObjectRequest& WithIfNoneMatch(Aws::String&& value) { SetIfNoneMatch(std::move(value)); return *this;}


inline GetObjectRequest& WithIfNoneMatch(const char* value) { SetIfNoneMatch(value); return *this;}



inline const Aws::Utils::DateTime& GetIfUnmodifiedSince() const{ return m_ifUnmodifiedSince; }


inline void SetIfUnmodifiedSince(const Aws::Utils::DateTime& value) { m_ifUnmodifiedSinceHasBeenSet = true; m_ifUnmodifiedSince = value; }


inline void SetIfUnmodifiedSince(Aws::Utils::DateTime&& value) { m_ifUnmodifiedSinceHasBeenSet = true; m_ifUnmodifiedSince = std::move(value); }


inline GetObjectRequest& WithIfUnmodifiedSince(const Aws::Utils::DateTime& value) { SetIfUnmodifiedSince(value); return *this;}


inline GetObjectRequest& WithIfUnmodifiedSince(Aws::Utils::DateTime&& value) { SetIfUnmodifiedSince(std::move(value)); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline GetObjectRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline GetObjectRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline GetObjectRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetRange() const{ return m_range; }


inline void SetRange(const Aws::String& value) { m_rangeHasBeenSet = true; m_range = value; }


inline void SetRange(Aws::String&& value) { m_rangeHasBeenSet = true; m_range = std::move(value); }


inline void SetRange(const char* value) { m_rangeHasBeenSet = true; m_range.assign(value); }


inline GetObjectRequest& WithRange(const Aws::String& value) { SetRange(value); return *this;}


inline GetObjectRequest& WithRange(Aws::String&& value) { SetRange(std::move(value)); return *this;}


inline GetObjectRequest& WithRange(const char* value) { SetRange(value); return *this;}



inline const Aws::String& GetResponseCacheControl() const{ return m_responseCacheControl; }


inline void SetResponseCacheControl(const Aws::String& value) { m_responseCacheControlHasBeenSet = true; m_responseCacheControl = value; }


inline void SetResponseCacheControl(Aws::String&& value) { m_responseCacheControlHasBeenSet = true; m_responseCacheControl = std::move(value); }


inline void SetResponseCacheControl(const char* value) { m_responseCacheControlHasBeenSet = true; m_responseCacheControl.assign(value); }


inline GetObjectRequest& WithResponseCacheControl(const Aws::String& value) { SetResponseCacheControl(value); return *this;}


inline GetObjectRequest& WithResponseCacheControl(Aws::String&& value) { SetResponseCacheControl(std::move(value)); return *this;}


inline GetObjectRequest& WithResponseCacheControl(const char* value) { SetResponseCacheControl(value); return *this;}



inline const Aws::String& GetResponseContentDisposition() const{ return m_responseContentDisposition; }


inline void SetResponseContentDisposition(const Aws::String& value) { m_responseContentDispositionHasBeenSet = true; m_responseContentDisposition = value; }


inline void SetResponseContentDisposition(Aws::String&& value) { m_responseContentDispositionHasBeenSet = true; m_responseContentDisposition = std::move(value); }


inline void SetResponseContentDisposition(const char* value) { m_responseContentDispositionHasBeenSet = true; m_responseContentDisposition.assign(value); }


inline GetObjectRequest& WithResponseContentDisposition(const Aws::String& value) { SetResponseContentDisposition(value); return *this;}


inline GetObjectRequest& WithResponseContentDisposition(Aws::String&& value) { SetResponseContentDisposition(std::move(value)); return *this;}


inline GetObjectRequest& WithResponseContentDisposition(const char* value) { SetResponseContentDisposition(value); return *this;}



inline const Aws::String& GetResponseContentEncoding() const{ return m_responseContentEncoding; }


inline void SetResponseContentEncoding(const Aws::String& value) { m_responseContentEncodingHasBeenSet = true; m_responseContentEncoding = value; }


inline void SetResponseContentEncoding(Aws::String&& value) { m_responseContentEncodingHasBeenSet = true; m_responseContentEncoding = std::move(value); }


inline void SetResponseContentEncoding(const char* value) { m_responseContentEncodingHasBeenSet = true; m_responseContentEncoding.assign(value); }


inline GetObjectRequest& WithResponseContentEncoding(const Aws::String& value) { SetResponseContentEncoding(value); return *this;}


inline GetObjectRequest& WithResponseContentEncoding(Aws::String&& value) { SetResponseContentEncoding(std::move(value)); return *this;}


inline GetObjectRequest& WithResponseContentEncoding(const char* value) { SetResponseContentEncoding(value); return *this;}



inline const Aws::String& GetResponseContentLanguage() const{ return m_responseContentLanguage; }


inline void SetResponseContentLanguage(const Aws::String& value) { m_responseContentLanguageHasBeenSet = true; m_responseContentLanguage = value; }


inline void SetResponseContentLanguage(Aws::String&& value) { m_responseContentLanguageHasBeenSet = true; m_responseContentLanguage = std::move(value); }


inline void SetResponseContentLanguage(const char* value) { m_responseContentLanguageHasBeenSet = true; m_responseContentLanguage.assign(value); }


inline GetObjectRequest& WithResponseContentLanguage(const Aws::String& value) { SetResponseContentLanguage(value); return *this;}


inline GetObjectRequest& WithResponseContentLanguage(Aws::String&& value) { SetResponseContentLanguage(std::move(value)); return *this;}


inline GetObjectRequest& WithResponseContentLanguage(const char* value) { SetResponseContentLanguage(value); return *this;}



inline const Aws::String& GetResponseContentType() const{ return m_responseContentType; }


inline void SetResponseContentType(const Aws::String& value) { m_responseContentTypeHasBeenSet = true; m_responseContentType = value; }


inline void SetResponseContentType(Aws::String&& value) { m_responseContentTypeHasBeenSet = true; m_responseContentType = std::move(value); }


inline void SetResponseContentType(const char* value) { m_responseContentTypeHasBeenSet = true; m_responseContentType.assign(value); }


inline GetObjectRequest& WithResponseContentType(const Aws::String& value) { SetResponseContentType(value); return *this;}


inline GetObjectRequest& WithResponseContentType(Aws::String&& value) { SetResponseContentType(std::move(value)); return *this;}


inline GetObjectRequest& WithResponseContentType(const char* value) { SetResponseContentType(value); return *this;}



inline const Aws::Utils::DateTime& GetResponseExpires() const{ return m_responseExpires; }


inline void SetResponseExpires(const Aws::Utils::DateTime& value) { m_responseExpiresHasBeenSet = true; m_responseExpires = value; }


inline void SetResponseExpires(Aws::Utils::DateTime&& value) { m_responseExpiresHasBeenSet = true; m_responseExpires = std::move(value); }


inline GetObjectRequest& WithResponseExpires(const Aws::Utils::DateTime& value) { SetResponseExpires(value); return *this;}


inline GetObjectRequest& WithResponseExpires(Aws::Utils::DateTime&& value) { SetResponseExpires(std::move(value)); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline GetObjectRequest& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline GetObjectRequest& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline GetObjectRequest& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline const Aws::String& GetSSECustomerAlgorithm() const{ return m_sSECustomerAlgorithm; }


inline void SetSSECustomerAlgorithm(const Aws::String& value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm = value; }


inline void SetSSECustomerAlgorithm(Aws::String&& value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm = std::move(value); }


inline void SetSSECustomerAlgorithm(const char* value) { m_sSECustomerAlgorithmHasBeenSet = true; m_sSECustomerAlgorithm.assign(value); }


inline GetObjectRequest& WithSSECustomerAlgorithm(const Aws::String& value) { SetSSECustomerAlgorithm(value); return *this;}


inline GetObjectRequest& WithSSECustomerAlgorithm(Aws::String&& value) { SetSSECustomerAlgorithm(std::move(value)); return *this;}


inline GetObjectRequest& WithSSECustomerAlgorithm(const char* value) { SetSSECustomerAlgorithm(value); return *this;}



inline const Aws::String& GetSSECustomerKey() const{ return m_sSECustomerKey; }


inline void SetSSECustomerKey(const Aws::String& value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey = value; }


inline void SetSSECustomerKey(Aws::String&& value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey = std::move(value); }


inline void SetSSECustomerKey(const char* value) { m_sSECustomerKeyHasBeenSet = true; m_sSECustomerKey.assign(value); }


inline GetObjectRequest& WithSSECustomerKey(const Aws::String& value) { SetSSECustomerKey(value); return *this;}


inline GetObjectRequest& WithSSECustomerKey(Aws::String&& value) { SetSSECustomerKey(std::move(value)); return *this;}


inline GetObjectRequest& WithSSECustomerKey(const char* value) { SetSSECustomerKey(value); return *this;}



inline const Aws::String& GetSSECustomerKeyMD5() const{ return m_sSECustomerKeyMD5; }


inline void SetSSECustomerKeyMD5(const Aws::String& value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5 = value; }


inline void SetSSECustomerKeyMD5(Aws::String&& value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5 = std::move(value); }


inline void SetSSECustomerKeyMD5(const char* value) { m_sSECustomerKeyMD5HasBeenSet = true; m_sSECustomerKeyMD5.assign(value); }


inline GetObjectRequest& WithSSECustomerKeyMD5(const Aws::String& value) { SetSSECustomerKeyMD5(value); return *this;}


inline GetObjectRequest& WithSSECustomerKeyMD5(Aws::String&& value) { SetSSECustomerKeyMD5(std::move(value)); return *this;}


inline GetObjectRequest& WithSSECustomerKeyMD5(const char* value) { SetSSECustomerKeyMD5(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline GetObjectRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline GetObjectRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}



inline int GetPartNumber() const{ return m_partNumber; }


inline void SetPartNumber(int value) { m_partNumberHasBeenSet = true; m_partNumber = value; }


inline GetObjectRequest& WithPartNumber(int value) { SetPartNumber(value); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_ifMatch;
bool m_ifMatchHasBeenSet;

Aws::Utils::DateTime m_ifModifiedSince;
bool m_ifModifiedSinceHasBeenSet;

Aws::String m_ifNoneMatch;
bool m_ifNoneMatchHasBeenSet;

Aws::Utils::DateTime m_ifUnmodifiedSince;
bool m_ifUnmodifiedSinceHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_range;
bool m_rangeHasBeenSet;

Aws::String m_responseCacheControl;
bool m_responseCacheControlHasBeenSet;

Aws::String m_responseContentDisposition;
bool m_responseContentDispositionHasBeenSet;

Aws::String m_responseContentEncoding;
bool m_responseContentEncodingHasBeenSet;

Aws::String m_responseContentLanguage;
bool m_responseContentLanguageHasBeenSet;

Aws::String m_responseContentType;
bool m_responseContentTypeHasBeenSet;

Aws::Utils::DateTime m_responseExpires;
bool m_responseExpiresHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;

Aws::String m_sSECustomerAlgorithm;
bool m_sSECustomerAlgorithmHasBeenSet;

Aws::String m_sSECustomerKey;
bool m_sSECustomerKeyHasBeenSet;

Aws::String m_sSECustomerKeyMD5;
bool m_sSECustomerKeyMD5HasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;

int m_partNumber;
bool m_partNumberHasBeenSet;
};

} 
} 
} 
