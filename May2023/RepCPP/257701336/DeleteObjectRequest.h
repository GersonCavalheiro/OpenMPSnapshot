

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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


class AWS_S3_API DeleteObjectRequest : public S3Request
{
public:
DeleteObjectRequest();

inline virtual const char* GetServiceRequestName() const override { return "DeleteObject"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline DeleteObjectRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline DeleteObjectRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline DeleteObjectRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline DeleteObjectRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline DeleteObjectRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline DeleteObjectRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline const Aws::String& GetMFA() const{ return m_mFA; }


inline void SetMFA(const Aws::String& value) { m_mFAHasBeenSet = true; m_mFA = value; }


inline void SetMFA(Aws::String&& value) { m_mFAHasBeenSet = true; m_mFA = std::move(value); }


inline void SetMFA(const char* value) { m_mFAHasBeenSet = true; m_mFA.assign(value); }


inline DeleteObjectRequest& WithMFA(const Aws::String& value) { SetMFA(value); return *this;}


inline DeleteObjectRequest& WithMFA(Aws::String&& value) { SetMFA(std::move(value)); return *this;}


inline DeleteObjectRequest& WithMFA(const char* value) { SetMFA(value); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline DeleteObjectRequest& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline DeleteObjectRequest& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline DeleteObjectRequest& WithVersionId(const char* value) { SetVersionId(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline DeleteObjectRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline DeleteObjectRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

Aws::String m_mFA;
bool m_mFAHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;
};

} 
} 
} 
