

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/Delete.h>
#include <aws/s3/model/RequestPayer.h>
#include <utility>

namespace Aws
{
namespace S3
{
namespace Model
{


class AWS_S3_API DeleteObjectsRequest : public S3Request
{
public:
DeleteObjectsRequest();

inline virtual const char* GetServiceRequestName() const override { return "DeleteObjects"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;

inline bool ShouldComputeContentMd5() const override { return true; }



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline DeleteObjectsRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline DeleteObjectsRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline DeleteObjectsRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Delete& GetDelete() const{ return m_delete; }


inline void SetDelete(const Delete& value) { m_deleteHasBeenSet = true; m_delete = value; }


inline void SetDelete(Delete&& value) { m_deleteHasBeenSet = true; m_delete = std::move(value); }


inline DeleteObjectsRequest& WithDelete(const Delete& value) { SetDelete(value); return *this;}


inline DeleteObjectsRequest& WithDelete(Delete&& value) { SetDelete(std::move(value)); return *this;}



inline const Aws::String& GetMFA() const{ return m_mFA; }


inline void SetMFA(const Aws::String& value) { m_mFAHasBeenSet = true; m_mFA = value; }


inline void SetMFA(Aws::String&& value) { m_mFAHasBeenSet = true; m_mFA = std::move(value); }


inline void SetMFA(const char* value) { m_mFAHasBeenSet = true; m_mFA.assign(value); }


inline DeleteObjectsRequest& WithMFA(const Aws::String& value) { SetMFA(value); return *this;}


inline DeleteObjectsRequest& WithMFA(Aws::String&& value) { SetMFA(std::move(value)); return *this;}


inline DeleteObjectsRequest& WithMFA(const char* value) { SetMFA(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline DeleteObjectsRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline DeleteObjectsRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}

private:

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Delete m_delete;
bool m_deleteHasBeenSet;

Aws::String m_mFA;
bool m_mFAHasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;
};

} 
} 
} 
