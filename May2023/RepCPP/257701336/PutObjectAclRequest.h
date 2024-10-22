

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Request.h>
#include <aws/s3/model/ObjectCannedACL.h>
#include <aws/s3/model/AccessControlPolicy.h>
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


class AWS_S3_API PutObjectAclRequest : public S3Request
{
public:
PutObjectAclRequest();

inline virtual const char* GetServiceRequestName() const override { return "PutObjectAcl"; }

Aws::String SerializePayload() const override;

void AddQueryStringParameters(Aws::Http::URI& uri) const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const ObjectCannedACL& GetACL() const{ return m_aCL; }


inline void SetACL(const ObjectCannedACL& value) { m_aCLHasBeenSet = true; m_aCL = value; }


inline void SetACL(ObjectCannedACL&& value) { m_aCLHasBeenSet = true; m_aCL = std::move(value); }


inline PutObjectAclRequest& WithACL(const ObjectCannedACL& value) { SetACL(value); return *this;}


inline PutObjectAclRequest& WithACL(ObjectCannedACL&& value) { SetACL(std::move(value)); return *this;}



inline const AccessControlPolicy& GetAccessControlPolicy() const{ return m_accessControlPolicy; }


inline void SetAccessControlPolicy(const AccessControlPolicy& value) { m_accessControlPolicyHasBeenSet = true; m_accessControlPolicy = value; }


inline void SetAccessControlPolicy(AccessControlPolicy&& value) { m_accessControlPolicyHasBeenSet = true; m_accessControlPolicy = std::move(value); }


inline PutObjectAclRequest& WithAccessControlPolicy(const AccessControlPolicy& value) { SetAccessControlPolicy(value); return *this;}


inline PutObjectAclRequest& WithAccessControlPolicy(AccessControlPolicy&& value) { SetAccessControlPolicy(std::move(value)); return *this;}



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucketHasBeenSet = true; m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucketHasBeenSet = true; m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucketHasBeenSet = true; m_bucket.assign(value); }


inline PutObjectAclRequest& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline PutObjectAclRequest& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline PutObjectAclRequest& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetContentMD5() const{ return m_contentMD5; }


inline void SetContentMD5(const Aws::String& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = value; }


inline void SetContentMD5(Aws::String&& value) { m_contentMD5HasBeenSet = true; m_contentMD5 = std::move(value); }


inline void SetContentMD5(const char* value) { m_contentMD5HasBeenSet = true; m_contentMD5.assign(value); }


inline PutObjectAclRequest& WithContentMD5(const Aws::String& value) { SetContentMD5(value); return *this;}


inline PutObjectAclRequest& WithContentMD5(Aws::String&& value) { SetContentMD5(std::move(value)); return *this;}


inline PutObjectAclRequest& WithContentMD5(const char* value) { SetContentMD5(value); return *this;}



inline const Aws::String& GetGrantFullControl() const{ return m_grantFullControl; }


inline void SetGrantFullControl(const Aws::String& value) { m_grantFullControlHasBeenSet = true; m_grantFullControl = value; }


inline void SetGrantFullControl(Aws::String&& value) { m_grantFullControlHasBeenSet = true; m_grantFullControl = std::move(value); }


inline void SetGrantFullControl(const char* value) { m_grantFullControlHasBeenSet = true; m_grantFullControl.assign(value); }


inline PutObjectAclRequest& WithGrantFullControl(const Aws::String& value) { SetGrantFullControl(value); return *this;}


inline PutObjectAclRequest& WithGrantFullControl(Aws::String&& value) { SetGrantFullControl(std::move(value)); return *this;}


inline PutObjectAclRequest& WithGrantFullControl(const char* value) { SetGrantFullControl(value); return *this;}



inline const Aws::String& GetGrantRead() const{ return m_grantRead; }


inline void SetGrantRead(const Aws::String& value) { m_grantReadHasBeenSet = true; m_grantRead = value; }


inline void SetGrantRead(Aws::String&& value) { m_grantReadHasBeenSet = true; m_grantRead = std::move(value); }


inline void SetGrantRead(const char* value) { m_grantReadHasBeenSet = true; m_grantRead.assign(value); }


inline PutObjectAclRequest& WithGrantRead(const Aws::String& value) { SetGrantRead(value); return *this;}


inline PutObjectAclRequest& WithGrantRead(Aws::String&& value) { SetGrantRead(std::move(value)); return *this;}


inline PutObjectAclRequest& WithGrantRead(const char* value) { SetGrantRead(value); return *this;}



inline const Aws::String& GetGrantReadACP() const{ return m_grantReadACP; }


inline void SetGrantReadACP(const Aws::String& value) { m_grantReadACPHasBeenSet = true; m_grantReadACP = value; }


inline void SetGrantReadACP(Aws::String&& value) { m_grantReadACPHasBeenSet = true; m_grantReadACP = std::move(value); }


inline void SetGrantReadACP(const char* value) { m_grantReadACPHasBeenSet = true; m_grantReadACP.assign(value); }


inline PutObjectAclRequest& WithGrantReadACP(const Aws::String& value) { SetGrantReadACP(value); return *this;}


inline PutObjectAclRequest& WithGrantReadACP(Aws::String&& value) { SetGrantReadACP(std::move(value)); return *this;}


inline PutObjectAclRequest& WithGrantReadACP(const char* value) { SetGrantReadACP(value); return *this;}



inline const Aws::String& GetGrantWrite() const{ return m_grantWrite; }


inline void SetGrantWrite(const Aws::String& value) { m_grantWriteHasBeenSet = true; m_grantWrite = value; }


inline void SetGrantWrite(Aws::String&& value) { m_grantWriteHasBeenSet = true; m_grantWrite = std::move(value); }


inline void SetGrantWrite(const char* value) { m_grantWriteHasBeenSet = true; m_grantWrite.assign(value); }


inline PutObjectAclRequest& WithGrantWrite(const Aws::String& value) { SetGrantWrite(value); return *this;}


inline PutObjectAclRequest& WithGrantWrite(Aws::String&& value) { SetGrantWrite(std::move(value)); return *this;}


inline PutObjectAclRequest& WithGrantWrite(const char* value) { SetGrantWrite(value); return *this;}



inline const Aws::String& GetGrantWriteACP() const{ return m_grantWriteACP; }


inline void SetGrantWriteACP(const Aws::String& value) { m_grantWriteACPHasBeenSet = true; m_grantWriteACP = value; }


inline void SetGrantWriteACP(Aws::String&& value) { m_grantWriteACPHasBeenSet = true; m_grantWriteACP = std::move(value); }


inline void SetGrantWriteACP(const char* value) { m_grantWriteACPHasBeenSet = true; m_grantWriteACP.assign(value); }


inline PutObjectAclRequest& WithGrantWriteACP(const Aws::String& value) { SetGrantWriteACP(value); return *this;}


inline PutObjectAclRequest& WithGrantWriteACP(Aws::String&& value) { SetGrantWriteACP(std::move(value)); return *this;}


inline PutObjectAclRequest& WithGrantWriteACP(const char* value) { SetGrantWriteACP(value); return *this;}



inline const Aws::String& GetKey() const{ return m_key; }


inline void SetKey(const Aws::String& value) { m_keyHasBeenSet = true; m_key = value; }


inline void SetKey(Aws::String&& value) { m_keyHasBeenSet = true; m_key = std::move(value); }


inline void SetKey(const char* value) { m_keyHasBeenSet = true; m_key.assign(value); }


inline PutObjectAclRequest& WithKey(const Aws::String& value) { SetKey(value); return *this;}


inline PutObjectAclRequest& WithKey(Aws::String&& value) { SetKey(std::move(value)); return *this;}


inline PutObjectAclRequest& WithKey(const char* value) { SetKey(value); return *this;}



inline const RequestPayer& GetRequestPayer() const{ return m_requestPayer; }


inline void SetRequestPayer(const RequestPayer& value) { m_requestPayerHasBeenSet = true; m_requestPayer = value; }


inline void SetRequestPayer(RequestPayer&& value) { m_requestPayerHasBeenSet = true; m_requestPayer = std::move(value); }


inline PutObjectAclRequest& WithRequestPayer(const RequestPayer& value) { SetRequestPayer(value); return *this;}


inline PutObjectAclRequest& WithRequestPayer(RequestPayer&& value) { SetRequestPayer(std::move(value)); return *this;}



inline const Aws::String& GetVersionId() const{ return m_versionId; }


inline void SetVersionId(const Aws::String& value) { m_versionIdHasBeenSet = true; m_versionId = value; }


inline void SetVersionId(Aws::String&& value) { m_versionIdHasBeenSet = true; m_versionId = std::move(value); }


inline void SetVersionId(const char* value) { m_versionIdHasBeenSet = true; m_versionId.assign(value); }


inline PutObjectAclRequest& WithVersionId(const Aws::String& value) { SetVersionId(value); return *this;}


inline PutObjectAclRequest& WithVersionId(Aws::String&& value) { SetVersionId(std::move(value)); return *this;}


inline PutObjectAclRequest& WithVersionId(const char* value) { SetVersionId(value); return *this;}

private:

ObjectCannedACL m_aCL;
bool m_aCLHasBeenSet;

AccessControlPolicy m_accessControlPolicy;
bool m_accessControlPolicyHasBeenSet;

Aws::String m_bucket;
bool m_bucketHasBeenSet;

Aws::String m_contentMD5;
bool m_contentMD5HasBeenSet;

Aws::String m_grantFullControl;
bool m_grantFullControlHasBeenSet;

Aws::String m_grantRead;
bool m_grantReadHasBeenSet;

Aws::String m_grantReadACP;
bool m_grantReadACPHasBeenSet;

Aws::String m_grantWrite;
bool m_grantWriteHasBeenSet;

Aws::String m_grantWriteACP;
bool m_grantWriteACPHasBeenSet;

Aws::String m_key;
bool m_keyHasBeenSet;

RequestPayer m_requestPayer;
bool m_requestPayerHasBeenSet;

Aws::String m_versionId;
bool m_versionIdHasBeenSet;
};

} 
} 
} 
