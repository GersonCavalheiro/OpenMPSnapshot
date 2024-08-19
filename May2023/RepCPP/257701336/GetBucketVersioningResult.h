

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/BucketVersioningStatus.h>
#include <aws/s3/model/MFADeleteStatus.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Xml
{
class XmlDocument;
} 
} 
namespace S3
{
namespace Model
{
class AWS_S3_API GetBucketVersioningResult
{
public:
GetBucketVersioningResult();
GetBucketVersioningResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketVersioningResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const BucketVersioningStatus& GetStatus() const{ return m_status; }


inline void SetStatus(const BucketVersioningStatus& value) { m_status = value; }


inline void SetStatus(BucketVersioningStatus&& value) { m_status = std::move(value); }


inline GetBucketVersioningResult& WithStatus(const BucketVersioningStatus& value) { SetStatus(value); return *this;}


inline GetBucketVersioningResult& WithStatus(BucketVersioningStatus&& value) { SetStatus(std::move(value)); return *this;}



inline const MFADeleteStatus& GetMFADelete() const{ return m_mFADelete; }


inline void SetMFADelete(const MFADeleteStatus& value) { m_mFADelete = value; }


inline void SetMFADelete(MFADeleteStatus&& value) { m_mFADelete = std::move(value); }


inline GetBucketVersioningResult& WithMFADelete(const MFADeleteStatus& value) { SetMFADelete(value); return *this;}


inline GetBucketVersioningResult& WithMFADelete(MFADeleteStatus&& value) { SetMFADelete(std::move(value)); return *this;}

private:

BucketVersioningStatus m_status;

MFADeleteStatus m_mFADelete;
};

} 
} 
} 
