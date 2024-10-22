

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/ServerSideEncryptionConfiguration.h>
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
class AWS_S3_API GetBucketEncryptionResult
{
public:
GetBucketEncryptionResult();
GetBucketEncryptionResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketEncryptionResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const ServerSideEncryptionConfiguration& GetServerSideEncryptionConfiguration() const{ return m_serverSideEncryptionConfiguration; }


inline void SetServerSideEncryptionConfiguration(const ServerSideEncryptionConfiguration& value) { m_serverSideEncryptionConfiguration = value; }


inline void SetServerSideEncryptionConfiguration(ServerSideEncryptionConfiguration&& value) { m_serverSideEncryptionConfiguration = std::move(value); }


inline GetBucketEncryptionResult& WithServerSideEncryptionConfiguration(const ServerSideEncryptionConfiguration& value) { SetServerSideEncryptionConfiguration(value); return *this;}


inline GetBucketEncryptionResult& WithServerSideEncryptionConfiguration(ServerSideEncryptionConfiguration&& value) { SetServerSideEncryptionConfiguration(std::move(value)); return *this;}

private:

ServerSideEncryptionConfiguration m_serverSideEncryptionConfiguration;
};

} 
} 
} 
