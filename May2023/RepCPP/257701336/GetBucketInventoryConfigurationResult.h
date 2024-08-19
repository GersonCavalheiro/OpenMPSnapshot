

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/InventoryConfiguration.h>
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
class AWS_S3_API GetBucketInventoryConfigurationResult
{
public:
GetBucketInventoryConfigurationResult();
GetBucketInventoryConfigurationResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketInventoryConfigurationResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const InventoryConfiguration& GetInventoryConfiguration() const{ return m_inventoryConfiguration; }


inline void SetInventoryConfiguration(const InventoryConfiguration& value) { m_inventoryConfiguration = value; }


inline void SetInventoryConfiguration(InventoryConfiguration&& value) { m_inventoryConfiguration = std::move(value); }


inline GetBucketInventoryConfigurationResult& WithInventoryConfiguration(const InventoryConfiguration& value) { SetInventoryConfiguration(value); return *this;}


inline GetBucketInventoryConfigurationResult& WithInventoryConfiguration(InventoryConfiguration&& value) { SetInventoryConfiguration(std::move(value)); return *this;}

private:

InventoryConfiguration m_inventoryConfiguration;
};

} 
} 
} 
