

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
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
class AWS_S3_API ListBucketInventoryConfigurationsResult
{
public:
ListBucketInventoryConfigurationsResult();
ListBucketInventoryConfigurationsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
ListBucketInventoryConfigurationsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::String& GetContinuationToken() const{ return m_continuationToken; }


inline void SetContinuationToken(const Aws::String& value) { m_continuationToken = value; }


inline void SetContinuationToken(Aws::String&& value) { m_continuationToken = std::move(value); }


inline void SetContinuationToken(const char* value) { m_continuationToken.assign(value); }


inline ListBucketInventoryConfigurationsResult& WithContinuationToken(const Aws::String& value) { SetContinuationToken(value); return *this;}


inline ListBucketInventoryConfigurationsResult& WithContinuationToken(Aws::String&& value) { SetContinuationToken(std::move(value)); return *this;}


inline ListBucketInventoryConfigurationsResult& WithContinuationToken(const char* value) { SetContinuationToken(value); return *this;}



inline const Aws::Vector<InventoryConfiguration>& GetInventoryConfigurationList() const{ return m_inventoryConfigurationList; }


inline void SetInventoryConfigurationList(const Aws::Vector<InventoryConfiguration>& value) { m_inventoryConfigurationList = value; }


inline void SetInventoryConfigurationList(Aws::Vector<InventoryConfiguration>&& value) { m_inventoryConfigurationList = std::move(value); }


inline ListBucketInventoryConfigurationsResult& WithInventoryConfigurationList(const Aws::Vector<InventoryConfiguration>& value) { SetInventoryConfigurationList(value); return *this;}


inline ListBucketInventoryConfigurationsResult& WithInventoryConfigurationList(Aws::Vector<InventoryConfiguration>&& value) { SetInventoryConfigurationList(std::move(value)); return *this;}


inline ListBucketInventoryConfigurationsResult& AddInventoryConfigurationList(const InventoryConfiguration& value) { m_inventoryConfigurationList.push_back(value); return *this; }


inline ListBucketInventoryConfigurationsResult& AddInventoryConfigurationList(InventoryConfiguration&& value) { m_inventoryConfigurationList.push_back(std::move(value)); return *this; }



inline bool GetIsTruncated() const{ return m_isTruncated; }


inline void SetIsTruncated(bool value) { m_isTruncated = value; }


inline ListBucketInventoryConfigurationsResult& WithIsTruncated(bool value) { SetIsTruncated(value); return *this;}



inline const Aws::String& GetNextContinuationToken() const{ return m_nextContinuationToken; }


inline void SetNextContinuationToken(const Aws::String& value) { m_nextContinuationToken = value; }


inline void SetNextContinuationToken(Aws::String&& value) { m_nextContinuationToken = std::move(value); }


inline void SetNextContinuationToken(const char* value) { m_nextContinuationToken.assign(value); }


inline ListBucketInventoryConfigurationsResult& WithNextContinuationToken(const Aws::String& value) { SetNextContinuationToken(value); return *this;}


inline ListBucketInventoryConfigurationsResult& WithNextContinuationToken(Aws::String&& value) { SetNextContinuationToken(std::move(value)); return *this;}


inline ListBucketInventoryConfigurationsResult& WithNextContinuationToken(const char* value) { SetNextContinuationToken(value); return *this;}

private:

Aws::String m_continuationToken;

Aws::Vector<InventoryConfiguration> m_inventoryConfigurationList;

bool m_isTruncated;

Aws::String m_nextContinuationToken;
};

} 
} 
} 
