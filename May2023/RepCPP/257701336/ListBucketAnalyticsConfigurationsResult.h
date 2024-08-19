

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/AnalyticsConfiguration.h>
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
class AWS_S3_API ListBucketAnalyticsConfigurationsResult
{
public:
ListBucketAnalyticsConfigurationsResult();
ListBucketAnalyticsConfigurationsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
ListBucketAnalyticsConfigurationsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline bool GetIsTruncated() const{ return m_isTruncated; }


inline void SetIsTruncated(bool value) { m_isTruncated = value; }


inline ListBucketAnalyticsConfigurationsResult& WithIsTruncated(bool value) { SetIsTruncated(value); return *this;}



inline const Aws::String& GetContinuationToken() const{ return m_continuationToken; }


inline void SetContinuationToken(const Aws::String& value) { m_continuationToken = value; }


inline void SetContinuationToken(Aws::String&& value) { m_continuationToken = std::move(value); }


inline void SetContinuationToken(const char* value) { m_continuationToken.assign(value); }


inline ListBucketAnalyticsConfigurationsResult& WithContinuationToken(const Aws::String& value) { SetContinuationToken(value); return *this;}


inline ListBucketAnalyticsConfigurationsResult& WithContinuationToken(Aws::String&& value) { SetContinuationToken(std::move(value)); return *this;}


inline ListBucketAnalyticsConfigurationsResult& WithContinuationToken(const char* value) { SetContinuationToken(value); return *this;}



inline const Aws::String& GetNextContinuationToken() const{ return m_nextContinuationToken; }


inline void SetNextContinuationToken(const Aws::String& value) { m_nextContinuationToken = value; }


inline void SetNextContinuationToken(Aws::String&& value) { m_nextContinuationToken = std::move(value); }


inline void SetNextContinuationToken(const char* value) { m_nextContinuationToken.assign(value); }


inline ListBucketAnalyticsConfigurationsResult& WithNextContinuationToken(const Aws::String& value) { SetNextContinuationToken(value); return *this;}


inline ListBucketAnalyticsConfigurationsResult& WithNextContinuationToken(Aws::String&& value) { SetNextContinuationToken(std::move(value)); return *this;}


inline ListBucketAnalyticsConfigurationsResult& WithNextContinuationToken(const char* value) { SetNextContinuationToken(value); return *this;}



inline const Aws::Vector<AnalyticsConfiguration>& GetAnalyticsConfigurationList() const{ return m_analyticsConfigurationList; }


inline void SetAnalyticsConfigurationList(const Aws::Vector<AnalyticsConfiguration>& value) { m_analyticsConfigurationList = value; }


inline void SetAnalyticsConfigurationList(Aws::Vector<AnalyticsConfiguration>&& value) { m_analyticsConfigurationList = std::move(value); }


inline ListBucketAnalyticsConfigurationsResult& WithAnalyticsConfigurationList(const Aws::Vector<AnalyticsConfiguration>& value) { SetAnalyticsConfigurationList(value); return *this;}


inline ListBucketAnalyticsConfigurationsResult& WithAnalyticsConfigurationList(Aws::Vector<AnalyticsConfiguration>&& value) { SetAnalyticsConfigurationList(std::move(value)); return *this;}


inline ListBucketAnalyticsConfigurationsResult& AddAnalyticsConfigurationList(const AnalyticsConfiguration& value) { m_analyticsConfigurationList.push_back(value); return *this; }


inline ListBucketAnalyticsConfigurationsResult& AddAnalyticsConfigurationList(AnalyticsConfiguration&& value) { m_analyticsConfigurationList.push_back(std::move(value)); return *this; }

private:

bool m_isTruncated;

Aws::String m_continuationToken;

Aws::String m_nextContinuationToken;

Aws::Vector<AnalyticsConfiguration> m_analyticsConfigurationList;
};

} 
} 
} 
