

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/LifecycleRule.h>
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
class AWS_S3_API GetBucketLifecycleConfigurationResult
{
public:
GetBucketLifecycleConfigurationResult();
GetBucketLifecycleConfigurationResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
GetBucketLifecycleConfigurationResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::Vector<LifecycleRule>& GetRules() const{ return m_rules; }


inline void SetRules(const Aws::Vector<LifecycleRule>& value) { m_rules = value; }


inline void SetRules(Aws::Vector<LifecycleRule>&& value) { m_rules = std::move(value); }


inline GetBucketLifecycleConfigurationResult& WithRules(const Aws::Vector<LifecycleRule>& value) { SetRules(value); return *this;}


inline GetBucketLifecycleConfigurationResult& WithRules(Aws::Vector<LifecycleRule>&& value) { SetRules(std::move(value)); return *this;}


inline GetBucketLifecycleConfigurationResult& AddRules(const LifecycleRule& value) { m_rules.push_back(value); return *this; }


inline GetBucketLifecycleConfigurationResult& AddRules(LifecycleRule&& value) { m_rules.push_back(std::move(value)); return *this; }

private:

Aws::Vector<LifecycleRule> m_rules;
};

} 
} 
} 
