

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/CORSRule.h>
#include <utility>

namespace Aws
{
namespace Utils
{
namespace Xml
{
class XmlNode;
} 
} 
namespace S3
{
namespace Model
{

class AWS_S3_API CORSConfiguration
{
public:
CORSConfiguration();
CORSConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
CORSConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<CORSRule>& GetCORSRules() const{ return m_cORSRules; }


inline void SetCORSRules(const Aws::Vector<CORSRule>& value) { m_cORSRulesHasBeenSet = true; m_cORSRules = value; }


inline void SetCORSRules(Aws::Vector<CORSRule>&& value) { m_cORSRulesHasBeenSet = true; m_cORSRules = std::move(value); }


inline CORSConfiguration& WithCORSRules(const Aws::Vector<CORSRule>& value) { SetCORSRules(value); return *this;}


inline CORSConfiguration& WithCORSRules(Aws::Vector<CORSRule>&& value) { SetCORSRules(std::move(value)); return *this;}


inline CORSConfiguration& AddCORSRules(const CORSRule& value) { m_cORSRulesHasBeenSet = true; m_cORSRules.push_back(value); return *this; }


inline CORSConfiguration& AddCORSRules(CORSRule&& value) { m_cORSRulesHasBeenSet = true; m_cORSRules.push_back(std::move(value)); return *this; }

private:

Aws::Vector<CORSRule> m_cORSRules;
bool m_cORSRulesHasBeenSet;
};

} 
} 
} 
