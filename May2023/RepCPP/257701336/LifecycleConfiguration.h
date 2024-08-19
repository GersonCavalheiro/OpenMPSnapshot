

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/Rule.h>
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

class AWS_S3_API LifecycleConfiguration
{
public:
LifecycleConfiguration();
LifecycleConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
LifecycleConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<Rule>& GetRules() const{ return m_rules; }


inline void SetRules(const Aws::Vector<Rule>& value) { m_rulesHasBeenSet = true; m_rules = value; }


inline void SetRules(Aws::Vector<Rule>&& value) { m_rulesHasBeenSet = true; m_rules = std::move(value); }


inline LifecycleConfiguration& WithRules(const Aws::Vector<Rule>& value) { SetRules(value); return *this;}


inline LifecycleConfiguration& WithRules(Aws::Vector<Rule>&& value) { SetRules(std::move(value)); return *this;}


inline LifecycleConfiguration& AddRules(const Rule& value) { m_rulesHasBeenSet = true; m_rules.push_back(value); return *this; }


inline LifecycleConfiguration& AddRules(Rule&& value) { m_rulesHasBeenSet = true; m_rules.push_back(std::move(value)); return *this; }

private:

Aws::Vector<Rule> m_rules;
bool m_rulesHasBeenSet;
};

} 
} 
} 
