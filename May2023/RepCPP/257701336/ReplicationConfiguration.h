

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/ReplicationRule.h>
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


class AWS_S3_API ReplicationConfiguration
{
public:
ReplicationConfiguration();
ReplicationConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
ReplicationConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetRole() const{ return m_role; }


inline void SetRole(const Aws::String& value) { m_roleHasBeenSet = true; m_role = value; }


inline void SetRole(Aws::String&& value) { m_roleHasBeenSet = true; m_role = std::move(value); }


inline void SetRole(const char* value) { m_roleHasBeenSet = true; m_role.assign(value); }


inline ReplicationConfiguration& WithRole(const Aws::String& value) { SetRole(value); return *this;}


inline ReplicationConfiguration& WithRole(Aws::String&& value) { SetRole(std::move(value)); return *this;}


inline ReplicationConfiguration& WithRole(const char* value) { SetRole(value); return *this;}



inline const Aws::Vector<ReplicationRule>& GetRules() const{ return m_rules; }


inline void SetRules(const Aws::Vector<ReplicationRule>& value) { m_rulesHasBeenSet = true; m_rules = value; }


inline void SetRules(Aws::Vector<ReplicationRule>&& value) { m_rulesHasBeenSet = true; m_rules = std::move(value); }


inline ReplicationConfiguration& WithRules(const Aws::Vector<ReplicationRule>& value) { SetRules(value); return *this;}


inline ReplicationConfiguration& WithRules(Aws::Vector<ReplicationRule>&& value) { SetRules(std::move(value)); return *this;}


inline ReplicationConfiguration& AddRules(const ReplicationRule& value) { m_rulesHasBeenSet = true; m_rules.push_back(value); return *this; }


inline ReplicationConfiguration& AddRules(ReplicationRule&& value) { m_rulesHasBeenSet = true; m_rules.push_back(std::move(value)); return *this; }

private:

Aws::String m_role;
bool m_roleHasBeenSet;

Aws::Vector<ReplicationRule> m_rules;
bool m_rulesHasBeenSet;
};

} 
} 
} 
