

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/ServerSideEncryptionRule.h>
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


class AWS_S3_API ServerSideEncryptionConfiguration
{
public:
ServerSideEncryptionConfiguration();
ServerSideEncryptionConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
ServerSideEncryptionConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<ServerSideEncryptionRule>& GetRules() const{ return m_rules; }


inline void SetRules(const Aws::Vector<ServerSideEncryptionRule>& value) { m_rulesHasBeenSet = true; m_rules = value; }


inline void SetRules(Aws::Vector<ServerSideEncryptionRule>&& value) { m_rulesHasBeenSet = true; m_rules = std::move(value); }


inline ServerSideEncryptionConfiguration& WithRules(const Aws::Vector<ServerSideEncryptionRule>& value) { SetRules(value); return *this;}


inline ServerSideEncryptionConfiguration& WithRules(Aws::Vector<ServerSideEncryptionRule>&& value) { SetRules(std::move(value)); return *this;}


inline ServerSideEncryptionConfiguration& AddRules(const ServerSideEncryptionRule& value) { m_rulesHasBeenSet = true; m_rules.push_back(value); return *this; }


inline ServerSideEncryptionConfiguration& AddRules(ServerSideEncryptionRule&& value) { m_rulesHasBeenSet = true; m_rules.push_back(std::move(value)); return *this; }

private:

Aws::Vector<ServerSideEncryptionRule> m_rules;
bool m_rulesHasBeenSet;
};

} 
} 
} 
