

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/Condition.h>
#include <aws/s3/model/Redirect.h>
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

class AWS_S3_API RoutingRule
{
public:
RoutingRule();
RoutingRule(const Aws::Utils::Xml::XmlNode& xmlNode);
RoutingRule& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Condition& GetCondition() const{ return m_condition; }


inline void SetCondition(const Condition& value) { m_conditionHasBeenSet = true; m_condition = value; }


inline void SetCondition(Condition&& value) { m_conditionHasBeenSet = true; m_condition = std::move(value); }


inline RoutingRule& WithCondition(const Condition& value) { SetCondition(value); return *this;}


inline RoutingRule& WithCondition(Condition&& value) { SetCondition(std::move(value)); return *this;}



inline const Redirect& GetRedirect() const{ return m_redirect; }


inline void SetRedirect(const Redirect& value) { m_redirectHasBeenSet = true; m_redirect = value; }


inline void SetRedirect(Redirect&& value) { m_redirectHasBeenSet = true; m_redirect = std::move(value); }


inline RoutingRule& WithRedirect(const Redirect& value) { SetRedirect(value); return *this;}


inline RoutingRule& WithRedirect(Redirect&& value) { SetRedirect(std::move(value)); return *this;}

private:

Condition m_condition;
bool m_conditionHasBeenSet;

Redirect m_redirect;
bool m_redirectHasBeenSet;
};

} 
} 
} 
