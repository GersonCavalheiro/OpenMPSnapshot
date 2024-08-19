

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/Payer.h>
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

class AWS_S3_API RequestPaymentConfiguration
{
public:
RequestPaymentConfiguration();
RequestPaymentConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
RequestPaymentConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Payer& GetPayer() const{ return m_payer; }


inline void SetPayer(const Payer& value) { m_payerHasBeenSet = true; m_payer = value; }


inline void SetPayer(Payer&& value) { m_payerHasBeenSet = true; m_payer = std::move(value); }


inline RequestPaymentConfiguration& WithPayer(const Payer& value) { SetPayer(value); return *this;}


inline RequestPaymentConfiguration& WithPayer(Payer&& value) { SetPayer(std::move(value)); return *this;}

private:

Payer m_payer;
bool m_payerHasBeenSet;
};

} 
} 
} 
