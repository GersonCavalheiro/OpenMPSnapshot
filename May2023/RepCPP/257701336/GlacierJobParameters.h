

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/Tier.h>
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

class AWS_S3_API GlacierJobParameters
{
public:
GlacierJobParameters();
GlacierJobParameters(const Aws::Utils::Xml::XmlNode& xmlNode);
GlacierJobParameters& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Tier& GetTier() const{ return m_tier; }


inline void SetTier(const Tier& value) { m_tierHasBeenSet = true; m_tier = value; }


inline void SetTier(Tier&& value) { m_tierHasBeenSet = true; m_tier = std::move(value); }


inline GlacierJobParameters& WithTier(const Tier& value) { SetTier(value); return *this;}


inline GlacierJobParameters& WithTier(Tier&& value) { SetTier(std::move(value)); return *this;}

private:

Tier m_tier;
bool m_tierHasBeenSet;
};

} 
} 
} 
