

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/GlacierJobParameters.h>
#include <aws/s3/model/RestoreRequestType.h>
#include <aws/s3/model/Tier.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/SelectParameters.h>
#include <aws/s3/model/OutputLocation.h>
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


class AWS_S3_API RestoreRequest
{
public:
RestoreRequest();
RestoreRequest(const Aws::Utils::Xml::XmlNode& xmlNode);
RestoreRequest& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline int GetDays() const{ return m_days; }


inline void SetDays(int value) { m_daysHasBeenSet = true; m_days = value; }


inline RestoreRequest& WithDays(int value) { SetDays(value); return *this;}



inline const GlacierJobParameters& GetGlacierJobParameters() const{ return m_glacierJobParameters; }


inline void SetGlacierJobParameters(const GlacierJobParameters& value) { m_glacierJobParametersHasBeenSet = true; m_glacierJobParameters = value; }


inline void SetGlacierJobParameters(GlacierJobParameters&& value) { m_glacierJobParametersHasBeenSet = true; m_glacierJobParameters = std::move(value); }


inline RestoreRequest& WithGlacierJobParameters(const GlacierJobParameters& value) { SetGlacierJobParameters(value); return *this;}


inline RestoreRequest& WithGlacierJobParameters(GlacierJobParameters&& value) { SetGlacierJobParameters(std::move(value)); return *this;}



inline const RestoreRequestType& GetType() const{ return m_type; }


inline void SetType(const RestoreRequestType& value) { m_typeHasBeenSet = true; m_type = value; }


inline void SetType(RestoreRequestType&& value) { m_typeHasBeenSet = true; m_type = std::move(value); }


inline RestoreRequest& WithType(const RestoreRequestType& value) { SetType(value); return *this;}


inline RestoreRequest& WithType(RestoreRequestType&& value) { SetType(std::move(value)); return *this;}



inline const Tier& GetTier() const{ return m_tier; }


inline void SetTier(const Tier& value) { m_tierHasBeenSet = true; m_tier = value; }


inline void SetTier(Tier&& value) { m_tierHasBeenSet = true; m_tier = std::move(value); }


inline RestoreRequest& WithTier(const Tier& value) { SetTier(value); return *this;}


inline RestoreRequest& WithTier(Tier&& value) { SetTier(std::move(value)); return *this;}



inline const Aws::String& GetDescription() const{ return m_description; }


inline void SetDescription(const Aws::String& value) { m_descriptionHasBeenSet = true; m_description = value; }


inline void SetDescription(Aws::String&& value) { m_descriptionHasBeenSet = true; m_description = std::move(value); }


inline void SetDescription(const char* value) { m_descriptionHasBeenSet = true; m_description.assign(value); }


inline RestoreRequest& WithDescription(const Aws::String& value) { SetDescription(value); return *this;}


inline RestoreRequest& WithDescription(Aws::String&& value) { SetDescription(std::move(value)); return *this;}


inline RestoreRequest& WithDescription(const char* value) { SetDescription(value); return *this;}



inline const SelectParameters& GetSelectParameters() const{ return m_selectParameters; }


inline void SetSelectParameters(const SelectParameters& value) { m_selectParametersHasBeenSet = true; m_selectParameters = value; }


inline void SetSelectParameters(SelectParameters&& value) { m_selectParametersHasBeenSet = true; m_selectParameters = std::move(value); }


inline RestoreRequest& WithSelectParameters(const SelectParameters& value) { SetSelectParameters(value); return *this;}


inline RestoreRequest& WithSelectParameters(SelectParameters&& value) { SetSelectParameters(std::move(value)); return *this;}



inline const OutputLocation& GetOutputLocation() const{ return m_outputLocation; }


inline void SetOutputLocation(const OutputLocation& value) { m_outputLocationHasBeenSet = true; m_outputLocation = value; }


inline void SetOutputLocation(OutputLocation&& value) { m_outputLocationHasBeenSet = true; m_outputLocation = std::move(value); }


inline RestoreRequest& WithOutputLocation(const OutputLocation& value) { SetOutputLocation(value); return *this;}


inline RestoreRequest& WithOutputLocation(OutputLocation&& value) { SetOutputLocation(std::move(value)); return *this;}

private:

int m_days;
bool m_daysHasBeenSet;

GlacierJobParameters m_glacierJobParameters;
bool m_glacierJobParametersHasBeenSet;

RestoreRequestType m_type;
bool m_typeHasBeenSet;

Tier m_tier;
bool m_tierHasBeenSet;

Aws::String m_description;
bool m_descriptionHasBeenSet;

SelectParameters m_selectParameters;
bool m_selectParametersHasBeenSet;

OutputLocation m_outputLocation;
bool m_outputLocationHasBeenSet;
};

} 
} 
} 
