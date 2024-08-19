

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/InputSerialization.h>
#include <aws/s3/model/ExpressionType.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/OutputSerialization.h>
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


class AWS_S3_API SelectParameters
{
public:
SelectParameters();
SelectParameters(const Aws::Utils::Xml::XmlNode& xmlNode);
SelectParameters& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const InputSerialization& GetInputSerialization() const{ return m_inputSerialization; }


inline void SetInputSerialization(const InputSerialization& value) { m_inputSerializationHasBeenSet = true; m_inputSerialization = value; }


inline void SetInputSerialization(InputSerialization&& value) { m_inputSerializationHasBeenSet = true; m_inputSerialization = std::move(value); }


inline SelectParameters& WithInputSerialization(const InputSerialization& value) { SetInputSerialization(value); return *this;}


inline SelectParameters& WithInputSerialization(InputSerialization&& value) { SetInputSerialization(std::move(value)); return *this;}



inline const ExpressionType& GetExpressionType() const{ return m_expressionType; }


inline void SetExpressionType(const ExpressionType& value) { m_expressionTypeHasBeenSet = true; m_expressionType = value; }


inline void SetExpressionType(ExpressionType&& value) { m_expressionTypeHasBeenSet = true; m_expressionType = std::move(value); }


inline SelectParameters& WithExpressionType(const ExpressionType& value) { SetExpressionType(value); return *this;}


inline SelectParameters& WithExpressionType(ExpressionType&& value) { SetExpressionType(std::move(value)); return *this;}



inline const Aws::String& GetExpression() const{ return m_expression; }


inline void SetExpression(const Aws::String& value) { m_expressionHasBeenSet = true; m_expression = value; }


inline void SetExpression(Aws::String&& value) { m_expressionHasBeenSet = true; m_expression = std::move(value); }


inline void SetExpression(const char* value) { m_expressionHasBeenSet = true; m_expression.assign(value); }


inline SelectParameters& WithExpression(const Aws::String& value) { SetExpression(value); return *this;}


inline SelectParameters& WithExpression(Aws::String&& value) { SetExpression(std::move(value)); return *this;}


inline SelectParameters& WithExpression(const char* value) { SetExpression(value); return *this;}



inline const OutputSerialization& GetOutputSerialization() const{ return m_outputSerialization; }


inline void SetOutputSerialization(const OutputSerialization& value) { m_outputSerializationHasBeenSet = true; m_outputSerialization = value; }


inline void SetOutputSerialization(OutputSerialization&& value) { m_outputSerializationHasBeenSet = true; m_outputSerialization = std::move(value); }


inline SelectParameters& WithOutputSerialization(const OutputSerialization& value) { SetOutputSerialization(value); return *this;}


inline SelectParameters& WithOutputSerialization(OutputSerialization&& value) { SetOutputSerialization(std::move(value)); return *this;}

private:

InputSerialization m_inputSerialization;
bool m_inputSerializationHasBeenSet;

ExpressionType m_expressionType;
bool m_expressionTypeHasBeenSet;

Aws::String m_expression;
bool m_expressionHasBeenSet;

OutputSerialization m_outputSerialization;
bool m_outputSerializationHasBeenSet;
};

} 
} 
} 
