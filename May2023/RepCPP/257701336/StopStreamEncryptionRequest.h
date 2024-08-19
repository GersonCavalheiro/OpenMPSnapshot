

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisRequest.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/kinesis/model/EncryptionType.h>
#include <utility>

namespace Aws
{
namespace Kinesis
{
namespace Model
{


class AWS_KINESIS_API StopStreamEncryptionRequest : public KinesisRequest
{
public:
StopStreamEncryptionRequest();

inline virtual const char* GetServiceRequestName() const override { return "StopStreamEncryption"; }

Aws::String SerializePayload() const override;

Aws::Http::HeaderValueCollection GetRequestSpecificHeaders() const override;



inline const Aws::String& GetStreamName() const{ return m_streamName; }


inline void SetStreamName(const Aws::String& value) { m_streamNameHasBeenSet = true; m_streamName = value; }


inline void SetStreamName(Aws::String&& value) { m_streamNameHasBeenSet = true; m_streamName = std::move(value); }


inline void SetStreamName(const char* value) { m_streamNameHasBeenSet = true; m_streamName.assign(value); }


inline StopStreamEncryptionRequest& WithStreamName(const Aws::String& value) { SetStreamName(value); return *this;}


inline StopStreamEncryptionRequest& WithStreamName(Aws::String&& value) { SetStreamName(std::move(value)); return *this;}


inline StopStreamEncryptionRequest& WithStreamName(const char* value) { SetStreamName(value); return *this;}



inline const EncryptionType& GetEncryptionType() const{ return m_encryptionType; }


inline void SetEncryptionType(const EncryptionType& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = value; }


inline void SetEncryptionType(EncryptionType&& value) { m_encryptionTypeHasBeenSet = true; m_encryptionType = std::move(value); }


inline StopStreamEncryptionRequest& WithEncryptionType(const EncryptionType& value) { SetEncryptionType(value); return *this;}


inline StopStreamEncryptionRequest& WithEncryptionType(EncryptionType&& value) { SetEncryptionType(std::move(value)); return *this;}



inline const Aws::String& GetKeyId() const{ return m_keyId; }


inline void SetKeyId(const Aws::String& value) { m_keyIdHasBeenSet = true; m_keyId = value; }


inline void SetKeyId(Aws::String&& value) { m_keyIdHasBeenSet = true; m_keyId = std::move(value); }


inline void SetKeyId(const char* value) { m_keyIdHasBeenSet = true; m_keyId.assign(value); }


inline StopStreamEncryptionRequest& WithKeyId(const Aws::String& value) { SetKeyId(value); return *this;}


inline StopStreamEncryptionRequest& WithKeyId(Aws::String&& value) { SetKeyId(std::move(value)); return *this;}


inline StopStreamEncryptionRequest& WithKeyId(const char* value) { SetKeyId(value); return *this;}

private:

Aws::String m_streamName;
bool m_streamNameHasBeenSet;

EncryptionType m_encryptionType;
bool m_encryptionTypeHasBeenSet;

Aws::String m_keyId;
bool m_keyIdHasBeenSet;
};

} 
} 
} 
