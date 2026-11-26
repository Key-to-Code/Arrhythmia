
#ifndef ECG_MODEL_DATA_H
#define ECG_MODEL_DATA_H
#include <stdint.h>
#define ECG_MODEL_INPUT_SIZE         250
#define ECG_MODEL_NUM_CLASSES        3
#define ECG_MODEL_CLASS_NORMAL       0
#define ECG_MODEL_CLASS_VENTRICULAR  1
#define ECG_MODEL_CLASS_SUPRAVENT    2
#define ECG_INPUT_SCALE              0.00392157f
#define ECG_INPUT_ZERO_POINT         -128
#define ECG_OUTPUT_SCALE             0.00390625f
#define ECG_OUTPUT_ZERO_POINT        -128
#define ECG_QUANTIZE_SAMPLE(val) ((int8_t)(((int32_t)(((val)/ECG_INPUT_SCALE)+0.5f)+ECG_INPUT_ZERO_POINT)>127?127:(((int32_t)(((val)/ECG_INPUT_SCALE)+0.5f)+ECG_INPUT_ZERO_POINT)<-128?-128:(int8_t)((int32_t)(((val)/ECG_INPUT_SCALE)+0.5f)+ECG_INPUT_ZERO_POINT))))
#define ECG_DEQUANTIZE_PROB(val) (((float)((val)-ECG_OUTPUT_ZERO_POINT))*ECG_OUTPUT_SCALE)
#if defined(__GNUC__)||defined(__clang__)
  #define ALIGN_ATTR __attribute__((aligned(16)))
#elif defined(__CC_ARM)||defined(__ARMCC_VERSION)
  #define ALIGN_ATTR __align(16)
#else
  #define ALIGN_ATTR
#endif
#endif // ECG_MODEL_DATA_H
