/**
 * ecg_cnn.h - Bare-metal INT8 1D-CNN inference engine (no TFLite-Micro needed)
 * Runs the layers exported from model_int8.tflite by export_cnn_weights.py.
 * Supported ops: CONV (valid, stride 1), per-channel MUL/ADD (BatchNorm), MAXPOOL, FC.
 */
#ifndef ECG_CNN_H
#define ECG_CNN_H

#include <stdint.h>

#define ECG_NUM_CLASSES 3
#define ECG_BEAT_LEN    250     /* samples per beat window fed to the network */

#ifndef ECG_CLASS_T_DEFINED
#define ECG_CLASS_T_DEFINED
typedef enum {
    ECG_CLASS_NORMAL           = 0,
    ECG_CLASS_VENTRICULAR      = 1,
    ECG_CLASS_SUPRAVENTRICULAR = 2
} ecg_class_t;
#endif

typedef enum { CNN_OP_CONV = 0, CNN_OP_MUL, CNN_OP_ADD, CNN_OP_POOL, CNN_OP_FC } cnn_op_t;

/* One layer descriptor (filled by the generated ecg_cnn_weights.h).
 * Activations are laid out [time][channel] (channel fastest). FC is stored as
 * a CONV with in_len=1, in_ch=N_in, k=1. */
typedef struct {
    uint8_t  type;
    uint16_t in_len, in_ch, out_len, out_ch, k;
    float    in_scale, out_scale, c_scale;
    int8_t   in_zp, out_zp, c_zp;
    const int8_t  *w;    /* weights [out_ch][k*in_ch] or MUL/ADD constant [ch] */
    const int32_t *b;    /* int32 bias [out_ch]                                */
    const float   *ws;   /* per-output-channel weight scale                    */
} cnn_layer_t;

int8_t ecg_cnn_quantize(float x01);                 /* [0,1] -> int8 model input */
int    ecg_cnn_predict(const int8_t *input, float probs[ECG_NUM_CLASSES], int8_t logits[ECG_NUM_CLASSES]);
int    ecg_cnn_selftest(void);                      /* 1 = matches TFLite reference */
float  ecg_cnn_selftest_maxdiff(void);              /* worst |prob - TFLite prob| seen by the last selftest */

/* Class labels, so firmware, OLED and telemetry all use the same strings */
const char *ecg_cnn_class_name(int cls);            /* "Normal (N)", "Ventricular (V)", ...  */
const char *ecg_cnn_class_letter(int cls);          /* "N", "V", "S"                         */

#endif
