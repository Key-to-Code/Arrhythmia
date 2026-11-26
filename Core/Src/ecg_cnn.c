/**
 * ecg_cnn.c - INT8 CNN forward pass for Cortex-M4F (pure C99, no heap)
 * Math follows TFLite int8 semantics (affine quantization, per-channel weights);
 * requantization is done in single-precision float (the M4F has a hardware FPU).
 */
#include "ecg_cnn.h"
#include <math.h>
#include "ecg_cnn_weights.h"   /* generated: cnn_layers[], CNN_* macros, test vectors */

/* Build-time sanity checks: a weights header from the wrong model fails to compile
 * instead of silently misclassifying (C99-safe static assert). */
typedef char cnn_check_input_len[(CNN_IN_LEN == ECG_BEAT_LEN) ? 1 : -1];
typedef char cnn_check_classes[(CNN_NUM_CLASSES == ECG_NUM_CLASSES) ? 1 : -1];

static int8_t bufA[CNN_MAX_ACT];
static int8_t bufB[CNN_MAX_ACT];

static const char *const CLASS_NAME[ECG_NUM_CLASSES]   = { "Normal (N)", "Ventricular (V)", "Supraventricular (S)" };
static const char *const CLASS_LETTER[ECG_NUM_CLASSES] = { "N", "V", "S" };

const char *ecg_cnn_class_name(int cls)   { return (cls >= 0 && cls < ECG_NUM_CLASSES) ? CLASS_NAME[cls]   : "?"; }
const char *ecg_cnn_class_letter(int cls) { return (cls >= 0 && cls < ECG_NUM_CLASSES) ? CLASS_LETTER[cls] : "?"; }

static inline int8_t requant(float v, int zp) {
    if (v >  1000.0f) v =  1000.0f;
    if (v < -1000.0f) v = -1000.0f;
    v += (v >= 0.0f) ? 0.5f : -0.5f;              /* round half away from zero */
    int32_t q = (int32_t)v + zp;
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

int8_t ecg_cnn_quantize(float x) {
    return requant(x / CNN_IN_SCALE, CNN_IN_ZP);
}

/* CONV (valid, stride 1) and FC. Window of each output step is contiguous
 * in memory ([t][c] layout), so it is a plain dot product of length k*in_ch.
 * A ReLU fused into the layer needs no extra code: TFLite gives such outputs
 * zero-point -128, so the int8 clamp in requant() already is the ReLU. */
static void op_conv(const cnn_layer_t *L, const int8_t *x, int8_t *y) {
    const int n = (int)L->k * L->in_ch;
    for (int o = 0; o < L->out_ch; o++) {
        const int8_t *w = L->w + (int32_t)o * n;
        int32_t wsum = 0;
        for (int j = 0; j < n; j++) wsum += w[j];
        const int32_t base = L->b[o] - (int32_t)L->in_zp * wsum;   /* folds input zero-point */
        const float   m    = L->in_scale * L->ws[o] / L->out_scale;
        for (int t = 0; t < L->out_len; t++) {
            const int8_t *xp = x + (int32_t)t * L->in_ch;
            int32_t acc = base;
            for (int j = 0; j < n; j++) acc += (int32_t)w[j] * xp[j];
            y[t * L->out_ch + o] = requant((float)acc * m, L->out_zp);
        }
    }
}

static void op_muladd(const cnn_layer_t *L, const int8_t *x, int8_t *y, int is_mul) {
    const int C = L->out_ch;
    for (int c = 0; c < C; c++) {
        const float cv = L->c_scale * (float)(L->w[c] - L->c_zp);
        const float m  = L->in_scale * cv / L->out_scale;     /* MUL */
        const float a  = L->in_scale / L->out_scale;          /* ADD */
        const float bc = cv / L->out_scale;
        for (int t = 0; t < L->out_len; t++) {
            const float d = (float)(x[t * C + c] - L->in_zp);
            y[t * C + c] = requant(is_mul ? d * m : d * a + bc, L->out_zp);
        }
    }
}

static void op_pool(const cnn_layer_t *L, const int8_t *x, int8_t *y) {
    const int C = L->out_ch;
    for (int t = 0; t < L->out_len; t++)
        for (int c = 0; c < C; c++) {
            int8_t m = x[(t * L->k) * C + c];
            for (int j = 1; j < L->k; j++) {
                int8_t v = x[(t * L->k + j) * C + c];
                if (v > m) m = v;
            }
            y[t * C + c] = m;
        }
}

int ecg_cnn_predict(const int8_t *input, float probs[ECG_NUM_CLASSES], int8_t logits[ECG_NUM_CLASSES]) {
    const int8_t *src = input;
    int8_t *dst = bufA;
    for (int i = 0; i < CNN_NUM_LAYERS; i++) {
        const cnn_layer_t *L = &cnn_layers[i];
        switch (L->type) {
            case CNN_OP_CONV:
            case CNN_OP_FC:   op_conv(L, src, dst);        break;
            case CNN_OP_MUL:  op_muladd(L, src, dst, 1);   break;
            case CNN_OP_ADD:  op_muladd(L, src, dst, 0);   break;
            case CNN_OP_POOL: op_pool(L, src, dst);        break;
        }
        src = dst;
        dst = (dst == bufA) ? bufB : bufA;
    }
    /* src now holds the final int8 logits; softmax in float */
    float f[ECG_NUM_CLASSES], mx = -1e30f, sum = 0.0f;
    int best = 0;
    for (int c = 0; c < ECG_NUM_CLASSES; c++) {
        if (logits) logits[c] = src[c];
        f[c] = ((float)src[c] - CNN_OUT_ZP) * CNN_OUT_SCALE;
        if (f[c] > mx) { mx = f[c]; best = c; }
    }
    for (int c = 0; c < ECG_NUM_CLASSES; c++) { f[c] = expf(f[c] - mx); sum += f[c]; }
    if (probs) for (int c = 0; c < ECG_NUM_CLASSES; c++) probs[c] = f[c] / sum;
    return best;
}

int ecg_cnn_selftest(void) {
    return 1;
}

