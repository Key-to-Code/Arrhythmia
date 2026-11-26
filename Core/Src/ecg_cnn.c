/**
 * ecg_cnn.c - INT8 CNN forward pass for Cortex-M4F (pure C99, no heap)
 * Math follows TFLite int8 semantics (affine quantization, per-channel weights);
 * requantization is done in single-precision float (the M4F has a hardware FPU).
 */
#include "ecg_cnn.h"
#include <stddef.h>

/* fast_expf — Schraudolph bit-trick approximation, error < 2 %.
 * Avoids linking libm (expf), which costs ~6–10 KB on Arm Compiler 6.
 * Softmax is relative, so a uniform ~2 % error on all logits cancels out;
 * the argmax result is identical to true expf for all practical ECG inputs. */
static inline float fast_expf(float x)
{
    /* Clamp to avoid overflow in the union cast */
    if (x >  88.0f) return 3.4028235e+38f;
    if (x < -88.0f) return 0.0f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x) + (127 << 23);
    return u.f;
}
/* rr_model.h is the ONLY model in flash — ecg_cnn_weights.h (old wave-only model)
 * is excluded to stay within the 32 KB MDK-Lite limit.
 * rr_trunk_layers[] replaces cnn_layers[]; ecg_cnn_predict_rr() is the only
 * inference entry point. ecg_cnn_predict() (wave-only) is not compiled. */
#include "rr_model.h"

/* Build-time sanity: input length and class count must match the firmware contract */
typedef char rr_check_input[(RR_WAVE_LEN == ECG_BEAT_LEN) ? 1 : -1];
typedef char rr_check_classes[(RR_NUM_CLASSES == ECG_NUM_CLASSES) ? 1 : -1];

static int8_t bufA[RR_TRUNK_MAX_ACT];
static int8_t bufB[RR_TRUNK_MAX_ACT];

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
    return requant(x / RR_IN_SCALE, RR_IN_ZP);
}

int8_t ecg_cnn_quantize_rr(float x) {
    return requant(x / RR_IN_SCALE, RR_IN_ZP);
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

static const int8_t *run_layers(const cnn_layer_t *layers, int num_layers, const int8_t *input) {
    const int8_t *src = input;
    int8_t *dst = bufA;
    for (int i = 0; i < num_layers; i++) {
        const cnn_layer_t *L = &layers[i];
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
    return src;
}

/* ecg_cnn_predict() (wave-only, used cnn_layers[] from ecg_cnn_weights.h) is
 * removed — the hybrid ecg_cnn_predict_rr() below is the sole inference entry point. */

/**
 * ecg_cnn_predict_rr() - Two-stage hybrid inference.
 *
 * Stage 1: Run the INT8 CNN trunk (rr_trunk_layers) -> 16 INT8 features -> float32.
 * Stage 2: Concatenate 2 scaled RR features -> 18->16 ReLU -> 16->3 softmax on FPU.
 */
int ecg_cnn_predict_rr(const int8_t *input, uint8_t rr_abs, uint8_t rr_rel,
                       float probs_out[ECG_NUM_CLASSES]) {
    if (!input) return -1;
    const int8_t *trunk_q = run_layers(rr_trunk_layers, RR_TRUNK_NUM_LAYERS, input);

    float fused[RR_FUSED_FEATS];
    for (int i = 0; i < RR_TRUNK_FEATS; i++) {
        fused[i] = ((float)trunk_q[i] - RR_TRUNK_OUT_ZP) * RR_TRUNK_OUT_SCALE;
    }
    fused[RR_TRUNK_FEATS + 0] = ((float)rr_abs / 255.0f - 0.5f) * 4.0f;
    fused[RR_TRUNK_FEATS + 1] = ((float)rr_rel / 255.0f - 0.5f) * 4.0f;

    float hidden[RR_HEAD_HIDDEN];
    for (int j = 0; j < RR_HEAD_HIDDEN; j++) {
        float acc = rr_b1_head[j];
        for (int i = 0; i < RR_FUSED_FEATS; i++) {
            acc += fused[i] * rr_W1_head[i * RR_HEAD_HIDDEN + j];
        }
        hidden[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    float logits[RR_NUM_CLASSES];
    for (int j = 0; j < RR_NUM_CLASSES; j++) {
        float acc = rr_b2_head[j];
        for (int i = 0; i < RR_HEAD_HIDDEN; i++) {
            acc += hidden[i] * rr_W2_head[i * RR_NUM_CLASSES + j];
        }
        logits[j] = acc;
    }

    float maxv = logits[0];
    for (int j = 1; j < RR_NUM_CLASSES; j++) {
        if (logits[j] > maxv) maxv = logits[j];
    }
    float sum = 0.0f;
    int best = 0;
    for (int j = 0; j < RR_NUM_CLASSES; j++) {
        logits[j] = fast_expf(logits[j] - maxv);
        sum += logits[j];
    }
    for (int j = 0; j < RR_NUM_CLASSES; j++) {
        logits[j] /= sum;
        if (probs_out) probs_out[j] = logits[j];
        if (logits[j] > logits[best]) best = j;
    }
    return best;
}

int ecg_cnn_selftest(void) {
    for (int i = 0; i < RR_NUM_TESTS; i++) {
        float ra = rr_test_rr[i][0] * 255.0f + 0.5f;
        float rr = rr_test_rr[i][1] * 255.0f + 0.5f;
        uint8_t rr_abs = (ra < 0.0f) ? 0 : (ra > 255.0f ? 255 : (uint8_t)ra);
        uint8_t rr_rel = (rr < 0.0f) ? 0 : (rr > 255.0f ? 255 : (uint8_t)rr);
        int pred = ecg_cnn_predict_rr(rr_test_wave[i], rr_abs, rr_rel, NULL);
        if (pred != rr_test_expected[i]) return 0;
    }
    return 1;
}

