
#include "stm32f4xx.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "ecg_cnn.h"

/* fabsf replacement — compiles to a single VABS.F32 on Cortex-M4F, no libm needed */
#define ECG_FABSF(x) ((x) < 0.0f ? -(x) : (x))

/* M_PI kept for any other users, defined as a constant (no math.h needed) */
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#define FS                      256.0f
#define SAMPLE_RATE             FS
#define BEAT_WINDOW_SIZE        250
#define PRE_R_SAMPLES           125
#define POST_R_SAMPLES          125
#define RING_BUF_SIZE           512
#define RING_BUF_MASK           (RING_BUF_SIZE - 1)
#define OLED_I2C_ADDR   0x3C   
#define OLED_WIDTH      128
#define OLED_HEIGHT     64
#define OLED_PAGES      8       // 64 pixel rows / 8 bits per page = 8 pages 
#define SSD1306_CTRL_CMD    0x00    
#define SSD1306_CTRL_DATA   0x40    

// MAX30003 ECG AFE on SPI1: SCK = PA5, MISO = PA6, MOSI = PA7, CS = PB0
#define MAX30003_CS_LOW()   (GPIOB->BSRR = (1UL << 16))
#define MAX30003_CS_HIGH()  (GPIOB->BSRR = (1UL << 0))

// ARRHYTHMIA CLASS LABELS
// ecg_class_t and the class-name helpers live in ecg_cnn.h / ecg_cnn.c so the
// firmware, OLED and telemetry all use exactly one definition.

volatile uint32_t msTicks = 0;

volatile float       watch_rawMv        = 0.0f;
volatile float       watch_filteredMv   = 0.0f;
volatile float       watch_qrsBandMv    = 0.0f;
volatile float       hr                 = 0.0f;
volatile bool        arrhythmia         = false;
volatile uint32_t    watch_sampleCount  = 0;

volatile ecg_class_t watch_predClass    = ECG_CLASS_NORMAL;
volatile const char* watch_className    = "Normal (N)";
volatile float       watch_prob_N       = 1.0f;
volatile float       watch_prob_V       = 0.0f;
volatile float       watch_prob_S       = 0.0f;
volatile uint32_t    watch_inferenceUs  = 0;
volatile uint32_t    watch_totalBeats   = 0;
volatile uint32_t    watch_ventBeats    = 0;
volatile uint32_t    watch_supraBeats   = 0;
volatile uint32_t    watch_fifoOverflows = 0;
volatile bool        watch_leadOff      = false;

/* RR interval state for hybrid CNN+RR inference */
static uint32_t rr_last_peak_tick = 0;      /* msTicks at previous QRS */
static uint32_t rr_history[8]     = {0};    /* rolling history of valid gaps (256-Hz samples) */
static uint8_t  rr_history_count  = 0;
static uint8_t  rr_history_head   = 0;

static inline uint32_t ms_to_samples(uint32_t ms) { return ms * 256 / 1000; }

static void rr_on_beat(uint32_t now_tick, uint8_t *rr_abs_out, uint8_t *rr_rel_out) {
    if (rr_last_peak_tick == 0) {
        *rr_abs_out = 128; *rr_rel_out = 128;
    } else {
        uint32_t gap_ms = now_tick - rr_last_peak_tick;
        uint32_t gap_s  = ms_to_samples(gap_ms);
        if (gap_s <= 1 || gap_s > 768) {
            *rr_abs_out = 128; *rr_rel_out = 128;
            rr_history_count = 0; rr_history_head = 0;
        } else {
            *rr_abs_out = (uint8_t)((gap_s > 512 ? 512 : gap_s) * 255 / 512);
            uint32_t mean = gap_s;
            if (rr_history_count > 0) {
                uint32_t sum = 0;
                for (uint8_t i = 0; i < rr_history_count; i++) sum += rr_history[i];
                mean = sum / rr_history_count;
            }
            uint32_t rel = gap_s * 128 / (mean > 0 ? mean : 1);
            *rr_rel_out = (uint8_t)(rel > 255 ? 255 : rel);
            rr_history[rr_history_head] = gap_s;
            rr_history_head = (rr_history_head + 1) & 7;
            if (rr_history_count < 8) rr_history_count++;
        }
    }
    rr_last_peak_tick = now_tick;
}

/* -------------------------------------------------------------------------
 * Small formatting helpers.
 * The full C library printf/strtof support costs ~15 KB of flash, which does
 * not fit the 32 KB MDK-Lite image together with the INT8 model. These tiny
 * helpers cover exactly the formats the firmware needs.
 * ---------------------------------------------------------------------- */
static char *append_str(char *p, const char *s) {
    while (*s) *p++ = *s++;
    return p;
}

static char *append_uint(char *p, uint32_t v, int min_width, char pad) {
    char tmp[10];
    int  n = 0;
    do {
        tmp[n++] = (char)('0' + (v % 10U));
        v /= 10U;
    } while (v != 0U);
    for (int i = n; i < min_width; i++) *p++ = pad;
    while (n > 0) *p++ = tmp[--n];
    return p;
}

static char *append_fixed(char *p, float v, int decimals) {
    if (v < 0.0f) { *p++ = '-'; v = -v; }
    uint32_t scale = 1U;
    for (int i = 0; i < decimals; i++) scale *= 10U;
    uint32_t q = (uint32_t)(v * (float)scale + 0.5f);
    p = append_uint(p, q / scale, 0, ' ');
    if (decimals > 0) {
        *p++ = '.';
        uint32_t rem = q % scale;
        uint32_t div = scale / 10U;
        while (div != 0U) {
            *p++ = (char)('0' + (rem / div) % 10U);
            div /= 10U;
        }
    }
    return p;
}

// OLED refresh flag - set by inference, consumed by main loop 
static volatile bool oled_update_pending = false;

void FPU_Init(void) {
    SCB->CPACR |= ((3UL << 10 * 2) | (3UL << 11 * 2));
    __DSB();
    __ISB();
}

void DWT_Init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL        |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT       = 0;
}

void SysTick_Init(uint32_t sysclk_hz) {
    SysTick->LOAD = (sysclk_hz / 1000U) - 1U;
    SysTick->VAL  = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
                  | SysTick_CTRL_TICKINT_Msk
                  | SysTick_CTRL_ENABLE_Msk;
}

void SysTick_Handler(void) { msTicks++; }

void delay_ms(uint32_t ms) {
    uint32_t start = msTicks;
    while ((msTicks - start) < ms) { __NOP(); }
}


void GPIO_Init_All(void) {
    // Enable GPIOA, GPIOB, GPIOC clocks
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN
                  | RCC_AHB1ENR_GPIOBEN
                  | RCC_AHB1ENR_GPIOCEN;
    GPIOC->MODER &= ~(3UL << (13 * 2));
    GPIOC->MODER |=  (1UL << (13 * 2));
    GPIOC->BSRR   =  (1UL << 13);   //LED OFF initially 
    //Alternate function (AF7) for pa2 and pa3
    GPIOA->MODER  &= ~((3UL << (2 * 2)) | (3UL << (3 * 2)));
    GPIOA->MODER  |=  ((2UL << (2 * 2)) | (2UL << (3 * 2)));
    GPIOA->AFR[0] &= ~((0xFUL << (2 * 4)) | (0xFUL << (3 * 4)));
    GPIOA->AFR[0] |=  ((7UL   << (2 * 4)) | (7UL   << (3 * 4)));
    // Internal pull-up on PA3 RX to avoid floating noise on idle line
    GPIOA->PUPDR  &= ~(3UL << (3 * 2));
    GPIOA->PUPDR  |=  (1UL << (3 * 2));
    GPIOA->MODER  &= ~((3UL << (9 * 2)) | (3UL << (10 * 2)));
    GPIOA->MODER  |=  ((2UL << (9 * 2)) | (2UL << (10 * 2)));
    GPIOA->AFR[1] &= ~((0xFUL << ((9  - 8) * 4)) | (0xFUL << ((10 - 8) * 4)));
    GPIOA->AFR[1] |=  ((7UL   << ((9  - 8) * 4)) | (7UL   << ((10 - 8) * 4)));
    // PB6 (I2C1_SCL) and PB7 (I2C1_SDA): Alternate Function 4
    GPIOB->MODER   &= ~((3UL << (6 * 2)) | (3UL << (7 * 2)));
    GPIOB->MODER   |=  ((2UL << (6 * 2)) | (2UL << (7 * 2)));  // AF mode 
    GPIOB->OTYPER  |=  ((1UL << 6) | (1UL << 7));               // Open-drain 
    GPIOB->OSPEEDR |=  ((3UL << (6 * 2)) | (3UL << (7 * 2)));  // High speed
    GPIOB->PUPDR   &= ~((3UL << (6 * 2)) | (3UL << (7 * 2)));
    GPIOB->PUPDR   |=  ((1UL << (6 * 2)) | (1UL << (7 * 2)));  // Pull-up 
    GPIOB->AFR[0]  &= ~((0xFUL << (6 * 4)) | (0xFUL << (7 * 4)));
    GPIOB->AFR[0]  |=  ((4UL   << (6 * 4)) | (4UL   << (7 * 4))); // AF4 = I2C1 
    // PA5/PA6/PA7: SPI1 SCK/MISO/MOSI, Alternate Function 5
    GPIOA->MODER   &= ~((3UL << (5 * 2)) | (3UL << (6 * 2)) | (3UL << (7 * 2)));
    GPIOA->MODER   |=  ((2UL << (5 * 2)) | (2UL << (6 * 2)) | (2UL << (7 * 2))); // AF mode
    GPIOA->OTYPER  &= ~((1UL << 5) | (1UL << 6) | (1UL << 7));                  // Push-pull
    GPIOA->OSPEEDR |=  ((3UL << (5 * 2)) | (3UL << (6 * 2)) | (3UL << (7 * 2))); // High speed
    GPIOA->PUPDR   &= ~((3UL << (5 * 2)) | (3UL << (6 * 2)) | (3UL << (7 * 2)));
    GPIOA->AFR[0]  &= ~((0xFUL << (5 * 4)) | (0xFUL << (6 * 4)) | (0xFUL << (7 * 4)));
    GPIOA->AFR[0]  |=  ((5UL   << (5 * 4)) | (5UL   << (6 * 4)) | (5UL   << (7 * 4))); // AF5 = SPI1
    // PB0: MAX30003 chip select, driven high (deselected) at boot
    GPIOB->MODER   &= ~(3UL << 0);
    GPIOB->MODER   |=  (1UL << 0);                     // General purpose output
    GPIOB->OTYPER  &= ~(1UL << 0);                     // Push-pull
    GPIOB->OSPEEDR |=  (3UL << 0);                     // High speed
    GPIOB->PUPDR   &= ~(3UL << 0);
    GPIOB->BSRR     =  (1UL << 0);                     // CS idle high
    // PB1: MAX30003 INT1 (EINT), open-drain active-low input with pull-up
    GPIOB->MODER   &= ~(3UL << (1 * 2));               // Input mode
    GPIOB->PUPDR   &= ~(3UL << (1 * 2));
    GPIOB->PUPDR   |=  (1UL << (1 * 2));               // Pull-up
}

// UART INITIALISATION

void UART_Init_All(uint32_t sysclk_hz) {
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN; 
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;  

    uint32_t ppre1 = (RCC->CFGR & RCC_CFGR_PPRE1) >> 10;
    uint32_t ppre2 = (RCC->CFGR & RCC_CFGR_PPRE2) >> 13;
    static const uint8_t APB_PrescTable[8] = {0, 0, 0, 0, 1, 2, 3, 4};
    uint32_t pclk1 = sysclk_hz >> APB_PrescTable[ppre1];
    uint32_t pclk2 = sysclk_hz >> APB_PrescTable[ppre2];


    USART2->BRR = (pclk1 + (115200U / 2U)) / 115200U;
    USART2->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_RXNEIE | USART_CR1_UE;
    NVIC_EnableIRQ(USART2_IRQn);


    USART1->BRR = (pclk2 + (9600U / 2U)) / 9600U;
    USART1->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}


// I2C1 INITIALISATION

void I2C1_Init(void) {
    RCC->APB1ENR |= RCC_APB1ENR_I2C1EN; 


    I2C1->CR1 |=  I2C_CR1_SWRST;
    I2C1->CR1 &= ~I2C_CR1_SWRST;

    // CR2: I2C peripheral clock in MHz, derived from the real APB1 clock
    uint32_t ppre1 = (RCC->CFGR & RCC_CFGR_PPRE1) >> 10;
    static const uint8_t APB_PrescTable[8] = {0, 0, 0, 0, 1, 2, 3, 4};
    uint32_t pclk1 = SystemCoreClock >> APB_PrescTable[ppre1];

    I2C1->CR2 = pclk1 / 1000000U;

    // CCR: standard mode, 100 kHz
    I2C1->CCR = pclk1 / 200000U;

    // TRISE: 1000 ns max rise time in PCLK1 periods, plus one
    I2C1->TRISE = (pclk1 / 1000000U) + 1U;
    

    // Enable I2C1 peripheral 
    I2C1->CR1 |= I2C_CR1_PE;
}


/* =========================================================================
 * MAX30003 ECG AFE on SPI1
 *   SCK  -> PA5
 *   MISO -> PA6
 *   MOSI -> PA7
 *   CS   -> PB0
 * The AFE streams ECG samples into its 32-word FIFO at 256 SPS. INT1/INT2 are
 * not wired, so the FIFO status flag (EINT) is polled instead of using an
 * interrupt. This replaces the synthetic samples the Python bridge used to
 * send over USART2.
 * ====================================================================== */
#define MAX30003_REG_STATUS     0x01
#define MAX30003_REG_MNGR_INT   0x04
#define MAX30003_REG_SW_RST     0x08
#define MAX30003_REG_SYNCH      0x09
#define MAX30003_REG_FIFO_RST   0x0A
#define MAX30003_REG_INFO       0x0F
#define MAX30003_REG_CNFG_GEN   0x10
#define MAX30003_REG_CNFG_CAL   0x12
#define MAX30003_REG_CNFG_EMUX  0x14
#define MAX30003_REG_CNFG_ECG   0x15
#define MAX30003_REG_ECG_FIFO   0x21

#define MAX30003_STATUS_EINT    0x800000UL
#define MAX30003_STATUS_EOVF    0x400000UL
#define MAX30003_STATUS_LOFF    0x00000FUL   /* lead-off PH/PL/NH/NL flags */
#define ECG_SAMPLE_LSB_MV       0.00048828125f   /* 0.48828125 uV per LSB */

#define SPI_TIMEOUT 10000U

void SPI1_Init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;

    uint32_t ppre2 = (RCC->CFGR & RCC_CFGR_PPRE2) >> 13;
    static const uint8_t APB_PrescTable[8] = {0, 0, 0, 0, 1, 2, 3, 4};
    uint32_t pclk2 = SystemCoreClock >> APB_PrescTable[ppre2];

    // Largest prescaler that keeps SCLK below the MAX30003 5 MHz limit
    uint32_t br = 0, div = 2;
    while (br < 7U && (pclk2 / div) > 5000000U) {
        div <<= 1;
        br++;
    }

    // Master, mode 0 (CPOL=0, CPHA=0), software NSS held high
    SPI1->CR1 = SPI_CR1_MSTR | SPI_CR1_SSM | SPI_CR1_SSI | (br << 3);
    SPI1->CR2 = 0;
    SPI1->CR1 |= SPI_CR1_SPE;
}

static uint8_t SPI1_Transfer(uint8_t data) {
    uint32_t to = SPI_TIMEOUT;
    while (!(SPI1->SR & SPI_SR_TXE) && --to) {}
    if (to == 0) return 0xFFU;

    *(volatile uint8_t *)&SPI1->DR = data;

    to = SPI_TIMEOUT;
    while (!(SPI1->SR & SPI_SR_RXNE) && --to) {}
    return (uint8_t)SPI1->DR;
}

static void MAX30003_WriteReg(uint8_t reg, uint32_t value) {
    MAX30003_CS_LOW();
    SPI1_Transfer((uint8_t)((reg << 1) | 0x00U));   // A[6:0] + W
    SPI1_Transfer((uint8_t)(value >> 16));
    SPI1_Transfer((uint8_t)(value >> 8));
    SPI1_Transfer((uint8_t)value);
    while (SPI1->SR & SPI_SR_BSY) {}
    MAX30003_CS_HIGH();
}

static uint32_t MAX30003_ReadReg(uint8_t reg) {
    MAX30003_CS_LOW();
    SPI1_Transfer((uint8_t)((reg << 1) | 0x01U));   // A[6:0] + R
    uint8_t b0 = SPI1_Transfer(0xFFU);
    uint8_t b1 = SPI1_Transfer(0xFFU);
    uint8_t b2 = SPI1_Transfer(0xFFU);
    while (SPI1->SR & SPI_SR_BSY) {}
    MAX30003_CS_HIGH();
    return ((uint32_t)b0 << 16) | ((uint32_t)b1 << 8) | b2;
}

// Returns true when the MAX30003 INFO register identifies the part.
static bool MAX30003_Init(void) {
    MAX30003_CS_HIGH();
    delay_ms(10);

    MAX30003_WriteReg(MAX30003_REG_SW_RST, 0x000000UL);
    delay_ms(100);

    uint32_t info = MAX30003_ReadReg(MAX30003_REG_INFO);

    MAX30003_WriteReg(MAX30003_REG_CNFG_GEN,  0x081007UL); // ECG channel enabled
    delay_ms(10);
    MAX30003_WriteReg(MAX30003_REG_CNFG_CAL,  0x720000UL); // calibration sources disabled
    delay_ms(10);
    MAX30003_WriteReg(MAX30003_REG_CNFG_EMUX, 0x0B0000UL); // electrodes on ECGP/ECGN
    delay_ms(10);
    MAX30003_WriteReg(MAX30003_REG_CNFG_ECG,  0x405000UL); // 256 SPS, 0.5 Hz HP, 40 Hz LP
    delay_ms(10);
    MAX30003_WriteReg(MAX30003_REG_MNGR_INT,  0x000000UL); // EINT at 1 FIFO sample
    MAX30003_WriteReg(MAX30003_REG_FIFO_RST,  0x000000UL);
    MAX30003_WriteReg(MAX30003_REG_SYNCH,     0x000000UL);
    delay_ms(10);

    return ((info >> 20) & 0x0FUL) == 0x05UL;   // MAX30003 ID nibble
}

// INT1 (PB1) is open-drain and pulled low while the FIFO holds a sample.
// If INT is left unconnected the internal pull-up keeps PB1 high, so the
// SPI status register is polled as a fallback.
static bool MAX30003_DataReady(void) {
    if ((GPIOB->IDR & (1UL << 1)) == 0) return true;
    return (MAX30003_ReadReg(MAX30003_REG_STATUS) & MAX30003_STATUS_EINT) != 0;
}

// Pops one ECG sample from the FIFO. Returns false for non-ECG/overflow entries.
static bool MAX30003_ReadSample(int32_t *sample) {
    uint32_t word = MAX30003_ReadReg(MAX30003_REG_ECG_FIFO);
    uint8_t  etag = (uint8_t)((word >> 3) & 0x07U);

    if (etag == 0x07U) {                          // FIFO overflow: flush it
        // No SYNCH here: restarting the signal chain makes the 0.5 Hz DHPF
        // step at the electrode offset, which shows up as a spike + decay.
        watch_fifoOverflows++;
        MAX30003_WriteReg(MAX30003_REG_FIFO_RST, 0x000000UL);
        return false;
    }
    if (etag != 0x00U && etag != 0x02U) return false;

    int32_t s = (int32_t)(word >> 6);             // 18-bit signed sample
    if (s & 0x20000) s -= 0x40000;                // sign-extend
    *sample = s;
    return true;
}


static bool oled_present = false;
#define I2C_TIMEOUT  10000U

static bool I2C1_Start(void) {
    uint32_t to = I2C_TIMEOUT;
    while ((I2C1->SR2 & I2C_SR2_BUSY) && --to) {}
    if (to == 0) return false;
    I2C1->CR1 |= I2C_CR1_START;
    to = I2C_TIMEOUT;
    while (!(I2C1->SR1 & I2C_SR1_SB) && --to) {}
    return (to > 0);
}

static bool I2C1_SendAddr(uint8_t addr7) {
    uint32_t to = I2C_TIMEOUT;
    I2C1->DR = (uint8_t)((addr7 << 1) & 0xFE);
    while (!(I2C1->SR1 & I2C_SR1_ADDR) && --to) {
        if (I2C1->SR1 & I2C_SR1_AF) {
            I2C1->SR1 &= ~I2C_SR1_AF;
            I2C1->CR1 |= I2C_CR1_STOP;
            return false;
        }
    }
    if (to == 0) { I2C1->CR1 |= I2C_CR1_STOP; return false; }
    (void)I2C1->SR1;
    (void)I2C1->SR2;
    return true;
}

static bool I2C1_WriteByte(uint8_t data) {
    uint32_t to = I2C_TIMEOUT;
    while (!(I2C1->SR1 & I2C_SR1_TXE) && --to) {}
    if (to == 0) return false;
    I2C1->DR = data;
    return true;
}

static void I2C1_Stop(void) {
    uint32_t to = I2C_TIMEOUT;
    while (!(I2C1->SR1 & I2C_SR1_BTF) && --to) {}
    I2C1->CR1 |= I2C_CR1_STOP;
}

static bool OLED_Cmd(uint8_t cmd) {
    if (!I2C1_Start()) return false;
    if (!I2C1_SendAddr(OLED_I2C_ADDR)) return false;
    I2C1_WriteByte(SSD1306_CTRL_CMD);
    I2C1_WriteByte(cmd);
    I2C1_Stop();
    return true;
}

__attribute__((unused)) static void OLED_Data(uint8_t data) {
    if (!oled_present) return;
    if (!I2C1_Start()) return;
    if (!I2C1_SendAddr(OLED_I2C_ADDR)) return;
    I2C1_WriteByte(SSD1306_CTRL_DATA);
    I2C1_WriteByte(data);
    I2C1_Stop();
}

static void OLED_DataBuf(const uint8_t *buf, uint16_t len) {
    if (!oled_present) return;
    if (!I2C1_Start()) return;
    if (!I2C1_SendAddr(OLED_I2C_ADDR)) return;
    I2C1_WriteByte(SSD1306_CTRL_DATA);
    for (uint16_t i = 0; i < len; i++) {
        if (!I2C1_WriteByte(buf[i])) break;
    }
    I2C1_Stop();
}

//OLED:
//VCC -> 5V 
//GND -> GND
//SDA -> PB7   
//SCL -> PB6 

void OLED_Init(void) {
    delay_ms(100);  

    static const uint8_t init_seq[] = {
        0xAE,           /* Display OFF */
        0xD5, 0x80,     /* Set display clock / oscillator frequency */
        0xA8, 0x3F,     /* Set multiplex ratio = 64 */
        0xD3, 0x00,     /* Set display offset = 0 */
        0x40,           /* Set display start line = 0 */
        0x8D, 0x14,     /* Charge pump: ENABLE (internal VCC) */
        0x20, 0x00,     /* Memory addressing mode: Horizontal */
        0xA1,           /* Segment remap: column 127 is SEG0 */
        0xC8,           /* COM scan direction: remapped (top-to-bottom) */
        0xDA, 0x12,     /* COM pins hardware configuration */
        0x81, 0xCF,     /* Contrast control = 207 */
        0xD9, 0xF1,     /* Pre-charge period: phase1=1, phase2=15 */
        0xDB, 0x40,     /* VCOMH deselect level */
        0xA4,           /* Entire display ON: resume from RAM */
        0xA6,           /* Normal display (non-inverted) */
        0xAF            /* Display ON */
    };

    if (!OLED_Cmd(init_seq[0])) {
        oled_present = false;
        return;
    }
    oled_present = true;
    for (uint32_t i = 1; i < sizeof(init_seq); i++) {
        OLED_Cmd(init_seq[i]);
    }
}

//OLED DISPLAY UTILITIES

void OLED_Clear(void) {
    static uint8_t zeros[OLED_WIDTH];
    memset(zeros, 0x00, sizeof(zeros));
    for (uint8_t page = 0; page < OLED_PAGES; page++) {
        OLED_Cmd((uint8_t)(0xB0 | page)); /* Set page address */
        OLED_Cmd(0x00);                    /* Lower column start = 0 */
        OLED_Cmd(0x10);                    /* Upper column start = 0 */
        OLED_DataBuf(zeros, OLED_WIDTH);
    }
}


static void OLED_SetCursor(uint8_t page, uint8_t col) {
    OLED_Cmd((uint8_t)(0xB0 | (page & 0x07)));
    OLED_Cmd((uint8_t)(0x00 | (col & 0x0F)));
    OLED_Cmd((uint8_t)(0x10 | ((col >> 4) & 0x0F)));
}

static const uint8_t font5x7[][5] = {
    { 0x00, 0x00, 0x00, 0x00, 0x00 }, /* 0x20 SPACE */
    { 0x00, 0x00, 0x5F, 0x00, 0x00 }, /* 0x21 ! */
    { 0x00, 0x07, 0x00, 0x07, 0x00 }, /* 0x22 " */
    { 0x14, 0x7F, 0x14, 0x7F, 0x14 }, /* 0x23 # */
    { 0x24, 0x2A, 0x7F, 0x2A, 0x12 }, /* 0x24 $ */
    { 0x23, 0x13, 0x08, 0x64, 0x62 }, /* 0x25 % */
    { 0x36, 0x49, 0x55, 0x22, 0x50 }, /* 0x26 & */
    { 0x00, 0x05, 0x03, 0x00, 0x00 }, /* 0x27 ' */
    { 0x00, 0x1C, 0x22, 0x41, 0x00 }, /* 0x28 ( */
    { 0x00, 0x41, 0x22, 0x1C, 0x00 }, /* 0x29 ) */
    { 0x14, 0x08, 0x3E, 0x08, 0x14 }, /* 0x2A * */
    { 0x08, 0x08, 0x3E, 0x08, 0x08 }, /* 0x2B + */
    { 0x00, 0x50, 0x30, 0x00, 0x00 }, /* 0x2C , */
    { 0x08, 0x08, 0x08, 0x08, 0x08 }, /* 0x2D - */
    { 0x00, 0x60, 0x60, 0x00, 0x00 }, /* 0x2E . */
    { 0x20, 0x10, 0x08, 0x04, 0x02 }, /* 0x2F / */
    { 0x3E, 0x51, 0x49, 0x45, 0x3E }, /* 0x30 0 */
    { 0x00, 0x42, 0x7F, 0x40, 0x00 }, /* 0x31 1 */
    { 0x42, 0x61, 0x51, 0x49, 0x46 }, /* 0x32 2 */
    { 0x21, 0x41, 0x45, 0x4B, 0x31 }, /* 0x33 3 */
    { 0x18, 0x14, 0x12, 0x7F, 0x10 }, /* 0x34 4 */
    { 0x27, 0x45, 0x45, 0x45, 0x39 }, /* 0x35 5 */
    { 0x3C, 0x4A, 0x49, 0x49, 0x30 }, /* 0x36 6 */
    { 0x01, 0x71, 0x09, 0x05, 0x03 }, /* 0x37 7 */
    { 0x36, 0x49, 0x49, 0x49, 0x36 }, /* 0x38 8 */
    { 0x06, 0x49, 0x49, 0x29, 0x1E }, /* 0x39 9 */
    { 0x00, 0x36, 0x36, 0x00, 0x00 }, /* 0x3A : */
    { 0x00, 0x56, 0x36, 0x00, 0x00 }, /* 0x3B ; */
    { 0x08, 0x14, 0x22, 0x41, 0x00 }, /* 0x3C < */
    { 0x14, 0x14, 0x14, 0x14, 0x14 }, /* 0x3D = */
    { 0x00, 0x41, 0x22, 0x14, 0x08 }, /* 0x3E > */
    { 0x02, 0x01, 0x51, 0x09, 0x06 }, /* 0x3F ? */
    { 0x32, 0x49, 0x79, 0x41, 0x3E }, /* 0x40 @ */
    { 0x7E, 0x11, 0x11, 0x11, 0x7E }, /* 0x41 A */
    { 0x7F, 0x49, 0x49, 0x49, 0x36 }, /* 0x42 B */
    { 0x3E, 0x41, 0x41, 0x41, 0x22 }, /* 0x43 C */
    { 0x7F, 0x41, 0x41, 0x22, 0x1C }, /* 0x44 D */
    { 0x7F, 0x49, 0x49, 0x49, 0x41 }, /* 0x45 E */
    { 0x7F, 0x09, 0x09, 0x09, 0x01 }, /* 0x46 F */
    { 0x3E, 0x41, 0x49, 0x49, 0x7A }, /* 0x47 G */
    { 0x7F, 0x08, 0x08, 0x08, 0x7F }, /* 0x48 H */
    { 0x00, 0x41, 0x7F, 0x41, 0x00 }, /* 0x49 I */
    { 0x20, 0x40, 0x41, 0x3F, 0x01 }, /* 0x4A J */
    { 0x7F, 0x08, 0x14, 0x22, 0x41 }, /* 0x4B K */
    { 0x7F, 0x40, 0x40, 0x40, 0x40 }, /* 0x4C L */
    { 0x7F, 0x02, 0x04, 0x02, 0x7F }, /* 0x4D M */
    { 0x7F, 0x04, 0x08, 0x10, 0x7F }, /* 0x4E N */
    { 0x3E, 0x41, 0x41, 0x41, 0x3E }, /* 0x4F O */
    { 0x7F, 0x09, 0x09, 0x09, 0x06 }, /* 0x50 P */
    { 0x3E, 0x41, 0x51, 0x21, 0x5E }, /* 0x51 Q */
    { 0x7F, 0x09, 0x19, 0x29, 0x46 }, /* 0x52 R */
    { 0x46, 0x49, 0x49, 0x49, 0x31 }, /* 0x53 S */
    { 0x01, 0x01, 0x7F, 0x01, 0x01 }, /* 0x54 T */
    { 0x3F, 0x40, 0x40, 0x40, 0x3F }, /* 0x55 U */
    { 0x1F, 0x20, 0x40, 0x20, 0x1F }, /* 0x56 V */
    { 0x3F, 0x40, 0x38, 0x40, 0x3F }, /* 0x57 W */
    { 0x63, 0x14, 0x08, 0x14, 0x63 }, /* 0x58 X */
    { 0x07, 0x08, 0x70, 0x08, 0x07 }, /* 0x59 Y */
    { 0x61, 0x51, 0x49, 0x45, 0x43 }, /* 0x5A Z */
    { 0x00, 0x7F, 0x41, 0x41, 0x00 }, /* 0x5B [ */
    { 0x02, 0x04, 0x08, 0x10, 0x20 }, /* 0x5C backslash */
    { 0x00, 0x41, 0x41, 0x7F, 0x00 }, /* 0x5D ] */
    { 0x04, 0x02, 0x01, 0x02, 0x04 }, /* 0x5E ^ */
    { 0x40, 0x40, 0x40, 0x40, 0x40 }, /* 0x5F _ */
    { 0x00, 0x01, 0x02, 0x04, 0x00 }, /* 0x60 ` */
    { 0x20, 0x54, 0x54, 0x54, 0x78 }, /* 0x61 a */
    { 0x7F, 0x48, 0x44, 0x44, 0x38 }, /* 0x62 b */
    { 0x38, 0x44, 0x44, 0x44, 0x20 }, /* 0x63 c */
    { 0x38, 0x44, 0x44, 0x48, 0x7F }, /* 0x64 d */
    { 0x38, 0x54, 0x54, 0x54, 0x18 }, /* 0x65 e */
    { 0x08, 0x7E, 0x09, 0x01, 0x02 }, /* 0x66 f */
    { 0x0C, 0x52, 0x52, 0x52, 0x3E }, /* 0x67 g */
    { 0x7F, 0x08, 0x04, 0x04, 0x78 }, /* 0x68 h */
    { 0x00, 0x44, 0x7D, 0x40, 0x00 }, /* 0x69 i */
    { 0x20, 0x40, 0x44, 0x3D, 0x00 }, /* 0x6A j */
    { 0x7F, 0x10, 0x28, 0x44, 0x00 }, /* 0x6B k */
    { 0x00, 0x41, 0x7F, 0x40, 0x00 }, /* 0x6C l */
    { 0x7C, 0x04, 0x18, 0x04, 0x78 }, /* 0x6D m */
    { 0x7C, 0x08, 0x04, 0x04, 0x78 }, /* 0x6E n */
    { 0x38, 0x44, 0x44, 0x44, 0x38 }, /* 0x6F o */
    { 0x7C, 0x14, 0x14, 0x14, 0x08 }, /* 0x70 p */
    { 0x08, 0x14, 0x14, 0x18, 0x7C }, /* 0x71 q */
    { 0x7C, 0x08, 0x04, 0x04, 0x08 }, /* 0x72 r */
    { 0x48, 0x54, 0x54, 0x54, 0x20 }, /* 0x73 s */
    { 0x04, 0x3F, 0x44, 0x40, 0x20 }, /* 0x74 t */
    { 0x3C, 0x40, 0x40, 0x20, 0x7C }, /* 0x75 u */
    { 0x1C, 0x20, 0x40, 0x20, 0x1C }, /* 0x76 v */
    { 0x3C, 0x40, 0x30, 0x40, 0x3C }, /* 0x77 w */
    { 0x44, 0x28, 0x10, 0x28, 0x44 }, /* 0x78 x */
    { 0x0C, 0x50, 0x50, 0x50, 0x3C }, /* 0x79 y */
    { 0x44, 0x64, 0x54, 0x4C, 0x44 }, /* 0x7A z */
    { 0x00, 0x08, 0x36, 0x41, 0x00 }, /* 0x7B { */
    { 0x00, 0x00, 0x7F, 0x00, 0x00 }, /* 0x7C | */
    { 0x00, 0x41, 0x36, 0x08, 0x00 }, /* 0x7D } */
    { 0x10, 0x08, 0x08, 0x10, 0x08 }, /* 0x7E ~ */
};

/**
 * Render one character at (page, col) on the OLED.
 * Returns the next column position (col + 6).
 * Each glyph = 5 pixel columns + 1 blank separator.
 */
static uint8_t OLED_PrintChar(uint8_t page, uint8_t col, char ch) {
    if (ch < 0x20 || ch > 0x7E) ch = ' ';
    const uint8_t *glyph = font5x7[(uint8_t)(ch - 0x20)];
    OLED_SetCursor(page, col);
    uint8_t buf[6] = { glyph[0], glyph[1], glyph[2], glyph[3], glyph[4], 0x00 };
    OLED_DataBuf(buf, 6);
    return (uint8_t)(col + 6);
}

/** Render a null-terminated string at (page, col). Clips at screen edge. */
void OLED_PrintStr(uint8_t page, uint8_t col, const char *str) {
    while (*str && col < OLED_WIDTH) {
        col = OLED_PrintChar(page, col, *str++);
    }
}

/* Per-page line cache: only characters that changed since the previous
 * refresh are written, so a routine update costs a few I2C bytes instead of
 * re-sending every glyph (which used to stall acquisition for ~70 ms). */
typedef struct {
    char prev[24];
    bool valid;
} OledLineCache;

static OledLineCache oled_cache[4];

static void OLED_PrintStrCached(uint8_t slot, uint8_t page, uint8_t col, const char *str) {
    OledLineCache *c = &oled_cache[slot];
    for (uint8_t i = 0; str[i] != '\0' && col < OLED_WIDTH && i < sizeof(c->prev); i++) {
        if (!c->valid || c->prev[i] != str[i]) {
            OLED_PrintChar(page, col, str[i]);
            c->prev[i] = str[i];
        }
        col = (uint8_t)(col + 6);
    }
    c->valid = true;
}

// Draw a thin horizontal separator line across the full page width. 
static void OLED_DrawLine(uint8_t page) {
    OLED_SetCursor(page, 0);
    uint8_t buf[OLED_WIDTH];
    memset(buf, 0x08, OLED_WIDTH);  /* Bit 3 = centre of 8-bit column */
    OLED_DataBuf(buf, OLED_WIDTH);
}


void OLED_ShowECGResult(void) {
    char line[24];
    char *p;

    // Page 3: Heart rate
    p = line;
    p = append_str(p, "HR: ");
    if (hr > 0.0f) {
        p = append_uint(p, (uint32_t)(hr + 0.5f), 3, ' ');
    } else {
        p = append_str(p, "---");
    }
    p = append_str(p, " bpm    ");
    *p = '\0';
    OLED_PrintStrCached(0, 3, 2, line);

    // Page 4: Total beat count
    p = line;
    p = append_str(p, "Beats: ");
    p = append_uint(p, watch_totalBeats, 4, ' ');
    p = append_str(p, "     ");
    *p = '\0';
    OLED_PrintStrCached(1, 4, 2, line);

    // Page 6: Classification label, space-padded to overwrite the previous one
    {
        const char *cn = ecg_cnn_class_name((int)watch_predClass);
        int n = 0;
        while (cn[n] != '\0' && n < 19) { line[n] = cn[n]; n++; }
        while (n < 19) line[n++] = ' ';
        line[n] = '\0';
        OLED_PrintStrCached(2, 6, 2, line);
    }

    // Page 7: Arrhythmia status
    if (arrhythmia) {
        OLED_PrintStrCached(3, 7, 2, "! ARRHYTHMIA !");
    } else {
        OLED_PrintStrCached(3, 7, 2, "  Rhythm OK    ");
    }
}


 //BIQUAD DIGITAL FILTERS
 //0.5 Hz HP + 40 Hz LP + 50 Hz Notch cascade

typedef struct { float b0, b1, b2, a1, a2, z1, z2; } Biquad;

/* Pre-computed biquad coefficients — eliminates cosf/sinf and libm entirely.
 * Values computed by: 2*pi*fc/256 -> cos/sin analytically, then normalised by a0=1+alpha.
 * To re-verify: python -c "import math; ..." (see ecg_training/train_rr.py firmware_sos()) */
static void biquad_set(Biquad *f, float b0, float b1, float b2, float a1, float a2) {
    f->b0 = b0; f->b1 = b1; f->b2 = b2;
    f->a1 = a1; f->a2 = a2;
    f->z1 = 0.0f; f->z2 = 0.0f;
}

float biquad_process(Biquad *f, float x) {
    float y = f->b0 * x + f->z1;
    f->z1   = f->b1 * x - f->a1 * y + f->z2;
    f->z2   = f->b2 * x - f->a2 * y;
    return y;
}

Biquad hp05, lp40, nt50, hp8, lp20;
void filters_init(void) {
    /* HP  0.5 Hz  Q=0.707  fs=256 */
    biquad_set(&hp05,  0.9913587464f, -1.9827174928f,  0.9913587464f, -1.9826428425f,  0.9827921431f);
    /* LP 40 Hz    Q=0.707  fs=256 */
    biquad_set(&lp40,  0.1399314618f,  0.2798629236f,  0.1399314618f, -0.6996989006f,  0.2594247477f);
    /* Notch 50 Hz Q=10.0   fs=256 */
    biquad_set(&nt50,  0.9550394153f, -0.6434861772f,  0.9550394153f, -0.6434861772f,  0.9100788306f);
    /* HP  8 Hz    Q=0.707  fs=256 */
    biquad_set(&hp8,   0.8703148444f, -1.7406296888f,  0.8703148444f, -1.7237446121f,  0.7575147655f);
    /* LP 20 Hz    Q=0.707  fs=256 */
    biquad_set(&lp20,  0.0442780368f,  0.0885560736f,  0.0442780368f, -1.3228374097f,  0.4999495569f);
}


//ELGENDI REAL-TIME QRS PEAK DETECTOR
#define ELG_W1 25
#define ELG_W2 156
static float zb1[64], zb2[256];
static float s1 = 0.0f, s2 = 0.0f, zmean = 0.0f, blkMax = 0.0f;
static int   i1 = 0, i2 = 0;
static long  nQ = 0, lastPeak = -1, blkStart = 0, blkIdx = 0;
static bool  inBlock = false;
static int   rrBuf[8];
static int   rrCount = 0, rrHead = 0;

long qrs_detect(float d) {
    float z = (d > 0.0f) ? (d * d) : 0.0f;
    s1 += z - zb1[i1]; zb1[i1] = z; if (++i1 >= ELG_W1) i1 = 0;
    s2 += z - zb2[i2]; zb2[i2] = z; if (++i2 >= ELG_W2) i2 = 0;
    zmean += (z - zmean) * 0.0001f;

    float mp  = s1 / (float)ELG_W1;
    float mb  = s2 / (float)ELG_W2;
    float thr = mb + 0.08f * zmean;
    long  res = -1;

    if (nQ > 768L) {
        if (mp > thr) {
            if (!inBlock) { inBlock = true; blkStart = nQ; blkMax = -1e9f; }
            if (d > blkMax) { blkMax = d; blkIdx = nQ; }
        } else if (inBlock) {
            inBlock = false;
            if ((nQ - blkStart) >= ELG_W1 &&
                (lastPeak < 0 || (blkIdx - lastPeak) > 51L)) {
                if (lastPeak >= 0) {
                    int rr = (int)(blkIdx - lastPeak);
                    rrBuf[rrHead] = rr;
                    rrHead = (rrHead + 1) % 8;
                    if (rrCount < 8) rrCount++;
                    if (rrCount == 8) {
                        float sum = 0.0f;
                        for (int i = 0; i < 8; i++) sum += (float)rrBuf[i];
                        hr = (60.0f * FS) / (sum / 8.0f);
                    }
                }
                lastPeak = blkIdx;
                res      = blkIdx;
            }
        }
    }
    nQ++;
    return res;
}

static float    ring_buf[RING_BUF_SIZE];
static uint32_t ring_head              = 0;
static long     peak_sample_pos        = -1;
static int      post_samples_remaining = -1;
static float    beat_window[BEAT_WINDOW_SIZE];
static int8_t   quantized_input[BEAT_WINDOW_SIZE];

void ring_buf_push(float sample) {
    ring_buf[ring_head] = sample;
    ring_head = (ring_head + 1) & RING_BUF_MASK;
}

float ring_buf_get_sample_at(long sample_index) {
    long offset = nQ - 1 - sample_index;
    if (offset < 0 || offset >= RING_BUF_SIZE) return 0.0f;
    uint32_t idx = (ring_head - 1 - (uint32_t)offset) & RING_BUF_MASK;
    return ring_buf[idx];
}


/* HC-05 alerts are queued and shifted out by the main loop. A full message
 * takes ~50 ms at 9600 baud, which used to stall acquisition past the
 * MAX30003 FIFO depth (32 samples = 125 ms). */
#define HC05_TX_SIZE 256U
static char     hc05TxBuf[HC05_TX_SIZE];
static uint16_t hc05TxHead = 0;
static uint16_t hc05TxTail = 0;

static void HC05_TxEnqueue(const char *s, uint16_t len) {
    uint16_t head = hc05TxHead;
    for (uint16_t i = 0; i < len; i++) {
        uint16_t next = (uint16_t)((head + 1U) & (HC05_TX_SIZE - 1U));
        if (next == hc05TxTail) break;          // full: drop the remainder
        hc05TxBuf[head] = s[i];
        head = next;
    }
    hc05TxHead = head;
}

static void HC05_TxDrain(void) {
    while (hc05TxTail != hc05TxHead && (USART1->SR & USART_SR_TXE)) {
        USART1->DR = (uint8_t)hc05TxBuf[hc05TxTail];
        hc05TxTail = (uint16_t)((hc05TxTail + 1U) & (HC05_TX_SIZE - 1U));
    }
}

void HC05_SendDiagnosticAlert(float hrVal, const char *clsStr, bool isArr);
void process_beat_inference(void) {
    // 1. Refine R-peak position on filtered signal (corrects QRS filter group delay) 
    long best_peak = peak_sample_pos;
    float max_abs_val = ECG_FABSF(ring_buf_get_sample_at(best_peak));
    for (long i = peak_sample_pos - 20; i <= peak_sample_pos; i++) {
        float v = ECG_FABSF(ring_buf_get_sample_at(i));
        if (v > max_abs_val) {
            max_abs_val = v;
            best_peak = i;
        }
    }

    // 2. Extract 250-sample beat window centred on refined R peak 
    for (int i = 0; i < BEAT_WINDOW_SIZE; i++) {
        long sample_idx = (best_peak - PRE_R_SAMPLES) + i;
        beat_window[i]  = ring_buf_get_sample_at(sample_idx);
    }

    // 2. Min-Max normalisation to [0.0, 1.0] - Paper Eq. 1 
    float min_val = beat_window[0];
    float max_val = beat_window[0];
    for (int i = 1; i < BEAT_WINDOW_SIZE; i++) {
        if (beat_window[i] < min_val) min_val = beat_window[i];
        if (beat_window[i] > max_val) max_val = beat_window[i];
    }
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;

    for (int i = 0; i < BEAT_WINDOW_SIZE; i++) {
        beat_window[i]     = (beat_window[i] - min_val) / range;
        quantized_input[i] = ecg_cnn_quantize_rr(beat_window[i]);
    }

    // 3. Hybrid INT8 CNN + RR inference (generated model in rr_model.h)
    float probs[ECG_NUM_CLASSES];
    uint8_t rr_abs = 128, rr_rel = 128;
    rr_on_beat(msTicks, &rr_abs, &rr_rel);
    uint32_t t0 = DWT->CYCCNT;
    int pred = ecg_cnn_predict_rr(quantized_input, rr_abs, rr_rel, probs);
    watch_inferenceUs = (DWT->CYCCNT - t0) / (SystemCoreClock / 1000000UL);

    // 4. Update watch variables
    watch_predClass = (ecg_class_t)pred;
    watch_className = ecg_cnn_class_name(pred);
    watch_prob_N    = probs[0];
    watch_prob_V    = probs[1];
    watch_prob_S    = probs[2];
    watch_totalBeats++;

    if (pred == ECG_CLASS_VENTRICULAR) {
        watch_ventBeats++;
        arrhythmia  = true;
        GPIOC->BSRR = (1UL << 13);           // LED OFF (active LOW)
    } else if (pred == ECG_CLASS_SUPRAVENTRICULAR) {
        watch_supraBeats++;
        arrhythmia  = true;
    } else {
        arrhythmia  = false;
        GPIOC->BSRR = (1UL << (13 + 16));    // LED ON (heartbeat pulse)
    }

    // 5. Send clean alert to phone via HC-05 
    HC05_SendDiagnosticAlert(hr, (const char *)watch_className, arrhythmia);

    // 6. Signal OLED update on next main loop iteration 
    oled_update_pending = true;
}


// Clean readable diagnostic alert sent to Mobile Phone Serial App
void HC05_SendDiagnosticAlert(float hrVal, const char *clsStr, bool isArr) {
    char  msg[96];
    char *p = msg;
    uint32_t hrInt = (hrVal > 0.0f) ? (uint32_t)(hrVal + 0.5f) : 0U;

    if (isArr) {
        p = append_str(p, "\r\n[ALERT] Arrhythmia Detected!\r\nType: ");
        p = append_str(p, clsStr);
        p = append_str(p, " | HR: ");
        p = append_uint(p, hrInt, 0, ' ');
        p = append_str(p, " BPM\r\n");
    } else {
        p = append_str(p, "\r\n[STATUS] Normal Sinus Rhythm | HR: ");
        p = append_uint(p, hrInt, 0, ' ');
        p = append_str(p, " BPM\r\n");
    }

    HC05_TxEnqueue(msg, (uint16_t)(p - msg));
}

// Fast high-speed telemetry sent ONLY to MATLAB on USART2 (COM4)
void StreamPacket_TinyML(uint32_t    timeMs,
                         float       rawMv,
                         float       filteredMv,
                         float       hrVal,
                         const char *clsStr,
                         bool        isArr,
                         uint32_t    infUs,
                         bool        leadOff,
                         uint32_t    overflows) {
    char  pkt[80];
    char *p = pkt;
    p = append_str(p, "ECG:");
    p = append_uint(p, timeMs, 0, ' ');
    *p++ = ',';
    p = append_fixed(p, rawMv, 3);
    *p++ = ',';
    p = append_fixed(p, filteredMv, 3);
    *p++ = ',';
    p = append_fixed(p, hrVal, 1);
    *p++ = ',';
    p = append_str(p, clsStr);
    *p++ = ',';
    *p++ = isArr ? '1' : '0';
    *p++ = ',';
    p = append_uint(p, infUs, 0, ' ');
    *p++ = ',';
    *p++ = leadOff ? '1' : '0';
    *p++ = ',';
    p = append_uint(p, overflows, 0, ' ');
    *p++ = '\r';
    *p++ = '\n';

    for (char *q = pkt; q < p; q++) {
        while (!(USART2->SR & USART_SR_TXE)) {}
        USART2->DR = (uint8_t)*q;
    }
}

/* USART2 RX ring buffer for MATLAB / PC ECG streaming */
#define UART2_RX_BUF_SZ 256U
static volatile char     uart2_rx_buf[UART2_RX_BUF_SZ];
static volatile uint16_t uart2_rx_head = 0;
static volatile uint16_t uart2_rx_tail = 0;
static uint32_t          last_uart_rx_tick = 0;
static bool              uart_stream_active = false;

void USART2_IRQHandler(void) {
    if (USART2->SR & USART_SR_RXNE) {
        char c = (char)(USART2->DR & 0xFFU);
        uint16_t next = (uint16_t)((uart2_rx_head + 1U) & (UART2_RX_BUF_SZ - 1U));
        if (next != uart2_rx_tail) {
            uart2_rx_buf[uart2_rx_head] = c;
            uart2_rx_head = next;
        }
    }
}

/* Lightweight float parser for "%.6f\n" lines (avoids pulling in ~15 KB strtof) */
static bool uart2_read_sample(float *out) {
    static char line[32];
    static uint8_t pos = 0;
    while (uart2_rx_tail != uart2_rx_head) {
        char c = uart2_rx_buf[uart2_rx_tail];
        uart2_rx_tail = (uint16_t)((uart2_rx_tail + 1U) & (UART2_RX_BUF_SZ - 1U));
        if (c == '\r') continue;
        if (c == '\n') {
            if (pos == 0) continue;
            line[pos] = '\0';
            pos = 0;
            const char *p = line;
            bool neg = false;
            if (*p == '-') { neg = true; p++; }
            else if (*p == '+') { p++; }
            if (*p < '0' || *p > '9') return false;
            float val = 0.0f;
            while (*p >= '0' && *p <= '9') { val = val * 10.0f + (float)(*p - '0'); p++; }
            if (*p == '.') {
                p++;
                float frac = 0.1f;
                while (*p >= '0' && *p <= '9') {
                    val += (float)(*p - '0') * frac;
                    frac *= 0.1f;
                    p++;
                }
            }
            *out = neg ? -val : val;
            return true;
        }
        if (pos < sizeof(line) - 1U) line[pos++] = c;
    }
    return false;
}

static void process_ecg_sample(float raw, bool from_max30003) {
    float y = biquad_process(&nt50,
                biquad_process(&lp40,
                  biquad_process(&hp05, raw)));

    // QRS detection band (8-20 Hz)
    float d = biquad_process(&lp20,
                biquad_process(&hp8, raw));

    // Push to segmentation ring buffer
    ring_buf_push(y);

    // R-peak detection (Elgendi algorithm)
    long r_peak_idx = qrs_detect(d);
    if (r_peak_idx >= 0) {
        peak_sample_pos        = r_peak_idx;
        post_samples_remaining = POST_R_SAMPLES;
    }

    // Collect 125 post-peak samples, then classify
    if (post_samples_remaining > 0) {
        post_samples_remaining--;
        if (post_samples_remaining == 0) {
            process_beat_inference();
        }
    }

    // Update Keil watch window scalars
    watch_rawMv      = raw;
    watch_filteredMv = y;
    watch_qrsBandMv  = d;

    uint32_t timeMs = (watch_sampleCount * 1000UL) / (unsigned long)FS;
    watch_sampleCount++;

    // Electrode contact check at ~4 Hz (only when reading MAX30003)
    if (from_max30003 && (watch_sampleCount & 0x3FU) == 0U) {
        uint32_t status = MAX30003_ReadReg(MAX30003_REG_STATUS);
        watch_leadOff = (status & MAX30003_STATUS_LOFF) != 0U;
    } else if (!from_max30003) {
        watch_leadOff = false;
    }

    // Telemetry at 128 Hz (every 2nd sample)
    if ((watch_sampleCount & 0x01U) == 0U) {
        StreamPacket_TinyML(
            timeMs, raw, y, hr,
            ecg_cnn_class_letter((int)watch_predClass),
            arrhythmia,
            watch_inferenceUs,
            watch_leadOff,
            watch_fifoOverflows
        );
    }

    HC05_TxDrain();
}

int main(void) {


    FPU_Init();                     // Enable Cortex-M4 FPU   
    DWT_Init();                     // Enable cycle counter
    SystemCoreClockUpdate();        // Resolve SystemCoreClock = 84 MHz                  
    SysTick_Init(SystemCoreClock);  // 1 ms SysTick  

    // Peripheral initialisation
    GPIO_Init_All();                // GPIO AF config for UART, I2C, SPI1, LED
    UART_Init_All(SystemCoreClock); // USART2 @ 115200 + USART1 @ 9600

    // Immediate UART boot heartbeat banner
    {
        const char *b2 = "\r\nSTM32 ECG MAX30003 Active\r\n";
        for (const char *p = b2; *p; p++) {
            while (!(USART2->SR & USART_SR_TXE)) {}
            USART2->DR = (uint8_t)*p;
        }
        const char *b1 = "\r\n[HC-05 BT] STM32 ECG Ready\r\n";
        for (const char *p = b1; *p; p++) {
            while (!(USART1->SR & USART_SR_TXE)) {}
            USART1->DR = (uint8_t)*p;
        }
    }

    I2C1_Init();                    // I2C1 @ 100 kHz for OLED (PB6/PB7)
    SPI1_Init();                    // SPI1 for MAX30003 (PA5/PA6/PA7, CS PB0)

    //DSP filter setup
    filters_init();

    //OLED boot sequence
    OLED_Init();
    OLED_Clear();

    //Splash screen
    OLED_PrintStr(0, 10, " STM32 ECG AI  ");
    OLED_PrintStr(1, 10, " MAX30003 AFE  ");
    OLED_DrawLine(2);
    OLED_PrintStr(3, 4,  "TinyML 1D-CNN  ");
    OLED_PrintStr(4, 4,  "INT8 Quantized ");
    OLED_DrawLine(5);
    OLED_PrintStr(6, 4,  "SPI1 Sensor    ");
    OLED_PrintStr(7, 4,  "ECG @ 256 Hz   ");
    delay_ms(2000);

    // Draw static portions of the result screen (redrawn only once) 
    OLED_Clear();
    OLED_PrintStr(0, 10, "  ECG Monitor  ");
    OLED_PrintStr(1, 4,  " STM32F401CCU6 ");
    OLED_DrawLine(2);
    OLED_PrintStr(3, 2,  "HR: --- bpm    ");
    OLED_PrintStr(4, 2,  "Beats:    0    ");
    OLED_DrawLine(5);
    OLED_PrintStr(6, 2,  "Starting ECG...");
    OLED_PrintStr(7, 2,  "               ");

    // Start the ECG stream last, so the FIFO does not fill while the
    // static screen is being drawn over the slow I2C bus.
    bool sensor_ok = MAX30003_Init();
    OLED_PrintStr(6, 2, sensor_ok ? "MAX30003 ready " : "MAX30003 absent");

    delay_ms(100);
    while (1) {
        HC05_TxDrain();

        // 1. Check for incoming MATLAB/PC ECG stream on USART2 (auto-disables MAX30003)
        float uart_raw = 0.0f;
        uint32_t burst = 0;
        while (burst < 32U && uart2_read_sample(&uart_raw)) {
            uart_stream_active = true;
            last_uart_rx_tick  = msTicks;
            process_ecg_sample(uart_raw, false);
            burst++;
        }

        // Fall back to MAX30003 if no UART samples arrived in the last 500 ms
        if (uart_stream_active && (msTicks - last_uart_rx_tick) > 500U) {
            uart_stream_active = false;
        }

        // 2. Otherwise drain the MAX30003 SPI FIFO
        if (!uart_stream_active) {
            int32_t adcSample;
            burst = 0;
            while (burst < 32U && MAX30003_DataReady() && MAX30003_ReadSample(&adcSample)) {
                float raw = (float)adcSample * ECG_SAMPLE_LSB_MV;
                process_ecg_sample(raw, true);
                burst++;
            }
        }
        //OLED refresh
        if (oled_update_pending) {
            oled_update_pending = false;
            OLED_ShowECGResult();   
        }

        // Sensor runs at 256 SPS; this bounds polling and idle CPU load
        delay_ms(1);
    }
}

