% Uses: Built-in websave() + Signal Processing Toolbox (resample) + serialport()
clear; clc; close all;

% --- Configuration ---
COM_PORT   = "COM4";
BAUD_RATE  = 115200;
ORIG_FS    = 360;        % Native MIT-BIH sampling rate (Hz)
TARGET_FS  = 256;        % Sampling rate expected by STM32 (Hz)
SAMPLE_T   = 1 / TARGET_FS;
DAT_FILE   = 'mitdb_213.dat';
DAT_URL    = 'https://physionet.org/files/mitdb/1.0.0/213.dat';
WINDOW_SEC = 10;         % Live scrolling window (seconds)
PLOT_EVERY = 16;         % Refresh plot every 16 samples (~16 FPS)

% --- 1. Download & Decode Real MIT-BIH Record 213 (Lead MLII) ---
if ~isfile(DAT_FILE)
    fprintf('[PhysioNet] Downloading real MIT-BIH Record 213 (~1.9 MB)...\n');
    websave(DAT_FILE, DAT_URL);
end

fprintf('[Decode] Reading MIT-BIH 212 format...\n');
fid = fopen(DAT_FILE, 'rb');
raw_bytes = fread(fid, [3, inf], 'uint8');
fclose(fid);

% Decode Channel 1 (Lead MLII) from 3-byte MIT-BIH 212 packets
b1 = raw_bytes(1, :);
b2 = raw_bytes(2, :);
ch1 = b1 + bitshift(bitand(b2, 15), 8);
ch1(ch1 >= 2048) = ch1(ch1 >= 2048) - 4096;  % 12-bit signed conversion

% Convert ADC units to mV (MIT-BIH standard: baseline=1024, gain=200 ADC/mV)
raw_mv = (ch1 - 1024) / 200;

% Resample from 360 Hz to 256 Hz
fprintf('[DSP] Resampling from %d Hz to %d Hz...\n', ORIG_FS, TARGET_FS);
ecg_signal = resample(raw_mv, TARGET_FS, ORIG_FS);
N = length(ecg_signal);
fprintf('[Ready] Loaded %.1f minutes (%d samples) of real MIT-BIH ECG.\n', (N/TARGET_FS)/60, N);

% --- 2. Setup Live Scrolling Plot ---
fig = figure('Name', 'Live MIT-BIH Record 213 -> STM32', ...
             'Color', [0.08 0.08 0.08], 'Position', [100, 100, 1100, 400]);
ax = axes('Parent', fig, 'Color', [0.05 0.05 0.05], ...
          'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], ...
          'GridColor', [0.3 0.3 0.3]);
hold(ax, 'on'); grid(ax, 'on');
ylim(ax, [prctile(ecg_signal, 0.5) - 0.3, prctile(ecg_signal, 99.5) + 0.3]);
xlabel(ax, 'Time (s)'); ylabel(ax, 'Amplitude (mV)');
title(ax, sprintf('Live MIT-BIH Record 213 Streaming to STM32 (%d Hz)', TARGET_FS), ...
      'Color', [0.2 1.0 0.4]);

win_len = WINDOW_SEC * TARGET_FS;
t_buf   = nan(1, win_len);
y_buf   = nan(1, win_len);
h_line  = plot(ax, t_buf, y_buf, 'Color', [0.0 0.9 1.0], 'LineWidth', 1.3);

% --- 3. Connect to STM32 via UART ---
fprintf('[UART] Connecting to STM32 on %s @ %d baud...\n', COM_PORT, BAUD_RATE);
ser = [];
try
    ser = serialport(COM_PORT, BAUD_RATE, "Timeout", 0.02);
    configureTerminator(ser, "LF");   % Appends '\n' automatically
    flush(ser);
    fprintf('[UART] Connected on %s!\n', COM_PORT);
catch ME
    fprintf('[Warning] %s unavailable (%s). Running live plot without UART.\n', COM_PORT, ME.message);
end
cleanupObj = onCleanup(@() clear("ser"));

% --- 4. Stream Numeric Values Continuously at 256 Hz ---
fprintf('[Live] Streaming numeric ECG values continuously @ %d Hz... (Close figure or Ctrl+C to stop)\n', TARGET_FS);

idx = 1;
sample_count = 0;
t_start = tic;
next_tx = toc(t_start);

while ishandle(fig)
    if toc(t_start) >= next_tx
        val   = ecg_signal(idx);
        t_now = sample_count / TARGET_FS;

        % Send numeric value formatted identically to ecg_bridge.py ("%.6f\n")
        if ~isempty(ser)
            writeline(ser, sprintf('%.6f', val));
        end

        % Update live scrolling plot
        t_buf = [t_buf(2:end), t_now];
        y_buf = [y_buf(2:end), val];
        if mod(sample_count, PLOT_EVERY) == 0
            set(h_line, 'XData', t_buf, 'YData', y_buf);
            xlim(ax, [max(0, t_now - WINDOW_SEC), max(WINDOW_SEC, t_now)]);
            drawnow limitrate;
        end

        idx = idx + 1;
        if idx > N
            idx = 1; % Loop full 30-minute recording continuously
        end
        sample_count = sample_count + 1;
        next_tx = next_tx + SAMPLE_T;
    end

    % Read returned telemetry from STM32
    if ~isempty(ser) && ser.NumBytesAvailable > 0
        raw_rx = readline(ser);
        if (isstring(raw_rx) || ischar(raw_rx)) && strlength(raw_rx) > 0
            rx_line = strtrim(raw_rx);
            if startsWith(rx_line, "ECG:")
                fprintf('[STM32] %s\n', rx_line);
            end
        end
    end
end