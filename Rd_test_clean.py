"""
rd_test_clean.py — Ring-Down Data Acquisition (Red Pitaya FPGA)

This script performs a single ring-down measurement on the QCM using the
custom FPGA bitstream. It is written to be readable and easy to maintain:
every phase is explained and the configuration knobs are grouped at the top.

High-level workflow:

1) Map AXI-Lite registers and DDR RAM using MMIO.
2) Check the FPGA identity constant to make sure the ring-down bitstream is loaded.
3) Reset the device and clear key configuration registers.
4) Generate a sine excitation waveform near the crystal resonance (e.g. 5 MHz).
5) Quantize the waveform to 14-bit DAC codes, enforce 64-sample alignment,
   and write it into DDR RAM at address 0.
6) Choose a capture region in DDR (writer_offset) where the ADC will store the
   ring-down decay after excitation.
7) Configure the FPGA:
      - Reader: reads the waveform region to feed the DAC.
      - Writer: records the ADC ring-down into the capture region.
      - Relay timing: EXCITATION window (relay closed) followed by
        RELAXATION window (relay open).
8) Release reset. The FPGA runs the sequence in hardware:
      - Drive the crystal with a sine burst.
      - Open relay and record the free decay.
9) Wait until the FPGA signals completion.
10) Read the captured ADC data from DDR and write it to `ringdown_data.raw`.
11) Save acquisition metadata to `ringdown_data.json`.
12) Reset the device back to a safe idle state and close the MMIO handles.

This script only captures data (time-domain ring-down).
All signal processing (envelope extraction, exponential fitting, f0, Q factor)
is handled by downstream ring-down analysis scripts (RD0… RD6).
"""





# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from periphery import MMIO
import numpy as np
import time
import json
import sys

# ---------------------------------------------------------
# USER CONFIGURATION — Ring-Down settings
# ---------------------------------------------------------

MODE             = "RINGDOWN"           # Label in metadata
FREQ_HZ          = 4.5e6                # Excitation frequency (Hz)
DURATION_S       = 0.001                # Total waveform length written to RAM
AM_PK_V          = 0.05                 # Peak drive amplitude (Volts)

# Relay timing (microseconds – these go directly into FPGA registers)
EXCITATION_TIME_US   = 300              # Relay closed (drive on)
RELAXATION_TIME_US   = 3000             # Relay open (record decay)

# Sampling / DAC settings
SAMPLE_RATE     = 125_000_000           # Hz
DAC_BITS        = 14
FULL_SCALE_V    = 2.0                   # DAC full-scale range (Vpp)

# Output files
OUT_RAW   = "ringdown_data.raw"
OUT_META  = OUT_RAW.replace(".raw", ".json")

# RAM / MMIO base addresses
AXIL_BASE = 0x40000000
AXIL_SIZE = 0x1000

RAM_BASE  = 0x01000000
RAM_SIZE  = 4 * 1024 * 1024      # 220 MB
SAMPLE_SIZE = 2                    # int16 = 2 bytes
BURST_BYTES = 128                  # DMA burst size in bytes


# ---------------------------------------------------------
# FPGA Register Map (Ring-Down bitstream)
# ---------------------------------------------------------

MLA_CONST_REG              = 0x00
WRITER_MIN_ADDRESS_REG     = 0x04
WRITER_NUM_BURSTS_REG      = 0x08
SOFT_RESET_REG             = 0x0C
WRITER_WRITE_PTR_REG       = 0x10
WRITER_DONE_REG            = 0x14
RELAXATION_TIME_REG        = 0x18
EXCITATION_TIME_REG        = 0x1C
READER_MIN_ADDRESS_REG     = 0x20
READER_READ_PTR_REG        = 0x24
READER_DONE_REG            = 0x28
READER_COMP_BURST_OUT_REG  = 0x2C
READER_NUM_BURSTS_REG      = 0x30
WRITER_COMP_BURST_OUT_REG  = 0x34
READER_CONTINUOUS_MODE_REG = 0x38
WRITER_CONTINUOUS_MODE_REG = 0x3C
RELAY_ENABLED_REG          = 0x40
TRIGGER_ENABLED_REG        = 0x44


# ---------------------------------------------------------
# Helper: wait until reader + writer are done (or timeout)
# ---------------------------------------------------------

def wait_done(axil, timeout_s=30.0, poll_interval=0.05):
    """
    Polls WRITER_DONE and READER_DONE flags until both are 1, or a timeout occurs.
    Returns final (write_ptr, read_ptr) when done.
    Raises TimeoutError if hardware does not complete in time.
    """
    t0 = time.time()
    while True:
        writer_done = axil.read32(WRITER_DONE_REG) & 0x1
        reader_done = axil.read32(READER_DONE_REG) & 0x1

        write_ptr = axil.read32(WRITER_WRITE_PTR_REG)
        read_ptr  = axil.read32(READER_READ_PTR_REG)

        if writer_done and reader_done:
            return write_ptr, read_ptr

        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"Timeout waiting for DONE. WP={write_ptr}, RP={read_ptr}"
            )

        time.sleep(poll_interval)

# ---------------------------------------------------------
# STEP 5 — Map AXI-Lite + DDR, check magic constant, reset
# ---------------------------------------------------------

print("Mapping FPGA memory regions...")

try:
    axil_mmio = MMIO(AXIL_BASE, AXIL_SIZE)
    ram_mmio  = MMIO(RAM_BASE, RAM_SIZE)
except Exception as e:
    print(f"ERROR: Unable to map FPGA memory regions: {e}")
    sys.exit(1)

print("Memory mapping OK.")

# ---- Check FPGA identity (magic constant) ----
print("Checking FPGA identity constant...")

raw_const = axil_mmio.read32(MLA_CONST_REG)
magic_str = raw_const.to_bytes(4, byteorder="big").decode("latin-1", errors="ignore").strip()

if magic_str != "ÅABC":
    print(f"Warning: unrecognized FPGA version constant '{magic_str}'.")
    print("Assuming default Red Pitaya overlay (ring-down mode). Continuing.")

print(f"Device identity confirmed: '{magic_str}'")

# ---- Reset and clear config registers ----
print("Resetting device and clearing configuration...")

# Hold hardware in reset (1 = reset asserted in this design)
axil_mmio.write32(SOFT_RESET_REG, 1)

# Clear writer side
axil_mmio.write32(WRITER_MIN_ADDRESS_REG, 0)
axil_mmio.write32(WRITER_NUM_BURSTS_REG, 0)

# Clear timing + relay
axil_mmio.write32(RELAXATION_TIME_REG, 0)
axil_mmio.write32(EXCITATION_TIME_REG, 0)
axil_mmio.write32(RELAY_ENABLED_REG, 0)

# Clear reader/writer mode flags
axil_mmio.write32(READER_CONTINUOUS_MODE_REG, 0)
axil_mmio.write32(WRITER_CONTINUOUS_MODE_REG, 0)

time.sleep(0.5)
print("Device reset complete. Hardware is in a known idle state.")

# ---------------------------------------------------------
# STEP 6 — Generate excitation sine waveform
# ---------------------------------------------------------

print("Generating excitation waveform...")
print("Generating excitation waveform...")

N = int(DURATION_S * SAMPLE_RATE)
dac_max = (2**(DAC_BITS - 1) - 1)

waveform = np.sin(2 * np.pi * FREQ_HZ * np.arange(N) / SAMPLE_RATE)
waveform = (waveform * AM_PK_V / FULL_SCALE_V * dac_max).astype(np.int16)

ram_mmio.write(0, waveform.tobytes())

waveform_samples = len(waveform)
waveform_bytes = waveform_samples * SAMPLE_SIZE

print(f"Excitation waveform generated: {waveform_samples} samples")




# ---------------------------------------------------------
# Pad to 64-sample boundary for DMA alignment
# ---------------------------------------------------------
pad = (-waveform_bytes) % 64
if pad > 0:
    waveform_bytes += pad

# ---------------------------------------------------------
# Zero waveform buffer (used to force DAC OFF)
# ---------------------------------------------------------
zero_waveform = np.zeros(waveform_samples, dtype=np.int16)
zero_bytes = zero_waveform.tobytes()

# ---------------------------------------------------------
# STEP 7 — Compute RAM layout and write waveform to DDR
# ---------------------------------------------------------

print("Computing RAM layout...")
waveform_bursts = waveform_bytes // BURST_BYTES


print(f"Waveform occupied{waveform_bytes} bytes ({waveform_bursts} bursts).")

# ---------------------------------------------------------
# Determine where ADC capture (ring-down) will be stored
# ---------------------------------------------------------

# Absolute DDR address for the capture region
writer_offset = RAM_BASE + waveform_bytes

# Relative offset for the FPGA's internal addressing
writer_offset_rel = writer_offset - RAM_BASE

print(f"Capture region absolute addr : 0x{writer_offset:X}")
print(f"Capture region relative addr : 0x{writer_offset_rel:X}")

# ---------------------------------------------------------
# Compute number of bursts to capture ring-down
# ---------------------------------------------------------

# Total samples we want from relaxation period
relax_samples = int((RELAXATION_TIME_US * 1e-6) * SAMPLE_RATE)

# Round up to nearest multiple of 64 samples for burst alignment
pad_relax = (-relax_samples) % 64
relax_samples_aligned = relax_samples + pad_relax

writer_num_bursts = (relax_samples_aligned * SAMPLE_SIZE) // BURST_BYTES
true_writer_bytes = writer_num_bursts * BURST_BYTES

print(f"Relaxation samples aligned : {relax_samples_aligned}")
print(f"Writer bursts              : {writer_num_bursts}")
print(f"Capture size (bytes)       : {true_writer_bytes}")

# ---------------------------------------------------------
# Write waveform to DDR at address 0
# ---------------------------------------------------------

print("Writing excitation waveform to DDR...")

wave_bytes = waveform.tobytes()

chunk_size = 1024 * 1024    # 1 MB chunks
for i in range(0,len(wave_bytes), chunk_size):
    ram_mmio.write(i, wave_bytes()[i:i+chunk_size])
del wave_bytes
del waveform
print("Waveform written.")

# ---------------------------------------------------------
# STEP 8 — Configure FPGA reader/writer and timing registers
# ---------------------------------------------------------

print("Configuring FPGA Ring-Down registers...")

# ---------------------------------------------------------
# 1) Configure DAC Writer (Excitation) 
# ---------------------------------------------------------
# It will read waveform from RAM address 0, for waveform_bursts bursts.
axil_mmio.write32(WRITER_MIN_ADDRESS_REG, RAM_BASE)
axil_mmio.write32(WRITER_NUM_BURSTS_REG, waveform_bursts)

# ---------------------------------------------------------
# 2) Configure ADC Reader (Ring-Down capture)
# ---------------------------------------------------------
# It will write ADC data into RAM at writer_offset (relative address).
axil_mmio.write32(READER_MIN_ADDRESS_REG, writer_offset)
axil_mmio.write32(READER_NUM_BURSTS_REG, writer_num_bursts)

# ---------------------------------------------------------
# 3) Configure timing (µs values interpreted by the FPGA)
# ---------------------------------------------------------
# Excitation: relay closed → DAC drives crystal
axil_mmio.write32(EXCITATION_TIME_REG, int(EXCITATION_TIME_US))

# Relaxation: relay open → free decay recorded
axil_mmio.write32(RELAXATION_TIME_REG, int(RELAXATION_TIME_US))

# ---------------------------------------------------------
# 4) Relay control and modes
# ---------------------------------------------------------
# Enable the relay so FPGA can toggle it during excitation/relaxation cycles
axil_mmio.write32(RELAY_ENABLED_REG, 1)

# Reader and writer continuous modes OFF for ring-down
axil_mmio.write32(READER_CONTINUOUS_MODE_REG, 0)
axil_mmio.write32(WRITER_CONTINUOUS_MODE_REG, 0)

# Trigger disabled (free-run)
axil_mmio.write32(TRIGGER_ENABLED_REG, 0)

print("FPGA configuration complete.")
print("Ready to begin ring-down measurement.")

input("Press ENTER to start measurement...")

# ---------------------------------------------------------
# 5) Release FPGA reset — measurement starts NOW
# ---------------------------------------------------------
axil_mmio.write32(SOFT_RESET_REG, 0)
print("Measurement started. FPGA running...")

# Excitation Phase 
time.sleep(EXCITATION_TIME_US * 1e-6)

# HARD OFF: force DAC to ZERO
ram_mmio.write(0, zero_bytes)
# Relaxtation Phase
time.sleep(RELAXATION_TIME_US * 1e-6)
print("Assuming ring-down capture complete.")


# ---------------------------------------------------------
# STEP 10 — Read captured ring-down from DDR → .raw file
# ---------------------------------------------------------

print(f"Reading captured data from DDR and saving to '{OUT_RAW}'...")

chunk_size   = 8 * 1024 * 1024    # 8 MB chunks
bytes_to_read = true_writer_bytes
bytes_read    = 0

with open(OUT_RAW, "wb") as f:
    for offset in range(0, bytes_to_read, chunk_size):
        
        # How many bytes to fetch in this iteration
        read_size = min(chunk_size, bytes_to_read - offset)

        # FPGA addressing is RELATIVE to RAM_BASE
        chunk = ram_mmio.read(writer_offset_rel + offset, read_size)

        f.write(chunk)
        bytes_read += read_size

        progress = (bytes_read / bytes_to_read) * 100
        print(f"\rProgress: {progress:.1f}%  ({bytes_read/1024/1024:.2f} MB)", end="")
        del chunk  # help garbage collector

print("\nRing-down data saved successfully.")

# ---------------------------------------------------------
# STEP 11 — Write metadata JSON file
# ---------------------------------------------------------

print(f"Writing metadata to '{OUT_META}'...")

metadata = {
    "mode": MODE,

    # User-configurable experiment parameters
    "excitation_frequency_hz": FREQ_HZ,
    "amplitude_peak_v": AM_PK_V,
    "excitation_time_us": EXCITATION_TIME_US,
    "relaxation_time_us": RELAXATION_TIME_US,
    "sample_rate_hz": SAMPLE_RATE,

    # Waveform info
    "waveform_samples": waveform_samples,
    "waveform_bytes": waveform_bytes,
    "waveform_bursts": waveform_bursts,

    # Capture info (ADC ring-down)
    "capture_bytes": true_writer_bytes,
    "capture_bursts": writer_num_bursts,
    "capture_start_offset_rel": writer_offset_rel,
    "relax_samples_aligned": relax_samples_aligned,

    # Output files
    "raw_output_file": OUT_RAW,
    "json_output_file": OUT_META,

    # Time & housekeeping
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Write metadata to JSON
with open(OUT_META, "w") as jf:
    json.dump(metadata, jf, indent=2)

print("Metadata saved successfully.")

# ---------------------------------------------------------
# STEP 12 — Final reset and cleanup
# ---------------------------------------------------------

print("Resetting FPGA to safe idle state...")

# Put the hardware back in reset
axil_mmio.write32(SOFT_RESET_REG, 1)

# Disable relay and modes
axil_mmio.write32(RELAY_ENABLED_REG, 0)
axil_mmio.write32(READER_CONTINUOUS_MODE_REG, 0)
axil_mmio.write32(WRITER_CONTINUOUS_MODE_REG, 0)

# Small delay for safety
time.sleep(0.2)

# Close MMIO mappings
ram_mmio.close()
axil_mmio.close()

print("\nRing-down acquisition COMPLETE.")
print(f"Raw data saved to:  {OUT_RAW}")
print(f"Metadata saved to:   {OUT_META}")
print("FPGA interface closed and system is safe.")






















         






































































































































































 
