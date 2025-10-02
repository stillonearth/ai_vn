#!/bin/bash

# Default values
REF_AUDIO=""
REF_TEXT=""
OUTPUT_FILE="./out.wav"
GEN_TEXT=""
MODEL_PATH="/home/cwiz/code/ai/aivn/F5-TTS/ckpts/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2/model_last_inference.safetensors"
VOCAB_PATH="/home/cwiz/code/ai/aivn/F5-TTS/ckpts/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -ref_audio)
            REF_AUDIO="$2"
            shift 2
            ;;
        -ref_text)
            REF_TEXT="$2"
            shift 2
            ;;
        -w)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -gen_text)
            GEN_TEXT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 -ref_audio <audio_file> -ref_text <text> -w <output_file> -gen_text <text>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$REF_AUDIO" ]]; then
    echo "Error: -ref_audio is required"
    exit 1
fi

if [[ -z "$REF_TEXT" ]]; then
    echo "Error: -ref_text is required"
    exit 1
fi

if [[ -z "$GEN_TEXT" ]]; then
    echo "Error: -gen_text is required"
    exit 1
fi

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate f5-tts

# Run F5-TTS inference
f5-tts_infer-cli \
    -p "$MODEL_PATH" \
    -v "$VOCAB_PATH" \
    --ref_audio "$REF_AUDIO" \
    --gen_text "$GEN_TEXT" \
    --ref_text "$REF_TEXT" \
    -w "$OUTPUT_FILE" \
    -o audio

# Deactivate conda environment
conda deactivate
