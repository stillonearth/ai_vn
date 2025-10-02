#!/bin/bash

# Default values
CHAR=""
REF_AUDIO=""
REF_TEXT=""
OUTPUT_FILE="./out.wav"
GEN_TEXT=""
MODEL_PATH="/home/cwiz/code/ai/aivn/F5-TTS/ckpts/F5-TTS_RUSSIAN/F5TTS_v1_Base_v2/model_last_inference.safetensors"
VOCAB_PATH="/home/cwiz/code/ai/aivn/F5-TTS/ckpts/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -char)
            CHAR="$2"
            shift 2
            ;;
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
        -model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -vocab_path)
            VOCAB_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 -char <character_name> -w <output_file> -gen_text <text> [-model_path <model_path>] [-vocab_path <vocab_path>]"
            echo "       OR"
            echo "       $0 -ref_audio <audio_file> -ref_text <text> -w <output_file> -gen_text <text> [-model_path <model_path>] [-vocab_path <vocab_path>]"
            exit 1
            ;;
    esac
done

# Validate arguments for character mode
if [[ -n "$CHAR" ]]; then
    REF_AUDIO="characters/${CHAR}.mp3"
    REF_TEXT_FILE="characters/${CHAR}.txt"

    # Check if character files exist
    if [[ ! -f "$REF_AUDIO" ]]; then
        echo "Error: Character audio file not found: $REF_AUDIO"
        exit 1
    fi

    if [[ ! -f "$REF_TEXT_FILE" ]]; then
        echo "Error: Character text file not found: $REF_TEXT_FILE"
        exit 1
    fi

    # Read reference text from file
    REF_TEXT=$(cat "$REF_TEXT_FILE")
    echo "Using character: $CHAR"
    echo "Reference audio: $REF_AUDIO"
    echo "Reference text: $REF_TEXT"
fi

# Validate required arguments
if [[ -z "$REF_AUDIO" ]]; then
    echo "Error: Either -char or -ref_audio is required"
    exit 1
fi

if [[ -z "$REF_TEXT" ]]; then
    echo "Error: Either -char or -ref_text is required"
    exit 1
fi

if [[ -z "$GEN_TEXT" ]]; then
    echo "Error: -gen_text is required"
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: -w (output file) is required"
    exit 1
fi

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate f5-tts

# Run F5-TTS inference
echo "Generating audio..."
f5-tts_infer-cli \
    -p "$MODEL_PATH" \
    -v "$VOCAB_PATH" \
    --ref_audio "$REF_AUDIO" \
    --gen_text "$GEN_TEXT" \
    --ref_text "$REF_TEXT" \
    -w "$OUTPUT_FILE" \
    -o audio

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo "Audio successfully generated: $OUTPUT_FILE"
else
    echo "Error: Audio generation failed"
    exit 1
fi

# Deactivate conda environment
conda deactivate
