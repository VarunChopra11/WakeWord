idf_component_register(
    SRCS
        "src/audio_preprocessor.cpp"
        "src/fft_util.cpp"
        "src/mel_filterbank.cpp"
    INCLUDE_DIRS
        "include"
        "src"
    REQUIRES
        esp-dsp
        log
        heap
)

# Enable aggressive optimisation on the DSP-heavy files
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O3
    -ffast-math
    -DESP_DSP_NO_ASSERT
)
