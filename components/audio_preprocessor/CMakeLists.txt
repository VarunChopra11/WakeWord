idf_component_register(
    SRCS
        "src/audio_preprocessor.cpp"
    INCLUDE_DIRS
        "include"
    REQUIRES
        micro_speech_features
        log
)

# Enable aggressive optimisation
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O3
    -ffast-math
)
