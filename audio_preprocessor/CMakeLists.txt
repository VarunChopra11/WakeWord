idf_component_register(
    SRCS
        "audio_preprocessor.cpp"
    INCLUDE_DIRS
        "."
    REQUIRES
        log
        espressif__esp-tflite-micro
)

# Enable aggressive optimisation
target_compile_options(${COMPONENT_LIB} PRIVATE
    -O3
    -ffast-math
)
