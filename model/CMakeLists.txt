idf_component_register(
    SRCS
        "model_runner.cpp"
        "model_data.cc"
    INCLUDE_DIRS
        "."
    REQUIRES
        esp-tflite-micro
        log
        heap
        esp_psram
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -O2
    -DESP_NN
)
