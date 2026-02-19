idf_component_register(
    SRCS
        "src/model_runner.cpp"
        "src/model_data.cc"
    INCLUDE_DIRS
        "include"
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
