if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/model_data.cc")
    message(FATAL_ERROR "Generate an INT8 model bundle into main/generated before building")
endif()
idf_component_register(SRCS "main.cc" "generated/model_data.cc"
                       INCLUDE_DIRS "." "generated"
                       REQUIRES esp_timer heap esp_system)
