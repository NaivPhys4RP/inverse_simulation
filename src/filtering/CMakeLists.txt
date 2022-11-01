project(rs_naivphs4r_filtering)

rs_add_library(rs_naivphs4r_Filter src/Filter.cpp)
target_link_libraries(rs_naivphs4r_Filter rs_core)

#if(X_FOUND)
# rs_add_library(rs_CaffeAnnotator src/CaffeAnnotator.cpp)
# target_link_libraries(rs_CaffeAnnotator rs_core rs_caffeProxy)
#endif()

rs_add_executable(Filter src/Filter.cpp)
target_link_libraries(Filter rs_naivphs4r_filtering ${catkin_LIBRARIES})
