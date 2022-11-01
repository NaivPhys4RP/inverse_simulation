project(rs_naivphs4r_imagination)

rs_add_library(rs_naivphs4r_Imaginator src/Imaginator.cpp)
target_link_libraries(rs_naivphs4r_Imaginator rs_core)

#if(X_FOUND)
# rs_add_library(rs_CaffeAnnotator src/CaffeAnnotator.cpp)
# target_link_libraries(rs_CaffeAnnotator rs_core rs_caffeProxy)
#endif()

rs_add_executable(Imaginator src/Imaginator.cpp)
target_link_libraries(Imaginator rs_naivphs4r_imagination ${catkin_LIBRARIES})
