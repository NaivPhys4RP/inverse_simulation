project(rs_naivphs4r_contextualization)

rs_add_library(rs_naivphs4r_Contextualizer src/Contextualizer.cpp)
target_link_libraries(rs_naivphs4r_Contextualizer rs_core)

#if(X_FOUND)
# rs_add_library(rs_CaffeAnnotator src/CaffeAnnotator.cpp)
# target_link_libraries(rs_CaffeAnnotator rs_core rs_caffeProxy)
#endif()

rs_add_executable(Contextualizer src/Contextualizer.cpp)
target_link_libraries(Contextualizer rs_naivphs4r_contextualization ${catkin_LIBRARIES})
