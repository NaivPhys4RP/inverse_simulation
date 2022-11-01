project(rs_naivphs4r_inverse_learning)

rs_add_library(rs_naivphs4r_InverseLearner src/InverseLearner.cpp)
target_link_libraries(rs_naivphs4r_inverse_learning rs_core)

#if(X_FOUND)
# rs_add_library(rs_CaffeAnnotator src/CaffeAnnotator.cpp)
# target_link_libraries(rs_CaffeAnnotator rs_core rs_caffeProxy)
#endif()

rs_add_executable(InverseLearner src/InverseLearner.cpp)
target_link_libraries(InverseLearner rs_naivphs4r_inverse_learning ${catkin_LIBRARIES})
