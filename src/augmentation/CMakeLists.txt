project(rs_naivphs4r_augmentation)

rs_add_library(rs_naivphs4r_Augmenter src/Augmenter.cpp)
target_link_libraries(rs_naivphs4r_Augmenter rs_core)

#if(X_FOUND)
# rs_add_library(rs_CaffeAnnotator src/CaffeAnnotator.cpp)
# target_link_libraries(rs_CaffeAnnotator rs_core rs_caffeProxy)
#endif()

rs_add_executable(Augmenter src/Augmenter.cpp)
target_link_libraries(Augmenter rs_naivphs4r_augmentation ${catkin_LIBRARIES})
