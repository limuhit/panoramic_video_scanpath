#include "main.hpp"
#include <torch/extension.h>
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    py::class_<projects_opt>(m,"ProjectsOp")
        .def(py::init<int, int , std::vector<float>, std::vector<float>, float, bool, int, bool>())
        .def("to", &projects_opt::to)
        .def("forward", &projects_opt::forward_cuda)
        .def("backward", &projects_opt::backward_cuda);
    

    py::class_<dtow_opt>(m,"DtowOp")
        .def(py::init<int, bool, int, bool>())
        .def("to", &dtow_opt::to)
        .def("forward", &dtow_opt::forward_cuda)
        .def("backward", &dtow_opt::backward_cuda);


    py::class_<viewport_opt>(m,"ViewportOp")
        .def(py::init<float, int, int, int, bool>())
        .def("to", &viewport_opt::to)
        .def("forward", &viewport_opt::forward_cuda)
        .def("backward", &viewport_opt::backward_cuda)
        .def("cal_rota_matrix",&viewport_opt::cal_rota_matrix)
        .def("get_viewport_xy", &viewport_opt::get_viewport_xy);

    py::class_<viewport_batch_opt>(m,"ViewportBatchOp")
        .def(py::init<float, int, int, int, int, int, int, bool>())
        .def("to", &viewport_batch_opt::to)
        .def("forward", &viewport_batch_opt::forward_cuda)
        .def("set_video", &viewport_batch_opt::set_video_sequence)
        .def("set_path", &viewport_batch_opt::set_path_sequence);

    py::class_<viewport_batch_eval_opt>(m,"ViewportBatchEvalOp")
        .def(py::init<float, int, int, int, int, int, int, bool>())
        .def("to", &viewport_batch_eval_opt::to)
        .def("forward", &viewport_batch_eval_opt::forward_cuda)
        .def("set_video", &viewport_batch_eval_opt::set_video_sequence);

    py::class_<InvTransSample_opt>(m,"InvtranssampleOp")
        .def(py::init< int, float, float, int, bool>())
        .def("to", &InvTransSample_opt::to)
        .def("forward", &InvTransSample_opt::forward_cuda)
        .def("select",&InvTransSample_opt::selcet_positions)
        .def("select_thr",&InvTransSample_opt::selcet_threshold);

    py::class_<erp2vp_opt>(m,"Erp2vpOp")
        .def(py::init<float, int, int, int, bool>())
        .def("to", &erp2vp_opt::to)
        .def("forward", &erp2vp_opt::forward_cuda)
        .def("backward", &erp2vp_opt::backward_cuda);

    py::class_<GMM_2D_Table_opt>(m,"Gmm2dTableOp")
        .def(py::init<int, int , int , float , float, int, bool>())
        .def("to", &GMM_2D_Table_opt::to)
        .def("forward", &GMM_2D_Table_opt::forward_cuda)
        .def("backward", &GMM_2D_Table_opt::backward_cuda);

    py::class_<vp2erp_opt>(m,"Vp2erpOp")
        .def(py::init<float, int, int, int, bool>())
        .def("to", &vp2erp_opt::to)
        .def("forward", &vp2erp_opt::forward_cuda);

    py::class_<linear_mask_opt>(m,"LinearMaskOp")
        .def(py::init<int, int, int, int, bool, bool, int, bool>())
        .def("to", &linear_mask_opt::to)
        .def("forward", &linear_mask_opt::forward_cuda);

    py::class_<pre_data_opt>(m,"PreDataOp")
        .def(py::init<float,float, int,int, int, int, int, bool, int, bool>())
        .def("to", &pre_data_opt::to)
        .def("forward", &pre_data_opt::forward_cuda)
        .def("forward2", &pre_data_opt::forward_cuda2);

    py::class_<data_manager_opt>(m,"DataManagerOp")
        .def(py::init<int, int, bool>())
        .def("to", &data_manager_opt::to)
        .def("forward", &data_manager_opt::forward_cuda)
        .def("clear",&data_manager_opt::clear_tensors)
        .def("push",&data_manager_opt::push_tensor);

    py::class_<gmm_sample_opt>(m,"GmmSampleOp")
        .def(py::init<int, float, int, bool>())
        .def("to", &gmm_sample_opt::to)
        .def("forward", &gmm_sample_opt::forward_cuda)
        .def("select",&gmm_sample_opt::select);
};