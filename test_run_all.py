# we run all the tests in one go to save time
from test_eval_f import TestEvalF
from test_eval_f_find_omega import main as sweep_omegas
from test_eval_f_real_world_data import TestEvalF_RealWorldData
from test_jacobian import Test_jacobian
from test_jacobian_two_methods import JacobianComparator
from test_visualizeNetwork import TestVisualizeNetwork

if __name__ == "__main__":

    # test_eval_f.py
    test_eval_f = TestEvalF()
    test_eval_f.run_all_tests(w=1, num_iter=84)

    # test_eval_f_find_omega.py
    sweep_omegas()

    # test_eval_f_real_world_data.py
    test_eval_f_real_world = TestEvalF_RealWorldData()
    test_eval_f_real_world.test_real_world_data(
        w=0.1,
        num_iter=100,
        data_path="data/fake_spatial_data_tumor_int.npy",
    )

    # test_jacobian.py
    test_instance = Test_jacobian()
    test_instance.setup_method()    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    total_tests = len(test_methods)
    failed_tests = 0
    
    for test_method in test_methods:
        try:
            getattr(test_instance, test_method)()
            print(f"✓ {test_method}")
        except Exception as e:
            print(f"✗ {test_method}: {e}")
            failed_tests += 1

    print("Tests completed!")
    print(f"Successful Jacobian tests: {total_tests - failed_tests}, Failed tests: {failed_tests}; Success rate: {(total_tests - failed_tests) / total_tests:.2%}")

    # test_jacobian_two_methods.py
    jacobian_comparator = JacobianComparator()
    info = jacobian_comparator.run()  # default dx grid
    info = jacobian_comparator.run()  # default dx grid
    path = jacobian_comparator.plot(show=False)
    print(f"Saved plot to: {path}")
    print(f"dx* (optimal): {info['dx_optimal']:.3e} | dx_machine: {info['dx_machine']:.3e} | dx_nitsol: {info['dx_nitsol']:.3e}")

    # test_visualizeNetwork.py
    test_visualize_network = TestVisualizeNetwork()
    test_visualize_network.setUp()
    test_visualize_network.test_saves_single_combined_image()
    test_visualize_network.test_bad_shape_raises()
    test_visualize_network.test_persistent_output_dir()
    test_visualize_network.test_fake_data_visualize()
    print("All tests in test_visualizeNetwork.py passed.")