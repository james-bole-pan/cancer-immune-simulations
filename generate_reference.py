import numpy as np
from forward_euler import forward_euler
from SimpleSolver import SimpleSolver

# Function to general a golden reference and save as a binary .npz
def transient_ref_simplesolve(eval_f, x_start, p, eval_u, maxiter,
                              n_min=3, n_max=10, save_path="Reference.npz"):

    prev_solution = None
    last_n = None

    # Try different values of n until something goes wrong or reach max n
    for n in range(n_min, n_max+1):
        dt = 10.0 ** (-n)
        print(f"\nRunning Forward Euler with dt = 1e-{n} ...")

        X, t = SimpleSolver(eval_f, x_start, p, eval_u, int(maxiter/dt), dt, visualize=False)

        current_solution = X[:, -1]

        if prev_solution is not None:
            error = np.max(np.abs(current_solution - prev_solution))
            print("Error vs previous dt", error)
        else:
            print("No previous solution to compare to.")
            error = None

        if n == n_max:
            print("Reached n_max, stopping refinement.")
            reference_error = error
            reference_solution = current_solution
            reference_time = t
            last_n = n
            break

        prev_solution = current_solution
        prev_time = t

    # Save to npz
    np.savez_compressed(save_path,
                        ReferenceSolution=reference_solution,
                        ReferenceTime=reference_time,
                        ReferenceConfidence=reference_error,
                        ReferenceExponent=last_n)

    print("Reference saved in ", save_path)
    print("n = ", last_n)
    print("ReferenceConfidence = ", reference_error)

    return reference_solution, reference_time, reference_error
