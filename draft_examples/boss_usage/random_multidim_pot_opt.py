# A simple script checking boss's ability to find minima of functions in many dimensions.
import numpy as np
import sys
from boss.bo.bo_main import BOMain


class RandFunc:
    def __init__(self, ndims=1):
        self.ndims = ndims
        self.prop_coeffs = np.random.random((self.ndims,))
        self.min_vec = np.random.random((self.ndims,))

    def __call__(self, x):
        return np.mean((x - self.min_vec) ** 2 * self.prop_coeffs)


def main():
    np.random.seed(1)

    if len(sys.argv) > 1:
        ndims = int(sys.argv[1])
    else:
        ndims = 1

    rf = RandFunc(ndims=ndims)
    #    print("True result:")
    #    print(rf.min_vec)
    bounds = np.array([[0.0, 1.0] for _ in range(ndims)])
    iterpts = int(1.5 * ndims ** (1.5))  # recommended value from boss manual.
    bo = BOMain(
        rf, bounds, yrange=[0.0, ndims], kernel="rbf", initpts=5, iterpts=iterpts
    )
    res = bo.run()
    print("Predicted global min: ", res.select("mu_glmin", -1))
    print("Different of predicted minimum location and the true result:")
    pred_min_pos = res.select("x_glmin", -1)
    print("###", ndims, np.sum(np.abs(pred_min_pos - rf.min_vec)))


if __name__ == "__main__":
    main()
