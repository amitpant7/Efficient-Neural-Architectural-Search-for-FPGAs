class ComputeLatencyIndex:
    """Predictor class that predicts the efficiency of architecture given the accuracy predictor,
    arthemetic intensity precictor and latency predictor. A new parameter is calculated and we call is compute_latency_index(cli)

    ref: [paper]

    """

    def __init__(
        self, accuracy_predictor, ai_predictor, latency_predictor, weights=[0.5, 0.5]
    ):
        self.ai = ai_predictor
        self.lat = latency_predictor
        self.acc = accuracy_predictor  # expects list of samples
        self.wts = weights

    def predict_efficiency(self, sample):  # Computes CLI of latency.
        arth_int = 1 / self.ai.predict_efficiency(
            sample
        )  # actualy returns 1/arth_intensity
        latency = self.lat.predict_efficiency(sample)
        acc = self.acc.predict_accuracy([sample]).item()

        cli = self.wts[1] * arth_int / latency + self.wts[0] * acc

        return 1 / cli  # inverse in order to make it the minimization problem
