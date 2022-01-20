import numpy as np
from datetime import timedelta


class DominoConverter:
    def __init__(self, min=None, max=None, fs=200):
        self.fs = fs
        self.min = min * self.fs if min else None
        self.max = max * self.fs if max else None

    def get_arousals(self, data):
        # Indices where one occurs
        indices = np.where(data == 1.0)[0]

        if indices.shape[0] == 0:
            return []

        # Find endings of arousal regions
        endings = [i for i in range(
            1, len(indices)) if ((indices[i] - 1 != indices[i - 1]) and ((indices[i] - indices[i - 1]) >= (3 * self.fs)))]

        # Find individual arousals
        arousals = []

        if endings:
            # Find individual arousals
            arousals = ([indices[: endings[0]]] + [indices[endings[i - 1]: endings[i]]
                                                   for i in range(1, len(endings))] + [indices[endings[-1]:]])
        else:
            arousals = [indices]

        if self.min:
            arousals = list(filter(lambda a: self.min <= len(a), arousals))

        if self.max:
            arousals = list(filter(lambda a: len(a) <= self.max, arousals))

        return arousals

    def get_number_arousals(self, data):
        return len(self.get_arousals(data))

    def write_anno(self, filename, data, start_time):

        arousals = self.get_arousals(data)

        # Write arousals down in file
        with open(filename, "w") as f:
            f.write("Signal ID: KorrelationMA\\MAK\n")

            st_str = start_time.strftime("%d.%m.%Y %H:%M:%S")
            f.write(f"Start Time: {st_str}\n\n")

            for a in arousals:
                start_delta = timedelta(seconds=a[0] / self.fs)
                end_delta = timedelta(seconds=a[-1] / self.fs)

                st = start_time + start_delta
                et = start_time + end_delta
                dur = end_delta - start_delta

                st_str = st.strftime("%d.%m.%Y %H:%M:%S")
                et_str = et.strftime("%H:%M:%S")

                f.write(f"{st_str},000-{et_str},000; {dur.seconds};0\n")

    def write_edf(self, filename, data, channel_names, fs=200, **kwargs):
        from pyedflib import highlevel as edf

        signal_headers = edf.make_signal_headers(
            channel_names,
            sample_rate=fs,
            physical_min=-32768,
            physical_max=32767)
        header = edf.make_header(**kwargs)
        edf.write_edf(filename, data, signal_headers, header)


if __name__ == "__main__":
    import h5py
    import scipy.io

    from datetime import datetime

    base_str = "training/tr04-0933/tr04-0933"
    X = scipy.io.loadmat(base_str + ".mat")["val"]
    A = np.array(h5py.File(base_str + "-arousal.mat", "r")["data"]["arousals"])
    A = np.transpose(A)

    indices = [0, 8, 10, 12]
    channel_names = ["F3-M2", "ABD", "AIRFLOW", "ECG", "Arousals"]

    d = DominoConverter()

    time = datetime.fromisoformat("2000-01-01T00:00:00")

    d.write_anno("john_doe.txt", A.squeeze(), time)

    data = np.concatenate((X[indices, :], A))
    d.write_edf(
        "john_doe.edf",
        data,
        channel_names,
        patientname="John Doe",
        gender="Male",
        startdate=time,
    )
