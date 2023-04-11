import numpy as np
import subprocess
import logging


def dist_regression(sample):
    data = np.zeros([int(sample['N'])])
    with open(sample['key_log'], 'r') as f:
        for line in f.readlines():
            last = ord(line.strip('\n')[-1])
            if last >= ord('A'):
                data[int(line.strip('\n')[:-1] + str(last - 65))] += 1
            else:
                data[int(line.strip('\n'))] += 1
    data = np.sort(np.squeeze(data[np.argwhere(data)]))[::-1]
    zipf_X = []
    zipf_Y = []
    for k, d in enumerate(data):
        x0 = np.log(k + 1)
        x1 = 1
        zipf_X.append([x0, x1])
        zipf_Y.append(np.log(d))
    zipf_X = np.array(zipf_X)
    zipf_Y = np.array(zipf_Y)
    alpha, c = np.linalg.lstsq(zipf_X, zipf_Y, rcond=-1)[0]
    alpha = -alpha
    return alpha, c


def generate_key_log(dist, skew, key_log, Q=200000, N=1e7):
    cmd = [
        'build/key_log',
        f'--entries {N}',
        f'--queries {Q}',
        f'--dist {dist}',
        f'--skew {skew}',
        f'--key-log-file {key_log}',
    ]

    cmd = ' '.join(cmd)
    logger = logging.getLogger("rlt_logger")
    logger.info(f'{cmd}')
    proc = subprocess.Popen(
        cmd,
        # stdin=None,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )
    timeout = 10 * 60 * 60
    _, _ = proc.communicate(timeout=timeout)
    return key_log
