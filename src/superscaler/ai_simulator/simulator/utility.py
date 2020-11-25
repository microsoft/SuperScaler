# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import humanreadable as hr
import bitmath


def transfer_rate_to_bps(rate_str):
    '''Get transfer rate based on the rate_str

    Args:
        rate_str: a string representing transfer rate,
                  e.g. '1Kibps' == '1024bps', '1Kbps' == '1000bps'
    Support suffix:
        "bps", "bit/s", "[kK]bps", "[kK]bit/s", "[kK]ibps", "[kK]ibit/s",
        "[mM]bps", "[mM]bit/s", "[mM]ibps", "[mM]ibit/s",
        "[gG]bps", "[gG]bit/s", "[gG]ibps", "[gG]ibit/s",
        "[tT]bps", "[tT]bit/s", "[tT]ibps", "[tT]ibit/s"
    '''
    # It is a little tricky here
    # In packet humanreadable, the hr.BitPerSecond(rate_str).bps is an int,
    # and the result is wrong if using this int value. Therefore, we need to
    # get an float value first (.kibi_bps), then manually convert it to bps
    return hr.BitPerSecond(rate_str).kibi_bps*1024


def data_size_to_bit(size_str):
    '''Get datasize based on the size_str

    Args:
        size_str: a string representing data size suffix,
                  e.g. '1Kib' == '1024b'
    Support suffix:
        'Bit', 'Byte', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB',
        'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB', 'Kib',
        'Mib', 'Gib', 'Tib', 'Pib', 'Eib', 'kb', 'Mb', 'Gb', 'Tb',
        'Pb', 'Eb', 'Zb', 'Yb'
    '''
    return bitmath.parse_string(size_str).to_Bit().value
