#! /usr/bin/env python

import os
import subprocess
from collections import defaultdict


def run_shell_cmd(cmd):
    p = subprocess.Popen(cmd,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out


def distribute_resources(deployment_setting,
                         local_resource_dir='/tmp/superscaler',
                         remote_resource_dir='/tmp/superscaler'):
    """
    this helper function helps to distribute resources to remote workers.

    @deployment_setting: a dict for specify how many workers over each host ip
    @local_resource_dir: generated resources by the master, it's organized in\
        the form of many global rank indexed folders in which contains all the\
        necessary per-process running resources
    @remote_working_dir: this is where the specified resource will be\
        delivered to on the remote host, it should be specified\
        in a full-path manner
    e.g., deployment_setting = {
        '10.0.0.21': 2,
    }
    """
    if not os.path.exists(local_resource_dir):
        raise Exception('local_resource_dir: %s is not existed!' %
                        (local_resource_dir))

    for ip in deployment_setting.keys():
        # TODO chgrp "/tmp/." failed: Operation not permitted
        run_shell_cmd('rsync -az %s %s:%s' %
                      (local_resource_dir, ip, remote_resource_dir))


def launch(rank2ip, rank2cmd, remote_wdir='/tmp'):
    """
    this helper function helps to launch cmds in a MPMD manner,\
        and it's assumed all the ssh passwdless connections are\
        built between workers before its use
    @rank2ip: a list of process ip, indexed by its global rank,\
        which means rank 0 will be assigned to the corresponding ip
    @rank2cmd:  a list of per process to-be-executed cmd
    """
    def parse_host_args(rank2ip):
        hosts_and_slots = defaultdict(int)
        for x in rank2ip:
            hosts_and_slots[x] += 1
        return hosts_and_slots

    hosts_and_slots = parse_host_args(rank2ip)

    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '-wdir {wdir} '
        '{cmds} '.format(
            wdir=remote_wdir,
            cmds=' : '.join('-np 1 -host {ip_slots} {cmd}'.format(
                ip_slots=ip + ':' + str(hosts_and_slots[ip]), cmd=(cmd))
                for ip, cmd in zip(rank2ip, rank2cmd))))
    subprocess.call(mpirun_command, shell=True)
