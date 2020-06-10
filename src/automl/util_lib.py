import sys
import getpass 

import paramiko
from scp import SCPClient

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def getinfo():
    user=input("input user name:")
    password=getpass.getpass(prompt='Password: ', stream=None)
    return user,password

server='lnxlgn.fyi.sas.com'
port=22
#if __name__ == "__main__":
user,password = getinfo()

