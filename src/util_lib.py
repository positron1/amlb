import sys
import getpass 

server='lnxlgn.fyi.sas.com'
port=22

def getinfo():
    user=input("input user name:")
    password=getpass.getpass(prompt='Password: ', stream=None)
    return user,password

#if __name__ == "__main__":
user,password = getinfo()

