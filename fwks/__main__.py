"""
Main subprogram of the FWKS module

install
service model.zip
"""

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", action="store", choices=['install', 'status'])
    parser.add_argument("args", nargs='*', action="store")
    args = parser.parse_args()
    if args.command == "install":
        import fwks.installer
        arguments = args.args
        if not arguments:
            print("fatal: No argument to the installer")
            quit()
        installer = fwks.installer.dependency_installer(arguments[0])
        installer()
    elif args.command == "service":
        pass  # to implement - launch a service from given module
    elif args.command == "status":
        import fwks.installer
        import fwks.meta
        status = fwks.installer.all_dependencies()
        status = {k: fwks.installer.is_installed(k) for k in status}
        todos = fwks.meta.ToDo.instances()
        print("Installation status:")
        for k, v in status:
            print("{}: {}".format(k, v))
        if todos:
            print("{} classes awaiting implementation:".format(len(todos)))
            for todo in todos:
                print(todo)
        else:
            print("No classes awaiting implementation")
