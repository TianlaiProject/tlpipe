"""Allow the pickling of instance methods.

Importing this module allows for the pickling of bound instance methods.  This
is nessisary for calling methods of classes using multiprocessing.  The code
here is a recipe I stole from these sources:

http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-map
"""

import copyreg
import types

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

