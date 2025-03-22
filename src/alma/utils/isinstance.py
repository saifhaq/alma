import importlib


def is_instance_of(obj, class_path: str):
    """
    Check if `obj` is an instance of the class defined by the string `class_path`.

    Inputs:
    - obj (any): the object to check the type of.
    - class_path (str): the name of the class we want to verify for `obj`.

    Outputs:
    (bool): whether or not `obj` is an instance of the class defined by the string
    `class_path`.
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return isinstance(obj, cls)
